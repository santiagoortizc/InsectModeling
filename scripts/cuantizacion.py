"""Cuantización entera completa a TFLite, para dispositivos livianos.

El modelo convertido recibe píxeles uint8 crudos (0-255) y lleva el
preprocess_input de su arquitectura por dentro, así que el dispositivo no
necesita saber qué arquitectura es. La salida queda en float32: con softmax
en uint8 las probabilidades se redondean a 1/256 y el top-3 empata.
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts import evalStats
from scripts import modelRegistry as registro
from scripts.architectures import IMG_SIZE, preprocessFn
from scripts.dataPipeline import SEMILLA, loadSplit

ARCHIVO = "model_int8.tflite"
MUESTRAS_CALIBRACION = 200
HILOS_EVALUACION = 4


def convertir(modelo, arquitectura, dirCalibracion, destino):
    """Calibra las activaciones con `MUESTRAS_CALIBRACION` imágenes de train."""
    alto, ancho = modelo.input_shape[1:3]
    entrada = tf.keras.Input((alto, ancho, 3))
    preprocesado = tf.keras.layers.Lambda(preprocessFn(arquitectura))(entrada)
    envoltorio = tf.keras.Model(entrada, modelo(preprocesado))

    calibracion, _ = loadSplit(dirCalibracion, (alto, ancho), 1, shuffle=True)

    def representativo():
        for x, _ in calibracion.take(MUESTRAS_CALIBRACION):
            yield [tf.cast(x, tf.float32)]

    # from_keras_model y no from_concrete_functions: esa vía deja las variables
    # de Keras 3 sin congelar y la calibración falla en READ_VARIABLE
    conversor = tf.lite.TFLiteConverter.from_keras_model(envoltorio)
    conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    conversor.representative_dataset = representativo
    conversor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conversor.inference_input_type = tf.uint8

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(conversor.convert())
    return destino


def convertirRun(runId, dirCalibracion="dataset/train"):
    """Cacheado en la carpeta del run: convertir cuesta minutos."""
    destino = registro.runDir(runId) / ARCHIVO
    if destino.exists():
        return destino
    tf.keras.utils.set_random_seed(SEMILLA)
    arquitectura = registro.loadRunConfig(runId)["architecture"]
    return convertir(registro.modeloInferencia(runId), arquitectura, dirCalibracion, destino)


def tamanoMB(ruta):
    return Path(ruta).stat().st_size / 1e6


def interprete(ruta, hilos):
    """XNNPACK si acepta el modelo; si no, los kernels de referencia, que son
    más lentos. Devuelve también cuál se usó: sus latencias no se comparan."""
    try:
        modelo = tf.lite.Interpreter(model_path=str(ruta), num_threads=hilos)
        modelo.allocate_tensors()
        return modelo, "xnnpack"
    except RuntimeError:
        sinDelegado = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        modelo = tf.lite.Interpreter(model_path=str(ruta), num_threads=hilos,
                                     experimental_op_resolver_type=sinDelegado)
        modelo.allocate_tensors()
        return modelo, "referencia"


def cuantizarEntrada(entrada, x):
    escala, cero = entrada["quantization"]
    if not escala:
        return x.astype(entrada["dtype"])
    info = np.iinfo(entrada["dtype"])
    return np.clip(np.round(x / escala + cero), info.min, info.max).astype(entrada["dtype"])


def predecir(modelo, lote):
    entrada = modelo.get_input_details()[0]
    salida = modelo.get_output_details()[0]
    modelo.set_tensor(entrada["index"], cuantizarEntrada(entrada, lote))
    modelo.invoke()
    return modelo.get_tensor(salida["index"])


def evaluar(ruta, directorio, classNames):
    """Mismas métricas que evalStats.evaluar, sobre el modelo cuantizado."""
    modelo, _ = interprete(ruta, HILOS_EVALUACION)
    ds, clases = loadSplit(directorio, IMG_SIZE, 1, shuffle=False)
    if clases != classNames:
        raise ValueError("Las clases del directorio no coinciden con las del run")

    probabilidades, yReal = [], []
    for x, y in ds:
        probabilidades.append(predecir(modelo, x.numpy().astype(np.float32))[0])
        yReal.append(int(np.argmax(y.numpy()[0])))
    return evalStats.metricas(np.array(yReal), np.array(probabilidades), classNames)


def medirLatencia(ruta, hilos=1, repeticiones=30, calentamiento=3):
    """Mediana y desviación en ms con un hilo, y el motor usado. Es lo más
    parecido a un dispositivo liviano que se puede medir en Kaggle, aunque una
    CPU ARM será más lenta."""
    modelo, motor = interprete(ruta, hilos)
    alto, ancho = modelo.get_input_details()[0]["shape"][1:3]
    lote = np.random.default_rng(SEMILLA).uniform(0, 255, (1, alto, ancho, 3)).astype(np.float32)

    for _ in range(calentamiento):
        predecir(modelo, lote)
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        predecir(modelo, lote)
        tiempos.append((time.perf_counter() - inicio) * 1000)
    return float(np.median(tiempos)), float(np.std(tiempos)), motor
