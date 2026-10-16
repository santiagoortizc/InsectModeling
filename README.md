# InsectModeling

Clasificación de insectos plaga agrícola del Tolima con destilación de
conocimiento. La pregunta es si un modelo lo bastante pequeño para correr en
un dispositivo liviano, cuantizado a int8, puede acercarse a uno grande, y cuánto de esa distancia la cierra
destilar en vez de simplemente entrenar mejor.

Este repositorio entrena y evalúa; el dataset lo produce el repositorio hermano
`DatasetInsectos`. Es una adaptación de
[ButterflyModeling](https://github.com/CCSandoval/ButterflyModeling), que hace
lo mismo con mariposas.

## El dataset

35 clases de plagas observadas en el Tolima, 13 625 imágenes recortadas a 448 px
(9 554 train / 2 724 test / 1 347 validate). *Atta cephalotes* y *A. colombica*
van fusionadas en `atta_spp` porque no se distinguen por foto. El reparto está
congelado en el `manifiesto.json` del repositorio hermano y se hace por
observación, así que fotos del mismo individuo no cruzan de train a evaluación.
Todos los runs comparten ese reparto: las diferencias entre ellos son del modelo
y no de los datos.

`corpus.json` apunta al repositorio hermano (`../DatasetInsectos`).
`scripts/corpus.py:materializar()` crea `dataset/<split>/<clase>/` con symlinks a
`procesado/`.

## El experimento

Tres fases, cada una en su carpeta de `notebooks/`:

- **`01_student`** — cuatro candidatos a estudiante: MobileNetV3Small,
  MobileNetV2, MobileNetV3Large y EfficientNetB0. Cada uno se convierte a
  TFLite con cuantización entera completa y se elige el mejor macro-F1 **int8**
  dentro de `PRESUPUESTO_MB` (8 MB). Cuantizar no le cuesta lo mismo a cada
  arquitectura: las MobileNetV3 pierden mucho más que MobileNetV2.
- **`02_teacher`** — cuatro candidatos a docente, todos claramente más grandes
  que cualquier estudiante: EfficientNetB3, ConvNeXtTiny, ResNet50V2 y
  DenseNet121. Se elige por macro-F1, sin mirar costo.
- **`03_destilacion`** — cuatro variantes que transfieren del docente al
  estudiante: respuesta con y sin descongelado, features y atención.

Cierran `04_comparacion_final` y `05_interpretabilidad_gradcam`. El `00_eda`
describe el dataset y el `06`/`07` son diagnósticos heredados de la plantilla.

La selección en cada fase usa validación y reporta test, y los runs se filtran
por dataset para que no se mezclen cifras de dos versiones del corpus.

## Entrenar en Kaggle

1. Subir como dataset de Kaggle `manifiesto.json` y `procesado/` de
   `DatasetInsectos`.
2. En cada notebook: adjuntar ese dataset, activar GPU y definir el secreto
   `GITHUB_TOKEN` (con permiso de escritura sobre este repo).
3. La primera celda clona este repo; al terminar, `scripts/kaggle.py` publica el
   run y los pesos (como release de GitHub).

Cada comparación mira solo los runs de su rol: una arquitectura entrenada como
docente nunca compite como estudiante.

## Cuantización

`scripts/cuantizacion.py` convierte un run a `model_int8.tflite`: entrada uint8
con los píxeles crudos (el preprocess va dentro), activaciones calibradas con
200 imágenes de train y salida en float32. Si XNNPACK no acepta el modelo se
evalúa con los kernels de referencia y la latencia queda marcada con `motor`.

## Resultados

Cada entrenamiento deja su `run.json`, `metrics.json`, `history.json` y figuras
en `outputs/runs/<run_id>/`, con los pesos en un release de GitHub. El estado
al día está en `outputs/runs/registry.json`.

## App

`streamlit run streamlit_app.py`. Necesita `outputs/student.json`, que escribe
`01_student/05_comparacion_estudiante` al elegir el estudiante, y el detector
horneado: correr una vez `from scripts import recorte; recorte.hornear()` con
`yolov8s-worldv2.pt` en la raíz.

## Pendiente

- Fichas de especie (nombre común, familia, imagen de referencia): la app
  todavía muestra solo el nombre científico.
