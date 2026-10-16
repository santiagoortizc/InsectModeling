"""Identificar un insecto plaga a partir de una foto."""

from urllib.parse import quote

import streamlit as st
from PIL import Image

from paginas import comun
from scripts import recorte
from scripts.fichas import nombreCientifico


def enlace(texto, url):
    return f"[{texto}]({quote(url, safe=':/#?&=')})"


def mostrarPasos(original, cuadrada, atencion, pasos):
    columnas = st.columns(3)
    columnas[0].image(recorte.conCaja(original, pasos["caja"]) if pasos else original,
                      caption="1 · Tu foto, con el insecto localizado")
    columnas[1].image(cuadrada, caption=f"2 · Recorte cuadrado a {recorte.LADO_SALIDA} px")
    columnas[2].image(atencion, caption="3 · En qué se fijó el modelo")

    if pasos is None:
        st.caption("Sin recorte automático: la foto entera se deformó a "
                   f"{recorte.LADO_SALIDA} px, que no es como se entrenó el modelo.")
        return
    if pasos["caja"] is None:
        st.warning("El detector no encontró ningún insecto, así que se usó el "
                   "cuadrado central de la foto. La predicción es poco fiable.")
        return

    x, y, lado = pasos["cuadrado"]
    escala = "ampliado" if pasos["ampliado"] else "reducido"
    st.caption(
        f"Insecto localizado con {pasos['confianza']:.0%} de confianza en una foto de "
        f"{pasos['ancho']}×{pasos['alto']} px. El cuadrado toma el lado mayor de la caja "
        f"más {recorte.MARGEN:.0%} de margen a cada lado: {lado} px desde ({x}, {y})"
        + (", desplazado para no salirse del borde" if pasos["desplazado"] else "")
        + f", y {escala} a {recorte.LADO_SALIDA} px.")


runId, _ = comun.modeloActual()

st.title("Identificar un insecto plaga")
st.write("Sube una foto y el modelo dirá de qué plaga del Tolima se trata.")

recortarAuto = st.checkbox(
    "Localizar el insecto antes de clasificar", value=True,
    help="Recorta un cuadrado alrededor del insecto, igual que en el corpus de "
         "entrenamiento. Sin esto el modelo ve la foto completa deformada.")

archivo = st.file_uploader("Foto del insecto", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.stop()

cuadrada, atencion, pasos, predicciones = comun.analizar(
    runId, archivo.getvalue(), recortarAuto)
clase, confianza = predicciones[0]

mostrarPasos(Image.open(archivo), cuadrada, atencion, pasos)

st.divider()
cientifico = nombreCientifico(clase)
st.subheader(f"*{cientifico}*")
st.progress(confianza, text=f"{confianza:.1%}")
busqueda = cientifico.removesuffix(" spp.")
st.caption(enlace("Ver en iNaturalist", f"https://www.inaturalist.org/taxa/search?q={busqueda}"))

st.divider()
st.caption("Las cinco clases más probables")
st.dataframe(
    {"clase": [nombreCientifico(c) for c, _ in predicciones],
     "confianza": [f"{p:.2%}" for _, p in predicciones]},
    hide_index=True, use_container_width=True,
)
