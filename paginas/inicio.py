"""Presentación del proyecto."""

import streamlit as st

from paginas import comun
from scripts import modelRegistry as registro

eleccion = registro.readJson(registro.OUTPUTS_DIR / "student.json")
modelos, predeterminado = comun.catalogoModelos()
manifiesto = comun.manifiestoCorpus()
sitio, mejor = modelos[predeterminado], modelos[next(iter(modelos))]
destilado = next((r for r in modelos.values() if r["role"] == "destilado"), None)

st.title("🐛 Insectos plaga del Tolima")
st.subheader(f"Reconoce {manifiesto['num_clases']} plagas agrícolas a partir de una "
             f"fotografía", divider="green")

izquierda, derecha = st.columns([3, 1], vertical_alignment="center")
izquierda.write(
    "Un clasificador entrenado con fotografías de ciencia ciudadana de insectos que son "
    "plaga de cultivos en el departamento del Tolima. Sube una foto y el prototipo "
    "localiza el insecto, propone una especie y muestra en qué se fijó para decidirlo."
)
derecha.page_link("paginas/clasificar.py", label="**Probar con una foto**", icon="🐛")

columnas = st.columns(4)
columnas[0].metric("Clases", manifiesto["num_clases"])
columnas[1].metric("Imágenes", f"{manifiesto['conteos']['total']:,}".replace(",", " "))
columnas[2].metric("Modelos entrenados", len(modelos))
columnas[3].metric("Mejor acierto", f"{mejor['test_accuracy']:.1%}")

st.subheader("La pregunta", divider="green")
izquierda, derecha = st.columns([2, 3])
izquierda.write(
    f"Un modelo que quepa en un teléfono o en un navegador tiene que ser pequeño, y lo "
    f"pequeño suele costar precisión. El proyecto mide cuánto cuesta, y si la destilación "
    f"de conocimiento —entrenar al modelo chico imitando a uno grande— cierra esa "
    f"distancia.\n\n"
    f"El mejor modelo, **{mejor['architecture']}**, acierta el "
    f"{mejor['test_accuracy']:.1%}; el que sirve este sitio, "
    f"**{sitio['architecture']}**, acierta el {sitio['test_accuracy']:.1%} pesando "
    f"{eleccion['pesos_MB']} MB, dentro del presupuesto de "
    f"{eleccion['presupuesto_MB']:.0f} MB. Son "
    f"{100 * (mejor['test_accuracy'] - sitio['test_accuracy']):.1f} puntos."
)
if destilado:
    izquierda.write(
        f"La versión destilada del modelo pequeño quedó en "
        f"{destilado['test_accuracy']:.1%}, frente al {sitio['test_accuracy']:.1%} "
        f"del mismo modelo entrenado de la forma corriente."
    )
derecha.bar_chart(
    [{"modelo": f"{r['architecture']}"
               f"{' destilado' if r['role'] == 'destilado' else ''}",
      "acierto": r["test_accuracy"]} for r in modelos.values()],
    x="modelo", y="acierto", horizontal=True, height=320,
    x_label="acierto en prueba", y_label="",
)

st.subheader("Cómo funciona", divider="green")
pasos = (
    ("🔍 Localizar", "Un detector encuentra el insecto y recorta un cuadrado a su "
                     "alrededor, igual que se construyó el corpus. Sin ese paso el modelo "
                     "vería la foto completa, deformada, y acertaría menos."),
    ("🧠 Clasificar", "Una red preentrenada en ImageNet, con la cabeza reemplazada y "
                      "reentrenada sobre las plagas del Tolima, reparte su confianza "
                      f"entre las {manifiesto['num_clases']}."),
    ("🔥 Explicar", "Grad-CAM colorea las zonas que más pesaron en la decisión. Sirve "
                    "para distinguir un acierto por el insecto de uno por la hoja o el "
                    "fondo."),
)
for columna, (titulo, cuerpo) in zip(st.columns(3), pasos):
    with columna.container(border=True):
        st.markdown(f"#### {titulo}")
        st.write(cuerpo)

st.subheader("Explora", divider="green")
izquierda, centro, derecha = st.columns(3)
izquierda.page_link("paginas/clasificar.py", label="**Identificar** · sube una foto",
                    icon="🐛")
centro.page_link("paginas/modelo.py", label="**El modelo** · capa por capa", icon="🧠")
derecha.page_link("paginas/corpus.py", label="**El dataset** · de dónde salió", icon="📚")
st.caption(f"El selector de la barra lateral cambia entre los {len(modelos)} modelos "
           f"entrenados sobre este mismo corpus.")

st.divider()
st.caption(
    "Las fotografías provienen de [iNaturalist](https://www.inaturalist.org) bajo "
    "licencias Creative Commons abiertas. El entrenamiento y la evaluación están en "
    "[InsectModeling](https://github.com/santiagoortizc/InsectModeling), adaptado de "
    "[ButterflyModeling](https://github.com/CCSandoval/ButterflyModeling)."
)
