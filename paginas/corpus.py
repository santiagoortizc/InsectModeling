"""De dónde salieron las imágenes con que se entrenó."""

import streamlit as st

from paginas import comun

MOTIVOS = {
    "sobreexpuesta": "Sobreexpuesta",
    "recorte_pequeno": "Insecto muy pequeño en la foto",
    "desenfocada": "Desenfocada",
    "casi_duplicado": "Casi idéntica a otra",
    "sin_deteccion": "No se encontró insecto",
    "sin_insecto_visible": "Hormiguero o suelo sin hormiga visible",
    "subexpuesta": "Demasiado oscura",
    "especie_escasa": "Especie con muy pocas fotos",
}

manifiesto = comun.manifiestoCorpus()
conteos, parametros = manifiesto["conteos"], manifiesto["parametros"]
umbrales = parametros["umbrales"]
porEspecie = {especie: sum(len(v) for v in splits.values())
              for especie, splits in manifiesto["reparto"].items()}

st.title("El conjunto de datos")
st.write("Fotos de iNaturalist con licencia abierta, recortadas alrededor del insecto.")

columnas = st.columns(4)
columnas[0].metric("Clases", manifiesto["num_clases"])
columnas[1].metric("Imágenes", f"{conteos['total']:,}".replace(",", " "))
columnas[2].metric("Por clase", f"{min(porEspecie.values())}–{max(porEspecie.values())}")
columnas[3].metric("Tamaño", f"{parametros['lado_salida']} px")

if parametros.get("fusiones"):
    st.caption("Clases que agrupan especies que no se distinguen por foto: " + ", ".join(
        f"{c} ← {', '.join(sorted(e for e, d in parametros['fusiones'].items() if d == c))}"
        for c in sorted(set(parametros["fusiones"].values()))) + ".")

st.caption(f"Se reparten en {conteos['train']} de entrenamiento, {conteos['test']} de "
           f"prueba y {conteos['validate']} de validación, con semilla "
           f"{parametros['semilla']}. Corpus del {manifiesto['creado'][:10]}.")

st.divider()
st.subheader("Imágenes por clase")
st.bar_chart([{"clase": e, "imágenes": n} for e, n in
              sorted(porEspecie.items(), key=lambda par: -par[1])],
             x="clase", y="imágenes", height=260,
             x_label="clase", y_label="imágenes")

st.divider()
st.subheader("Qué se descartó")
st.caption(f"Pasan el filtro las fotos con nitidez sobre {umbrales['nitidez']:.0f}, un "
           f"recorte de al menos {umbrales['ladoOriginal']} px, menos de "
           f"{umbrales['fracSaturados']:.0%} de píxeles quemados y "
           f"{umbrales['fracNegros']:.0%} en negro.")

if st.checkbox("Ver el detalle", help="Baja 5 MB de auditoría"):
    motivos = comun.motivosCorpus()
    total, aceptadas = sum(motivos.values()), motivos["aceptada"]
    st.caption(f"De {total} fotos candidatas se aceptaron {aceptadas}, "
               f"el {aceptadas / total:.0%}.")
    st.bar_chart([{"motivo": MOTIVOS.get(k, k), "imágenes": v}
                  for k, v in sorted(motivos.items(), key=lambda par: -par[1])
                  if k != "aceptada"],
                 x="motivo", y="imágenes", horizontal=True, height=260,
                 x_label="imágenes", y_label="")
