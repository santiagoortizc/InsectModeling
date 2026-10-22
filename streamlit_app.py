"""Prototipo de clasificación de insectos plaga del Tolima."""

import streamlit as st
import torchvision  # noqa: F401  antes que TensorFlow: si triton carga después, choca con su LLVM y el proceso se cae

from paginas import comun

st.set_page_config(page_title="Insectos plaga del Tolima", page_icon="🐛", layout="wide")

comun.selectorModelo()

st.navigation([
    st.Page("paginas/inicio.py", title="El proyecto", icon="🏠", default=True),
    st.Page("paginas/clasificar.py", title="Identificar", icon="🐛"),
    st.Page("paginas/modelo.py", title="Métricas del modelo", icon="🧠"),
    st.Page("paginas/corpus.py", title="Métricas del dataset", icon="📚"),
]).run()
