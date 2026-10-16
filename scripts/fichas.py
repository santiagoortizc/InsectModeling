"""Datos del corpus: del repo hermano en local y del repo público al desplegar."""

import csv
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

CRUDO = "https://raw.githubusercontent.com/santiagoortizc/InsectDataset/main"
ROOT_DIR = Path(__file__).resolve().parent.parent


def _repoLocal():
    with open(ROOT_DIR / "corpus.json", encoding="utf-8") as handle:
        ruta = (ROOT_DIR / json.load(handle)["dataset_repo"]).resolve()
    return ruta if (ruta / "manifiesto.json").exists() else None


def _texto(nombre):
    local = _repoLocal()
    if local:
        return (local / nombre).read_text(encoding="utf-8")
    with urllib.request.urlopen(f"{CRUDO}/{nombre}") as respuesta:
        return respuesta.read().decode("utf-8")


def manifiesto():
    return json.loads(_texto("manifiesto.json"))


def motivos():
    """Veredictos de la auditoría. Es pesada: pedirla solo cuando se muestre."""
    filas = csv.DictReader(io.StringIO(_texto("auditoria.csv")))
    return Counter(fila["motivo"] for fila in filas)


def nombreCientifico(clase):
    """`abracris_flavolineata` -> `Abracris flavolineata`; `atta_spp` -> `Atta spp.`"""
    genero, _, resto = clase.partition("_")
    resto = resto.replace("_", " ")
    return f"{genero.capitalize()} {'spp.' if resto == 'spp' else resto}".strip()
