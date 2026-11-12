"""Saca una muestra aleatoria de ~60.000 filas del CSV grande de Saber 11.

Uso (con la caja .venv activada):
    python sacar_muestra.py "C:\\ruta\\al\\archivo_grande.csv"

Si no le pasas la ruta, el script te la pide.
"""
import sys
from itertools import chain
from pathlib import Path

import pandas as pd

FILAS_OBJETIVO = 60_000        # tamano final de la muestra
PORCENTAJE_POR_PEDAZO = 0.015  # se toma 1,5% de cada pedazo (luego se recorta a FILAS_OBJETIVO)
TAMANO_PEDAZO = 200_000        # filas que se leen en memoria cada vez
SEMILLA = 42                   # para que la muestra sea siempre la misma


def abrir(ruta):
    """Prueba utf-8 y, si falla, latin-1. Devuelve el primer pedazo y el resto."""
    for codificacion in ("utf-8", "latin-1"):
        try:
            lector = pd.read_csv(ruta, dtype=str, chunksize=TAMANO_PEDAZO, encoding=codificacion)
            primer = next(lector)
            return codificacion, chain([primer], lector)
        except UnicodeDecodeError:
            continue
    raise SystemExit("No pude leer el archivo con codificacion utf-8 ni latin-1.")


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else input("Arrastra aqui el CSV grande y presiona Enter: ")
    ruta = Path(ruta.strip().strip('"').strip("'"))
    if not ruta.exists():
        raise SystemExit(f"No encuentro el archivo: {ruta}")

    codificacion, pedazos = abrir(ruta)
    print(f"Leyendo {ruta.name} (codificacion {codificacion})... esto puede tardar unos minutos.")

    partes, leidas = [], 0
    for i, pedazo in enumerate(pedazos, start=1):
        leidas += len(pedazo)
        partes.append(pedazo.sample(frac=PORCENTAJE_POR_PEDAZO, random_state=SEMILLA + i))
        print(f"  {leidas:,} filas leidas", end="\r")

    muestra = pd.concat(partes, ignore_index=True)
    if len(muestra) > FILAS_OBJETIVO:
        muestra = muestra.sample(FILAS_OBJETIVO, random_state=SEMILLA)

    salida = ruta.with_name("muestra_saber11.csv")
    muestra.to_csv(salida, index=False, encoding="utf-8")
    print(f"\nListo: {leidas:,} filas leidas en total, muestra de {len(muestra):,} filas.")
    print(f"Guardada en: {salida}")


if __name__ == "__main__":
    main()
