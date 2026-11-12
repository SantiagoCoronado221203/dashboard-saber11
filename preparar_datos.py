"""Limpia la muestra de Saber 11 y crea la variable objetivo `bajo_puntaje`.

Entrada : data/muestra_saber11.csv  (muestra aleatoria del archivo "Resultados unicos Saber 11")
Salida  : data/saber11_preparado.csv y outputs/corte_bajo_puntaje.json
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / "data" / "muestra_saber11.csv"
SALIDA = ROOT / "data" / "saber11_preparado.csv"
CORTE = ROOT / "outputs" / "corte_bajo_puntaje.json"
PERCENTIL_CORTE = 0.25  # el 25% de puntajes mas bajos se marca como "bajo puntaje"

# Niveles educativos ordenados de menor a mayor (0 = ninguno, 9 = postgrado)
EDUCACION = {
    "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9,
}
# "No sabe" y "No Aplica" quedan como dato faltante.

# Personas en el hogar: el ICFES cambio de rangos a numeros en algunos periodos.
PERSONAS = {
    "Una": 1, "1 a 2": 1.5, "Dos": 2, "Tres": 3, "3 a 4": 3.5, "Cuatro": 4, "Cinco": 5,
    "5 a 6": 5.5, "Seis": 6, "Siete": 7, "7 a 8": 7.5, "Ocho": 8, "Nueve": 9, "9 o más": 9,
    "Diez": 10, "Once": 11, "Doce o más": 12,
}
CUARTOS = {
    "Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5, "Seis": 6, "Seis o mas": 6,
    "Siete": 7, "Ocho": 8, "Nueve": 9, "Diez o más": 10,
}
SI_NO = {"Si": "Si", "No": "No", "S": "Si", "N": "No"}


def main():
    raw = pd.read_csv(ENTRADA, dtype=str)
    d = raw.copy()

    # 1. Solo periodos con puntaje global (desde 2014-2), estudiantes y resultados publicados.
    d["PUNT_GLOBAL"] = pd.to_numeric(d["PUNT_GLOBAL"], errors="coerce")
    d = d[d["PUNT_GLOBAL"].notna() & (d["PUNT_GLOBAL"] > 0)]
    d = d[d["ESTU_ESTADOINVESTIGACION"].eq("PUBLICAR")]
    d = d.drop_duplicates("ESTU_CONSECUTIVO")

    out = pd.DataFrame({
        "estudiante_id": d["ESTU_CONSECUTIVO"],
        "periodo": d["PERIODO"],
        "anio": d["PERIODO"].str[:4].astype(int),
        "departamento": d["COLE_DEPTO_UBICACION"].str.strip().str.upper().replace({"BOGOTÁ": "BOGOTA"}),
        "naturaleza": d["COLE_NATURALEZA"].str.title(),
        "area": d["COLE_AREA_UBICACION"].str.title(),
        "jornada": d["COLE_JORNADA"].str.title(),
        "calendario": d["COLE_CALENDARIO"],
        "bilingue": d["COLE_BILINGUE"].map(SI_NO),
        "caracter": d["COLE_CARACTER"].str.title(),
        "genero_colegio": d["COLE_GENERO"].str.title(),
        "genero": d["ESTU_GENERO"].map({"F": "Femenino", "M": "Masculino"}),
    })

    # Edad al presentar la prueba
    nacimiento = pd.to_datetime(d["ESTU_FECHANACIMIENTO"], format="%d/%m/%Y", errors="coerce")
    edad = out["anio"] - nacimiento.dt.year
    out["edad"] = edad.where(edad.between(13, 60))

    # Estrato: "Sin Estrato" = 0, Estrato 1..6
    out["estrato"] = (d["FAMI_ESTRATOVIVIENDA"].str.extract(r"(\d)")[0].astype(float)
                      .where(d["FAMI_ESTRATOVIVIENDA"].ne("Sin Estrato"), 0))
    out["educacion_madre"] = d["FAMI_EDUCACIONMADRE"].map(EDUCACION)
    out["educacion_padre"] = d["FAMI_EDUCACIONPADRE"].map(EDUCACION)
    out["personas_hogar"] = d["FAMI_PERSONASHOGAR"].map(PERSONAS)
    out["cuartos_hogar"] = d["FAMI_CUARTOSHOGAR"].map(CUARTOS)
    for col, fuente in [("tiene_internet", "FAMI_TIENEINTERNET"), ("tiene_computador", "FAMI_TIENECOMPUTADOR"),
                        ("tiene_automovil", "FAMI_TIENEAUTOMOVIL"), ("tiene_lavadora", "FAMI_TIENELAVADORA")]:
        out[col] = d[fuente].map(SI_NO)

    # Puntajes: se guardan para mostrarlos, pero NO se usan como pistas del modelo.
    out["punt_global"] = d["PUNT_GLOBAL"].astype(int)
    out["punt_matematicas"] = pd.to_numeric(d["PUNT_MATEMATICAS"], errors="coerce")
    out["punt_lectura_critica"] = pd.to_numeric(d["PUNT_LECTURA_CRITICA"], errors="coerce")

    # 2. Variable objetivo
    corte = float(out["punt_global"].quantile(PERCENTIL_CORTE))
    out["bajo_puntaje"] = (out["punt_global"] < corte).astype(int)

    # Los textos faltantes se marcan como "Sin dato" para que se vean en los filtros.
    texto = ["departamento", "naturaleza", "area", "jornada", "calendario", "bilingue", "caracter",
             "genero_colegio", "genero", "tiene_internet", "tiene_computador", "tiene_automovil", "tiene_lavadora"]
    out[texto] = out[texto].fillna("Sin dato")

    out = out.reset_index(drop=True)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    CORTE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SALIDA, index=False, encoding="utf-8")
    CORTE.write_text(json.dumps({"percentil": PERCENTIL_CORTE, "corte_puntaje_global": corte}, indent=2), encoding="utf-8")

    print(f"Filas en la muestra original: {len(raw):,}")
    print(f"Filas utiles (2014-2 en adelante, publicadas): {len(out):,}")
    print(f"Corte de bajo puntaje: puntaje global menor a {corte:.0f}")
    print(f"Proporcion con bajo puntaje: {out['bajo_puntaje'].mean():.1%}")


if __name__ == "__main__":
    main()
