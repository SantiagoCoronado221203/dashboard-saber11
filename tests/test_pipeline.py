from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_archivos_y_esquema():
    data = pd.read_csv(ROOT / "data" / "saber11_preparado.csv", dtype={"periodo": str})
    pred = pd.read_csv(ROOT / "outputs" / "predicciones_prueba.csv")
    results = pd.read_csv(ROOT / "outputs" / "comparacion_modelos.csv")
    corte = json.loads((ROOT / "outputs" / "corte_bajo_puntaje.json").read_text(encoding="utf-8"))["corte_puntaje_global"]
    assert len(data) > 1000
    assert data.estudiante_id.is_unique
    assert set(data.bajo_puntaje.unique()) <= {0, 1}
    assert data.periodo.str[:5].ge("20142").all()           # solo periodos con puntaje global
    assert ((data.punt_global < corte) == (data.bajo_puntaje == 1)).all()
    assert 0.15 < data.bajo_puntaje.mean() < 0.35
    assert pred.probabilidad_bajo.between(0, 1).all()
    assert len(results) == 3
    assert results.roc_auc_prueba.between(0.5, 1).all()


def test_sin_fuga_de_puntajes():
    """Ningun puntaje del examen puede usarse como pista del modelo."""
    imp = pd.read_csv(ROOT / "outputs" / "importancia_variables.csv")
    assert not imp.variable.str.startswith("punt_").any()


def test_particion_sin_duplicados():
    data = pd.read_csv(ROOT / "data" / "saber11_preparado.csv")
    pred = pd.read_csv(ROOT / "outputs" / "predicciones_prueba.csv")
    assert pred.estudiante_id.is_unique
    assert abs(len(pred) - 0.25 * len(data)) <= 1
