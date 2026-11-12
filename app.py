"""Dashboard interactivo: alerta temprana de bajo puntaje en Saber 11."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(ROOT / "data" / "saber11_preparado.csv", dtype={"periodo": str})
PRED = pd.read_csv(ROOT / "outputs" / "predicciones_prueba.csv", dtype={"periodo": str})
MODELS = pd.read_csv(ROOT / "outputs" / "comparacion_modelos.csv")
IMPORTANCE = pd.read_csv(ROOT / "outputs" / "importancia_variables.csv").head(12)
CORTE = json.loads((ROOT / "outputs" / "corte_bajo_puntaje.json").read_text(encoding="utf-8"))["corte_puntaje_global"]
COLORS = {"navy": "#12304A", "blue": "#1976D2", "cyan": "#26A6A1", "gold": "#F4B942", "red": "#D95D5D", "paper": "#F4F7FA"}
ANIOS = sorted(int(a) for a in DATA.anio.unique())
EDUCACION_TXT = {0: "Ninguno", 1: "Primaria incompleta", 2: "Primaria completa", 3: "Secundaria incompleta",
                 4: "Secundaria completa", 5: "Tecnica incompleta", 6: "Tecnica completa",
                 7: "Profesional incompleta", 8: "Profesional completa", 9: "Postgrado"}
NOMBRES = {"departamento": "Departamento", "jornada": "Jornada", "edad": "Edad", "anio": "Año", "cuartos_hogar": "Cuartos del hogar",
           "area": "Area (urbana/rural)", "educacion_madre": "Educacion madre", "personas_por_cuarto": "Personas por cuarto",
           "educacion_padre": "Educacion padre", "genero_colegio": "Genero del colegio", "calendario": "Calendario",
           "estrato": "Estrato", "naturaleza": "Naturaleza colegio", "tiene_computador": "Tiene computador",
           "tiene_internet": "Tiene internet", "genero": "Genero", "educacion_max_padres": "Educacion max. padres",
           "bilingue": "Colegio bilingue", "personas_hogar": "Personas en el hogar", "tiene_automovil": "Tiene automovil",
           "tiene_lavadora": "Tiene lavadora", "bienes_hogar": "Bienes del hogar (0-4)", "caracter": "Caracter del colegio"}

app = Dash(__name__, title="Alerta temprana Saber 11", suppress_callback_exceptions=True)
server = app.server


def card(title, value_id, note):
    return html.Div([html.P(title, className="kpi-title"), html.H2(id=value_id), html.P(note, className="kpi-note")], className="kpi-card")


app.layout = html.Div([
    html.Header([
        html.Div([html.P("PRUEBAS SABER 11 · ICFES · DATOS ABIERTOS", className="eyebrow"),
                  html.H1("Alerta temprana de bajo puntaje en Saber 11"),
                  html.P(f"Exploracion, desempeño predictivo y factores asociados. 'Bajo puntaje' = puntaje global menor a {CORTE:.0f} "
                         "(el 25% mas bajo de la muestra). Muestra aleatoria de resultados 2014-2 a 2022.", className="subtitle")]),
        html.Div("DASH + PLOTLY", className="badge"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H3("Filtros interactivos"),
            html.Label("Departamento del colegio"),
            dcc.Dropdown(sorted(DATA.departamento.unique()), multi=True, id="departamento", placeholder="Todos los departamentos"),
            html.Label("Naturaleza del colegio"),
            dcc.Dropdown(sorted(DATA.naturaleza.unique()), multi=True, id="naturaleza", placeholder="Oficial y no oficial"),
            html.Label("Area"),
            dcc.Dropdown(sorted(DATA.area.unique()), multi=True, id="area", placeholder="Urbana y rural"),
            html.Label("Años"),
            dcc.RangeSlider(ANIOS[0], ANIOS[-1], 1, value=[ANIOS[0], ANIOS[-1]], marks={a: str(a)[2:] for a in ANIOS}, id="anios"),
            html.Hr(),
            html.Label("Umbral de clasificacion"),
            dcc.Slider(0.20, 0.80, 0.05, value=0.50, marks={x: f"{x:.1f}" for x in [0.2, 0.4, 0.6, 0.8]}, id="umbral"),
            html.P("El umbral solo cambia la decision Alto/No alto y las metricas; no reentrena el modelo.", className="help"),
            html.Button("Restablecer filtros", id="reset", n_clicks=0),
        ], className="sidebar"),
        html.Main([
            html.Div([card("Estudiantes", "kpi-n", "poblacion filtrada"), card("Bajo puntaje observado", "kpi-rate", f"puntaje global < {CORTE:.0f}"),
                      card("Puntaje global medio", "kpi-att", "de 0 a 500"), card("Riesgo alto", "kpi-high", "en muestra de prueba")], className="kpi-grid"),
            dcc.Tabs([
                dcc.Tab(label="Panorama", children=[
                    html.Div([dcc.Graph(id="depto-chart"), dcc.Graph(id="estrato-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="anio-chart"), dcc.Graph(id="educ-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="bienes-chart"), dcc.Graph(id="risk-chart")], className="grid-2"),
                ]),
                dcc.Tab(label="Rendimiento del modelo", children=[
                    html.Div(id="metrics-strip", className="metric-strip"),
                    html.Div([dcc.Graph(id="confusion-chart"), dcc.Graph(id="model-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="importance-chart"), html.Div(id="narrative", className="narrative")], className="grid-2"),
                ]),
                dcc.Tab(label="Casos de mayor riesgo", children=[
                    html.P("Estudiantes del conjunto de prueba ordenados por probabilidad estimada de bajo puntaje. "
                           "El identificador es el consecutivo anonimo del ICFES; no hay nombres ni documentos."),
                    dash_table.DataTable(id="risk-table", page_size=12, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px"},
                        style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"},
                        style_data_conditional=[{"if": {"filter_query": "{nivel_riesgo} = Alto"}, "backgroundColor": "#FCE8E8"}]),
                ]),
            ]),
            html.Footer("Fuente: ICFES, 'Resultados unicos Saber 11' (datos.gov.co), muestra aleatoria. "
                        "Evaluacion en conjunto de prueba estratificado del 25%."),
        ], className="content")
    ], className="shell")
], className="app")


@app.callback(
    Output("departamento", "value"), Output("naturaleza", "value"), Output("area", "value"), Output("anios", "value"), Output("umbral", "value"),
    Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [ANIOS[0], ANIOS[-1]], 0.50


def tasa(d, col):
    return d.groupby(col, as_index=False).agg(tasa=("bajo_puntaje", "mean"), estudiantes=("estudiante_id", "count"))


@app.callback(
    Output("kpi-n", "children"), Output("kpi-rate", "children"), Output("kpi-att", "children"), Output("kpi-high", "children"),
    Output("depto-chart", "figure"), Output("estrato-chart", "figure"), Output("anio-chart", "figure"), Output("educ-chart", "figure"),
    Output("bienes-chart", "figure"), Output("risk-chart", "figure"),
    Output("metrics-strip", "children"), Output("confusion-chart", "figure"), Output("model-chart", "figure"), Output("importance-chart", "figure"),
    Output("narrative", "children"), Output("risk-table", "data"), Output("risk-table", "columns"),
    Input("departamento", "value"), Input("naturaleza", "value"), Input("area", "value"), Input("anios", "value"), Input("umbral", "value"))
def update(deptos, naturalezas, areas, anios, umbral):
    def filtrado(df):
        d = df
        if deptos: d = d[d.departamento.isin(deptos)]
        if naturalezas: d = d[d.naturaleza.isin(naturalezas)]
        if areas: d = d[d.area.isin(areas)]
        return d[d.anio.between(anios[0], anios[1])]
    d, p = filtrado(DATA), filtrado(PRED)
    template = "plotly_white"
    pct = lambda s: s.map(lambda x: f"{x:.1%}")

    # 1. Tasa por departamento (solo departamentos con al menos 30 estudiantes para no exagerar casos pequeños)
    by_dep = tasa(d, "departamento").query("estudiantes >= 30").sort_values("tasa")
    fig_dep = px.bar(by_dep, x="tasa", y="departamento", orientation="h", text=pct(by_dep.tasa), hover_data={"estudiantes": True},
                     title="Bajo puntaje por departamento", color="tasa", color_continuous_scale=["#CDE8E6", COLORS["cyan"]])
    fig_dep.update_layout(template=template, coloraxis_showscale=False, xaxis_tickformat=".0%", height=max(420, 22 * len(by_dep) + 120),
                          xaxis_title="Proporcion con bajo puntaje", yaxis_title="")

    # 2. Puntaje global por estrato y naturaleza
    e = d.dropna(subset=["estrato"]).assign(estrato_txt=lambda x: x.estrato.astype(int).map(lambda v: "Sin estrato" if v == 0 else f"Estrato {v}"))
    orden = ["Sin estrato"] + [f"Estrato {i}" for i in range(1, 7)]
    fig_est = px.box(e, x="estrato_txt", y="punt_global", color="naturaleza", category_orders={"estrato_txt": orden},
                     title="Puntaje global por estrato y tipo de colegio", points=False,
                     color_discrete_map={"Oficial": COLORS["blue"], "No Oficial": COLORS["gold"]})
    fig_est.add_hline(y=CORTE, line_dash="dash", line_color=COLORS["red"], annotation_text=f"corte {CORTE:.0f}")
    fig_est.update_layout(template=template, xaxis_title="", yaxis_title="Puntaje global", legend_title="Colegio",
                          height=max(420, 22 * len(by_dep) + 120))

    # 3. Tendencia por año y naturaleza
    by_year = d.groupby(["anio", "naturaleza"], as_index=False).agg(tasa=("bajo_puntaje", "mean"), estudiantes=("estudiante_id", "count"))
    by_year = by_year[by_year.estudiantes >= 30]
    fig_year = px.line(by_year, x="anio", y="tasa", color="naturaleza", markers=True, hover_data={"estudiantes": True},
                       title="Tendencia por año", color_discrete_map={"Oficial": COLORS["blue"], "No Oficial": COLORS["gold"]})
    fig_year.update_traces(line_width=3)
    fig_year.update_layout(template=template, yaxis_tickformat=".0%", yaxis_title="Proporcion con bajo puntaje", xaxis_title="",
                           xaxis=dict(type="category"), legend_title="Colegio")

    # 4. Educacion de la madre
    by_ed = tasa(d.dropna(subset=["educacion_madre"]), "educacion_madre").sort_values("educacion_madre")
    by_ed["nivel"] = by_ed.educacion_madre.astype(int).map(EDUCACION_TXT)
    fig_ed = px.bar(by_ed, x="tasa", y="nivel", orientation="h", text=pct(by_ed.tasa), hover_data={"estudiantes": True},
                    title="Bajo puntaje segun educacion de la madre", color_discrete_sequence=[COLORS["blue"]])
    fig_ed.update_layout(template=template, xaxis_tickformat=".0%", xaxis_title="Proporcion con bajo puntaje", yaxis_title="")

    # 5. Bienes del hogar: internet y computador
    bienes = []
    for col, nombre in [("tiene_internet", "Internet"), ("tiene_computador", "Computador"), ("tiene_automovil", "Automovil"), ("tiene_lavadora", "Lavadora")]:
        t = d[d[col].isin(["Si", "No"])].groupby(col).bajo_puntaje.mean()
        for k, v in t.items():
            bienes.append({"bien": nombre, "tiene": k, "tasa": v})
    bienes = pd.DataFrame(bienes, columns=["bien", "tiene", "tasa"])
    fig_bienes = px.bar(bienes, x="bien", y="tasa", color="tiene", barmode="group", text=pct(bienes.tasa) if len(bienes) else None,
                        title="Bajo puntaje segun bienes del hogar", color_discrete_map={"Si": COLORS["cyan"], "No": COLORS["red"]})
    fig_bienes.update_traces(textposition="outside", textangle=0, cliponaxis=False)
    fig_bienes.update_layout(template=template, yaxis_tickformat=".0%", yaxis_title="Proporcion con bajo puntaje", xaxis_title="",
                             legend_title="¿Lo tiene?")

    # 6. Distribucion del puntaje de riesgo
    p_plot = p.assign(resultado=p.bajo_puntaje.map({0: "No bajo", 1: "Bajo puntaje"}))
    fig_risk = px.histogram(p_plot, x="probabilidad_bajo", color="resultado", nbins=25, barmode="overlay", opacity=0.65,
                            title="Distribucion de la probabilidad estimada (prueba)",
                            color_discrete_map={"No bajo": COLORS["blue"], "Bajo puntaje": COLORS["red"]})
    fig_risk.add_vline(x=umbral, line_dash="dash", line_color=COLORS["gold"])
    fig_risk.update_layout(template=template, xaxis_tickformat=".0%", xaxis_title="Probabilidad de bajo puntaje",
                           yaxis_title="Estudiantes", legend_title="Resultado real")

    # Metricas del modelo
    if len(p) and p.bajo_puntaje.nunique() == 2:
        y = p.bajo_puntaje.to_numpy(); pred = (p.probabilidad_bajo.to_numpy() >= umbral).astype(int)
        auc = roc_auc_score(y, p.probabilidad_bajo); acc = accuracy_score(y, pred); pre = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0); f1 = f1_score(y, pred, zero_division=0); cm = confusion_matrix(y, pred, labels=[0, 1])
        cells = [("ROC-AUC", auc), ("Exactitud", acc), ("Precision", pre), ("Recall", rec), ("F1", f1)]
        strip = [html.Div([html.Span(k), html.Strong(f"{v:.1%}")]) for k, v in cells]
    else:
        cm = np.zeros((2, 2), dtype=int); auc = acc = pre = rec = f1 = np.nan
        strip = [html.Div([html.Span("Sin muestra suficiente"), html.Strong("--")])]
    fig_cm = go.Figure(go.Heatmap(z=cm, x=["Predice no bajo", "Predice bajo"], y=["Real no bajo", "Real bajo"], text=cm,
                                  texttemplate="%{text}", colorscale=[[0, "#EAF1F7"], [1, COLORS["blue"]]], showscale=False))
    fig_cm.update_layout(template=template, title=f"Matriz de confusion (umbral {umbral:.0%})")

    mm = MODELS.melt(id_vars="version", value_vars=["roc_auc_prueba", "f1", "recall"], var_name="metrica", value_name="valor")
    mm["metrica"] = mm.metrica.map({"roc_auc_prueba": "ROC-AUC", "f1": "F1", "recall": "Recall"})
    fig_models = px.bar(mm, x="version", y="valor", color="metrica", barmode="group", title="Comparacion de iteraciones",
                        color_discrete_sequence=[COLORS["blue"], COLORS["cyan"], COLORS["gold"]])
    fig_models.update_layout(template=template, yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Valor", legend_title="Metrica")

    imp = IMPORTANCE.sort_values("importancia").assign(nombre=lambda x: x.variable.map(NOMBRES).fillna(x.variable))
    fig_imp = px.bar(imp, x="importancia", y="nombre", orientation="h", title="Importancia por permutacion", color_discrete_sequence=[COLORS["cyan"]])
    fig_imp.update_layout(template=template, xaxis_title="Cuanto empeora el ROC-AUC si se desordena la variable", yaxis_title="")

    high = int((p.probabilidad_bajo >= umbral).sum()) if len(p) else 0
    narrative = [html.H3("Lectura ejecutiva"),
        html.P(f"La vista contiene {len(d):,} estudiantes. El {d.bajo_puntaje.mean():.1%} obtuvo un puntaje global menor a {CORTE:.0f}.") if len(d) else html.P("No hay registros con estos filtros."),
        html.P(f"Con un umbral de {umbral:.0%}, el modelo marca {high:,} estudiantes de la muestra de prueba. El recall es {rec:.1%} "
               f"(de cada 10 estudiantes con bajo puntaje, detecta {rec*10:.0f}) y la precision {pre:.1%}." if not np.isnan(rec) else "La seleccion no contiene ambas clases para calcular todas las metricas."),
        html.P("Mover el umbral cambia el equilibrio entre detectar mas estudiantes y producir mas alertas falsas. Se debe definir segun la capacidad de apoyo (refuerzos, tutorias)."),
        html.P("Los factores muestran asociacion predictiva, no causalidad: el estrato o la jornada no 'causan' el puntaje, reflejan condiciones del contexto. "
               "Los años 2018, 2020 y 2021 tienen pocos registros en la muestra; interpretelos con cuidado."),
    ]

    cols = ["estudiante_id", "departamento", "anio", "naturaleza", "area", "jornada", "estrato", "punt_global", "probabilidad_bajo", "nivel_riesgo", "bajo_puntaje"]
    table = p.sort_values("probabilidad_bajo", ascending=False)[cols].head(100).copy()
    table["probabilidad_bajo"] = table["probabilidad_bajo"].round(3)
    nombres_tabla = {"estudiante_id": "Id estudiante", "anio": "Año", "punt_global": "Puntaje global", "probabilidad_bajo": "Probabilidad bajo",
                     "nivel_riesgo": "Nivel riesgo", "bajo_puntaje": "Bajo puntaje real"}
    columns = [{"name": nombres_tabla.get(c, c.replace("_", " ").title()), "id": c} for c in cols]
    return (f"{len(d):,}", f"{d.bajo_puntaje.mean():.1%}" if len(d) else "--", f"{d.punt_global.mean():.0f}" if len(d) else "--", f"{high:,}",
            fig_dep, fig_est, fig_year, fig_ed, fig_bienes, fig_risk, strip, fig_cm, fig_models, fig_imp, narrative,
            table.to_dict("records"), columns)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
