# Dashboard de alerta temprana de bajo puntaje en Saber 11 (Dash)

Adaptacion del proyecto base "Dashboard de desercion academica" a datos reales del ICFES:
[Resultados unicos Saber 11](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe/about_data) (datos.gov.co).

**Pregunta:** con datos del estudiante, su hogar y su colegio (sin usar sus puntajes), ¿se puede estimar la
probabilidad de que quede en el 25% de puntajes globales mas bajos?

## Estructura

```text
app.py                         Dashboard Dash
sacar_muestra.py               Saca una muestra aleatoria del CSV completo (~3,6 GB)
preparar_datos.py              Limpieza y creacion de la variable objetivo bajo_puntaje
entrenar_modelo.py             Entrenamiento, ajuste y exportacion de resultados (3 iteraciones)
data/muestra_saber11.csv       Muestra aleatoria de 60.000 registros (todas las columnas originales)
data/saber11_preparado.csv     Datos limpios usados por el modelo y el dashboard
data/diccionario_datos.csv     Diccionario de variables
outputs/                       Corte, metricas, predicciones e importancia
models/                        Modelo serializado
assets/style.css               Estilos del dashboard
tests/test_pipeline.py         Pruebas de integridad
ENLACES.txt                    Plantilla para GitHub y Binder
```

## Decisiones principales

- Solo se usan los periodos **2014-2 a 2022**: antes de 2014-2 el examen tenia otra estructura y no existe el puntaje global.
- Se excluyen registros con puntaje 0 o con estado distinto a "PUBLICAR".
- **Variable objetivo:** `bajo_puntaje = 1` si el puntaje global es menor al percentil 25 de la muestra (214 puntos).
- **Sin fuga de informacion:** los puntajes por materia no se usan como variables del modelo, porque el puntaje global se calcula a partir de ellos.
- Tres iteraciones: linea base (3 variables), regresion logistica regularizada y bosque aleatorio. Se elige la de mayor ROC-AUC en prueba.

## Ejecucion rapida

Requiere Python 3.11 a 3.13 (con 3.15 algunas librerias aun no traen instaladores listos).

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Abra `http://127.0.0.1:8050`. Los datos preparados y los resultados ya vienen incluidos.
Tambien puede abrir el dashboard con doble clic en `ejecutar_dashboard.bat` (despues de crear `.venv`).

Para regenerar todo desde la muestra:

```powershell
python preparar_datos.py
python entrenar_modelo.py   # tarda 1-3 minutos
```

Para usar una muestra distinta del archivo completo:

```powershell
python sacar_muestra.py "C:\ruta\Resultados_unicos_Saber_11.csv"
```

y copie el `muestra_saber11.csv` resultante dentro de `data/`.

## Pruebas

```bash
pytest -q
```

## Publicacion en GitHub y Binder

1. Cree un repositorio publico y cargue el contenido de esta carpeta (sin `.venv`).
2. Reemplace `USUARIO` y `REPOSITORIO` en `ENLACES.txt`.
3. Abra el enlace Binder. El archivo `binder/start` inicia Dash y lo expone mediante `jupyter-server-proxy`.

## Nota etica

Los datos son publicos y anonimos (sin nombres ni documentos). El modelo muestra asociaciones, no causas:
variables como estrato, departamento o jornada reflejan desigualdades del contexto y no deben usarse para
etiquetar ni excluir estudiantes. Es un ejercicio academico, no una herramienta de decision.
