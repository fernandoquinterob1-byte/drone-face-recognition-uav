# Proyecto Rostros

Baseline academico de reconocimiento facial sobre DroneFace usando InsightFace/ArcFace preentrenado. El proyecto construye un flujo reproducible de preparacion de datos, embeddings, evaluacion closed-set, metricas y figuras finales para informe.

No se entrena una red desde cero. No se integra control de dron ni DJI Tello. SCface fue removido porque no esta disponible.

## DroneFace

DroneFace aproxima captura aerea desde dron. En este baseline:

- `portraits` se usa como galeria (`gallery`).
- `photos_all_faces` se usa como conjunto probe.
- `photos_all`, `trio_cam` y `trio_gp` quedan indexadas como `unused` y no se usan en la evaluacion principal.

La ruta esperada del dataset es:

```text
data/DroneFace/open_data_set
```

## Estructura

```text
Proyecto_rostros/
|-- data/
|   `-- DroneFace/
|       `-- open_data_set/
|           |-- photos_all/
|           |-- photos_all_faces/
|           |-- portraits/
|           |-- trio_cam/
|           |-- trio_gp/
|           `-- LICENCE.md
|-- figs/
|   |-- dataset/
|   |-- detection/
|   |-- recognition/
|   `-- metrics/
|-- outputs/
|   |-- indexes/
|   |-- embeddings/
|   |-- results/
|   `-- tables/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- dataset_index.py
|   |-- face_model.py
|   |-- evaluation.py
|   `-- visualization.py
|-- avance_preliminar.ipynb
|-- requirements.txt
`-- README.md
```

## Instalacion en Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecucion

```powershell
jupyter notebook avance_preliminar.ipynb
```

## Archivos de salida

- `outputs/indexes/droneface_index.csv`
- `outputs/embeddings/droneface_gallery_embeddings.npz`
- `outputs/results/droneface_results.csv`
- `outputs/tables/summary_global.csv`
- `outputs/tables/droneface_by_height.csv`
- `outputs/tables/droneface_by_distance.csv`
- `outputs/tables/droneface_by_height_distance.csv`
- `outputs/tables/preliminary_discussion.txt`
- `figs/dataset/droneface_samples.png`
- `figs/detection/droneface_detections.png`
- `figs/recognition/droneface_correct_incorrect.png`
- `figs/metrics/droneface_confusion_matrix.png`
- `figs/metrics/droneface_accuracy_by_height.png`
- `figs/metrics/droneface_accuracy_by_distance.png`
- `figs/metrics/droneface_detection_vs_accuracy.png`
- `figs/metrics/droneface_accuracy_height_distance.png`

## Metodologia

La evaluacion es closed-set: cada probe evaluado pertenece a un sujeto presente en la galeria. Cada embedding probe se compara contra todos los embeddings de galeria mediante similitud coseno y se predice el sujeto con mayor similitud. En esta etapa no se usa umbral de desconocido.

El modelo se usa preentrenado mediante InsightFace/ArcFace (`buffalo_l`). Si InsightFace no puede importar o descargar el modelo, el codigo muestra el error exacto y no cambia automaticamente a otro modelo.

## Limitaciones

- Solo se usa DroneFace porque SCface no esta disponible.
- El modelo es preentrenado y no fue ajustado al dataset.
- El experimento es closed-set, por lo que no evalua sujetos desconocidos.
- El rendimiento depende de la deteccion facial previa.
- `portraits` y `photos_all_faces` representan un baseline simplificado.
- `photos_all`, `trio_cam` y `trio_gp` no se usan aun en la evaluacion principal.
- No se implementa control de dron.
- No se evalua tiempo real.
- El uso de modelos preentrenados debe mantenerse en contexto academico/no comercial y respetar licencias.

## Trabajo futuro

- Incorporar `photos_all` con deteccion sobre imagenes completas.
- Evaluar imagenes con multiples personas de `trio_cam` y `trio_gp`.
- Probar open-set recognition con umbral de desconocido.
- Comparar ArcFace con otro metodo.
- Analizar sensibilidad a resolucion, blur, altura y distancia.
- Estudiar factibilidad de procesamiento en tiempo real.
- Eventualmente conectar con un sistema de captura o seguimiento, pero no en este avance.
