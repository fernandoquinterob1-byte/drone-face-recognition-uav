# Proyecto Rostros

Baseline academico de reconocimiento facial sobre DroneFace usando InsightFace/ArcFace preentrenado. El proyecto construye un flujo reproducible de preparacion de datos, embeddings, evaluacion closed-set, metricas y figuras finales para informe.

No se entrena una red desde cero. No se integra control de dron ni DJI Tello. SCface fue removido porque no esta disponible.

## DroneFace

DroneFace aproxima captura aerea desde dron. En este baseline:

- `portraits` se usa como galeria (`gallery`).
- `photos_all_faces` se usa en el Experimento A como probe controlado con rostros ya recortados.
- `photos_all` se usa en el Experimento B como imagen completa del dron.
- `trio_cam` y `trio_gp` quedan indexadas como `unused` y se dejan para trabajo futuro.

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
|   |-- crops/
|   |-- indexes/
|   |-- embeddings/
|   |-- results/
|   `-- tables/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- dataset_index.py
|   |-- detection_experiments.py
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
- `outputs/embeddings/droneface_gallery_embeddings_det320.npz`
- `outputs/embeddings/droneface_gallery_embeddings_det640.npz`
- `outputs/embeddings/droneface_gallery_embeddings_det1024.npz`
- `outputs/results/droneface_results.csv`
- `outputs/crops/full_images_det1280/`
- `outputs/results/droneface_full_image_crops.csv`
- `outputs/results/droneface_full_image_crop_recognition.csv`
- `outputs/results/droneface_full_images_detections.csv`
- `outputs/results/droneface_full_images_results.csv`
- `outputs/tables/summary_global.csv`
- `outputs/tables/droneface_by_height.csv`
- `outputs/tables/droneface_by_distance.csv`
- `outputs/tables/droneface_by_height_distance.csv`
- `outputs/tables/droneface_full_images_summary.csv`
- `outputs/tables/droneface_full_images_by_height.csv`
- `outputs/tables/droneface_full_images_by_distance.csv`
- `outputs/tables/droneface_full_images_by_height_distance.csv`
- `outputs/tables/droneface_experiment_a_vs_b.csv`
- `outputs/tables/droneface_experiment_a_b_common_metrics.csv`
- `outputs/tables/droneface_full_images_det_size_summary.csv`
- `outputs/tables/preliminary_discussion.txt`
- `outputs/tables/gallery_failures_droneface_det320.csv`
- `outputs/tables/gallery_failures_droneface_det640.csv`
- `outputs/tables/gallery_failures_droneface_det1024.csv`
- `outputs/results/droneface_det_size_results.csv`
- `outputs/results/droneface_results_det320.csv`
- `outputs/results/droneface_results_det640.csv`
- `outputs/results/droneface_results_det1024.csv`
- `outputs/tables/droneface_det_size_summary.csv`
- `figs/dataset/droneface_samples.png`
- `figs/detection/droneface_detections.png`
- `figs/detection/full_images_review_det1280/`
- `figs/detection/droneface_full_image_detections.png`
- `figs/detection/droneface_det_size_examples.png`
- `figs/recognition/droneface_correct_incorrect.png`
- `figs/metrics/droneface_confusion_matrix.png`
- `figs/metrics/droneface_accuracy_by_height.png`
- `figs/metrics/droneface_accuracy_by_distance.png`
- `figs/metrics/droneface_detection_vs_accuracy.png`
- `figs/metrics/droneface_accuracy_height_distance.png`
- `figs/metrics/droneface_experiment_a_vs_b.png`
- `figs/metrics/droneface_experiment_a_b_common_metrics.png`
- `figs/metrics/droneface_full_image_recall_by_distance.png`
- `figs/metrics/droneface_full_image_recall_height_distance.png`
- `figs/metrics/droneface_full_image_det_size_sweep.png`
- `figs/metrics/droneface_det_size_detection_accuracy.png`

## Metodologia

El Experimento A es closed-set sobre rostros recortados: cada probe evaluado pertenece a un sujeto presente en la galeria. Cada embedding probe se compara contra todos los embeddings de galeria mediante similitud coseno y se predice el sujeto con mayor similitud. En esta etapa no se usa umbral de desconocido.

El Experimento B usa `photos_all`, es decir, imagenes completas del dron. Primero se detectan rostros en la imagen completa, se guardan imagenes de revision con bounding boxes y se guardan crops de cada rostro detectado para auditoria visual. La metrica principal usa los embeddings que InsightFace calcula en esa deteccion original; no se vuelve a detectar dentro del crop rectangular guardado. En esas imagenes las etiquetas son compuestas (`ab`, `cd`, `ef`, `gh`, `ijk`), por lo que se reporta recuperacion de identidades esperadas por imagen. Los sujetos sin galeria (`b`, `c`, `f`, `h`) se informan aparte y no se cuentan como error de reconocimiento.

El modelo se usa preentrenado mediante InsightFace/ArcFace (`buffalo_l`). El Experimento A usa `det_size=(320, 320)` porque fue la configuracion con mejor desempeno sobre rostros ya recortados. El Experimento B usa `det_size=(1280, 1280)` porque en imagenes completas del dron entrega la mejor deteccion y recuperacion de identidades esperadas dentro del barrido corregido. Si InsightFace no puede importar o descargar el modelo, el codigo muestra el error exacto y no cambia automaticamente a otro modelo.

Resumen comparativo generado:

```text
experimento  entrada            metrica principal
A           photos_all_faces    accuracy_global sobre rostro recortado
B           photos_all          expected_identity_recall sobre imagen completa
```

Resultados actuales:

```text
Experimento A: accuracy_global = 90.78%, detection_rate = 99.31%
Experimento B: expected_identity_recall_global = 78.80%, image_detection_rate = 91.45%
```

Tambien se genera una comparacion con metricas comunes. En A la unidad es un rostro recortado con una sola identidad esperada. En B la unidad es una imagen completa y se comparan conjuntos de identidades esperadas contra identidades predichas, porque `photos_all` no tiene identidad anotada por bounding box.

```text
experimento        precision_micro  recall_micro  f1_micro  exact_match_rate
A_cropped_faces   91.42%           90.78%        91.10%    90.78%
B_full_images     63.16%           78.80%        70.12%    25.48%
```

El F1 es util porque hay varias identidades/clases y porque B puede producir predicciones extra. En A, `recall_micro` coincide con `accuracy_global`; en B, `recall_micro` mide cuantas identidades esperadas se recuperan, mientras que `precision_micro` penaliza identidades predichas que no estaban esperadas en la imagen.

Tambien se probo el Experimento B con `det_size` mayores:

```text
det_size  image_detection_rate  expected_identity_recall_global
320       9.03%                 6.68%
640       41.13%                33.29%
1024      75.32%                63.82%
1280      91.45%                78.80%
```

El aumento de `det_size` mejora la cantidad de cajas detectadas y, con la metrica corregida, tambien mejora la recuperacion de identidades esperadas. La version anterior que redetectaba dentro de crops rectangulares castigaba artificialmente el resultado, porque muchos crops no volvian a generar una caja facial aunque el rostro ya habia sido detectado en la imagen original.

Los graficos por distancia agregan todas las alturas disponibles. Para analizar distancia y altura simultaneamente, el notebook genera heatmaps altura-distancia para el Experimento A y para el Experimento B.

## Experimento de sensibilidad del detector

El notebook incluye un experimento adicional para estudiar si el cuello de botella esta en la deteccion facial. Se repite el mismo protocolo closed-set con `det_size` igual a 320, 640 y 1024 usando el mismo modelo `buffalo_l`.

Este experimento no cambia el metodo de reconocimiento ni reemplaza InsightFace. Solo modifica el tamano de entrada usado por el detector facial antes de extraer embeddings. Un resultado `no detection` significa que el detector no genero una caja facial valida, aunque visualmente una persona pueda reconocer una cara en la imagen.

Resumen del barrido generado en este proyecto:

```text
det_size  detection_rate  accuracy_conditional  accuracy_global
320       0.9931          0.9142                0.9078
640       0.5599          0.8724                0.4885
1024      0.0027          0.0000                0.0000
```

Por esto, los archivos principales `droneface_results.csv`, `summary_global.csv` y las figuras finales corresponden al baseline ajustado con `det_size=320`. Los archivos `det320`, `det640`, `det1024` se conservan para justificar y analizar el ajuste del detector.

## Limitaciones

- Solo se usa DroneFace porque SCface no esta disponible.
- El modelo es preentrenado y no fue ajustado al dataset.
- El Experimento A es closed-set, por lo que no evalua sujetos desconocidos.
- El rendimiento depende de la deteccion facial previa.
- `photos_all_faces` representa un baseline simplificado con rostros ya recortados.
- `photos_all` se evalua como imagen completa y se guardan crops detectados, pero no existen anotaciones de bounding box por identidad para saber manualmente que crop corresponde a cada sujeto real.
- `trio_cam` y `trio_gp` no se usan aun en la evaluacion principal.
- No se implementa control de dron.
- No se evalua tiempo real.
- El uso de modelos preentrenados debe mantenerse en contexto academico/no comercial y respetar licencias.

## Trabajo futuro

- Mejorar la deteccion y asociacion de identidades en `photos_all`.
- Evaluar imagenes con multiples personas de `trio_cam` y `trio_gp`.
- Probar open-set recognition con umbral de desconocido.
- Comparar ArcFace con otro metodo.
- Analizar sensibilidad a resolucion, blur, altura y distancia.
- Estudiar factibilidad de procesamiento en tiempo real.
- Eventualmente conectar con un sistema de captura o seguimiento, pero no en este avance.
