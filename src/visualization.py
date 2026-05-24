from __future__ import annotations

from pathlib import Path
import math
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .dataset_index import _read_rgb, plot_dataset_samples, plot_droneface_samples
from .face_model import _maybe_upscale


def _select_detection_examples(index_df, max_images: int):
    gallery = index_df[index_df["split"] == "gallery"].sort_values("subject_id").head(max(1, max_images // 2))
    gallery_subjects = gallery["subject_id"].dropna().astype(str).tolist()
    probe_candidates = index_df[
        (index_df["split"] == "probe") & (index_df["subject_id"].astype(str).isin(gallery_subjects))
    ].copy()
    probe = (
        probe_candidates.sort_values(["subject_id", "height_m", "distance_m"])
        .groupby("subject_id", as_index=False, dropna=True)
        .head(1)
        .head(max_images - len(gallery))
    )
    examples = []
    if not gallery.empty:
        examples.append(gallery)
    if not probe.empty:
        examples.append(probe)
    if not examples:
        return index_df.head(max_images).index.to_numpy()
    return np.concatenate([df.index.to_numpy() for df in examples])


def plot_detection_examples(index_df, model, out_path, max_images: int = 8):
    """Plot DroneFace gallery/probe examples with InsightFace detections."""
    if index_df.empty:
        warnings.warn("Cannot plot detections because the index is empty.", stacklevel=2)
        return None

    selected_indices = _select_detection_examples(index_df, max_images)
    sample_df = index_df.loc[selected_indices].head(max_images)
    if sample_df.empty:
        warnings.warn("No gallery or probe images available for detection examples.", stacklevel=2)
        return None

    n_images = len(sample_df)
    n_cols = min(4, n_images)
    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes_array, sample_df.itertuples(index=False)):
        ax.axis("off")
        path = Path(row.path)
        title = f"{row.split} | {row.subject_id}\n{path.name}"
        try:
            image_bgr = model.load_image(path)
            extraction = model.extract_single_face_embedding(path)
        except Exception as exc:
            ax.set_title(f"{title}\nerror", fontsize=8)
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True)
            continue

        image_used = image_bgr
        if extraction.get("upscaled_for_detection"):
            image_used, _ = _maybe_upscale(image_bgr)

        image_rgb = cv2.cvtColor(image_used, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        if extraction["detected"] and extraction.get("bbox") is not None:
            x1, y1, x2, y2 = extraction["bbox"]
            rect = Rectangle(
                (x1, y1),
                max(0, x2 - x1),
                max(0, y2 - y1),
                fill=False,
                linewidth=2,
                edgecolor="lime",
            )
            ax.add_patch(rect)
            score = extraction.get("detection_score", np.nan)
            ax.set_title(f"{title}\nscore={score:.3f}", fontsize=8)
        else:
            ax.set_title(f"{title}\nno detection", fontsize=8)

    for ax in axes_array[n_images:]:
        ax.axis("off")

    fig.suptitle("DroneFace - ejemplos de deteccion InsightFace")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved detection examples: {out_path}")
    return fig


def _save_or_warn(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    return fig


def _format_value(value, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{value:g}{suffix}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)


def _plot_image_grid(rows, title: str, out_path, n_cols: int = 4):
    if not rows:
        warnings.warn(f"No images available for figure: {title}", stacklevel=2)
        return None
    n_images = len(rows)
    n_cols = min(n_cols, n_images)
    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.7 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, item in zip(axes_array, rows):
        ax.axis("off")
        image = _read_rgb(Path(item["path"]))
        ax.set_title(item["title"], fontsize=8)
        if image is None:
            ax.text(0.5, 0.5, "unreadable image", ha="center", va="center", wrap=True)
            continue
        ax.imshow(image)

    for ax in axes_array[n_images:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_dataset_samples_droneface(index_df, out_path):
    """Plot gallery and probe examples from the DroneFace baseline protocol."""
    if index_df.empty:
        warnings.warn("Cannot plot DroneFace samples because the index is empty.", stacklevel=2)
        return None

    gallery = index_df[index_df["split"] == "gallery"].sort_values("subject_id")
    gallery_subjects = gallery["subject_id"].dropna().astype(str).tolist()
    probe_candidates = index_df[
        (index_df["split"] == "probe") & (index_df["subject_id"].astype(str).isin(gallery_subjects))
    ].copy()
    probe = (
        probe_candidates.sort_values(["subject_id", "height_m", "distance_m"])
        .groupby("subject_id", as_index=False, dropna=True)
        .head(1)
    )
    rows = []
    for row in gallery.itertuples(index=False):
        rows.append(
            {
                "path": row.path,
                "title": f"gallery | sujeto {row.subject_id}\n{Path(row.path).name}",
            }
        )
    for row in probe.itertuples(index=False):
        rows.append(
            {
                "path": row.path,
                "title": (
                    f"probe | sujeto {row.subject_id}\n"
                    f"h={_format_value(row.height_m, 'm')} d={_format_value(row.distance_m, 'm')}"
                ),
            }
        )
    return _plot_image_grid(rows, "DroneFace - muestras gallery/probe por sujeto", out_path)


def _select_diverse_results(results_df, mask, max_rows: int):
    subset = results_df[mask].copy()
    if subset.empty:
        return subset
    return (
        subset.sort_values(["subject_id", "height_m", "distance_m"])
        .groupby("subject_id", as_index=False, dropna=True)
        .head(1)
        .head(max_rows)
    )


def _gallery_lookup(index_df):
    if index_df is None or index_df.empty:
        return {}
    gallery = index_df[index_df["split"] == "gallery"].sort_values("subject_id")
    return {
        str(row.subject_id): row.path
        for row in gallery.itertuples(index=False)
        if row.subject_id is not None
    }


def plot_recognition_examples(results_df, out_path, max_correct: int = 4, max_incorrect: int = 4, index_df=None):
    """Plot automatically selected correct and failed recognition examples."""
    if results_df.empty:
        warnings.warn("Cannot plot recognition examples because results are empty.", stacklevel=2)
        return None

    correct = _select_diverse_results(
        results_df,
        (results_df["detected"] == True) & (results_df["correct"] == True),
        max_correct,
    )
    incorrect = _select_diverse_results(
        results_df,
        (results_df["detected"] == True) & (results_df["correct"] == False),
        max_incorrect,
    )
    if incorrect.empty:
        incorrect = _select_diverse_results(results_df, results_df["correct"] == False, max_incorrect)
    if incorrect.empty:
        warnings.warn("No incorrect examples found; plotting only correct examples.", stacklevel=2)
    if correct.empty:
        warnings.warn("No correct examples found; plotting only incorrect examples.", stacklevel=2)

    selected = [("correcto", correct), ("fallo", incorrect)]
    selected_rows = [(label, row) for label, df in selected for row in df.itertuples(index=False)]
    if not selected_rows:
        warnings.warn("No recognition examples available.", stacklevel=2)
        return None

    gallery_paths = _gallery_lookup(index_df)
    n_cases = len(selected_rows)
    fig, axes = plt.subplots(n_cases, 2, figsize=(8, max(3.2, 3.0 * n_cases)))
    axes = np.atleast_2d(axes)

    for row_idx, (label, row) in enumerate(selected_rows):
        gallery_path = gallery_paths.get(str(row.subject_id))
        gallery_ax = axes[row_idx, 0]
        probe_ax = axes[row_idx, 1]
        for ax in [gallery_ax, probe_ax]:
            ax.axis("off")

        if gallery_path is None:
            gallery_ax.text(0.5, 0.5, "sin galeria", ha="center", va="center")
        else:
            gallery_img = _read_rgb(Path(gallery_path))
            if gallery_img is None:
                gallery_ax.text(0.5, 0.5, "galeria no legible", ha="center", va="center")
            else:
                gallery_ax.imshow(gallery_img)
        gallery_ax.set_title(f"galeria conocida\nsujeto {row.subject_id}", fontsize=9)

        probe_img = _read_rgb(Path(row.path))
        if probe_img is None:
            probe_ax.text(0.5, 0.5, "probe no legible", ha="center", va="center")
        else:
            probe_ax.imshow(probe_img)
        status = "OK" if label == "correcto" else "ERROR"
        probe_ax.set_title(
            (
                f"{status}: real {row.subject_id} | pred {_format_value(row.predicted_subject)}\n"
                f"sim={_format_value(row.similarity)} det={row.detected} "
                f"h={_format_value(row.height_m, 'm')} d={_format_value(row.distance_m, 'm')}"
            ),
            fontsize=9,
        )

    fig.suptitle("DroneFace - reconocimiento: galeria vs probe")
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_metric_bar(metric_df, x_col: str, y_col: str, title: str, ylabel: str, out_path):
    if metric_df.empty:
        warnings.warn(f"Cannot plot {title}: metric dataframe is empty.", stacklevel=2)
        return None
    plot_df = metric_df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        warnings.warn(f"Cannot plot {title}: missing {x_col} or {y_col}.", stacklevel=2)
        return None

    plot_df = plot_df.sort_values(x_col)
    x_labels = [str(value) for value in plot_df[x_col]]
    values = plot_df[y_col].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(max(6, len(plot_df) * 0.7), 4.5))
    ax.bar(x_labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_detection_vs_accuracy(summary_or_results_df, out_path):
    if summary_or_results_df.empty:
        warnings.warn("Cannot plot detection vs accuracy: dataframe is empty.", stacklevel=2)
        return None

    if {"detection_rate", "accuracy_conditional", "accuracy_global"}.issubset(summary_or_results_df.columns):
        row = summary_or_results_df.iloc[0]
        values = [row["detection_rate"], row["accuracy_conditional"], row["accuracy_global"]]
    else:
        detected = summary_or_results_df["detected"].astype(bool)
        correct = summary_or_results_df["correct"].astype(bool)
        total = len(summary_or_results_df)
        detected_count = int(detected.sum())
        values = [
            detected_count / total if total else np.nan,
            int(correct.sum()) / detected_count if detected_count else np.nan,
            int(correct.sum()) / total if total else np.nan,
        ]

    labels = ["detection_rate", "accuracy_conditional", "accuracy_global"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=["#59a14f", "#f28e2b", "#4e79a7"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("proporcion")
    ax.set_title("DroneFace - deteccion y accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_accuracy_heatmap_height_distance(metric_df, out_path):
    if metric_df.empty:
        warnings.warn("Cannot plot height-distance heatmap: metric dataframe is empty.", stacklevel=2)
        return None
    required = {"height_m", "distance_m", "accuracy_global"}
    if not required.issubset(metric_df.columns):
        warnings.warn("Cannot plot height-distance heatmap: required columns are missing.", stacklevel=2)
        return None

    pivot = metric_df.pivot_table(
        index="height_m",
        columns="distance_m",
        values="accuracy_global",
        aggfunc="mean",
        dropna=False,
    ).sort_index(axis=0).sort_index(axis=1)
    if pivot.empty:
        warnings.warn("Cannot plot height-distance heatmap: pivot table is empty.", stacklevel=2)
        return None

    fig_width = max(8, 0.45 * len(pivot.columns))
    fig_height = max(4, 0.55 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("DroneFace - accuracy_global por altura y distancia")
    ax.set_xlabel("distance_m")
    ax.set_ylabel("height_m")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_format_value(value) for value in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_format_value(value) for value in pivot.index])
    fig.colorbar(image, ax=ax, label="accuracy_global")
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_confusion_matrix(results_df, out_path):
    detected_df = results_df[(results_df["detected"] == True) & results_df["predicted_subject"].notna()].copy()
    if detected_df.empty:
        warnings.warn("Cannot plot confusion matrix: no detected rows with predictions.", stacklevel=2)
        return None

    labels = sorted(
        set(detected_df["subject_id"].dropna().astype(str)).union(
            set(detected_df["predicted_subject"].dropna().astype(str))
        )
    )
    matrix = confusion_matrix(
        detected_df["subject_id"].astype(str),
        detected_df["predicted_subject"].astype(str),
        labels=labels,
    )
    size = max(6, 0.75 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("DroneFace - matriz de confusion (detectadas)")
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def _split_subject_text(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part for part in text.split(",") if part]


def plot_full_image_detections(image_results_df, detections_df, out_path, max_images: int = 6):
    if image_results_df.empty:
        warnings.warn("Cannot plot full-image detections: image results are empty.", stacklevel=2)
        return None

    sort_cols = ["composite_subject_id", "height_m", "distance_m", "filename"]
    sample_df = (
        image_results_df.sort_values(sort_cols, na_position="last")
        .groupby("composite_subject_id", as_index=False, dropna=True)
        .head(1)
        .head(max_images)
    )
    if sample_df.empty:
        warnings.warn("No full-image examples available.", stacklevel=2)
        return None

    n_images = len(sample_df)
    n_cols = min(3, n_images)
    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes_array, sample_df.itertuples(index=False)):
        ax.axis("off")
        path = Path(row.path)
        image = _read_rgb(path)
        expected = _split_subject_text(row.expected_subjects_with_gallery)
        found = _split_subject_text(row.found_expected_subjects)
        title = (
            f"{row.composite_subject_id} | esperados con galeria: {','.join(expected) or 'NA'}\n"
            f"encontrados: {','.join(found) or 'ninguno'} | h={_format_value(row.height_m, 'm')} d={_format_value(row.distance_m, 'm')}"
        )
        ax.set_title(title, fontsize=8)
        if image is None:
            ax.text(0.5, 0.5, "unreadable image", ha="center", va="center", wrap=True)
            continue
        ax.imshow(image)

        det_path_col = "path" if "path" in detections_df.columns else "source_path"
        det_subset = (
            detections_df[detections_df[det_path_col].astype(str) == str(path)]
            if not detections_df.empty and det_path_col in detections_df.columns
            else detections_df
        )
        for det in det_subset.itertuples(index=False):
            if np.isnan(getattr(det, "bbox_x1", np.nan)):
                continue
            x1, y1, x2, y2 = det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2
            predicted = _format_value(det.predicted_subject)
            similarity = _format_value(det.similarity)
            color = "lime" if str(det.predicted_subject) in expected else "orange"
            rect = Rectangle(
                (x1, y1),
                max(0, x2 - x1),
                max(0, y2 - y1),
                fill=False,
                linewidth=2,
                edgecolor=color,
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0, y1 - 4),
                f"{predicted} ({similarity})",
                color=color,
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )

        if int(row.num_faces_detected) == 0:
            ax.text(0.5, 0.05, "no detection", ha="center", va="bottom", color="red", transform=ax.transAxes)

    for ax in axes_array[n_images:]:
        ax.axis("off")

    fig.suptitle("DroneFace - Experimento B: detecciones en imagen completa")
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_experiment_a_vs_b(comparison_df, out_path):
    if comparison_df.empty:
        warnings.warn("Cannot plot A vs B comparison: dataframe is empty.", stacklevel=2)
        return None

    plot_df = comparison_df.copy()
    labels = plot_df["experiment"].astype(str).tolist()
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(x - width / 2, plot_df["detection_rate"].astype(float), width=width, label="detection_rate", color="#59a14f")
    ax.bar(x + width / 2, plot_df["primary_metric"].astype(float), width=width, label="metrica principal", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("proporcion")
    ax.set_title("DroneFace - Experimento A vs B")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_experiment_a_b_common_metrics(common_metrics_df, out_path):
    if common_metrics_df.empty:
        warnings.warn("Cannot plot common A/B metrics: dataframe is empty.", stacklevel=2)
        return None
    required = {"experiment", "precision_micro", "recall_micro", "f1_micro", "exact_match_rate"}
    if not required.issubset(common_metrics_df.columns):
        missing = ", ".join(sorted(required - set(common_metrics_df.columns)))
        warnings.warn(f"Cannot plot common A/B metrics. Missing columns: {missing}", stacklevel=2)
        return None

    plot_df = common_metrics_df.copy()
    labels = plot_df["experiment"].astype(str).tolist()
    metrics = ["precision_micro", "recall_micro", "f1_micro", "exact_match_rate"]
    x = np.arange(len(plot_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metrics))
    for metric, offset, color in zip(metrics, offsets, colors):
        ax.bar(x + offset, plot_df[metric].astype(float), width=width, label=metric, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("proporcion")
    ax.set_title("DroneFace - metricas comparables A vs B")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_full_image_recall_by_distance(metric_df, out_path):
    if metric_df.empty:
        warnings.warn("Cannot plot full-image recall by distance: dataframe is empty.", stacklevel=2)
        return None
    required = {"distance_m", "expected_identity_recall_global"}
    if not required.issubset(metric_df.columns):
        missing = ", ".join(sorted(required - set(metric_df.columns)))
        warnings.warn(f"Cannot plot full-image recall by distance. Missing columns: {missing}", stacklevel=2)
        return None

    plot_df = metric_df.dropna(subset=["distance_m", "expected_identity_recall_global"]).copy()
    if plot_df.empty:
        warnings.warn("Cannot plot full-image recall by distance: no valid rows.", stacklevel=2)
        return None

    plot_df = plot_df.sort_values("distance_m")
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(plot_df)), 4.8))
    ax.bar(
        [str(_format_value(value)) for value in plot_df["distance_m"]],
        plot_df["expected_identity_recall_global"].astype(float),
        color="#4c78a8",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("distance_m")
    ax.set_ylabel("expected_identity_recall")
    ax.set_title("DroneFace - Experimento B: recall por distancia (promedio sobre alturas)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


def plot_full_image_recall_heatmap_height_distance(metric_df, out_path):
    if metric_df.empty:
        warnings.warn("Cannot plot full-image height-distance heatmap: dataframe is empty.", stacklevel=2)
        return None
    required = {"height_m", "distance_m", "expected_identity_recall_global"}
    if not required.issubset(metric_df.columns):
        missing = ", ".join(sorted(required - set(metric_df.columns)))
        warnings.warn(f"Cannot plot full-image height-distance heatmap. Missing columns: {missing}", stacklevel=2)
        return None

    plot_df = metric_df.dropna(subset=["height_m", "distance_m", "expected_identity_recall_global"]).copy()
    if plot_df.empty:
        warnings.warn("Cannot plot full-image height-distance heatmap: no valid rows.", stacklevel=2)
        return None

    pivot = plot_df.pivot_table(
        index="height_m",
        columns="distance_m",
        values="expected_identity_recall_global",
        aggfunc="mean",
        dropna=False,
    ).sort_index(axis=0).sort_index(axis=1)
    if pivot.empty:
        warnings.warn("Cannot plot full-image height-distance heatmap: pivot table is empty.", stacklevel=2)
        return None

    fig_width = max(8, 0.45 * len(pivot.columns))
    fig_height = max(4, 0.6 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("DroneFace - Experimento B: recall por altura y distancia")
    ax.set_xlabel("distance_m")
    ax.set_ylabel("height_m")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_format_value(value) for value in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_format_value(value) for value in pivot.index])
    fig.colorbar(image, ax=ax, label="expected_identity_recall_global")
    fig.tight_layout()
    return _save_or_warn(fig, out_path)


__all__ = [
    "plot_dataset_samples",
    "plot_droneface_samples",
    "plot_detection_examples",
    "plot_dataset_samples_droneface",
    "plot_recognition_examples",
    "plot_metric_bar",
    "plot_detection_vs_accuracy",
    "plot_accuracy_heatmap_height_distance",
    "plot_confusion_matrix",
    "plot_full_image_detections",
    "plot_experiment_a_vs_b",
    "plot_experiment_a_b_common_metrics",
    "plot_full_image_recall_by_distance",
    "plot_full_image_recall_heatmap_height_distance",
]
