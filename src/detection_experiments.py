from __future__ import annotations

from pathlib import Path
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .config import EMBEDDINGS_DIR, FIGS_DIR, RESULTS_DIR, TABLES_DIR, ensure_project_dirs
from .evaluation import build_gallery, compute_global_metrics, evaluate_closed_set
from .face_model import FaceRecognizerArcFace, _maybe_upscale


DET_SIZE_RESULT_COLUMNS = [
    "det_size",
    "dataset",
    "path",
    "filename",
    "source_folder",
    "subject_id",
    "predicted_subject",
    "detected",
    "num_faces_detected",
    "detection_score",
    "similarity",
    "correct",
    "upscaled_for_detection",
    "height_m",
    "distance_m",
    "image_type",
    "error",
]


DET_SIZE_SUMMARY_COLUMNS = [
    "det_size",
    "dataset",
    "n_gallery_subjects",
    "total_probe",
    "detected_count",
    "detection_rate",
    "correct_count",
    "accuracy_conditional",
    "accuracy_global",
    "mean_similarity_correct",
    "mean_similarity_incorrect",
    "gallery_subjects_before_detection",
    "gallery_subjects_with_embedding",
    "probes_excluded_no_gallery",
    "probes_evaluated",
    "runtime_seconds",
]


def _normalize_det_sizes(det_sizes) -> list[tuple[int, int]]:
    normalized = []
    for value in det_sizes:
        if isinstance(value, int):
            normalized.append((value, value))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            normalized.append((int(value[0]), int(value[1])))
        else:
            raise ValueError(f"Invalid det_size value: {value!r}")
    return normalized


def _det_size_label(det_size: tuple[int, int]) -> str:
    width, height = det_size
    return str(width) if width == height else f"{width}x{height}"


def _sample_probes_per_subject(index_df: pd.DataFrame, max_probes_per_subject: int | None) -> pd.DataFrame:
    if max_probes_per_subject is None:
        return index_df.copy()
    if max_probes_per_subject <= 0:
        raise ValueError("max_probes_per_subject must be positive or None.")

    non_probe = index_df[index_df["split"] != "probe"].copy()
    probe = (
        index_df[index_df["split"] == "probe"]
        .sort_values(["subject_id", "height_m", "distance_m", "filename"], na_position="last")
        .groupby("subject_id", as_index=False, dropna=True)
        .head(max_probes_per_subject)
    )
    return pd.concat([non_probe, probe], ignore_index=True)


def run_det_size_sweep(
    index_df: pd.DataFrame,
    det_sizes=(320, 640, 1024),
    model_name: str = "buffalo_l",
    allow_cuda: bool = False,
    max_probes_per_subject: int | None = None,
    results_path=None,
    summary_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the DroneFace closed-set baseline with multiple detector input sizes."""
    ensure_project_dirs()

    if index_df.empty:
        raise ValueError("Cannot run det_size sweep because the DroneFace index is empty.")
    if "split" not in index_df.columns or "subject_id" not in index_df.columns:
        raise ValueError("DroneFace index must contain split and subject_id columns.")

    det_sizes = _normalize_det_sizes(det_sizes)
    results_path = Path(results_path) if results_path is not None else RESULTS_DIR / "droneface_det_size_results.csv"
    summary_path = Path(summary_path) if summary_path is not None else TABLES_DIR / "droneface_det_size_summary.csv"

    experiment_df = _sample_probes_per_subject(index_df, max_probes_per_subject)
    gallery_df = experiment_df[experiment_df["split"] == "gallery"].copy()
    probe_df = experiment_df[experiment_df["split"] == "probe"].copy()

    if gallery_df.empty:
        raise ValueError("Cannot run det_size sweep because there are no gallery rows.")
    if probe_df.empty:
        raise ValueError("Cannot run det_size sweep because there are no probe rows.")

    all_results = []
    summary_rows = []

    for det_size in det_sizes:
        label = _det_size_label(det_size)
        print(f"\n=== det_size={label} ===")
        started = time.perf_counter()

        model = FaceRecognizerArcFace(model_name=model_name, det_size=det_size, allow_cuda=allow_cuda)
        gallery_save_path = EMBEDDINGS_DIR / f"droneface_gallery_embeddings_det{label}.npz"
        failures_path = TABLES_DIR / f"gallery_failures_droneface_det{label}.csv"

        gallery_subjects_before = gallery_df["subject_id"].dropna().astype(str).nunique()
        gallery_embeddings, _, _ = build_gallery(
            experiment_df,
            model,
            dataset_name="DroneFace",
            save_path=gallery_save_path,
            failures_path=failures_path,
        )
        if not gallery_embeddings:
            warnings.warn(f"det_size={label}: no valid gallery embeddings; skipping probe evaluation.", stacklevel=2)
            results_df = pd.DataFrame(columns=DET_SIZE_RESULT_COLUMNS)
        else:
            valid_subjects = set(gallery_embeddings.keys())
            probes_excluded_no_gallery = int(
                ((probe_df["split"] == "probe") & ~probe_df["subject_id"].astype(str).isin(valid_subjects)).sum()
            )
            results_df = evaluate_closed_set(
                probe_df,
                gallery_embeddings,
                model,
                save_path=RESULTS_DIR / f"droneface_results_det{label}.csv",
            )
            results_df.insert(0, "det_size", label)
            results_df = results_df[DET_SIZE_RESULT_COLUMNS]
            all_results.append(results_df)

        elapsed = time.perf_counter() - started
        if gallery_embeddings:
            metric_df = compute_global_metrics(results_df, gallery_embeddings, dataset_name="DroneFace")
            metric_row = metric_df.iloc[0].to_dict()
        else:
            probes_excluded_no_gallery = len(probe_df)
            metric_row = {
                "dataset": "DroneFace",
                "n_gallery_subjects": 0,
                "total_probe": 0,
                "detected_count": 0,
                "detection_rate": np.nan,
                "correct_count": 0,
                "accuracy_conditional": np.nan,
                "accuracy_global": np.nan,
                "mean_similarity_correct": np.nan,
                "mean_similarity_incorrect": np.nan,
            }

        metric_row.update(
            {
                "det_size": label,
                "gallery_subjects_before_detection": gallery_subjects_before,
                "gallery_subjects_with_embedding": len(gallery_embeddings),
                "probes_excluded_no_gallery": probes_excluded_no_gallery,
                "probes_evaluated": int(metric_row["total_probe"]),
                "runtime_seconds": elapsed,
            }
        )
        summary_rows.append(metric_row)

    combined_results = (
        pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(columns=DET_SIZE_RESULT_COLUMNS)
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[DET_SIZE_SUMMARY_COLUMNS]

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined_results.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved det_size results: {results_path}")
    print(f"Saved det_size summary: {summary_path}")
    return combined_results, summary_df


def plot_det_size_summary(summary_df: pd.DataFrame, out_path=None):
    """Plot detection and recognition metrics for each detector input size."""
    if summary_df.empty:
        warnings.warn("Cannot plot det_size summary because the dataframe is empty.", stacklevel=2)
        return None

    required = {"det_size", "detection_rate", "accuracy_conditional", "accuracy_global"}
    if not required.issubset(summary_df.columns):
        missing = ", ".join(sorted(required - set(summary_df.columns)))
        warnings.warn(f"Cannot plot det_size summary. Missing columns: {missing}", stacklevel=2)
        return None

    plot_df = summary_df.copy()
    plot_df["det_size_num"] = pd.to_numeric(plot_df["det_size"], errors="coerce")
    plot_df = plot_df.sort_values(["det_size_num", "det_size"], na_position="last")

    metrics = ["detection_rate", "accuracy_conditional", "accuracy_global"]
    colors = ["#59a14f", "#f28e2b", "#4e79a7"]
    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, metric, color in zip([-width, 0, width], metrics, colors):
        ax.bar(x + offset, plot_df[metric].astype(float).to_numpy(), width=width, label=metric, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["det_size"].astype(str).tolist())
    ax.set_xlabel("det_size")
    ax.set_ylabel("proporcion")
    ax.set_ylim(0, 1)
    ax.set_title("DroneFace - sensibilidad a det_size")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path) if out_path is not None else FIGS_DIR / "metrics" / "droneface_det_size_detection_accuracy.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved det_size summary figure: {out_path}")
    return fig


def _select_common_probe_examples(index_df: pd.DataFrame, max_subjects: int = 4) -> pd.DataFrame:
    gallery_subjects = (
        index_df[index_df["split"] == "gallery"]["subject_id"].dropna().astype(str).sort_values().unique().tolist()
    )
    probe = index_df[
        (index_df["split"] == "probe") & (index_df["subject_id"].astype(str).isin(gallery_subjects))
    ].copy()
    if probe.empty:
        return probe
    return (
        probe.sort_values(["subject_id", "height_m", "distance_m", "filename"], na_position="last")
        .groupby("subject_id", as_index=False, dropna=True)
        .head(1)
        .head(max_subjects)
    )


def plot_det_size_detection_examples(
    index_df: pd.DataFrame,
    det_sizes=(320, 640, 1024),
    model_name: str = "buffalo_l",
    allow_cuda: bool = False,
    out_path=None,
    max_subjects: int = 4,
):
    """Show the same probe images detected with several det_size values."""
    if index_df.empty:
        warnings.warn("Cannot plot det_size examples because the index is empty.", stacklevel=2)
        return None

    det_sizes = _normalize_det_sizes(det_sizes)
    sample_df = _select_common_probe_examples(index_df, max_subjects=max_subjects)
    if sample_df.empty:
        warnings.warn("No probe examples available for det_size detection figure.", stacklevel=2)
        return None

    models = {
        _det_size_label(det_size): FaceRecognizerArcFace(
            model_name=model_name,
            det_size=det_size,
            allow_cuda=allow_cuda,
        )
        for det_size in det_sizes
    }

    n_rows = len(sample_df)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
    axes = np.atleast_2d(axes)

    for row_idx, row in enumerate(sample_df.itertuples(index=False)):
        for col_idx, (label, model) in enumerate(models.items()):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            path = Path(row.path)
            title = f"sujeto {row.subject_id} | det_size={label}\n{path.name}"
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

    fig.suptitle("DroneFace - mismo probe con distintos det_size")
    fig.tight_layout()

    out_path = Path(out_path) if out_path is not None else FIGS_DIR / "detection" / "droneface_det_size_examples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved det_size detection examples: {out_path}")
    return fig


__all__ = [
    "run_det_size_sweep",
    "plot_det_size_summary",
    "plot_det_size_detection_examples",
]
