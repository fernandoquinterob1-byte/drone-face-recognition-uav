from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from .config import FIGS_DIR, RESULTS_DIR, TABLES_DIR, ensure_project_dirs
from .face_model import _bbox_area, _l2_normalize, _maybe_upscale


SYNTHETIC_RESULT_COLUMNS = [
    "dataset",
    "path",
    "filename",
    "source_folder",
    "subject_id",
    "degradation_level",
    "downscale_factor",
    "blur_kernel",
    "jpeg_quality",
    "shift_x",
    "shift_y",
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

SYNTHETIC_SUMMARY_COLUMNS = [
    "dataset",
    "degradation_level",
    "downscale_factor",
    "blur_kernel",
    "jpeg_quality",
    "max_abs_shift",
    "total_synthetic_probes",
    "detected_count",
    "detection_rate",
    "correct_count",
    "rank1_accuracy_conditional",
    "rank1_accuracy_global",
    "mean_similarity_correct",
    "mean_similarity_incorrect",
]

DEFAULT_DEGRADATIONS = [
    {
        "degradation_level": "clean_reference",
        "downscale_factor": 1.0,
        "blur_kernel": 0,
        "jpeg_quality": 100,
        "max_abs_shift": 0,
    },
    {
        "degradation_level": "mild",
        "downscale_factor": 0.75,
        "blur_kernel": 3,
        "jpeg_quality": 80,
        "max_abs_shift": 2,
    },
    {
        "degradation_level": "medium",
        "downscale_factor": 0.50,
        "blur_kernel": 5,
        "jpeg_quality": 60,
        "max_abs_shift": 4,
    },
    {
        "degradation_level": "strong",
        "downscale_factor": 0.35,
        "blur_kernel": 7,
        "jpeg_quality": 45,
        "max_abs_shift": 6,
    },
]


def _safe_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _read_bgr(path) -> np.ndarray | None:
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        image = cv2.imread(str(path))
    return image


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def _apply_degradation(image: np.ndarray, config: dict, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    degraded = image.copy()
    original_height, original_width = degraded.shape[:2]

    scale = float(config.get("downscale_factor", 1.0))
    if 0 < scale < 1:
        small_width = max(1, int(round(original_width * scale)))
        small_height = max(1, int(round(original_height * scale)))
        degraded = cv2.resize(degraded, (small_width, small_height), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(degraded, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

    blur_kernel = int(config.get("blur_kernel", 0))
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        degraded = cv2.GaussianBlur(degraded, (blur_kernel, blur_kernel), sigmaX=0)

    jpeg_quality = int(config.get("jpeg_quality", 100))
    if jpeg_quality < 100:
        degraded = _jpeg_roundtrip(degraded, jpeg_quality)

    max_abs_shift = int(config.get("max_abs_shift", 0))
    if max_abs_shift > 0:
        shift_x = int(rng.integers(-max_abs_shift, max_abs_shift + 1))
        shift_y = int(rng.integers(-max_abs_shift, max_abs_shift + 1))
        transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        degraded = cv2.warpAffine(
            degraded,
            transform,
            (original_width, original_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    else:
        shift_x = 0
        shift_y = 0

    return degraded, shift_x, shift_y


def _extract_single_face_embedding_from_image(model, image: np.ndarray, allow_upscale: bool = True) -> dict:
    result = {
        "detected": False,
        "embedding": None,
        "bbox": None,
        "detection_score": np.nan,
        "num_faces_detected": 0,
        "upscaled_for_detection": False,
        "error": "",
    }

    image_used = image
    if allow_upscale:
        image_used, upscaled = _maybe_upscale(image)
        result["upscaled_for_detection"] = upscaled

    try:
        faces = model.detect_faces(image_used)
    except Exception as exc:
        result["error"] = f"InsightFace detection failed: {exc}"
        return result

    result["num_faces_detected"] = len(faces)
    if not faces:
        return result

    face = max(faces, key=_bbox_area)
    embedding = getattr(face, "embedding", None)
    if embedding is None:
        result["error"] = "Face detected but embedding is missing."
        return result

    bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
    result["detected"] = True
    result["embedding"] = _l2_normalize(embedding)
    result["bbox"] = bbox.tolist() if bbox.size else None
    result["detection_score"] = float(getattr(face, "det_score", np.nan))
    return result


def _best_gallery_match(embedding, gallery_embeddings: dict[str, np.ndarray], model):
    best_subject = None
    best_similarity = -np.inf
    for subject_id, gallery_embedding in gallery_embeddings.items():
        similarity = model.cosine_similarity(embedding, gallery_embedding)
        if np.isnan(similarity):
            continue
        if similarity > best_similarity:
            best_similarity = similarity
            best_subject = subject_id
    if best_subject is None:
        return None, np.nan
    return best_subject, float(best_similarity)


def _select_known_identity_probes(
    index_df: pd.DataFrame,
    experiment_a_results_df: pd.DataFrame | None,
    gallery_subjects: set[str],
    max_probes_per_subject: int | None,
) -> pd.DataFrame:
    if experiment_a_results_df is not None and not experiment_a_results_df.empty:
        detected = _safe_bool_series(experiment_a_results_df["detected"])
        correct = _safe_bool_series(experiment_a_results_df["correct"])
        probe_df = experiment_a_results_df[
            detected
            & correct
            & experiment_a_results_df["subject_id"].astype(str).isin(gallery_subjects)
            & experiment_a_results_df["source_folder"].astype(str).eq("photos_all_faces")
        ].copy()
    else:
        probe_df = index_df[
            index_df["split"].astype(str).eq("probe")
            & index_df["subject_id"].astype(str).isin(gallery_subjects)
            & index_df["source_folder"].astype(str).eq("photos_all_faces")
        ].copy()

    if probe_df.empty:
        raise ValueError("No known-identity probes are available for synthetic degradation.")

    sort_cols = [col for col in ["subject_id", "height_m", "distance_m", "filename"] if col in probe_df.columns]
    probe_df = probe_df.sort_values(sort_cols, na_position="last")
    if max_probes_per_subject is not None:
        if max_probes_per_subject <= 0:
            raise ValueError("max_probes_per_subject must be positive or None.")
        probe_df = probe_df.groupby("subject_id", as_index=False, dropna=True).head(max_probes_per_subject)
    return probe_df.reset_index(drop=True)


def compute_synthetic_degradation_summary(results_df: pd.DataFrame, dataset_name: str = "DroneFace") -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=SYNTHETIC_SUMMARY_COLUMNS)

    rows = []
    for level, group in results_df.groupby("degradation_level", sort=False, dropna=False):
        total = len(group)
        detected = _safe_bool_series(group["detected"])
        correct = _safe_bool_series(group["correct"])
        detected_count = int(detected.sum())
        correct_count = int(correct.sum())
        correct_detected = group[detected & correct]
        incorrect_detected = group[detected & ~correct]
        first = group.iloc[0]
        rows.append(
            {
                "dataset": dataset_name,
                "degradation_level": level,
                "downscale_factor": first.get("downscale_factor", np.nan),
                "blur_kernel": first.get("blur_kernel", np.nan),
                "jpeg_quality": first.get("jpeg_quality", np.nan),
                "max_abs_shift": max(group["shift_x"].abs().max(), group["shift_y"].abs().max()),
                "total_synthetic_probes": total,
                "detected_count": detected_count,
                "detection_rate": detected_count / total if total else np.nan,
                "correct_count": correct_count,
                "rank1_accuracy_conditional": correct_count / detected_count if detected_count else np.nan,
                "rank1_accuracy_global": correct_count / total if total else np.nan,
                "mean_similarity_correct": correct_detected["similarity"].mean(),
                "mean_similarity_incorrect": incorrect_detected["similarity"].mean(),
            }
        )

    return pd.DataFrame(rows, columns=SYNTHETIC_SUMMARY_COLUMNS)


def run_synthetic_degradation_experiment(
    index_df: pd.DataFrame,
    experiment_a_results_df: pd.DataFrame | None,
    gallery_embeddings: dict[str, np.ndarray],
    model,
    degradation_levels: list[dict] | None = None,
    max_probes_per_subject: int | None = 30,
    seed: int = 42,
    results_path=None,
    summary_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Rank-1 recognition on known identities after synthetic degradation."""
    ensure_project_dirs()
    if not gallery_embeddings:
        raise ValueError("Cannot run synthetic degradation without gallery embeddings.")

    degradation_levels = degradation_levels or DEFAULT_DEGRADATIONS
    gallery_subjects = set(str(subject) for subject in gallery_embeddings.keys())
    probes_df = _select_known_identity_probes(
        index_df,
        experiment_a_results_df,
        gallery_subjects,
        max_probes_per_subject,
    )

    rows = []
    rng = np.random.default_rng(seed)
    for row_idx, row in enumerate(
        tqdm(probes_df.itertuples(index=False), total=len(probes_df), desc="Synthetic degradation probes")
    ):
        image = _read_bgr(row.path)
        for level_idx, degradation in enumerate(degradation_levels):
            level_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + row_idx * 101 + level_idx)
            predicted_subject = None
            similarity = np.nan
            correct = False
            error = ""
            shift_x = 0
            shift_y = 0
            extraction = {
                "detected": False,
                "num_faces_detected": 0,
                "detection_score": np.nan,
                "upscaled_for_detection": False,
            }

            if image is None:
                error = f"Could not read image: {row.path}"
            else:
                degraded, shift_x, shift_y = _apply_degradation(image, degradation, level_rng)
                extraction = _extract_single_face_embedding_from_image(model, degraded)
                error = extraction.get("error", "")
                if extraction["detected"] and extraction.get("embedding") is not None:
                    predicted_subject, similarity = _best_gallery_match(
                        extraction["embedding"],
                        gallery_embeddings,
                        model,
                    )
                    correct = str(predicted_subject) == str(row.subject_id)

            rows.append(
                {
                    "dataset": getattr(row, "dataset", "DroneFace"),
                    "path": row.path,
                    "filename": row.filename,
                    "source_folder": getattr(row, "source_folder", "photos_all_faces"),
                    "subject_id": row.subject_id,
                    "degradation_level": degradation["degradation_level"],
                    "downscale_factor": degradation.get("downscale_factor", 1.0),
                    "blur_kernel": degradation.get("blur_kernel", 0),
                    "jpeg_quality": degradation.get("jpeg_quality", 100),
                    "shift_x": shift_x,
                    "shift_y": shift_y,
                    "predicted_subject": predicted_subject,
                    "detected": extraction["detected"],
                    "num_faces_detected": extraction["num_faces_detected"],
                    "detection_score": extraction["detection_score"],
                    "similarity": similarity,
                    "correct": correct,
                    "upscaled_for_detection": extraction["upscaled_for_detection"],
                    "height_m": getattr(row, "height_m", np.nan),
                    "distance_m": getattr(row, "distance_m", np.nan),
                    "image_type": getattr(row, "image_type", None),
                    "error": error,
                }
            )

    results_df = pd.DataFrame(rows, columns=SYNTHETIC_RESULT_COLUMNS)
    summary_df = compute_synthetic_degradation_summary(results_df)

    results_path = (
        Path(results_path)
        if results_path is not None
        else RESULTS_DIR / "droneface_synthetic_degradation_results.csv"
    )
    summary_path = (
        Path(summary_path)
        if summary_path is not None
        else TABLES_DIR / "droneface_synthetic_degradation_summary.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved synthetic degradation results: {results_path}")
    print(f"Saved synthetic degradation summary: {summary_path}")
    return results_df, summary_df


def plot_synthetic_degradation_summary(summary_df: pd.DataFrame, out_path=None):
    if summary_df.empty:
        return None

    plot_df = summary_df.copy()
    labels = plot_df["degradation_level"].astype(str).tolist()
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    metrics = ["detection_rate", "rank1_accuracy_conditional", "rank1_accuracy_global"]
    colors = ["#59a14f", "#f28e2b", "#4e79a7"]
    offsets = [-width, 0, width]
    for metric, color, offset in zip(metrics, colors, offsets):
        ax.bar(x + offset, plot_df[metric].astype(float), width=width, label=metric, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("proporcion")
    ax.set_title("DroneFace - Experimento C: Rank-1 bajo degradacion sintetica")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = (
        Path(out_path)
        if out_path is not None
        else FIGS_DIR / "metrics" / "droneface_synthetic_degradation_rank1.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved synthetic degradation summary figure: {out_path}")
    return fig


def plot_synthetic_degradation_examples(
    index_df: pd.DataFrame,
    degradation_levels: list[dict] | None = None,
    out_path=None,
    max_subjects: int = 4,
    subject_ids=None,
    seed: int = 42,
):
    if index_df.empty:
        return None

    degradation_levels = degradation_levels or DEFAULT_DEGRADATIONS
    sample_df = index_df[
        index_df["split"].astype(str).eq("probe")
        & index_df["source_folder"].astype(str).eq("photos_all_faces")
    ].copy()
    if subject_ids is not None:
        valid_subjects = set(str(subject) for subject in subject_ids)
        sample_df = sample_df[sample_df["subject_id"].astype(str).isin(valid_subjects)]

    sample_df = (
        sample_df.sort_values(["subject_id", "height_m", "distance_m", "filename"], na_position="last")
        .groupby("subject_id", as_index=False, dropna=True)
        .head(1)
        .head(max_subjects)
    )
    if sample_df.empty:
        return None

    n_rows = len(sample_df)
    n_cols = len(degradation_levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.0 * n_rows))
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(seed)

    for row_idx, row in enumerate(sample_df.itertuples(index=False)):
        image = _read_bgr(row.path)
        for col_idx, degradation in enumerate(degradation_levels):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if image is None:
                ax.text(0.5, 0.5, "unreadable image", ha="center", va="center")
                continue
            level_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + row_idx * 101 + col_idx)
            degraded, _, _ = _apply_degradation(image, degradation, level_rng)
            ax.imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"sujeto {row.subject_id} | {degradation['degradation_level']}\n{Path(row.path).name}",
                fontsize=8,
            )

    fig.suptitle("DroneFace - Experimento C: ejemplos con degradacion sintetica")
    fig.tight_layout()
    out_path = (
        Path(out_path)
        if out_path is not None
        else FIGS_DIR / "recognition" / "droneface_synthetic_degradation_examples.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved synthetic degradation examples: {out_path}")
    return fig


__all__ = [
    "DEFAULT_DEGRADATIONS",
    "SYNTHETIC_RESULT_COLUMNS",
    "SYNTHETIC_SUMMARY_COLUMNS",
    "compute_synthetic_degradation_summary",
    "run_synthetic_degradation_experiment",
    "plot_synthetic_degradation_summary",
    "plot_synthetic_degradation_examples",
]
