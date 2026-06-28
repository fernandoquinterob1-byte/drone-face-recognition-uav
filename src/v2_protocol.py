from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import PROJECT_ROOT
from .evaluation import (
    _best_gallery_match,
    _join_subjects,
    _metric_row,
    _safe_bool_series,
    compute_full_image_grouped_metrics,
    compute_full_image_summary,
    compute_global_metrics,
    compute_grouped_metrics,
    evaluate_closed_set,
    evaluate_full_images,
    parse_composite_subject_id,
    save_gallery_npz,
)
from .face_model import _l2_normalize


OUTPUTS_V2_DIR = PROJECT_ROOT / "outputs_v2"
V2_ANNOTATIONS_DIR = OUTPUTS_V2_DIR / "annotations"
V2_EMBEDDINGS_DIR = OUTPUTS_V2_DIR / "embeddings"
V2_RESULTS_DIR = OUTPUTS_V2_DIR / "results"
V2_TABLES_DIR = OUTPUTS_V2_DIR / "tables"
V2_FIGS_DIR = OUTPUTS_V2_DIR / "figs"
V2_CROPS_DIR = OUTPUTS_V2_DIR / "crops"

V2_DIRS = [
    V2_ANNOTATIONS_DIR,
    V2_EMBEDDINGS_DIR,
    V2_RESULTS_DIR,
    V2_TABLES_DIR,
    V2_FIGS_DIR / "dataset",
    V2_FIGS_DIR / "detection",
    V2_FIGS_DIR / "recognition",
    V2_FIGS_DIR / "metrics",
    V2_CROPS_DIR,
]

POSE_GALLERY_SPECS = {
    "trio_front": ("porF",),
    "trio_side": ("porL", "porR"),
    "trio_all": ("porF", "porL", "porR"),
}

TRIO_SOURCES = ("trio_cam", "trio_gp")
PHOTOS_ALL_PREFIXES = ("ab", "cd", "ef", "gh", "ijk")
SPATIAL_ANNOTATION_COLUMNS = ["prefix", "camera_left_to_right", "notes"]

LABELED_DETECTION_COLUMNS = [
    "gallery_name",
    "dataset",
    "path",
    "filename",
    "source_folder",
    "composite_subject_id",
    "expected_subjects",
    "spatial_rank",
    "expected_subject",
    "detection_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_x_center",
    "detection_score",
    "predicted_subject",
    "similarity",
    "match_type",
    "tp",
    "fp",
    "fn",
    "height_m",
    "distance_m",
    "image_type",
    "error",
]

BBOX_METRIC_COLUMNS = [
    "gallery_name",
    "group",
    "total_slots",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]


@dataclass(frozen=True)
class V2OutputPaths:
    root: Path = OUTPUTS_V2_DIR
    annotations: Path = V2_ANNOTATIONS_DIR
    embeddings: Path = V2_EMBEDDINGS_DIR
    results: Path = V2_RESULTS_DIR
    tables: Path = V2_TABLES_DIR
    figs: Path = V2_FIGS_DIR
    crops: Path = V2_CROPS_DIR


def ensure_outputs_v2() -> V2OutputPaths:
    """Create the V2 output tree without touching the original outputs/ tree."""
    for directory in V2_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    return V2OutputPaths()


def create_spatial_annotation_template(path: Path | None = None, overwrite: bool = False) -> Path:
    """Create the manual photos_all spatial-order template if it does not exist."""
    ensure_outputs_v2()
    path = Path(path) if path is not None else V2_ANNOTATIONS_DIR / "photos_all_spatial_order_manual.csv"
    if path.exists() and not overwrite:
        return path

    rows = [
        {
            "prefix": prefix,
            "camera_left_to_right": "",
            "notes": "Completar con separador |; ejemplo b|a",
        }
        for prefix in PHOTOS_ALL_PREFIXES
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SPATIAL_ANNOTATION_COLUMNS).to_csv(path, index=False)
    return path


def _split_subject_order(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    # Accept b|a, b,a, b a, or b>a. The pipe form avoids CSV quoting issues.
    parts = [part for part in re.split(r"[|,>\s]+", text) if part]
    return parts


def validate_spatial_annotation(path: Path | None = None) -> tuple[pd.DataFrame, bool, list[str]]:
    """Validate the manual prefix-to-left-right annotation file."""
    path = Path(path) if path is not None else V2_ANNOTATIONS_DIR / "photos_all_spatial_order_manual.csv"
    if not path.exists():
        create_spatial_annotation_template(path)

    try:
        annotation_df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        return pd.DataFrame(columns=SPATIAL_ANNOTATION_COLUMNS), False, [f"No se pudo leer {path}: {exc}"]

    errors: list[str] = []
    missing_cols = set(SPATIAL_ANNOTATION_COLUMNS) - set(annotation_df.columns)
    if missing_cols:
        errors.append(f"Faltan columnas en {path}: {', '.join(sorted(missing_cols))}")
        return annotation_df, False, errors

    prefixes = annotation_df["prefix"].astype(str).str.strip().str.lower().tolist()
    missing_prefixes = sorted(set(PHOTOS_ALL_PREFIXES) - set(prefixes))
    extra_prefixes = sorted(set(prefixes) - set(PHOTOS_ALL_PREFIXES))
    if missing_prefixes:
        errors.append(f"Faltan prefijos: {', '.join(missing_prefixes)}")
    if extra_prefixes:
        errors.append(f"Prefijos no esperados: {', '.join(extra_prefixes)}")

    parsed_orders = []
    for row in annotation_df.itertuples(index=False):
        prefix = str(row.prefix).strip().lower()
        order = _split_subject_order(row.camera_left_to_right)
        parsed_orders.append(order)
        if prefix in PHOTOS_ALL_PREFIXES:
            expected_subjects = parse_composite_subject_id(prefix)
            if not order:
                errors.append(f"Prefijo {prefix}: falta camera_left_to_right.")
            elif sorted(order) != sorted(expected_subjects):
                errors.append(
                    f"Prefijo {prefix}: orden {order} no coincide con sujetos esperados {expected_subjects}."
                )
            elif len(order) != len(expected_subjects):
                errors.append(f"Prefijo {prefix}: cantidad de sujetos no coincide con {expected_subjects}.")

    annotation_df = annotation_df.copy()
    annotation_df["parsed_camera_left_to_right"] = ["|".join(order) for order in parsed_orders]
    return annotation_df, not errors, errors


def select_gallery_rows(
    index_df: pd.DataFrame,
    sources: tuple[str, ...] = TRIO_SOURCES,
    poses: tuple[str, ...] = ("porF", "porL", "porR"),
) -> pd.DataFrame:
    """Select configurable gallery rows from DroneFace without changing the base split column."""
    if index_df.empty:
        return index_df.copy()
    required = {"source_folder", "image_type", "subject_id", "parsed"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"index_df missing required columns: {', '.join(sorted(missing))}")

    pose_set = set(poses)
    source_set = set(sources)
    selected = index_df[
        index_df["source_folder"].astype(str).isin(source_set)
        & index_df["image_type"].astype(str).isin(pose_set)
        & (index_df["parsed"] == True)
        & index_df["subject_id"].notna()
    ].copy()
    selected["split_v2"] = "gallery"
    return selected.sort_values(["subject_id", "source_folder", "image_type", "filename"]).reset_index(drop=True)


def select_photos_all_faces_probes(index_df: pd.DataFrame) -> pd.DataFrame:
    probes = index_df[
        (index_df["source_folder"].astype(str) == "photos_all_faces")
        & (index_df["image_type"].astype(str).str.lower() == "ef")
        & (index_df["parsed"] == True)
        & index_df["subject_id"].notna()
    ].copy()
    probes["split"] = "probe"
    return probes.sort_values(["subject_id", "height_m", "distance_m", "filename"]).reset_index(drop=True)


def build_configurable_gallery(
    gallery_df: pd.DataFrame,
    model,
    gallery_name: str,
    save_path: Path | None = None,
    failures_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Build mean normalized embeddings for a caller-selected gallery dataframe."""
    ensure_outputs_v2()
    save_path = Path(save_path) if save_path is not None else V2_EMBEDDINGS_DIR / f"droneface_gallery_{gallery_name}.npz"
    failures_path = (
        Path(failures_path)
        if failures_path is not None
        else V2_TABLES_DIR / f"gallery_failures_{gallery_name}.csv"
    )
    if gallery_df.empty:
        raise ValueError(f"Cannot build gallery {gallery_name}: no gallery rows selected.")

    gallery_embeddings: dict[str, np.ndarray] = {}
    info_rows = []
    failure_rows = []

    for subject_id, subject_df in tqdm(
        gallery_df.groupby("subject_id", dropna=True),
        desc=f"Building gallery {gallery_name}",
        total=gallery_df["subject_id"].dropna().nunique(),
    ):
        valid_embeddings = []
        subject_errors = []
        for row in subject_df.itertuples(index=False):
            extraction = model.extract_single_face_embedding(row.path)
            if extraction["detected"] and extraction["embedding"] is not None:
                valid_embeddings.append(extraction["embedding"])
            else:
                error = extraction.get("error", "No face detected.")
                subject_errors.append(error)
                failure_rows.append(
                    {
                        "gallery_name": gallery_name,
                        "subject_id": subject_id,
                        "path": row.path,
                        "filename": row.filename,
                        "source_folder": row.source_folder,
                        "image_type": row.image_type,
                        "detected": extraction["detected"],
                        "num_faces_detected": extraction["num_faces_detected"],
                        "error": error,
                    }
                )

        if valid_embeddings:
            mean_embedding = _l2_normalize(np.mean(np.vstack(valid_embeddings), axis=0))
            gallery_embeddings[str(subject_id)] = mean_embedding
            included = True
            error = ""
        else:
            included = False
            error = "; ".join(sorted(set(filter(None, subject_errors)))) or "No valid gallery embedding."

        info_rows.append(
            {
                "gallery_name": gallery_name,
                "subject_id": subject_id,
                "n_gallery_images": len(subject_df),
                "n_valid_embeddings": len(valid_embeddings),
                "n_failed_images": len(subject_df) - len(valid_embeddings),
                "included": included,
                "source_folders": ",".join(sorted(subject_df["source_folder"].astype(str).unique())),
                "poses": ",".join(sorted(subject_df["image_type"].astype(str).unique())),
                "error": error,
            }
        )

    info_df = pd.DataFrame(info_rows)
    failures_df = pd.DataFrame(failure_rows)
    if gallery_embeddings:
        save_gallery_npz(gallery_embeddings, save_path)
        print(f"Saved V2 gallery embeddings: {save_path}")
    else:
        warnings.warn(f"Gallery {gallery_name} produced no valid embeddings.", stacklevel=2)

    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_df.to_csv(failures_path, index=False)
    print(f"Saved V2 gallery failures: {failures_path}")
    return gallery_embeddings, info_df, failures_df


def run_pose_gallery_experiments(
    index_df: pd.DataFrame,
    model,
    gallery_specs: dict[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Build trio pose galleries and evaluate photos_all_faces for each one."""
    ensure_outputs_v2()
    gallery_specs = gallery_specs or POSE_GALLERY_SPECS
    probes_df = select_photos_all_faces_probes(index_df)
    if probes_df.empty:
        raise ValueError("No photos_all_faces probes were found for V2 experiment A.")

    experiment_outputs: dict[str, dict] = {}
    summary_rows = []
    gallery_info_frames = []

    for gallery_name, poses in gallery_specs.items():
        gallery_df = select_gallery_rows(index_df, poses=poses)
        gallery_embeddings, gallery_info_df, failures_df = build_configurable_gallery(
            gallery_df,
            model,
            gallery_name=gallery_name,
            save_path=V2_EMBEDDINGS_DIR / f"droneface_gallery_{gallery_name}.npz",
            failures_path=V2_TABLES_DIR / f"gallery_failures_{gallery_name}.csv",
        )
        gallery_info_frames.append(gallery_info_df)

        results_path = V2_RESULTS_DIR / f"photos_all_faces_{gallery_name}.csv"
        results_df = evaluate_closed_set(probes_df, gallery_embeddings, model, save_path=results_path)
        results_df.insert(0, "gallery_name", gallery_name)
        results_df.to_csv(results_path, index=False)

        summary_df = compute_global_metrics(results_df, gallery_embeddings, dataset_name="DroneFace")
        summary_row = summary_df.iloc[0].to_dict()
        summary_row.update(
            {
                "gallery_name": gallery_name,
                "poses": ",".join(poses),
                "gallery_rows": len(gallery_df),
                "gallery_subjects_before_detection": gallery_df["subject_id"].dropna().astype(str).nunique(),
                "gallery_subjects_with_embedding": len(gallery_embeddings),
                "total_available_probes": len(probes_df),
                "probes_excluded_no_gallery": len(probes_df) - len(results_df),
            }
        )
        summary_rows.append(summary_row)

        by_subject = compute_grouped_metrics(results_df, ["subject_id"])
        by_height = compute_grouped_metrics(results_df, ["height_m"])
        by_distance = compute_grouped_metrics(results_df, ["distance_m"])
        by_height_distance = compute_grouped_metrics(results_df, ["height_m", "distance_m"])
        for table_name, table_df in [
            ("by_subject", by_subject),
            ("by_height", by_height),
            ("by_distance", by_distance),
            ("by_height_distance", by_height_distance),
        ]:
            table_df.insert(0, "gallery_name", gallery_name)
            table_df.to_csv(V2_TABLES_DIR / f"photos_all_faces_{gallery_name}_{table_name}.csv", index=False)

        experiment_outputs[gallery_name] = {
            "gallery_df": gallery_df,
            "gallery_embeddings": gallery_embeddings,
            "gallery_info_df": gallery_info_df,
            "gallery_failures_df": failures_df,
            "results_df": results_df,
            "by_subject": by_subject,
            "by_height": by_height,
            "by_distance": by_distance,
            "by_height_distance": by_height_distance,
        }

    comparison_df = pd.DataFrame(summary_rows)
    comparison_df = comparison_df[
        [
            "gallery_name",
            "poses",
            "gallery_rows",
            "gallery_subjects_before_detection",
            "gallery_subjects_with_embedding",
            "n_gallery_subjects",
            "total_available_probes",
            "total_probe",
            "probes_excluded_no_gallery",
            "detected_count",
            "detection_rate",
            "correct_count",
            "accuracy_conditional",
            "accuracy_global",
            "mean_similarity_correct",
            "mean_similarity_incorrect",
        ]
    ]
    comparison_path = V2_TABLES_DIR / "photos_all_faces_pose_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    gallery_info_all_df = pd.concat(gallery_info_frames, ignore_index=True) if gallery_info_frames else pd.DataFrame()
    gallery_info_all_df.to_csv(V2_TABLES_DIR / "gallery_pose_info.csv", index=False)
    return experiment_outputs, comparison_df, gallery_info_all_df


def run_full_image_set_experiments(
    index_df: pd.DataFrame,
    model,
    gallery_outputs: dict[str, dict],
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Evaluate photos_all with the existing image-level expected-subject set metric."""
    ensure_outputs_v2()
    full_outputs = {}
    summary_frames = []

    for gallery_name, payload in gallery_outputs.items():
        gallery_embeddings = payload["gallery_embeddings"]
        detections_path = V2_RESULTS_DIR / f"photos_all_{gallery_name}_detections.csv"
        results_path = V2_RESULTS_DIR / f"photos_all_{gallery_name}.csv"
        detections_df, results_df = evaluate_full_images(
            index_df,
            gallery_embeddings,
            model,
            detections_path=detections_path,
            results_path=results_path,
        )
        detections_df.insert(0, "gallery_name", gallery_name)
        results_df.insert(0, "gallery_name", gallery_name)
        detections_df.to_csv(detections_path, index=False)
        results_df.to_csv(results_path, index=False)

        summary_df = compute_full_image_summary(results_df, dataset_name="DroneFace")
        summary_df.insert(0, "gallery_name", gallery_name)
        summary_frames.append(summary_df)

        for group_cols, table_name in [
            (["height_m"], "by_height"),
            (["distance_m"], "by_distance"),
            (["height_m", "distance_m"], "by_height_distance"),
        ]:
            grouped = compute_full_image_grouped_metrics(results_df, group_cols)
            grouped.insert(0, "gallery_name", gallery_name)
            grouped.to_csv(V2_TABLES_DIR / f"photos_all_{gallery_name}_{table_name}.csv", index=False)

        full_outputs[gallery_name] = {"detections_df": detections_df, "results_df": results_df}

    summary_all_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_all_df.to_csv(V2_TABLES_DIR / "photos_all_set_metrics_by_gallery.csv", index=False)
    return full_outputs, summary_all_df


def _row_value(row: pd.Series | None, column: str, default=np.nan):
    if row is None or column not in row:
        return default
    return row[column]


def label_full_image_detections_from_spatial_order(
    index_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    gallery_name: str,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Assign expected identities to photos_all detections by horizontal rank."""
    ensure_outputs_v2()
    if "parsed_camera_left_to_right" not in annotation_df.columns:
        annotation_df = annotation_df.copy()
        annotation_df["parsed_camera_left_to_right"] = annotation_df["camera_left_to_right"].apply(
            lambda value: "|".join(_split_subject_order(value))
        )

    order_by_prefix = {
        str(row.prefix).strip().lower(): _split_subject_order(row.parsed_camera_left_to_right)
        for row in annotation_df.itertuples(index=False)
    }
    eval_df = index_df[
        (index_df["source_folder"].astype(str) == "photos_all")
        & (index_df["image_type"].astype(str).str.lower() == "eo")
    ].copy()

    rows = []
    for image_row in eval_df.itertuples(index=False):
        prefix = str(image_row.subject_id).strip().lower()
        expected_order = order_by_prefix.get(prefix, [])
        image_detections = detections_df[detections_df["filename"].astype(str) == str(image_row.filename)].copy()
        image_detections = image_detections.dropna(subset=["bbox_x1", "bbox_x2"], how="any")
        if not image_detections.empty:
            image_detections["bbox_x_center"] = (
                pd.to_numeric(image_detections["bbox_x1"], errors="coerce")
                + pd.to_numeric(image_detections["bbox_x2"], errors="coerce")
            ) / 2
            image_detections = image_detections.sort_values("bbox_x_center").reset_index(drop=True)

        n_slots = max(len(expected_order), len(image_detections))
        for spatial_rank in range(n_slots):
            expected_subject = expected_order[spatial_rank] if spatial_rank < len(expected_order) else ""
            det = image_detections.iloc[spatial_rank] if spatial_rank < len(image_detections) else None
            predicted_subject = str(_row_value(det, "predicted_subject", "") or "")
            has_detection = det is not None

            if expected_subject and has_detection and predicted_subject == expected_subject:
                match_type, tp, fp, fn = "correct_identity", 1, 0, 0
            elif expected_subject and has_detection:
                match_type, tp, fp, fn = "wrong_identity", 0, 1, 1
            elif expected_subject and not has_detection:
                match_type, tp, fp, fn = "missing_detection", 0, 0, 1
            elif not expected_subject and has_detection:
                match_type, tp, fp, fn = "extra_detection", 0, 1, 0
            else:
                match_type, tp, fp, fn = "empty", 0, 0, 0

            rows.append(
                {
                    "gallery_name": gallery_name,
                    "dataset": getattr(image_row, "dataset", "DroneFace"),
                    "path": getattr(image_row, "path", ""),
                    "filename": image_row.filename,
                    "source_folder": image_row.source_folder,
                    "composite_subject_id": prefix,
                    "expected_subjects": _join_subjects(expected_order),
                    "spatial_rank": spatial_rank,
                    "expected_subject": expected_subject,
                    "detection_index": _row_value(det, "detection_index"),
                    "bbox_x1": _row_value(det, "bbox_x1"),
                    "bbox_y1": _row_value(det, "bbox_y1"),
                    "bbox_x2": _row_value(det, "bbox_x2"),
                    "bbox_y2": _row_value(det, "bbox_y2"),
                    "bbox_x_center": _row_value(det, "bbox_x_center"),
                    "detection_score": _row_value(det, "detection_score"),
                    "predicted_subject": predicted_subject,
                    "similarity": _row_value(det, "similarity"),
                    "match_type": match_type,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "height_m": getattr(image_row, "height_m", np.nan),
                    "distance_m": getattr(image_row, "distance_m", np.nan),
                    "image_type": getattr(image_row, "image_type", ""),
                    "error": _row_value(det, "error", ""),
                }
            )

    labeled_df = pd.DataFrame(rows, columns=LABELED_DETECTION_COLUMNS)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_csv(save_path, index=False)
    return labeled_df


def compute_bbox_level_metrics(labeled_df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute identity-aware bbox-level TP/FP/FN metrics from labeled detection rows."""
    if labeled_df.empty:
        return pd.DataFrame(columns=BBOX_METRIC_COLUMNS)
    group_cols = group_cols or ["gallery_name"]

    rows = []
    for group_key, group_df in labeled_df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        tp = int(pd.to_numeric(group_df["tp"], errors="coerce").fillna(0).sum())
        fp = int(pd.to_numeric(group_df["fp"], errors="coerce").fillna(0).sum())
        fn = int(pd.to_numeric(group_df["fn"], errors="coerce").fillna(0).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        if pd.isna(precision) or pd.isna(recall):
            f1 = np.nan
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        group_label = ", ".join(f"{col}={value}" for col, value in zip(group_cols, group_key))
        rows.append(
            {
                "gallery_name": group_df["gallery_name"].iloc[0] if "gallery_name" in group_df.columns else "",
                "group": group_label,
                "total_slots": int(len(group_df)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows, columns=BBOX_METRIC_COLUMNS)


def run_bbox_level_if_annotations_complete(
    index_df: pd.DataFrame,
    full_outputs: dict[str, dict],
    annotation_path: Path | None = None,
) -> tuple[bool, pd.DataFrame, pd.DataFrame, list[str]]:
    """Run optional bbox-level evaluation only when manual annotations are complete."""
    annotation_df, is_complete, errors = validate_spatial_annotation(annotation_path)
    if not is_complete:
        return False, pd.DataFrame(columns=LABELED_DETECTION_COLUMNS), pd.DataFrame(columns=BBOX_METRIC_COLUMNS), errors

    labeled_frames = []
    metric_frames = []
    for gallery_name, payload in full_outputs.items():
        labeled_df = label_full_image_detections_from_spatial_order(
            index_df,
            payload["detections_df"],
            annotation_df,
            gallery_name=gallery_name,
            save_path=V2_RESULTS_DIR / f"photos_all_{gallery_name}_detections_labeled.csv",
        )
        labeled_frames.append(labeled_df)
        metrics_global = compute_bbox_level_metrics(labeled_df, ["gallery_name"])
        metrics_by_prefix = compute_bbox_level_metrics(labeled_df, ["gallery_name", "composite_subject_id"])
        metrics_global.to_csv(V2_TABLES_DIR / f"photos_all_{gallery_name}_bbox_metrics.csv", index=False)
        metrics_by_prefix.to_csv(V2_TABLES_DIR / f"photos_all_{gallery_name}_bbox_metrics_by_prefix.csv", index=False)
        metric_frames.extend([metrics_global, metrics_by_prefix])

    all_labeled_df = pd.concat(labeled_frames, ignore_index=True) if labeled_frames else pd.DataFrame()
    all_metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    all_labeled_df.to_csv(V2_RESULTS_DIR / "photos_all_detections_labeled_all_galleries.csv", index=False)
    all_metrics_df.to_csv(V2_TABLES_DIR / "photos_all_bbox_metrics_by_gallery.csv", index=False)
    return True, all_labeled_df, all_metrics_df, []


def plot_pose_comparison(comparison_df: pd.DataFrame, out_path: Path):
    if comparison_df.empty:
        warnings.warn("Cannot plot pose comparison: dataframe is empty.", stacklevel=2)
        return None
    required = {"gallery_name", "accuracy_global", "accuracy_conditional", "detection_rate"}
    missing = required - set(comparison_df.columns)
    if missing:
        warnings.warn(f"Cannot plot pose comparison. Missing columns: {', '.join(sorted(missing))}", stacklevel=2)
        return None

    plot_df = comparison_df.sort_values("gallery_name").copy()
    labels = plot_df["gallery_name"].astype(str).tolist()
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width, plot_df["detection_rate"].astype(float), width=width, label="detection_rate")
    ax.bar(x, plot_df["accuracy_global"].astype(float), width=width, label="accuracy_global")
    ax.bar(x + width, plot_df["accuracy_conditional"].astype(float), width=width, label="accuracy_conditional")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("proporcion")
    ax.set_title("DroneFace V2 - comparacion por poses de galeria")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pose comparison figure: {out_path}")
    return out_path


def _bbox_metric_top_level(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=BBOX_METRIC_COLUMNS)
    required = {"gallery_name", "group"}
    missing = required - set(metrics_df.columns)
    if missing:
        warnings.warn(f"Cannot select bbox metrics. Missing columns: {', '.join(sorted(missing))}", stacklevel=2)
        return pd.DataFrame(columns=BBOX_METRIC_COLUMNS)

    df = metrics_df.copy()
    mask = df.apply(lambda row: str(row["group"]) == f"gallery_name={row['gallery_name']}", axis=1)
    top = df[mask].copy()
    numeric_cols = ["total_slots", "tp", "fp", "fn", "precision", "recall", "f1"]
    for col in numeric_cols:
        if col in top.columns:
            top[col] = pd.to_numeric(top[col], errors="coerce")
    return top.sort_values("gallery_name").reset_index(drop=True)


def _save_bbox_metric_fig(fig, out_path: Path, label: str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {label} figure: {out_path}")
    return out_path


def plot_bbox_metrics_by_gallery(metrics_df: pd.DataFrame, out_path: Path):
    """Plot bbox-level precision, recall, and F1 for the top-level gallery rows."""
    top = _bbox_metric_top_level(metrics_df)
    if top.empty:
        warnings.warn("Cannot plot bbox metrics by gallery: no top-level rows found.", stacklevel=2)
        return None

    labels = top["gallery_name"].astype(str).tolist()
    x = np.arange(len(top))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width, top["precision"], width=width, label="precision")
    ax.bar(x, top["recall"], width=width, label="recall")
    ax.bar(x + width, top["f1"], width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("proporcion")
    ax.set_title("DroneFace V2 - bbox-level precision/recall/F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_bbox_metric_fig(fig, out_path, "bbox metrics by gallery")


def plot_bbox_counts_by_gallery(metrics_df: pd.DataFrame, out_path: Path):
    """Plot bbox-level TP, FP, and FN counts for the top-level gallery rows."""
    top = _bbox_metric_top_level(metrics_df)
    if top.empty:
        warnings.warn("Cannot plot bbox counts by gallery: no top-level rows found.", stacklevel=2)
        return None

    labels = top["gallery_name"].astype(str).tolist()
    x = np.arange(len(top))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width, top["tp"], width=width, label="TP")
    ax.bar(x, top["fp"], width=width, label="FP")
    ax.bar(x + width, top["fn"], width=width, label="FN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("conteo")
    ax.set_title("DroneFace V2 - bbox-level TP/FP/FN")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_bbox_metric_fig(fig, out_path, "bbox counts by gallery")


def plot_bbox_f1_by_prefix(metrics_df: pd.DataFrame, out_path: Path):
    """Plot bbox-level F1 by photos_all prefix and gallery."""
    if metrics_df.empty:
        warnings.warn("Cannot plot bbox F1 by prefix: dataframe is empty.", stacklevel=2)
        return None
    required = {"gallery_name", "group", "f1"}
    missing = required - set(metrics_df.columns)
    if missing:
        warnings.warn(f"Cannot plot bbox F1 by prefix. Missing columns: {', '.join(sorted(missing))}", stacklevel=2)
        return None

    rows = []
    for row in metrics_df.itertuples(index=False):
        group = str(getattr(row, "group"))
        if "composite_subject_id=" not in group:
            continue
        prefix = group.split("composite_subject_id=", 1)[1].split(",", 1)[0].strip()
        rows.append(
            {
                "gallery_name": str(getattr(row, "gallery_name")),
                "prefix": prefix,
                "f1": pd.to_numeric(getattr(row, "f1"), errors="coerce"),
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        warnings.warn("Cannot plot bbox F1 by prefix: no prefix rows found.", stacklevel=2)
        return None

    pivot = plot_df.pivot(index="prefix", columns="gallery_name", values="f1").sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylim(0, 1)
    ax.set_xlabel("prefijo photos_all")
    ax.set_ylabel("F1")
    ax.set_title("DroneFace V2 - bbox-level F1 por prefijo")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="galeria")
    fig.tight_layout()
    return _save_bbox_metric_fig(fig, out_path, "bbox F1 by prefix")
