from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import CROPS_DIR, EMBEDDINGS_DIR, FIGS_DIR, RESULTS_DIR, TABLES_DIR, ensure_project_dirs
from .face_model import _bbox_area, _l2_normalize, _maybe_upscale

GALLERY_FAILURE_COLUMNS = [
    "dataset",
    "subject_id",
    "path",
    "filename",
    "detected",
    "num_faces_detected",
    "error",
]

RESULT_COLUMNS = [
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

FULL_IMAGE_DETECTION_COLUMNS = [
    "dataset",
    "path",
    "filename",
    "source_folder",
    "composite_subject_id",
    "expected_subjects",
    "expected_subjects_with_gallery",
    "expected_subjects_without_gallery",
    "detection_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "detection_score",
    "predicted_subject",
    "similarity",
    "height_m",
    "distance_m",
    "image_type",
    "error",
]

FULL_IMAGE_CROP_COLUMNS = [
    "dataset",
    "source_path",
    "source_filename",
    "crop_path",
    "crop_filename",
    "review_path",
    "source_folder",
    "composite_subject_id",
    "expected_subjects",
    "expected_subjects_with_gallery",
    "expected_subjects_without_gallery",
    "detection_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "detection_score",
    "height_m",
    "distance_m",
    "image_type",
    "upscaled_for_detection",
    "image_shape_original",
    "image_shape_used",
    "error",
]

FULL_IMAGE_CROP_RECOGNITION_COLUMNS = [
    "dataset",
    "source_path",
    "source_filename",
    "crop_path",
    "crop_filename",
    "source_folder",
    "composite_subject_id",
    "expected_subjects",
    "expected_subjects_with_gallery",
    "expected_subjects_without_gallery",
    "detection_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "source_detection_score",
    "recognition_source",
    "crop_detected",
    "crop_num_faces_detected",
    "crop_detection_score",
    "predicted_subject",
    "similarity",
    "height_m",
    "distance_m",
    "image_type",
    "error",
]

FULL_IMAGE_RESULT_COLUMNS = [
    "dataset",
    "path",
    "filename",
    "source_folder",
    "composite_subject_id",
    "expected_subjects",
    "expected_subjects_with_gallery",
    "expected_subjects_without_gallery",
    "predicted_subjects",
    "predicted_unexpected_subjects",
    "num_faces_detected",
    "num_recognized_faces",
    "found_expected_subjects",
    "n_expected_with_gallery",
    "n_expected_found",
    "expected_identity_recall",
    "any_expected_found",
    "all_expected_found",
    "height_m",
    "distance_m",
    "image_type",
    "error",
]

FULL_IMAGE_SUMMARY_COLUMNS = [
    "dataset",
    "total_images",
    "images_with_detections",
    "image_detection_rate",
    "total_expected_identities_with_gallery",
    "expected_identities_found",
    "expected_identity_recall_global",
    "images_any_expected_found",
    "any_expected_found_rate",
    "images_all_expected_found",
    "all_expected_found_rate",
    "mean_expected_identity_recall",
    "expected_identities_without_gallery",
]

COMMON_EXPERIMENT_METRIC_COLUMNS = [
    "dataset",
    "experiment",
    "unit",
    "metric_scope",
    "total_units",
    "detected_units",
    "detection_rate",
    "total_expected_identities",
    "total_predicted_identities",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "exact_match_count",
    "exact_match_rate",
    "macro_f1",
    "notes",
]


def _safe_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _metric_row(results_df: pd.DataFrame, dataset_name: str, n_gallery_subjects=np.nan) -> dict:
    total_probe = len(results_df)
    if total_probe == 0:
        return {
            "dataset": dataset_name,
            "n_gallery_subjects": n_gallery_subjects,
            "total_probe": 0,
            "detected_count": 0,
            "detection_rate": np.nan,
            "correct_count": 0,
            "accuracy_conditional": np.nan,
            "accuracy_global": np.nan,
            "mean_similarity_correct": np.nan,
            "mean_similarity_incorrect": np.nan,
        }

    detected = _safe_bool_series(results_df["detected"])
    correct = _safe_bool_series(results_df["correct"])
    detected_count = int(detected.sum())
    correct_count = int(correct.sum())
    correct_detected = results_df[detected & correct]
    incorrect_detected = results_df[detected & ~correct]

    return {
        "dataset": dataset_name,
        "n_gallery_subjects": n_gallery_subjects,
        "total_probe": total_probe,
        "detected_count": detected_count,
        "detection_rate": detected_count / total_probe,
        "correct_count": correct_count,
        "accuracy_conditional": correct_count / detected_count if detected_count else np.nan,
        "accuracy_global": correct_count / total_probe,
        "mean_similarity_correct": correct_detected["similarity"].mean(),
        "mean_similarity_incorrect": incorrect_detected["similarity"].mean(),
    }


def compute_global_metrics(
    results_df: pd.DataFrame,
    gallery_embeddings=None,
    dataset_name: str = "DroneFace",
) -> pd.DataFrame:
    n_gallery_subjects = len(gallery_embeddings) if gallery_embeddings is not None else np.nan
    return pd.DataFrame([_metric_row(results_df, dataset_name, n_gallery_subjects)])


def compute_grouped_metrics(results_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if results_df.empty:
        columns = [
            "dataset",
            *group_cols,
            "total_probe",
            "detected_count",
            "detection_rate",
            "correct_count",
            "accuracy_conditional",
            "accuracy_global",
            "mean_similarity_correct",
            "mean_similarity_incorrect",
        ]
        return pd.DataFrame(columns=columns)

    rows = []
    for group_values, group_df in results_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        dataset_name = group_df["dataset"].dropna().iloc[0] if "dataset" in group_df and not group_df.empty else "DroneFace"
        row = _metric_row(group_df, dataset_name)
        row.pop("n_gallery_subjects", None)
        for col, value in zip(group_cols, group_values):
            row[col] = value
        rows.append(row)

    ordered_cols = [
        "dataset",
        *group_cols,
        "total_probe",
        "detected_count",
        "detection_rate",
        "correct_count",
        "accuracy_conditional",
        "accuracy_global",
        "mean_similarity_correct",
        "mean_similarity_incorrect",
    ]
    df = pd.DataFrame(rows)
    return df[ordered_cols].sort_values(group_cols, na_position="last").reset_index(drop=True)


def _format_percent(value) -> str:
    if pd.isna(value):
        return "no disponible"
    return f"{value:.2%}"


def generate_preliminary_discussion(
    summary_global_df: pd.DataFrame,
    droneface_by_height: pd.DataFrame,
    droneface_by_distance: pd.DataFrame,
    droneface_by_height_distance: pd.DataFrame,
) -> str:
    lines = []

    if summary_global_df.empty:
        lines.append("No se pudo generar una discusion global porque la tabla de resumen esta vacia.")
    else:
        row = summary_global_df.iloc[0]
        lines.append(
            "El baseline closed-set sobre DroneFace obtuvo una accuracy_global de "
            f"{_format_percent(row.get('accuracy_global'))}, con detection_rate de "
            f"{_format_percent(row.get('detection_rate'))} y accuracy_conditional de "
            f"{_format_percent(row.get('accuracy_conditional'))}."
        )
        if pd.notna(row.get("accuracy_conditional")) and pd.notna(row.get("accuracy_global")):
            gap = row["accuracy_conditional"] - row["accuracy_global"]
            if gap > 0.05:
                lines.append(
                    "La diferencia entre accuracy_conditional y accuracy_global sugiere que una parte importante "
                    "del error proviene de la etapa de deteccion facial previa."
                )
            else:
                lines.append(
                    "La diferencia entre accuracy_conditional y accuracy_global no muestra por si sola que la "
                    "deteccion sea el unico factor dominante del error."
                )
        else:
            lines.append("No se puede concluir el efecto de la deteccion porque faltan metricas globales.")

    if not droneface_by_height.empty and "accuracy_global" in droneface_by_height:
        valid = droneface_by_height.dropna(subset=["accuracy_global"])
        if valid.empty:
            lines.append("No se puede identificar la altura con menor accuracy_global por falta de datos validos.")
        else:
            worst = valid.sort_values("accuracy_global").iloc[0]
            lines.append(
                f"La altura con menor accuracy_global fue {worst['height_m']} m "
                f"({_format_percent(worst['accuracy_global'])})."
            )
    else:
        lines.append("No se puede concluir sobre altura porque la tabla por altura esta vacia.")

    if not droneface_by_distance.empty and "accuracy_global" in droneface_by_distance:
        valid = droneface_by_distance.dropna(subset=["accuracy_global"])
        if valid.empty:
            lines.append("No se puede identificar la distancia con menor accuracy_global por falta de datos validos.")
        else:
            worst = valid.sort_values("accuracy_global").iloc[0]
            lines.append(
                f"La distancia con menor accuracy_global fue {worst['distance_m']} m "
                f"({_format_percent(worst['accuracy_global'])})."
            )
    else:
        lines.append("No se puede concluir sobre distancia porque la tabla por distancia esta vacia.")

    if not droneface_by_height_distance.empty and "accuracy_global" in droneface_by_height_distance:
        valid = droneface_by_height_distance.dropna(subset=["accuracy_global"])
        if valid.empty:
            lines.append(
                "No se puede identificar la combinacion altura-distancia con menor accuracy_global por falta de datos validos."
            )
        else:
            worst = valid.sort_values("accuracy_global").iloc[0]
            lines.append(
                "La combinacion altura-distancia con menor accuracy_global fue "
                f"altura {worst['height_m']} m y distancia {worst['distance_m']} m "
                f"({_format_percent(worst['accuracy_global'])})."
            )
    else:
        lines.append(
            "No se puede concluir sobre combinaciones altura-distancia porque la tabla correspondiente esta vacia."
        )

    lines.append(
        "Estos resultados corresponden a un baseline closed-set con un modelo preentrenado; no representan todavia "
        "un sistema final de vigilancia real."
    )
    return "\n\n".join(lines)


def save_gallery_npz(gallery_embeddings: dict[str, np.ndarray], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subject_ids = np.array(list(gallery_embeddings.keys()), dtype=str)
    embeddings = np.vstack([gallery_embeddings[subject_id] for subject_id in subject_ids]).astype(np.float32)
    np.savez(path, subject_ids=subject_ids, embeddings=embeddings)


def load_gallery_npz(path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    subject_ids = data["subject_ids"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    return {subject_id: embedding for subject_id, embedding in zip(subject_ids, embeddings)}


def build_gallery(
    index_df: pd.DataFrame,
    model,
    dataset_name: str = "DroneFace",
    save_path=None,
    failures_path=None,
):
    ensure_project_dirs()
    save_path = Path(save_path) if save_path is not None else EMBEDDINGS_DIR / "droneface_gallery_embeddings.npz"
    failures_path = (
        Path(failures_path) if failures_path is not None else TABLES_DIR / "gallery_failures_droneface.csv"
    )

    gallery_df = index_df[index_df["split"] == "gallery"].copy()
    gallery_embeddings: dict[str, np.ndarray] = {}
    info_rows = []
    failure_rows = []

    for subject_id, subject_df in tqdm(
        gallery_df.groupby("subject_id", dropna=True),
        desc="Building gallery",
        total=gallery_df["subject_id"].dropna().nunique(),
    ):
        valid_embeddings = []
        subject_errors = []

        for row in subject_df.itertuples(index=False):
            extraction = model.extract_single_face_embedding(row.path)
            if extraction["detected"] and extraction["embedding"] is not None:
                valid_embeddings.append(extraction["embedding"])
            else:
                subject_errors.append(extraction.get("error", "No face detected."))
                failure_rows.append(
                    {
                        "dataset": dataset_name,
                        "subject_id": subject_id,
                        "path": row.path,
                        "filename": row.filename,
                        "detected": extraction["detected"],
                        "num_faces_detected": extraction["num_faces_detected"],
                        "error": extraction.get("error", ""),
                    }
                )

        if valid_embeddings:
            mean_embedding = _l2_normalize(np.mean(np.vstack(valid_embeddings), axis=0))
            gallery_embeddings[str(subject_id)] = mean_embedding
            info_rows.append(
                {
                    "dataset": dataset_name,
                    "subject_id": subject_id,
                    "n_gallery_images": len(subject_df),
                    "n_valid_embeddings": len(valid_embeddings),
                    "n_failed_images": len(subject_df) - len(valid_embeddings),
                    "included": True,
                    "error": "",
                }
            )
        else:
            failure_message = "; ".join(sorted(set(filter(None, subject_errors)))) or "No valid gallery embedding."
            info_rows.append(
                {
                    "dataset": dataset_name,
                    "subject_id": subject_id,
                    "n_gallery_images": len(subject_df),
                    "n_valid_embeddings": 0,
                    "n_failed_images": len(subject_df),
                    "included": False,
                    "error": failure_message,
                }
            )

    gallery_info_df = pd.DataFrame(info_rows)
    failures_df = pd.DataFrame(failure_rows, columns=GALLERY_FAILURE_COLUMNS)

    if gallery_embeddings:
        save_gallery_npz(gallery_embeddings, save_path)
        print(f"Saved gallery embeddings: {save_path}")
    else:
        print("No gallery embeddings were saved because no valid gallery faces were detected.")

    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_df.to_csv(failures_path, index=False)
    print(f"Saved gallery failures: {failures_path}")

    return gallery_embeddings, gallery_info_df, failures_df


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


def _join_subjects(subjects) -> str:
    return ",".join(str(subject) for subject in subjects)


def parse_composite_subject_id(subject_id) -> list[str]:
    if pd.isna(subject_id):
        return []
    value = str(subject_id).strip().lower()
    valid_subjects = set("abcdefghijk")
    return [char for char in value if char in valid_subjects]


def _safe_divide(numerator, denominator) -> float:
    denominator = float(denominator)
    if denominator == 0:
        return float("nan")
    return float(numerator) / denominator


def _f1_score(precision, recall) -> float:
    if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def _subject_list_from_csv(value) -> list[str]:
    return parse_composite_subject_id(value)


def _bbox_columns(bbox) -> dict:
    if bbox is None or len(bbox) < 4:
        return {"bbox_x1": np.nan, "bbox_y1": np.nan, "bbox_x2": np.nan, "bbox_y2": np.nan}
    return {
        "bbox_x1": float(bbox[0]),
        "bbox_y1": float(bbox[1]),
        "bbox_x2": float(bbox[2]),
        "bbox_y2": float(bbox[3]),
    }


def _safe_image_stem(filename: str) -> str:
    return Path(str(filename)).stem.replace(" ", "_")


def _write_image(path, image) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Could not encode image for {path}")
    encoded.tofile(str(path))


def _clip_bbox(bbox, width: int, height: int, padding: float = 0.0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if padding > 0:
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        x1 -= box_w * padding
        x2 += box_w * padding
        y1 -= box_h * padding
        y2 += box_h * padding
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x2))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y2))))
    return x1, y1, x2, y2


def _draw_review_image(image, detections, out_path) -> None:
    review = image.copy()
    for det in detections:
        bbox = det.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = _clip_bbox(bbox, image.shape[1], image.shape[0], padding=0.0)
        label = f"{det.get('face_index', '')}:{det.get('detection_score', np.nan):.3f}"
        cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            review,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    _write_image(out_path, review)


def _full_image_eval_rows(index_df: pd.DataFrame) -> pd.DataFrame:
    return index_df[
        (index_df["source_folder"] == "photos_all")
        & (index_df["image_type"].astype(str).str.lower() == "eo")
    ].copy()


def detect_and_save_full_image_crops(
    index_df: pd.DataFrame,
    model,
    crops_dir=None,
    review_dir=None,
    crops_path=None,
    gallery_subjects=None,
    padding: float = 0.20,
) -> pd.DataFrame:
    """Detect faces in photos_all, save bbox review images and face crops."""
    ensure_project_dirs()
    crops_dir = Path(crops_dir) if crops_dir is not None else CROPS_DIR / "full_images"
    review_dir = Path(review_dir) if review_dir is not None else FIGS_DIR / "detection" / "full_images_review"
    crops_path = Path(crops_path) if crops_path is not None else RESULTS_DIR / "droneface_full_image_crops.csv"
    crops_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    eval_df = _full_image_eval_rows(index_df)
    if eval_df.empty:
        raise ValueError("No photos_all / eo rows found for full-image crop extraction.")

    valid_gallery_subjects = (
        set(str(subject) for subject in gallery_subjects)
        if gallery_subjects is not None
        else set("adegijk")
    )
    crop_rows = []

    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df), desc="Detecting and saving full-image crops"):
        expected_subjects = parse_composite_subject_id(row.subject_id)
        expected_with_gallery = [subject for subject in expected_subjects if subject in valid_gallery_subjects]
        expected_without_gallery = [subject for subject in expected_subjects if subject not in valid_gallery_subjects]
        source_path = Path(row.path)
        review_path = review_dir / f"{_safe_image_stem(row.filename)}_boxes.jpg"

        try:
            image = model.load_image(source_path)
            image_used, upscaled = _maybe_upscale(image)
            faces = model.detect_faces(image_used)
        except Exception as exc:
            crop_rows.append(
                {
                    "dataset": getattr(row, "dataset", "DroneFace"),
                    "source_path": row.path,
                    "source_filename": row.filename,
                    "crop_path": "",
                    "crop_filename": "",
                    "review_path": str(review_path),
                    "source_folder": row.source_folder,
                    "composite_subject_id": row.subject_id,
                    "expected_subjects": _join_subjects(expected_subjects),
                    "expected_subjects_with_gallery": _join_subjects(expected_with_gallery),
                    "expected_subjects_without_gallery": _join_subjects(expected_without_gallery),
                    "detection_index": np.nan,
                    **_bbox_columns(None),
                    "crop_x1": np.nan,
                    "crop_y1": np.nan,
                    "crop_x2": np.nan,
                    "crop_y2": np.nan,
                    "detection_score": np.nan,
                    "height_m": getattr(row, "height_m", np.nan),
                    "distance_m": getattr(row, "distance_m", np.nan),
                    "image_type": getattr(row, "image_type", None),
                    "upscaled_for_detection": False,
                    "image_shape_original": "",
                    "image_shape_used": "",
                    "error": str(exc),
                }
            )
            continue

        faces = sorted(faces, key=_bbox_area, reverse=True)
        image_shape_original = tuple(int(v) for v in image.shape)
        image_shape_used = tuple(int(v) for v in image_used.shape)
        detections_for_review = []

        for detection_index, face in enumerate(faces):
            bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
            if bbox.size < 4:
                continue
            bbox_list = bbox.tolist()
            detections_for_review.append(
                {
                    "bbox": bbox_list,
                    "face_index": detection_index,
                    "detection_score": float(getattr(face, "det_score", np.nan)),
                }
            )
            crop_x1, crop_y1, crop_x2, crop_y2 = _clip_bbox(
                bbox_list,
                image_used.shape[1],
                image_used.shape[0],
                padding=padding,
            )
            crop = image_used[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_filename = f"{_safe_image_stem(row.filename)}_face{detection_index:02d}.jpg"
            crop_path = crops_dir / crop_filename
            _write_image(crop_path, crop)

            crop_rows.append(
                {
                    "dataset": getattr(row, "dataset", "DroneFace"),
                    "source_path": row.path,
                    "source_filename": row.filename,
                    "crop_path": str(crop_path),
                    "crop_filename": crop_filename,
                    "review_path": str(review_path),
                    "source_folder": row.source_folder,
                    "composite_subject_id": row.subject_id,
                    "expected_subjects": _join_subjects(expected_subjects),
                    "expected_subjects_with_gallery": _join_subjects(expected_with_gallery),
                    "expected_subjects_without_gallery": _join_subjects(expected_without_gallery),
                    "detection_index": detection_index,
                    **_bbox_columns(bbox_list),
                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "crop_x2": crop_x2,
                    "crop_y2": crop_y2,
                    "detection_score": float(getattr(face, "det_score", np.nan)),
                    "height_m": getattr(row, "height_m", np.nan),
                    "distance_m": getattr(row, "distance_m", np.nan),
                    "image_type": getattr(row, "image_type", None),
                    "upscaled_for_detection": upscaled,
                    "image_shape_original": str(image_shape_original),
                    "image_shape_used": str(image_shape_used),
                    "error": "",
                }
            )

        _draw_review_image(image_used, detections_for_review, review_path)

        if not faces:
            crop_rows.append(
                {
                    "dataset": getattr(row, "dataset", "DroneFace"),
                    "source_path": row.path,
                    "source_filename": row.filename,
                    "crop_path": "",
                    "crop_filename": "",
                    "review_path": str(review_path),
                    "source_folder": row.source_folder,
                    "composite_subject_id": row.subject_id,
                    "expected_subjects": _join_subjects(expected_subjects),
                    "expected_subjects_with_gallery": _join_subjects(expected_with_gallery),
                    "expected_subjects_without_gallery": _join_subjects(expected_without_gallery),
                    "detection_index": np.nan,
                    **_bbox_columns(None),
                    "crop_x1": np.nan,
                    "crop_y1": np.nan,
                    "crop_x2": np.nan,
                    "crop_y2": np.nan,
                    "detection_score": np.nan,
                    "height_m": getattr(row, "height_m", np.nan),
                    "distance_m": getattr(row, "distance_m", np.nan),
                    "image_type": getattr(row, "image_type", None),
                    "upscaled_for_detection": upscaled,
                    "image_shape_original": str(image_shape_original),
                    "image_shape_used": str(image_shape_used),
                    "error": "No face detected.",
                }
            )

    crops_df = pd.DataFrame(crop_rows, columns=FULL_IMAGE_CROP_COLUMNS)
    crops_path.parent.mkdir(parents=True, exist_ok=True)
    crops_df.to_csv(crops_path, index=False)
    print(f"Saved full-image crop metadata: {crops_path}")
    return crops_df


def recognize_full_image_crops(
    crops_df: pd.DataFrame,
    gallery_embeddings: dict[str, np.ndarray],
    model,
    recognition_path=None,
) -> pd.DataFrame:
    ensure_project_dirs()
    if crops_df.empty:
        raise ValueError("Cannot recognize full-image crops because crop metadata is empty.")
    if not gallery_embeddings:
        raise ValueError("Cannot recognize full-image crops without gallery embeddings.")

    rows = []
    crop_df = crops_df[crops_df["crop_path"].fillna("").astype(str).str.len() > 0].copy()

    for row in tqdm(crop_df.itertuples(index=False), total=len(crop_df), desc="Recognizing saved full-image crops"):
        extraction = model.extract_single_face_embedding(row.crop_path)
        predicted_subject = None
        similarity = np.nan
        error = extraction.get("error", "")

        if extraction["detected"] and extraction["embedding"] is not None:
            predicted_subject, similarity = _best_gallery_match(extraction["embedding"], gallery_embeddings, model)
        elif not error:
            error = "No face detected in saved crop."

        rows.append(
            {
                "dataset": row.dataset,
                "source_path": row.source_path,
                "source_filename": row.source_filename,
                "crop_path": row.crop_path,
                "crop_filename": row.crop_filename,
                "source_folder": row.source_folder,
                "composite_subject_id": row.composite_subject_id,
                "expected_subjects": row.expected_subjects,
                "expected_subjects_with_gallery": row.expected_subjects_with_gallery,
                "expected_subjects_without_gallery": row.expected_subjects_without_gallery,
                "detection_index": row.detection_index,
                "bbox_x1": row.bbox_x1,
                "bbox_y1": row.bbox_y1,
                "bbox_x2": row.bbox_x2,
                "bbox_y2": row.bbox_y2,
                "crop_x1": row.crop_x1,
                "crop_y1": row.crop_y1,
                "crop_x2": row.crop_x2,
                "crop_y2": row.crop_y2,
                "source_detection_score": row.detection_score,
                "recognition_source": "saved_crop_redetect",
                "crop_detected": extraction["detected"],
                "crop_num_faces_detected": extraction["num_faces_detected"],
                "crop_detection_score": extraction["detection_score"],
                "predicted_subject": predicted_subject,
                "similarity": similarity,
                "height_m": row.height_m,
                "distance_m": row.distance_m,
                "image_type": row.image_type,
                "error": error,
            }
        )

    recognition_df = pd.DataFrame(rows, columns=FULL_IMAGE_CROP_RECOGNITION_COLUMNS)
    recognition_path = (
        Path(recognition_path)
        if recognition_path is not None
        else RESULTS_DIR / "droneface_full_image_crop_recognition.csv"
    )
    recognition_path.parent.mkdir(parents=True, exist_ok=True)
    recognition_df.to_csv(recognition_path, index=False)
    print(f"Saved full-image crop recognition: {recognition_path}")
    return recognition_df


def build_crop_recognition_from_full_image_detections(
    crops_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    recognition_path=None,
) -> pd.DataFrame:
    """Build per-crop recognition rows from the original full-image detections.

    InsightFace already computes an aligned face embedding for every detected face.
    This function links saved rectangular crops to those source detections, avoiding
    a second detector pass inside the crop that would measure a different problem.
    """
    ensure_project_dirs()
    if crops_df.empty:
        raise ValueError("Cannot build crop recognition because crop metadata is empty.")
    if detections_df.empty:
        raise ValueError("Cannot build crop recognition because detections are empty.")

    crop_df = crops_df[crops_df["crop_path"].fillna("").astype(str).str.len() > 0].copy()
    det_df = detections_df.copy()
    det_path_col = "path" if "path" in det_df.columns else "source_path"
    det_df["_path_key"] = det_df[det_path_col].astype(str)
    det_df["_det_key"] = pd.to_numeric(det_df["detection_index"], errors="coerce")

    rows = []
    for row in crop_df.itertuples(index=False):
        det_key = pd.to_numeric(pd.Series([row.detection_index]), errors="coerce").iloc[0]
        matched = det_df[
            (det_df["_path_key"] == str(row.source_path))
            & (det_df["_det_key"] == det_key)
        ]
        det = matched.iloc[0] if not matched.empty else None
        error = "" if det is not None else "No matching source detection row."
        predicted_subject = None if det is None else det.get("predicted_subject", None)
        similarity = np.nan if det is None else det.get("similarity", np.nan)
        detection_score = np.nan if det is None else det.get("detection_score", np.nan)

        rows.append(
            {
                "dataset": row.dataset,
                "source_path": row.source_path,
                "source_filename": row.source_filename,
                "crop_path": row.crop_path,
                "crop_filename": row.crop_filename,
                "source_folder": row.source_folder,
                "composite_subject_id": row.composite_subject_id,
                "expected_subjects": row.expected_subjects,
                "expected_subjects_with_gallery": row.expected_subjects_with_gallery,
                "expected_subjects_without_gallery": row.expected_subjects_without_gallery,
                "detection_index": row.detection_index,
                "bbox_x1": row.bbox_x1,
                "bbox_y1": row.bbox_y1,
                "bbox_x2": row.bbox_x2,
                "bbox_y2": row.bbox_y2,
                "crop_x1": row.crop_x1,
                "crop_y1": row.crop_y1,
                "crop_x2": row.crop_x2,
                "crop_y2": row.crop_y2,
                "source_detection_score": row.detection_score,
                "recognition_source": "source_detection_embedding",
                "crop_detected": det is not None,
                "crop_num_faces_detected": 1 if det is not None else 0,
                "crop_detection_score": detection_score,
                "predicted_subject": predicted_subject,
                "similarity": similarity,
                "height_m": row.height_m,
                "distance_m": row.distance_m,
                "image_type": row.image_type,
                "error": error,
            }
        )

    recognition_df = pd.DataFrame(rows, columns=FULL_IMAGE_CROP_RECOGNITION_COLUMNS)
    recognition_path = (
        Path(recognition_path)
        if recognition_path is not None
        else RESULTS_DIR / "droneface_full_image_crop_recognition.csv"
    )
    recognition_path.parent.mkdir(parents=True, exist_ok=True)
    recognition_df.to_csv(recognition_path, index=False)
    print(f"Saved full-image crop recognition from source detections: {recognition_path}")
    return recognition_df


def aggregate_full_image_crop_recognition(
    index_df: pd.DataFrame,
    crops_df: pd.DataFrame,
    crop_recognition_df: pd.DataFrame,
    results_path=None,
) -> pd.DataFrame:
    eval_df = _full_image_eval_rows(index_df)
    image_rows = []

    for row in eval_df.itertuples(index=False):
        expected_subjects = parse_composite_subject_id(row.subject_id)
        # Use the gallery availability encoded in crop metadata when available.
        image_crop_rows = crops_df[crops_df["source_path"].astype(str) == str(row.path)]
        if image_crop_rows.empty:
            expected_with_gallery = [subject for subject in expected_subjects if subject in set("adegijk")]
            expected_without_gallery = [subject for subject in expected_subjects if subject not in set("adegijk")]
        else:
            first_crop = image_crop_rows.iloc[0]
            expected_with_gallery = parse_composite_subject_id(first_crop["expected_subjects_with_gallery"])
            expected_without_gallery = parse_composite_subject_id(first_crop["expected_subjects_without_gallery"])

        recog_rows = crop_recognition_df[crop_recognition_df["source_path"].astype(str) == str(row.path)]
        predicted_subjects = sorted(
            set(recog_rows["predicted_subject"].dropna().astype(str).tolist())
            if not recog_rows.empty
            else []
        )
        found_expected = [subject for subject in expected_with_gallery if subject in predicted_subjects]
        predicted_unexpected = [subject for subject in predicted_subjects if subject not in expected_with_gallery]
        n_expected = len(expected_with_gallery)
        n_found = len(found_expected)
        recall = n_found / n_expected if n_expected else np.nan

        image_rows.append(
            {
                "dataset": getattr(row, "dataset", "DroneFace"),
                "path": row.path,
                "filename": row.filename,
                "source_folder": row.source_folder,
                "composite_subject_id": row.subject_id,
                "expected_subjects": _join_subjects(expected_subjects),
                "expected_subjects_with_gallery": _join_subjects(expected_with_gallery),
                "expected_subjects_without_gallery": _join_subjects(expected_without_gallery),
                "predicted_subjects": _join_subjects(predicted_subjects),
                "predicted_unexpected_subjects": _join_subjects(predicted_unexpected),
                "num_faces_detected": int(image_crop_rows["crop_path"].fillna("").astype(str).str.len().gt(0).sum()),
                "num_recognized_faces": int(recog_rows["crop_detected"].fillna(False).astype(bool).sum()) if not recog_rows.empty else 0,
                "found_expected_subjects": _join_subjects(found_expected),
                "n_expected_with_gallery": n_expected,
                "n_expected_found": n_found,
                "expected_identity_recall": recall,
                "any_expected_found": n_found > 0 if n_expected else False,
                "all_expected_found": n_found == n_expected if n_expected else False,
                "height_m": getattr(row, "height_m", np.nan),
                "distance_m": getattr(row, "distance_m", np.nan),
                "image_type": getattr(row, "image_type", None),
                "error": "",
            }
        )

    results_df = pd.DataFrame(image_rows, columns=FULL_IMAGE_RESULT_COLUMNS)
    results_path = (
        Path(results_path)
        if results_path is not None
        else RESULTS_DIR / "droneface_full_images_results.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved full-image crop-based results: {results_path}")
    return results_df


def evaluate_closed_set(
    probe_df: pd.DataFrame,
    gallery_embeddings: dict[str, np.ndarray],
    model,
    save_path=None,
) -> pd.DataFrame:
    ensure_project_dirs()
    valid_subjects = set(gallery_embeddings.keys())
    eval_df = probe_df[(probe_df["split"] == "probe") & (probe_df["subject_id"].isin(valid_subjects))].copy()

    rows = []
    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df), desc="Evaluating probes"):
        extraction = model.extract_single_face_embedding(row.path)
        predicted_subject = None
        similarity = np.nan
        correct = False

        if extraction["detected"] and extraction["embedding"] is not None:
            predicted_subject, similarity = _best_gallery_match(extraction["embedding"], gallery_embeddings, model)
            correct = predicted_subject == row.subject_id

        rows.append(
            {
                "dataset": getattr(row, "dataset", "DroneFace"),
                "path": row.path,
                "filename": row.filename,
                "source_folder": row.source_folder,
                "subject_id": row.subject_id,
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
                "error": extraction.get("error", ""),
            }
        )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_path = Path(save_path) if save_path is not None else RESULTS_DIR / "droneface_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved closed-set results: {results_path}")
    return results_df


def evaluate_full_images(
    index_df: pd.DataFrame,
    gallery_embeddings: dict[str, np.ndarray],
    model,
    detections_path=None,
    results_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate DroneFace original full drone images from photos_all."""
    ensure_project_dirs()
    if not gallery_embeddings:
        raise ValueError("Cannot evaluate full images without gallery embeddings.")

    eval_df = index_df[
        (index_df["source_folder"] == "photos_all")
        & (index_df["image_type"].astype(str).str.lower() == "eo")
    ].copy()
    if eval_df.empty:
        raise ValueError("No photos_all / eo rows found for full-image evaluation.")

    valid_gallery_subjects = set(str(subject) for subject in gallery_embeddings.keys())
    image_rows = []
    detection_rows = []

    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df), desc="Evaluating full drone images"):
        expected_subjects = parse_composite_subject_id(row.subject_id)
        expected_with_gallery = [subject for subject in expected_subjects if subject in valid_gallery_subjects]
        expected_without_gallery = [subject for subject in expected_subjects if subject not in valid_gallery_subjects]
        extraction = model.extract_all_face_embeddings(row.path)

        predicted_subjects = []
        for detection in extraction["detections"]:
            predicted_subject = None
            similarity = np.nan
            if detection.get("embedding") is not None:
                predicted_subject, similarity = _best_gallery_match(
                    detection["embedding"],
                    gallery_embeddings,
                    model,
                )
                if predicted_subject is not None:
                    predicted_subjects.append(str(predicted_subject))

            detection_row = {
                "dataset": getattr(row, "dataset", "DroneFace"),
                "path": row.path,
                "filename": row.filename,
                "source_folder": row.source_folder,
                "composite_subject_id": row.subject_id,
                "expected_subjects": _join_subjects(expected_subjects),
                "expected_subjects_with_gallery": _join_subjects(expected_with_gallery),
                "expected_subjects_without_gallery": _join_subjects(expected_without_gallery),
                "detection_index": detection.get("face_index", np.nan),
                **_bbox_columns(detection.get("bbox")),
                "detection_score": detection.get("detection_score", np.nan),
                "predicted_subject": predicted_subject,
                "similarity": similarity,
                "height_m": getattr(row, "height_m", np.nan),
                "distance_m": getattr(row, "distance_m", np.nan),
                "image_type": getattr(row, "image_type", None),
                "error": extraction.get("error", ""),
            }
            detection_rows.append(detection_row)

        predicted_unique = sorted(set(predicted_subjects))
        found_expected = [subject for subject in expected_with_gallery if subject in predicted_unique]
        predicted_unexpected = [subject for subject in predicted_unique if subject not in expected_with_gallery]
        n_expected = len(expected_with_gallery)
        n_found = len(found_expected)
        recall = n_found / n_expected if n_expected else np.nan

        image_rows.append(
            {
                "dataset": getattr(row, "dataset", "DroneFace"),
                "path": row.path,
                "filename": row.filename,
                "source_folder": row.source_folder,
                "composite_subject_id": row.subject_id,
                "expected_subjects": _join_subjects(expected_subjects),
                "expected_subjects_with_gallery": _join_subjects(expected_with_gallery),
                "expected_subjects_without_gallery": _join_subjects(expected_without_gallery),
                "predicted_subjects": _join_subjects(predicted_unique),
                "predicted_unexpected_subjects": _join_subjects(predicted_unexpected),
                "num_faces_detected": extraction["num_faces_detected"],
                "num_recognized_faces": len(predicted_subjects),
                "found_expected_subjects": _join_subjects(found_expected),
                "n_expected_with_gallery": n_expected,
                "n_expected_found": n_found,
                "expected_identity_recall": recall,
                "any_expected_found": n_found > 0 if n_expected else False,
                "all_expected_found": n_found == n_expected if n_expected else False,
                "height_m": getattr(row, "height_m", np.nan),
                "distance_m": getattr(row, "distance_m", np.nan),
                "image_type": getattr(row, "image_type", None),
                "error": extraction.get("error", ""),
            }
        )

    detections_df = pd.DataFrame(detection_rows, columns=FULL_IMAGE_DETECTION_COLUMNS)
    results_df = pd.DataFrame(image_rows, columns=FULL_IMAGE_RESULT_COLUMNS)

    detections_path = (
        Path(detections_path)
        if detections_path is not None
        else RESULTS_DIR / "droneface_full_images_detections.csv"
    )
    results_path = (
        Path(results_path)
        if results_path is not None
        else RESULTS_DIR / "droneface_full_images_results.csv"
    )
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    detections_df.to_csv(detections_path, index=False)
    results_df.to_csv(results_path, index=False)
    print(f"Saved full-image detections: {detections_path}")
    print(f"Saved full-image results: {results_path}")
    return detections_df, results_df


def _full_image_summary_row(results_df: pd.DataFrame, dataset_name: str = "DroneFace") -> dict:
    total_images = len(results_df)
    if total_images == 0:
        return {
            "dataset": dataset_name,
            "total_images": 0,
            "images_with_detections": 0,
            "image_detection_rate": np.nan,
            "total_expected_identities_with_gallery": 0,
            "expected_identities_found": 0,
            "expected_identity_recall_global": np.nan,
            "images_any_expected_found": 0,
            "any_expected_found_rate": np.nan,
            "images_all_expected_found": 0,
            "all_expected_found_rate": np.nan,
            "mean_expected_identity_recall": np.nan,
            "expected_identities_without_gallery": 0,
        }

    images_with_detections = int((pd.to_numeric(results_df["num_faces_detected"], errors="coerce") > 0).sum())
    total_expected = int(pd.to_numeric(results_df["n_expected_with_gallery"], errors="coerce").fillna(0).sum())
    expected_found = int(pd.to_numeric(results_df["n_expected_found"], errors="coerce").fillna(0).sum())
    any_found = _safe_bool_series(results_df["any_expected_found"])
    all_found = _safe_bool_series(results_df["all_expected_found"])
    without_gallery_count = results_df["expected_subjects_without_gallery"].fillna("").apply(
        lambda value: 0 if not str(value).strip() else len(str(value).split(","))
    )

    return {
        "dataset": dataset_name,
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "image_detection_rate": images_with_detections / total_images,
        "total_expected_identities_with_gallery": total_expected,
        "expected_identities_found": expected_found,
        "expected_identity_recall_global": expected_found / total_expected if total_expected else np.nan,
        "images_any_expected_found": int(any_found.sum()),
        "any_expected_found_rate": int(any_found.sum()) / total_images,
        "images_all_expected_found": int(all_found.sum()),
        "all_expected_found_rate": int(all_found.sum()) / total_images,
        "mean_expected_identity_recall": results_df["expected_identity_recall"].mean(),
        "expected_identities_without_gallery": int(without_gallery_count.sum()),
    }


def compute_full_image_summary(results_df: pd.DataFrame, dataset_name: str = "DroneFace") -> pd.DataFrame:
    return pd.DataFrame([_full_image_summary_row(results_df, dataset_name)])[FULL_IMAGE_SUMMARY_COLUMNS]


def compute_full_image_grouped_metrics(results_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=[*group_cols, *FULL_IMAGE_SUMMARY_COLUMNS])

    rows = []
    for group_values, group_df in results_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        dataset_name = group_df["dataset"].dropna().iloc[0] if "dataset" in group_df and not group_df.empty else "DroneFace"
        row = _full_image_summary_row(group_df, dataset_name)
        for col, value in zip(group_cols, group_values):
            row[col] = value
        rows.append(row)

    ordered_cols = [*group_cols, *FULL_IMAGE_SUMMARY_COLUMNS]
    return pd.DataFrame(rows)[ordered_cols].sort_values(group_cols, na_position="last").reset_index(drop=True)


def _macro_f1_single_label(results_df: pd.DataFrame) -> float:
    if results_df.empty or "subject_id" not in results_df:
        return float("nan")

    true_subjects = results_df["subject_id"].dropna().astype(str)
    pred_subjects = results_df["predicted_subject"].dropna().astype(str)
    labels = sorted(set(true_subjects).union(set(pred_subjects)))
    if not labels:
        return float("nan")

    y_true = results_df["subject_id"].astype(str)
    y_pred = results_df["predicted_subject"].where(results_df["predicted_subject"].notna(), "").astype(str)
    f1_values = []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1_values.append(_f1_score(precision, recall))

    valid = [value for value in f1_values if pd.notna(value)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_experiment_a_common_metrics(
    results_df: pd.DataFrame,
    dataset_name: str = "DroneFace",
) -> pd.DataFrame:
    """Compute common retrieval/classification metrics for Experiment A.

    Experiment A has one expected identity per probe image. A wrong detected
    prediction contributes one false positive for the predicted class and one
    false negative for the true class. A non-detection contributes one false
    negative and no false positive.
    """
    total_units = len(results_df)
    if total_units == 0:
        return pd.DataFrame(columns=COMMON_EXPERIMENT_METRIC_COLUMNS)

    detected = _safe_bool_series(results_df["detected"])
    correct = _safe_bool_series(results_df["correct"])
    has_prediction = results_df["predicted_subject"].notna()
    tp = int(correct.sum())
    fp = int((has_prediction & ~correct).sum())
    fn = int(total_units - tp)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)

    row = {
        "dataset": dataset_name,
        "experiment": "A_cropped_faces",
        "unit": "cropped_face",
        "metric_scope": "single_label_closed_set",
        "total_units": total_units,
        "detected_units": int(detected.sum()),
        "detection_rate": _safe_divide(int(detected.sum()), total_units),
        "total_expected_identities": total_units,
        "total_predicted_identities": int(has_prediction.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": _f1_score(precision, recall),
        "exact_match_count": tp,
        "exact_match_rate": _safe_divide(tp, total_units),
        "macro_f1": _macro_f1_single_label(results_df),
        "notes": "A tiene una identidad esperada por probe; recall_micro coincide con accuracy_global.",
    }
    return pd.DataFrame([row], columns=COMMON_EXPERIMENT_METRIC_COLUMNS)


def compute_experiment_b_common_metrics(
    results_df: pd.DataFrame,
    dataset_name: str = "DroneFace",
) -> pd.DataFrame:
    """Compute common set-based metrics for Experiment B.

    Experiment B has a set of expected identities per full drone image. Because
    DroneFace photos_all does not provide identity labels per bounding box, the
    metrics compare the set of expected gallery identities with the set of
    predicted identities in the image.
    """
    total_units = len(results_df)
    if total_units == 0:
        return pd.DataFrame(columns=COMMON_EXPERIMENT_METRIC_COLUMNS)

    tp = fp = fn = exact = total_expected = total_predicted = 0
    image_f1_values = []
    detected_units = int((pd.to_numeric(results_df["num_faces_detected"], errors="coerce") > 0).sum())

    for row in results_df.itertuples(index=False):
        expected = set(_subject_list_from_csv(getattr(row, "expected_subjects_with_gallery", "")))
        predicted = set(_subject_list_from_csv(getattr(row, "predicted_subjects", "")))
        intersection = expected & predicted
        row_tp = len(intersection)
        row_fp = len(predicted - expected)
        row_fn = len(expected - predicted)
        tp += row_tp
        fp += row_fp
        fn += row_fn
        total_expected += len(expected)
        total_predicted += len(predicted)
        if expected == predicted:
            exact += 1

        precision_i = _safe_divide(row_tp, row_tp + row_fp)
        recall_i = _safe_divide(row_tp, row_tp + row_fn)
        f1_i = _f1_score(precision_i, recall_i)
        if pd.notna(f1_i):
            image_f1_values.append(f1_i)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    row = {
        "dataset": dataset_name,
        "experiment": "B_full_images",
        "unit": "full_image_identity_set",
        "metric_scope": "set_retrieval_without_bbox_identity",
        "total_units": total_units,
        "detected_units": detected_units,
        "detection_rate": _safe_divide(detected_units, total_units),
        "total_expected_identities": total_expected,
        "total_predicted_identities": total_predicted,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": _f1_score(precision, recall),
        "exact_match_count": exact,
        "exact_match_rate": _safe_divide(exact, total_units),
        "macro_f1": float(np.mean(image_f1_values)) if image_f1_values else float("nan"),
        "notes": "B compara conjuntos de identidades por imagen; no es F1 por bounding box anotado.",
    }
    return pd.DataFrame([row], columns=COMMON_EXPERIMENT_METRIC_COLUMNS)


def compute_experiment_a_b_common_metrics(
    experiment_a_results_df: pd.DataFrame,
    experiment_b_results_df: pd.DataFrame,
    dataset_name: str = "DroneFace",
) -> pd.DataFrame:
    metrics = [
        compute_experiment_a_common_metrics(experiment_a_results_df, dataset_name),
        compute_experiment_b_common_metrics(experiment_b_results_df, dataset_name),
    ]
    non_empty = [df for df in metrics if not df.empty]
    if not non_empty:
        return pd.DataFrame(columns=COMMON_EXPERIMENT_METRIC_COLUMNS)
    return pd.concat(non_empty, ignore_index=True)[COMMON_EXPERIMENT_METRIC_COLUMNS]


def build_experiment_a_b_comparison(summary_a_df: pd.DataFrame, summary_b_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not summary_a_df.empty:
        row = summary_a_df.iloc[0]
        rows.append(
            {
                "experiment": "A_cropped_faces",
                "probe_source": "photos_all_faces",
                "unit": "cropped_face",
                "total_units": row.get("total_probe", np.nan),
                "detection_rate": row.get("detection_rate", np.nan),
                "primary_metric": row.get("accuracy_global", np.nan),
                "primary_metric_name": "accuracy_global",
            }
        )
    if not summary_b_df.empty:
        row = summary_b_df.iloc[0]
        rows.append(
            {
                "experiment": "B_full_images",
                "probe_source": "photos_all",
                "unit": "full_image_expected_identity",
                "total_units": row.get("total_images", np.nan),
                "detection_rate": row.get("image_detection_rate", np.nan),
                "primary_metric": row.get("expected_identity_recall_global", np.nan),
                "primary_metric_name": "expected_identity_recall_global",
            }
        )
    return pd.DataFrame(rows)
