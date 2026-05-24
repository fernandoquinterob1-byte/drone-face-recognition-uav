from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import EMBEDDINGS_DIR, RESULTS_DIR, TABLES_DIR, ensure_project_dirs
from .face_model import _l2_normalize

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


def evaluate_closed_set(probe_df: pd.DataFrame, gallery_embeddings: dict[str, np.ndarray], model) -> pd.DataFrame:
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
    results_path = RESULTS_DIR / "droneface_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved closed-set results: {results_path}")
    return results_df
