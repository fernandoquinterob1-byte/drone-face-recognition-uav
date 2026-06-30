from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for lightweight analysis environments.
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .config import PROJECT_ROOT


OUTPUTS_V2_DIR = PROJECT_ROOT / "outputs_v2"
V2_EMBEDDINGS_DIR = OUTPUTS_V2_DIR / "embeddings"
V2_RESULTS_DIR = OUTPUTS_V2_DIR / "results"
V2_TABLES_DIR = OUTPUTS_V2_DIR / "tables"
V2_FIGS_DIR = OUTPUTS_V2_DIR / "figs"

PAPER_DISTANCE_BINS_M = np.array(
    [2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5, 16.0],
    dtype=float,
)

VERIFICATION_COLUMNS = [
    "gallery_name",
    "dataset",
    "protocol",
    "probe_key",
    "path",
    "filename",
    "probe_subject",
    "expected_subject",
    "candidate_subject",
    "score",
    "is_match",
    "detected",
    "height_m",
    "distance_m",
    "image_type",
    "source_folder",
    "composite_subject_id",
    "spatial_rank",
    "detection_index",
    "match_type",
    "error",
]


@dataclass(frozen=True)
class VerificationOutputPaths:
    faces_scores: Path
    bbox_scores: Path


def verification_score_paths(gallery_name: str = "trio_front") -> VerificationOutputPaths:
    return VerificationOutputPaths(
        faces_scores=V2_RESULTS_DIR / f"photos_all_faces_verification_scores_{gallery_name}.csv",
        bbox_scores=V2_RESULTS_DIR / f"photos_all_bbox_verification_scores_{gallery_name}.csv",
    )


def load_gallery_embeddings(gallery_name: str = "trio_front") -> dict[str, np.ndarray]:
    path = V2_EMBEDDINGS_DIR / f"droneface_gallery_{gallery_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No existe la galeria {path}. Ejecuta primero el pipeline pesado V2.")
    data = np.load(path, allow_pickle=False)
    subject_ids = data["subject_ids"].astype(str)
    embeddings = data["embeddings"].astype(np.float32)
    return {subject_id: _l2_normalize(embedding) for subject_id, embedding in zip(subject_ids, embeddings)}


def _l2_normalize(embedding) -> np.ndarray | None:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _cosine_score(embedding, candidate_embedding) -> float:
    embedding = _l2_normalize(embedding)
    candidate_embedding = _l2_normalize(candidate_embedding)
    if embedding is None or candidate_embedding is None:
        return 0.0
    return float(np.dot(embedding, candidate_embedding))


def _clean_subject(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_float(value) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_int(value) -> int | None:
    number = _safe_float(value)
    if np.isnan(number):
        return None
    return int(round(number))


def _candidate_rows(
    base_row: dict,
    embedding,
    expected_subject: str,
    gallery_embeddings: dict[str, np.ndarray],
    detected: bool,
) -> list[dict]:
    rows = []
    for candidate_subject in sorted(gallery_embeddings):
        score = _cosine_score(embedding, gallery_embeddings[candidate_subject]) if detected else 0.0
        rows.append(
            {
                **base_row,
                "candidate_subject": candidate_subject,
                "score": score,
                "is_match": bool(expected_subject and candidate_subject == expected_subject),
                "detected": bool(detected),
            }
        )
    return rows


def build_photos_all_faces_verification_scores(
    faces_results_df: pd.DataFrame,
    model,
    gallery_embeddings: dict[str, np.ndarray],
    gallery_name: str = "trio_front",
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Generate one verification score per probe face and candidate identity."""
    if faces_results_df.empty:
        raise ValueError("faces_results_df esta vacio.")

    rows = []
    for row in tqdm(
        faces_results_df.itertuples(index=False),
        total=len(faces_results_df),
        desc=f"Verification scores photos_all_faces {gallery_name}",
    ):
        expected_subject = _clean_subject(getattr(row, "subject_id", ""))
        extraction = model.extract_single_face_embedding(getattr(row, "path"))
        detected = bool(extraction.get("detected") and extraction.get("embedding") is not None)
        embedding = extraction.get("embedding") if detected else None
        base_row = {
            "gallery_name": gallery_name,
            "dataset": getattr(row, "dataset", "DroneFace"),
            "protocol": "photos_all_faces",
            "probe_key": str(getattr(row, "path")),
            "path": getattr(row, "path"),
            "filename": getattr(row, "filename"),
            "probe_subject": expected_subject,
            "expected_subject": expected_subject,
            "height_m": getattr(row, "height_m", np.nan),
            "distance_m": getattr(row, "distance_m", np.nan),
            "image_type": getattr(row, "image_type", ""),
            "source_folder": getattr(row, "source_folder", "photos_all_faces"),
            "composite_subject_id": "",
            "spatial_rank": np.nan,
            "detection_index": np.nan,
            "match_type": "",
            "error": extraction.get("error", ""),
        }
        rows.extend(_candidate_rows(base_row, embedding, expected_subject, gallery_embeddings, detected))

    scores_df = pd.DataFrame(rows, columns=VERIFICATION_COLUMNS)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(save_path, index=False)
    return scores_df


def _extract_full_image_embeddings_by_path(labeled_df: pd.DataFrame, model) -> dict[str, dict[int, np.ndarray]]:
    embeddings_by_path: dict[str, dict[int, np.ndarray]] = {}
    unique_paths = labeled_df["path"].dropna().astype(str).drop_duplicates().tolist()
    for path in tqdm(unique_paths, total=len(unique_paths), desc="Extracting full-image embeddings"):
        extraction = model.extract_all_face_embeddings(path)
        detection_map: dict[int, np.ndarray] = {}
        for detection in extraction.get("detections", []):
            face_index = detection.get("face_index")
            embedding = detection.get("embedding")
            if face_index is not None and embedding is not None:
                detection_map[int(face_index)] = embedding
        embeddings_by_path[path] = detection_map
    return embeddings_by_path


def build_photos_all_bbox_verification_scores(
    labeled_bbox_df: pd.DataFrame,
    model,
    gallery_embeddings: dict[str, np.ndarray],
    gallery_name: str = "trio_front",
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Generate verification scores for photos_all spatially labeled bbox slots."""
    if labeled_bbox_df.empty:
        raise ValueError("labeled_bbox_df esta vacio.")

    gallery_df = labeled_bbox_df[labeled_bbox_df["gallery_name"].astype(str) == gallery_name].copy()
    if gallery_df.empty:
        raise ValueError(f"No hay filas bbox-level para gallery_name={gallery_name}.")

    embeddings_by_path = _extract_full_image_embeddings_by_path(gallery_df, model)
    rows = []
    for row in tqdm(
        gallery_df.itertuples(index=False),
        total=len(gallery_df),
        desc=f"Verification scores photos_all bbox {gallery_name}",
    ):
        path = str(getattr(row, "path"))
        expected_subject = _clean_subject(getattr(row, "expected_subject", ""))
        detection_index = _safe_int(getattr(row, "detection_index", np.nan))
        detection_map = embeddings_by_path.get(path, {})
        embedding = detection_map.get(detection_index) if detection_index is not None else None
        detected = embedding is not None
        spatial_rank = getattr(row, "spatial_rank", np.nan)
        match_type = str(getattr(row, "match_type", ""))
        if expected_subject:
            probe_key = f"{path}|slot={spatial_rank}"
        else:
            probe_key = f"{path}|extra_detection={detection_index}"

        base_row = {
            "gallery_name": gallery_name,
            "dataset": getattr(row, "dataset", "DroneFace"),
            "protocol": "photos_all_bbox",
            "probe_key": probe_key,
            "path": path,
            "filename": getattr(row, "filename"),
            "probe_subject": expected_subject,
            "expected_subject": expected_subject,
            "height_m": getattr(row, "height_m", np.nan),
            "distance_m": getattr(row, "distance_m", np.nan),
            "image_type": getattr(row, "image_type", ""),
            "source_folder": getattr(row, "source_folder", "photos_all"),
            "composite_subject_id": getattr(row, "composite_subject_id", ""),
            "spatial_rank": spatial_rank,
            "detection_index": detection_index if detection_index is not None else np.nan,
            "match_type": match_type,
            "error": getattr(row, "error", ""),
        }
        rows.extend(_candidate_rows(base_row, embedding, expected_subject, gallery_embeddings, detected))

    scores_df = pd.DataFrame(rows, columns=VERIFICATION_COLUMNS)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(save_path, index=False)
    return scores_df


def _require_sklearn_metrics():
    try:
        from sklearn.metrics import auc, roc_auc_score, roc_curve
    except ImportError as exc:
        raise ImportError("Se requiere scikit-learn para calcular ROC/AUC.") from exc
    return auc, roc_auc_score, roc_curve


def compute_auc_metrics(scores_df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute ROC AUC globally or by grouping columns from verification score rows."""
    if scores_df.empty:
        return pd.DataFrame()
    _, roc_auc_score, _ = _require_sklearn_metrics()
    group_cols = group_cols or []
    df = scores_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["is_match"] = df["is_match"].apply(_bool_value)

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [((), df)]

    rows = []
    for group_key, group_df in grouped:
        if group_cols and not isinstance(group_key, tuple):
            group_key = (group_key,)
        labels = group_df["is_match"].astype(int).to_numpy()
        scores = group_df["score"].astype(float).to_numpy()
        n_match = int(labels.sum())
        n_mismatch = int(len(labels) - n_match)
        auc_value = np.nan
        if n_match > 0 and n_mismatch > 0:
            auc_value = float(roc_auc_score(labels, scores))
        row = {
            "protocol": group_df["protocol"].iloc[0] if "protocol" in group_df else "",
            "gallery_name": group_df["gallery_name"].iloc[0] if "gallery_name" in group_df else "",
            "group": "global" if not group_cols else ", ".join(
                f"{col}={value}" for col, value in zip(group_cols, group_key)
            ),
            "n_pairs": int(len(group_df)),
            "n_match": n_match,
            "n_mismatch": n_mismatch,
            "auc": auc_value,
        }
        for col, value in zip(group_cols, group_key if group_cols else ()):
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def add_paper_distance_bins(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Assign each raw 0.5 m distance to the 1.5 m bins used in the paper figures."""
    df = scores_df.copy()
    if "distance_m" not in df.columns:
        df["paper_distance_m"] = np.nan
        return df

    distances = pd.to_numeric(df["distance_m"], errors="coerce")
    df["paper_distance_m"] = np.nan
    valid = distances.notna()
    if valid.any():
        values = distances.loc[valid].to_numpy(dtype=float)
        bin_idx = np.abs(values[:, None] - PAPER_DISTANCE_BINS_M[None, :]).argmin(axis=1)
        df.loc[valid, "paper_distance_m"] = PAPER_DISTANCE_BINS_M[bin_idx]
    return df


def decision_table_from_scores(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()
    df = scores_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["is_match"] = df["is_match"].apply(_bool_value)
    idx = df.groupby("probe_key")["score"].idxmax()
    best_df = df.loc[idx].copy().reset_index(drop=True)
    best_df["best_candidate_subject"] = best_df["candidate_subject"].astype(str)
    best_df["best_score"] = best_df["score"].astype(float)
    best_df["threshold"] = float(threshold)
    best_df["predicted_subject_thresholded"] = np.where(
        best_df["best_score"] >= threshold,
        best_df["best_candidate_subject"],
        "None",
    )
    expected = best_df["expected_subject"].fillna("").astype(str)
    predicted = best_df["predicted_subject_thresholded"].astype(str)
    best_df["has_expected_subject"] = expected.str.len() > 0
    best_df["correct_thresholded"] = best_df["has_expected_subject"] & (predicted == expected)
    best_df["unknown_thresholded"] = predicted == "None"
    return best_df


def compute_threshold_metrics(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    decision_df = decision_table_from_scores(scores_df, threshold)
    if decision_df.empty:
        return pd.DataFrame()

    expected = decision_df["expected_subject"].fillna("").astype(str)
    predicted = decision_df["predicted_subject_thresholded"].astype(str)
    has_expected = expected.str.len() > 0
    predicted_known = predicted != "None"
    tp = int((has_expected & (predicted == expected)).sum())
    fp = int((predicted_known & ((~has_expected) | (predicted != expected))).sum())
    fn = int((has_expected & (predicted != expected)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = np.nan if pd.isna(precision) or pd.isna(recall) else (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )

    df = scores_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["is_match"] = df["is_match"].apply(_bool_value)
    match_mask = df["is_match"]
    mismatch_mask = ~match_mask
    false_accept_rate = float(((df["score"] >= threshold) & mismatch_mask).sum() / mismatch_mask.sum())
    false_reject_rate = float(((df["score"] < threshold) & match_mask).sum() / match_mask.sum())

    return pd.DataFrame(
        [
            {
                "protocol": df["protocol"].iloc[0] if "protocol" in df else "",
                "gallery_name": df["gallery_name"].iloc[0] if "gallery_name" in df else "",
                "threshold": float(threshold),
                "n_decisions": int(len(decision_df)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "unknown_rate": float(decision_df["unknown_thresholded"].mean()),
                "false_accept_rate": false_accept_rate,
                "false_reject_rate": false_reject_rate,
            }
        ]
    )


def threshold_curve(scores_df: pd.DataFrame, n_thresholds: int = 501) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()
    scores = pd.to_numeric(scores_df["score"], errors="coerce").fillna(0.0)
    if scores.empty:
        return pd.DataFrame()
    low = float(scores.min()) - 1e-6
    high = float(scores.max()) + 1e-6
    thresholds = np.linspace(low, high, n_thresholds)
    frames = [compute_threshold_metrics(scores_df, threshold) for threshold in thresholds]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def choose_best_threshold(curve_df: pd.DataFrame) -> float:
    if curve_df.empty:
        return 0.0
    ranked = curve_df.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
        na_position="last",
    )
    return float(ranked.iloc[0]["threshold"])


def compute_pair_threshold_metrics(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Verification metrics when every probe-candidate pair is accepted/rejected by threshold."""
    if scores_df.empty:
        return pd.DataFrame()
    df = scores_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["is_match"] = df["is_match"].apply(_bool_value)

    accepted = df["score"] >= threshold
    labels = df["is_match"]
    tp = int((accepted & labels).sum())
    fp = int((accepted & ~labels).sum())
    fn = int((~accepted & labels).sum())
    tn = int((~accepted & ~labels).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = np.nan if pd.isna(precision) or pd.isna(recall) else (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    false_accept_rate = fp / (fp + tn) if (fp + tn) else np.nan
    false_reject_rate = fn / (tp + fn) if (tp + fn) else np.nan
    youden_j = recall - false_accept_rate if not pd.isna(recall) and not pd.isna(false_accept_rate) else np.nan

    return pd.DataFrame(
        [
            {
                "protocol": df["protocol"].iloc[0] if "protocol" in df else "",
                "gallery_name": df["gallery_name"].iloc[0] if "gallery_name" in df else "",
                "threshold": float(threshold),
                "n_pairs": int(len(df)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_accept_rate": false_accept_rate,
                "false_reject_rate": false_reject_rate,
                "youden_j": youden_j,
            }
        ]
    )


def pair_threshold_curve(scores_df: pd.DataFrame, n_thresholds: int = 501) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()
    scores = pd.to_numeric(scores_df["score"], errors="coerce").fillna(0.0)
    if scores.empty:
        return pd.DataFrame()
    low = float(scores.min()) - 1e-6
    high = float(scores.max()) + 1e-6
    thresholds = np.linspace(low, high, n_thresholds)
    frames = [compute_pair_threshold_metrics(scores_df, threshold) for threshold in thresholds]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def choose_pair_threshold(curve_df: pd.DataFrame, strategy: str = "youden_j") -> float:
    if curve_df.empty:
        return 0.0
    strategy = str(strategy).strip().lower()
    if strategy in {"f1", "pair_f1"}:
        sort_cols = ["f1", "precision", "threshold"]
        ascending = [False, False, False]
    else:
        sort_cols = ["youden_j", "f1", "threshold"]
        ascending = [False, False, False]
    ranked = curve_df.sort_values(sort_cols, ascending=ascending, na_position="last")
    return float(ranked.iloc[0]["threshold"])


def _save_fig(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(scores_df: pd.DataFrame, out_path: Path, title: str) -> Path | None:
    if scores_df.empty:
        return None
    _, roc_auc_score, roc_curve = _require_sklearn_metrics()
    df = scores_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["is_match"] = df["is_match"].apply(_bool_value)
    labels = df["is_match"].astype(int).to_numpy()
    if labels.sum() == 0 or labels.sum() == len(labels):
        warnings.warn(f"No se puede graficar ROC para {title}: falta una clase.", stacklevel=2)
        return None
    fpr, tpr, _ = roc_curve(labels, df["score"].astype(float).to_numpy())
    auc_value = roc_auc_score(labels, df["score"].astype(float).to_numpy())
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot(fpr, tpr, label=f"AUC={auc_value:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="azar")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_score_histogram(
    scores_df: pd.DataFrame,
    out_path: Path,
    title: str,
    selected_threshold: float | None = None,
) -> Path | None:
    if scores_df.empty:
        return None
    df = scores_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["is_match"] = df["is_match"].apply(_bool_value)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(df.loc[~df["is_match"], "score"], bins=50, alpha=0.65, label="mismatch", density=True)
    ax.hist(df.loc[df["is_match"], "score"], bins=50, alpha=0.65, label="match", density=True)
    if selected_threshold is not None:
        ax.axvline(float(selected_threshold), color="black", linestyle="--", linewidth=1.5, label="threshold")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_threshold_curve(
    curve_df: pd.DataFrame,
    out_path: Path,
    title: str,
    selected_threshold: float | None = None,
) -> Path | None:
    if curve_df.empty:
        return None
    df = curve_df.copy()
    for col in ["threshold", "precision", "recall", "f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(df["threshold"], df["precision"], label="precision")
    ax.plot(df["threshold"], df["recall"], label="recall")
    ax.plot(df["threshold"], df["f1"], label="F1", linewidth=2)
    if selected_threshold is not None:
        ax.axvline(float(selected_threshold), color="black", linestyle="--", linewidth=1.5, label="selected threshold")
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_auc_heatmap(
    auc_df: pd.DataFrame,
    out_path: Path,
    title: str,
    distance_col: str = "distance_m",
    xlabel: str = "ground distance (m)",
) -> Path | None:
    required = {"height_m", distance_col, "auc"}
    if auc_df.empty or not required.issubset(auc_df.columns):
        return None
    df = auc_df.copy()
    for col in ["height_m", distance_col, "auc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    pivot = df.pivot_table(index="height_m", columns=distance_col, values="auc", aggfunc="mean").sort_index()
    pivot = pivot.sort_index(axis=1)
    if pivot.empty:
        return None
    fig_width = max(9.5, 0.72 * len(pivot.columns))
    fig_height = max(4.8, 0.78 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("height (m)", fontsize=11)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index], fontsize=10)
    for y_idx, height in enumerate(pivot.index):
        for x_idx, distance in enumerate(pivot.columns):
            value = pivot.loc[height, distance]
            if pd.notna(value):
                text_color = "white" if float(value) < 0.55 else "black"
                ax.text(
                    x_idx,
                    y_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="semibold",
                )
    fig.colorbar(image, ax=ax, label="AUC", fraction=0.035, pad=0.025)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def analyze_verification_scores(
    scores_df: pd.DataFrame,
    output_prefix: str,
    threshold: float | None = None,
    threshold_strategy: str = "youden_j",
    figures: bool = True,
) -> dict[str, pd.DataFrame | float | list[Path]]:
    """Compute AUC tables, threshold metrics, and figures from saved verification scores."""
    V2_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (V2_FIGS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    auc_global = compute_auc_metrics(scores_df)
    detected_scores = scores_df[scores_df["detected"].apply(_bool_value)].copy() if "detected" in scores_df else scores_df
    auc_detected_global = compute_auc_metrics(detected_scores)
    if not auc_detected_global.empty:
        auc_detected_global["group"] = "detected_only"
    auc_by_height = compute_auc_metrics(scores_df, ["height_m"])
    auc_by_distance = compute_auc_metrics(scores_df, ["distance_m"])
    auc_by_height_distance = compute_auc_metrics(scores_df, ["height_m", "distance_m"])
    binned_scores_df = add_paper_distance_bins(scores_df)
    auc_by_paper_distance = compute_auc_metrics(binned_scores_df, ["paper_distance_m"])
    auc_by_paper_height_distance = compute_auc_metrics(binned_scores_df, ["height_m", "paper_distance_m"])
    decision_curve_df = threshold_curve(scores_df)
    pair_curve_df = pair_threshold_curve(scores_df)
    threshold_source = threshold_strategy if threshold is None else "provided"
    chosen_threshold = choose_pair_threshold(pair_curve_df, threshold_strategy) if threshold is None else float(threshold)
    threshold_metrics = compute_threshold_metrics(scores_df, chosen_threshold)
    if not threshold_metrics.empty:
        threshold_metrics.insert(3, "threshold_source", threshold_source)
    decisions_df = decision_table_from_scores(scores_df, chosen_threshold)

    auc_global.to_csv(V2_TABLES_DIR / f"{output_prefix}_auc_global.csv", index=False)
    auc_detected_global.to_csv(V2_TABLES_DIR / f"{output_prefix}_auc_detected_only_global.csv", index=False)
    auc_by_height.to_csv(V2_TABLES_DIR / f"{output_prefix}_auc_by_height.csv", index=False)
    auc_by_distance.to_csv(V2_TABLES_DIR / f"{output_prefix}_auc_by_distance.csv", index=False)
    auc_by_height_distance.to_csv(V2_TABLES_DIR / f"{output_prefix}_auc_by_height_distance.csv", index=False)
    auc_by_paper_distance.to_csv(V2_TABLES_DIR / f"{output_prefix}_auc_by_paper_distance.csv", index=False)
    auc_by_paper_height_distance.to_csv(
        V2_TABLES_DIR / f"{output_prefix}_auc_by_paper_height_distance.csv",
        index=False,
    )
    pair_curve_df.to_csv(V2_TABLES_DIR / f"{output_prefix}_threshold_curve.csv", index=False)
    pair_curve_df.to_csv(V2_TABLES_DIR / f"{output_prefix}_pair_threshold_curve.csv", index=False)
    decision_curve_df.to_csv(V2_TABLES_DIR / f"{output_prefix}_decision_threshold_curve.csv", index=False)
    threshold_metrics.to_csv(V2_TABLES_DIR / f"{output_prefix}_threshold_metrics.csv", index=False)
    decisions_df.to_csv(V2_RESULTS_DIR / f"{output_prefix}_threshold_decisions.csv", index=False)

    figure_paths: list[Path] = []
    if figures:
        metrics_dir = V2_FIGS_DIR / "metrics"
        plot_specs = [
            plot_roc_curve(scores_df, metrics_dir / f"{output_prefix}_roc.png", f"{output_prefix}: ROC"),
            plot_roc_curve(
                detected_scores,
                metrics_dir / f"{output_prefix}_roc_detected_only.png",
                f"{output_prefix}: ROC detectados",
            ),
            plot_score_histogram(
                scores_df,
                metrics_dir / f"{output_prefix}_score_histogram.png",
                f"{output_prefix}: scores",
                selected_threshold=chosen_threshold,
            ),
            plot_score_histogram(
                detected_scores,
                metrics_dir / f"{output_prefix}_score_histogram_detected_only.png",
                f"{output_prefix}: scores detectados",
                selected_threshold=chosen_threshold,
            ),
            plot_threshold_curve(
                pair_curve_df,
                metrics_dir / f"{output_prefix}_f1_vs_threshold.png",
                f"{output_prefix}: verification threshold",
                selected_threshold=chosen_threshold,
            ),
            plot_threshold_curve(
                decision_curve_df,
                metrics_dir / f"{output_prefix}_decision_f1_vs_threshold.png",
                f"{output_prefix}: decision threshold",
                selected_threshold=chosen_threshold,
            ),
            plot_auc_heatmap(
                auc_by_height_distance,
                metrics_dir / f"{output_prefix}_auc_height_distance.png",
                f"{output_prefix}: AUC por altura/distancia",
            ),
            plot_auc_heatmap(
                auc_by_paper_height_distance,
                metrics_dir / f"{output_prefix}_auc_paper_height_distance.png",
                f"{output_prefix}: AUC por altura/distancia (bins paper)",
                distance_col="paper_distance_m",
                xlabel="paper-binned ground distance (m)",
            ),
        ]
        figure_paths = [path for path in plot_specs if path is not None]

    return {
        "auc_global": auc_global,
        "auc_detected_global": auc_detected_global,
        "auc_by_height": auc_by_height,
        "auc_by_distance": auc_by_distance,
        "auc_by_height_distance": auc_by_height_distance,
        "auc_by_paper_distance": auc_by_paper_distance,
        "auc_by_paper_height_distance": auc_by_paper_height_distance,
        "threshold_curve": pair_curve_df,
        "pair_threshold_curve": pair_curve_df,
        "decision_threshold_curve": decision_curve_df,
        "threshold": chosen_threshold,
        "threshold_metrics": threshold_metrics,
        "decisions": decisions_df,
        "figure_paths": figure_paths,
    }


def run_saved_verification_analysis(gallery_name: str = "trio_front") -> dict[str, object]:
    paths = verification_score_paths(gallery_name)
    if not paths.faces_scores.exists() or not paths.bbox_scores.exists():
        raise FileNotFoundError(
            "Faltan CSV de scores verification. Ejecuta primero el scoring semi-pesado."
        )
    faces_scores = pd.read_csv(paths.faces_scores)
    bbox_scores = pd.read_csv(paths.bbox_scores)
    faces_analysis = analyze_verification_scores(
        faces_scores,
        output_prefix=f"photos_all_faces_verification_{gallery_name}",
        threshold=None,
    )
    threshold = float(faces_analysis["threshold"])
    bbox_analysis = analyze_verification_scores(
        bbox_scores,
        output_prefix=f"photos_all_bbox_verification_{gallery_name}",
        threshold=threshold,
    )
    threshold_summary = pd.concat(
        [faces_analysis["threshold_metrics"], bbox_analysis["threshold_metrics"]],
        ignore_index=True,
    )
    threshold_summary.to_csv(
        V2_TABLES_DIR / f"verification_threshold_summary_{gallery_name}.csv",
        index=False,
    )
    auc_summary = pd.concat(
        [
            faces_analysis["auc_global"],
            faces_analysis["auc_detected_global"],
            bbox_analysis["auc_global"],
            bbox_analysis["auc_detected_global"],
        ],
        ignore_index=True,
    )
    auc_summary.to_csv(V2_TABLES_DIR / f"verification_auc_summary_{gallery_name}.csv", index=False)
    return {
        "faces_scores": faces_scores,
        "bbox_scores": bbox_scores,
        "faces_analysis": faces_analysis,
        "bbox_analysis": bbox_analysis,
        "threshold": threshold,
        "threshold_summary": threshold_summary,
        "auc_summary": auc_summary,
    }
