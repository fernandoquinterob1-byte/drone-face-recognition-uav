from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT


OUTPUTS_V2_DIR = PROJECT_ROOT / "outputs_v2"
V2_RESULTS_DIR = OUTPUTS_V2_DIR / "results"
V2_TABLES_DIR = OUTPUTS_V2_DIR / "tables"
V2_FIGS_DIR = OUTPUTS_V2_DIR / "figs"
V2_METRICS_FIGS_DIR = V2_FIGS_DIR / "metrics"

GALLERY_NAMES = ("trio_front", "trio_side", "trio_all")


@dataclass
class V2LightResults:
    pose_comparison: pd.DataFrame
    full_set_summary: pd.DataFrame
    bbox_metrics: pd.DataFrame
    labeled_bbox: pd.DataFrame
    faces_results: dict[str, pd.DataFrame]
    full_results: dict[str, pd.DataFrame]


def ensure_v2_figure_dirs() -> None:
    V2_METRICS_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    V2_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"No existe {path}. Ejecuta primero el pipeline pesado con RUN_HEAVY_PIPELINE=True."
            )
        return pd.DataFrame()
    return pd.read_csv(path)


def load_v2_light_results(
    results_dir: Path = V2_RESULTS_DIR,
    tables_dir: Path = V2_TABLES_DIR,
) -> V2LightResults:
    """Load V2 outputs from CSV only. This function does not import or initialize InsightFace."""
    results_dir = Path(results_dir)
    tables_dir = Path(tables_dir)

    faces_results = {
        gallery: _read_csv(results_dir / f"photos_all_faces_{gallery}.csv")
        for gallery in GALLERY_NAMES
    }
    full_results = {
        gallery: _read_csv(results_dir / f"photos_all_{gallery}.csv")
        for gallery in GALLERY_NAMES
    }
    return V2LightResults(
        pose_comparison=_read_csv(tables_dir / "photos_all_faces_pose_comparison.csv"),
        full_set_summary=_read_csv(tables_dir / "photos_all_set_metrics_by_gallery.csv"),
        bbox_metrics=_read_csv(tables_dir / "photos_all_bbox_metrics_by_gallery.csv", required=False),
        labeled_bbox=_read_csv(results_dir / "photos_all_detections_labeled_all_galleries.csv", required=False),
        faces_results=faces_results,
        full_results=full_results,
    )


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _as_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sort_for_plot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    numeric = pd.to_numeric(out[column], errors="coerce")
    if numeric.notna().any():
        out["_sort_key"] = numeric
    else:
        out["_sort_key"] = out[column].astype(str)
    return out.sort_values("_sort_key").drop(columns=["_sort_key"])


def _save_fig(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _format_value(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:g}"
    return str(value)


def compute_closed_set_metrics(results_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for group_values, group_df in results_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        detected = _safe_bool(group_df["detected"])
        correct = _safe_bool(group_df["correct"])
        total = int(len(group_df))
        detected_count = int(detected.sum())
        correct_count = int(correct.sum())
        row = {
            "total_probe": total,
            "detected_count": detected_count,
            "detection_rate": detected_count / total if total else np.nan,
            "correct_count": correct_count,
            "accuracy_conditional": correct_count / detected_count if detected_count else np.nan,
            "accuracy_global": correct_count / total if total else np.nan,
        }
        for col, value in zip(group_cols, group_values):
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def compute_full_image_metrics(results_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    numeric_cols = [
        "num_faces_detected",
        "n_expected_with_gallery",
        "n_expected_found",
        "expected_identity_recall",
        "jaccard_set_similarity",
    ]
    df = _as_numeric(results_df, numeric_cols)
    rows = []
    for group_values, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        total_images = int(len(group_df))
        expected_total = float(group_df["n_expected_with_gallery"].fillna(0).sum())
        found_total = float(group_df["n_expected_found"].fillna(0).sum())
        row = {
            "total_images": total_images,
            "images_with_detections": int((group_df["num_faces_detected"].fillna(0) > 0).sum()),
            "total_expected_identities_with_gallery": expected_total,
            "expected_identities_found": found_total,
            "expected_identity_recall_global": found_total / expected_total if expected_total else np.nan,
            "mean_expected_identity_recall": group_df["expected_identity_recall"].mean(),
            "mean_jaccard_set_similarity": group_df["jaccard_set_similarity"].mean(),
            "exact_expected_set_match_rate": _safe_bool(group_df["exact_expected_set_match"]).mean(),
        }
        for col, value in zip(group_cols, group_values):
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def compute_bbox_metrics(labeled_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if labeled_df.empty:
        return pd.DataFrame()
    df = _as_numeric(labeled_df, ["tp", "fp", "fn", "height_m", "distance_m"])
    rows = []
    for group_values, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        tp = int(group_df["tp"].fillna(0).sum())
        fp = int(group_df["fp"].fillna(0).sum())
        fn = int(group_df["fn"].fillna(0).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = np.nan if pd.isna(precision) or pd.isna(recall) else (
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
        row = {
            "group": ", ".join(f"{col}={value}" for col, value in zip(group_cols, group_values)),
            "total_slots": int(len(group_df)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for col, value in zip(group_cols, group_values):
            row[col] = value
        if "gallery_name" not in row and "gallery_name" in group_df.columns:
            row["gallery_name"] = group_df["gallery_name"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def bbox_top_level_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    required = {"gallery_name", "group"}
    if not required.issubset(metrics_df.columns):
        return metrics_df.copy()
    out = metrics_df[
        metrics_df.apply(lambda row: str(row["group"]) == f"gallery_name={row['gallery_name']}", axis=1)
    ].copy()
    return _as_numeric(out, ["total_slots", "tp", "fp", "fn", "precision", "recall", "f1"])


def plot_pose_comparison(comparison_df: pd.DataFrame, out_path: Path) -> Path | None:
    if comparison_df.empty:
        warnings.warn("No hay datos para graficar comparacion por poses.", stacklevel=2)
        return None
    plot_df = _as_numeric(
        comparison_df,
        ["accuracy_global", "accuracy_conditional"],
    ).sort_values("gallery_name")
    labels = plot_df["gallery_name"].astype(str).tolist()
    x = np.arange(len(plot_df))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars_global = ax.bar(
        x - width / 2,
        plot_df["accuracy_global"],
        width=width,
        label="accuracy_global",
        color="#4c78a8",
    )
    bars_conditional = ax.bar(
        x + width / 2,
        plot_df["accuracy_conditional"],
        width=width,
        label="accuracy_conditional",
        color="#f58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("proporcion")
    ax.set_title("V2 - photos_all_faces: reconocimiento por galeria")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(bars_global, labels=[f"{v:.1%}" for v in plot_df["accuracy_global"]], padding=3, fontsize=8)
    ax.bar_label(
        bars_conditional,
        labels=[f"{v:.1%}" for v in plot_df["accuracy_conditional"]],
        padding=3,
        fontsize=8,
    )
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_bar(metric_df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out_path: Path):
    if metric_df.empty or x_col not in metric_df or y_col not in metric_df:
        warnings.warn(f"No hay datos para graficar {title}.", stacklevel=2)
        return None
    plot_df = _sort_for_plot(_as_numeric(metric_df, [y_col]).dropna(subset=[x_col, y_col]), x_col)
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(plot_df)), 4.5))
    ax.bar(plot_df[x_col].map(_format_value), plot_df[y_col], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_heatmap(metric_df: pd.DataFrame, value_col: str, title: str, out_path: Path):
    required = {"height_m", "distance_m", value_col}
    if metric_df.empty or not required.issubset(metric_df.columns):
        warnings.warn(f"No hay datos para graficar {title}.", stacklevel=2)
        return None
    df = _as_numeric(metric_df, ["height_m", "distance_m", value_col])
    pivot = df.pivot_table(
        index="height_m",
        columns="distance_m",
        values=value_col,
        aggfunc="mean",
        dropna=False,
    ).sort_index(axis=0).sort_index(axis=1)
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(pivot.columns)), max(4, 0.55 * len(pivot.index))))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("distance_m")
    ax.set_ylabel("height_m")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([_format_value(value) for value in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_format_value(value) for value in pivot.index])
    fig.colorbar(image, ax=ax, label=value_col)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_confusion_matrix(results_df: pd.DataFrame, out_path: Path):
    if results_df.empty:
        return None
    df = results_df[_safe_bool(results_df["detected"]) & results_df["predicted_subject"].notna()].copy()
    if df.empty:
        return None
    matrix = pd.crosstab(
        df["subject_id"].astype(str),
        df["predicted_subject"].astype(str),
        dropna=False,
    )
    labels_x = matrix.columns.tolist()
    labels_y = matrix.index.tolist()
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels_x)), max(6, 0.7 * len(labels_y))))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_title("V2 - matriz de confusion (detectadas)")
    ax.set_xlabel("predicted_subject")
    ax.set_ylabel("subject_id")
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x)
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix.iat[row_idx, col_idx])
            if value:
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="conteo")
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_bbox_metrics_by_gallery(metrics_df: pd.DataFrame, out_path: Path):
    top = bbox_top_level_metrics(metrics_df)
    if top.empty:
        return None
    top = _as_numeric(top.sort_values("gallery_name"), ["precision", "recall", "f1"])
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
    ax.set_title("V2 - bbox-level precision/recall/F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_bbox_counts_by_gallery(metrics_df: pd.DataFrame, out_path: Path):
    top = bbox_top_level_metrics(metrics_df)
    if top.empty:
        return None
    top = _as_numeric(top.sort_values("gallery_name"), ["tp", "fp", "fn"])
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
    ax.set_title("V2 - bbox-level TP/FP/FN")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_bbox_f1_by_gallery(metrics_df: pd.DataFrame, out_path: Path):
    top = bbox_top_level_metrics(metrics_df)
    return plot_bar(top, "gallery_name", "f1", "V2 - bbox-level F1 por galeria", "F1", out_path)


def plot_bbox_f1_grouped(metrics_df: pd.DataFrame, group_col: str, title: str, out_path: Path):
    if metrics_df.empty or group_col not in metrics_df.columns:
        return None
    df = _as_numeric(metrics_df, ["f1"]).dropna(subset=[group_col, "f1"]).copy()
    if df.empty:
        return None
    pivot = df.pivot_table(index=group_col, columns="gallery_name", values="f1", aggfunc="mean")
    if pd.to_numeric(pd.Series(pivot.index), errors="coerce").notna().any():
        pivot["_sort_key"] = pd.to_numeric(pd.Series(pivot.index, index=pivot.index), errors="coerce")
        pivot = pivot.sort_values("_sort_key").drop(columns=["_sort_key"])
    else:
        pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(pivot.index)), 4.8))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylim(0, 1)
    ax.set_xlabel(group_col)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="galeria")
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_bbox_f1_lines_grouped(metrics_df: pd.DataFrame, group_col: str, title: str, out_path: Path):
    if metrics_df.empty or group_col not in metrics_df.columns:
        return None
    df = _as_numeric(metrics_df, ["f1"]).dropna(subset=[group_col, "f1"]).copy()
    if df.empty:
        return None

    pivot = df.pivot_table(index=group_col, columns="gallery_name", values="f1", aggfunc="mean")
    numeric_index = pd.to_numeric(pd.Series(pivot.index, index=pivot.index), errors="coerce")
    if numeric_index.notna().any():
        pivot["_sort_key"] = numeric_index
        pivot = pivot.sort_values("_sort_key").drop(columns=["_sort_key"])
        x = pd.to_numeric(pd.Series(pivot.index), errors="coerce").to_numpy()
        x_labels = [f"{value:g}" if pd.notna(value) else str(label) for value, label in zip(x, pivot.index)]
    else:
        pivot = pivot.sort_index()
        x = np.arange(len(pivot.index))
        x_labels = [str(label) for label in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    for gallery in pivot.columns:
        ax.plot(x, pivot[gallery].to_numpy(), marker="o", linewidth=2.4, markersize=5.5, label=str(gallery))

    ax.set_ylim(0, 1)
    ax.set_xlabel(group_col, fontsize=12)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title(title, fontsize=15)
    if len(x) > 18:
        tick_idx = np.arange(0, len(x), 2)
        ax.set_xticks(x[tick_idx])
        ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=45, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="both", alpha=0.25)
    ax.legend(title="galeria", ncol=min(3, len(pivot.columns)), fontsize=10, title_fontsize=11)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def regenerate_v2_figures(
    results: V2LightResults | None = None,
    results_dir: Path = V2_RESULTS_DIR,
    tables_dir: Path = V2_TABLES_DIR,
    figs_dir: Path = V2_FIGS_DIR,
) -> list[Path]:
    """Regenerate all V2 metric figures from existing CSV outputs only."""
    ensure_v2_figure_dirs()
    results = results or load_v2_light_results(results_dir=results_dir, tables_dir=tables_dir)
    figs_dir = Path(figs_dir)
    metrics_dir = figs_dir / "metrics"
    tables_dir = Path(tables_dir)
    figure_paths: list[Path] = []

    path = plot_pose_comparison(results.pose_comparison, metrics_dir / "photos_all_faces_pose_comparison.png")
    if path is not None:
        figure_paths.append(path)

    for gallery, df in results.faces_results.items():
        by_subject = compute_closed_set_metrics(df, ["subject_id"])
        by_height = compute_closed_set_metrics(df, ["height_m"])
        by_distance = compute_closed_set_metrics(df, ["distance_m"])
        by_height_distance = compute_closed_set_metrics(df, ["height_m", "distance_m"])
        by_subject.to_csv(tables_dir / f"photos_all_faces_{gallery}_by_subject.csv", index=False)
        by_height.to_csv(tables_dir / f"photos_all_faces_{gallery}_by_height.csv", index=False)
        by_distance.to_csv(tables_dir / f"photos_all_faces_{gallery}_by_distance.csv", index=False)
        by_height_distance.to_csv(tables_dir / f"photos_all_faces_{gallery}_by_height_distance.csv", index=False)

        for maybe_path in [
            plot_bar(
                by_subject,
                "subject_id",
                "accuracy_global",
                f"V2 - {gallery}: accuracy por sujeto",
                "accuracy_global",
                metrics_dir / f"photos_all_faces_{gallery}_accuracy_by_subject.png",
            ),
            plot_bar(
                by_height,
                "height_m",
                "accuracy_global",
                f"V2 - {gallery}: accuracy por altura",
                "accuracy_global",
                metrics_dir / f"photos_all_faces_{gallery}_accuracy_by_height.png",
            ),
            plot_bar(
                by_distance,
                "distance_m",
                "accuracy_global",
                f"V2 - {gallery}: accuracy por distancia",
                "accuracy_global",
                metrics_dir / f"photos_all_faces_{gallery}_accuracy_by_distance.png",
            ),
            plot_heatmap(
                by_height_distance,
                "accuracy_global",
                f"V2 - {gallery}: accuracy por altura/distancia",
                metrics_dir / f"photos_all_faces_{gallery}_accuracy_height_distance.png",
            ),
            plot_confusion_matrix(df, metrics_dir / f"photos_all_faces_{gallery}_confusion_matrix.png"),
        ]:
            if maybe_path is not None:
                figure_paths.append(maybe_path)

    for gallery, df in results.full_results.items():
        by_height = compute_full_image_metrics(df, ["height_m"])
        by_distance = compute_full_image_metrics(df, ["distance_m"])
        by_height_distance = compute_full_image_metrics(df, ["height_m", "distance_m"])
        by_height.to_csv(tables_dir / f"photos_all_{gallery}_by_height.csv", index=False)
        by_distance.to_csv(tables_dir / f"photos_all_{gallery}_by_distance.csv", index=False)
        by_height_distance.to_csv(tables_dir / f"photos_all_{gallery}_by_height_distance.csv", index=False)

        for maybe_path in [
            plot_bar(
                by_height,
                "height_m",
                "expected_identity_recall_global",
                f"V2 - {gallery}: recall por altura",
                "expected_identity_recall_global",
                metrics_dir / f"photos_all_{gallery}_recall_by_height.png",
            ),
            plot_bar(
                by_distance,
                "distance_m",
                "expected_identity_recall_global",
                f"V2 - {gallery}: recall por distancia",
                "expected_identity_recall_global",
                metrics_dir / f"photos_all_{gallery}_recall_by_distance.png",
            ),
            plot_heatmap(
                by_height_distance,
                "expected_identity_recall_global",
                f"V2 - {gallery}: recall por altura/distancia",
                metrics_dir / f"photos_all_{gallery}_recall_height_distance.png",
            ),
        ]:
            if maybe_path is not None:
                figure_paths.append(maybe_path)

    if not results.labeled_bbox.empty:
        bbox_by_gallery = compute_bbox_metrics(results.labeled_bbox, ["gallery_name"])
        bbox_by_prefix = compute_bbox_metrics(results.labeled_bbox, ["gallery_name", "composite_subject_id"])
        bbox_by_height = compute_bbox_metrics(results.labeled_bbox, ["gallery_name", "height_m"])
        bbox_by_distance = compute_bbox_metrics(results.labeled_bbox, ["gallery_name", "distance_m"])
        bbox_combined = pd.concat([bbox_by_gallery, bbox_by_prefix], ignore_index=True)
        bbox_combined.to_csv(tables_dir / "photos_all_bbox_metrics_by_gallery.csv", index=False)
        bbox_by_height.to_csv(tables_dir / "photos_all_bbox_metrics_by_height.csv", index=False)
        bbox_by_distance.to_csv(tables_dir / "photos_all_bbox_metrics_by_distance.csv", index=False)
    else:
        bbox_combined = results.bbox_metrics
        bbox_by_prefix = results.bbox_metrics[
            results.bbox_metrics["group"].astype(str).str.contains("composite_subject_id=", regex=False)
        ].copy() if not results.bbox_metrics.empty and "group" in results.bbox_metrics else pd.DataFrame()
        bbox_by_height = pd.DataFrame()
        bbox_by_distance = pd.DataFrame()

    for maybe_path in [
        plot_bbox_f1_by_gallery(bbox_combined, metrics_dir / "photos_all_bbox_f1_by_gallery.png"),
        plot_bbox_metrics_by_gallery(
            bbox_combined,
            metrics_dir / "photos_all_bbox_precision_recall_f1_by_gallery.png",
        ),
        plot_bbox_counts_by_gallery(bbox_combined, metrics_dir / "photos_all_bbox_tp_fp_fn_by_gallery.png"),
        plot_bbox_f1_grouped(
            bbox_by_prefix.rename(columns={"composite_subject_id": "prefix"}),
            "prefix",
            "V2 - bbox-level F1 por prefijo",
            metrics_dir / "photos_all_bbox_f1_by_prefix.png",
        ),
        plot_bbox_f1_grouped(
            bbox_by_height,
            "height_m",
            "V2 - bbox-level F1 por altura",
            metrics_dir / "photos_all_bbox_f1_by_height.png",
        ),
        plot_bbox_f1_lines_grouped(
            bbox_by_distance,
            "distance_m",
            "V2 - bbox-level F1 por distancia",
            metrics_dir / "photos_all_bbox_f1_by_distance.png",
        ),
    ]:
        if maybe_path is not None:
            figure_paths.append(maybe_path)

    return figure_paths
