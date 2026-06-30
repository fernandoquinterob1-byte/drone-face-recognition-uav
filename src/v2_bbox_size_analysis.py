from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import PROJECT_ROOT
from .v2_figures import V2_FIGS_DIR, V2_RESULTS_DIR, V2_TABLES_DIR


BBOX_HEIGHT_BINS = [-np.inf, 30, 40, 50, 70, 100, np.inf]
BBOX_HEIGHT_LABELS = ["<30 px", "30-39 px", "40-49 px", "50-69 px", "70-99 px", ">=100 px"]


@dataclass
class BBoxSizeAnalysis:
    detections: pd.DataFrame
    metrics: pd.DataFrame
    figure_paths: list[Path]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float).ne(0)
    return series.astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _resolve_image_path(value) -> Path:
    text = str(value)
    candidates = [Path(text)]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(Path(text.encode(encoding).decode("utf-8")))
        except UnicodeError:
            pass
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = int(max(0, min(width - 1, np.floor(x1))))
    top = int(max(0, min(height - 1, np.floor(y1))))
    right = int(max(left + 1, min(width, np.ceil(x2))))
    bottom = int(max(top + 1, min(height, np.ceil(y2))))
    return left, top, right, bottom


def _center_crop_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    scale: float,
    min_side: int,
) -> tuple[int, int, int, int]:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(float(x2 - x1), float(y2 - y1)) * scale
    side = max(side, float(min_side))
    half = side / 2
    left = max(0, int(round(cx - half)))
    top = max(0, int(round(cy - half)))
    right = min(width, int(round(cx + half)))
    bottom = min(height, int(round(cy + half)))
    if right - left < min_side:
        deficit = min_side - (right - left)
        left = max(0, left - deficit // 2)
        right = min(width, left + min_side)
    if bottom - top < min_side:
        deficit = min_side - (bottom - top)
        top = max(0, top - deficit // 2)
        bottom = min(height, top + min_side)
    return left, top, right, bottom


def load_labeled_bbox_detections(
    path: Path = V2_RESULTS_DIR / "photos_all_detections_labeled_all_galleries.csv",
    gallery_name: str = "trio_front",
) -> pd.DataFrame:
    """Load saved bbox-level rows. This does not initialize InsightFace."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero el pipeline V2 pesado.")
    df = pd.read_csv(path)
    if "gallery_name" in df.columns:
        df = df[df["gallery_name"].astype(str) == gallery_name].copy()
    return df.reset_index(drop=True)


def add_bbox_size_columns(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Add bbox size and correctness columns for detected expected slots."""
    if labeled_df.empty:
        return pd.DataFrame()
    bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "tp", "fp", "fn", "height_m", "distance_m"]
    df = _numeric(labeled_df, bbox_cols)
    required_bbox = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    has_expected = df["expected_subject"].notna() if "expected_subject" in df else pd.Series(False, index=df.index)
    has_bbox = df[required_bbox].notna().all(axis=1)
    df = df[has_expected & has_bbox].copy()
    if df.empty:
        return df

    df["bbox_width"] = df["bbox_x2"] - df["bbox_x1"]
    df["bbox_height"] = df["bbox_y2"] - df["bbox_y1"]
    df["bbox_min_side"] = df[["bbox_width", "bbox_height"]].min(axis=1)
    df["bbox_area"] = df["bbox_width"] * df["bbox_height"]
    df = df[(df["bbox_width"] > 0) & (df["bbox_height"] > 0)].copy()
    df["correct_recognition"] = _as_bool(df["tp"])
    df["bbox_height_bin"] = pd.cut(
        df["bbox_height"],
        bins=BBOX_HEIGHT_BINS,
        labels=BBOX_HEIGHT_LABELS,
        right=False,
    )
    return df.reset_index(drop=True)


def compute_bbox_height_bin_metrics(size_df: pd.DataFrame) -> pd.DataFrame:
    if size_df.empty:
        return pd.DataFrame()
    rows = []
    for label in BBOX_HEIGHT_LABELS:
        group = size_df[size_df["bbox_height_bin"].astype(str) == label]
        if group.empty:
            rows.append(
                {
                    "bbox_height_bin": label,
                    "detected_slots": 0,
                    "correct_count": 0,
                    "incorrect_count": 0,
                    "accuracy_conditional": np.nan,
                    "mean_bbox_height": np.nan,
                    "median_bbox_height": np.nan,
                    "mean_bbox_min_side": np.nan,
                    "mean_distance_m": np.nan,
                }
            )
            continue
        correct = int(group["correct_recognition"].sum())
        total = int(len(group))
        rows.append(
            {
                "bbox_height_bin": label,
                "detected_slots": total,
                "correct_count": correct,
                "incorrect_count": total - correct,
                "accuracy_conditional": correct / total if total else np.nan,
                "mean_bbox_height": float(group["bbox_height"].mean()),
                "median_bbox_height": float(group["bbox_height"].median()),
                "mean_bbox_min_side": float(group["bbox_min_side"].mean()),
                "mean_distance_m": float(group["distance_m"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _save_fig(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_by_bbox_height(metrics_df: pd.DataFrame, out_path: Path) -> Path | None:
    if metrics_df.empty:
        warnings.warn("No hay metricas bbox-size para graficar accuracy por altura.", stacklevel=2)
        return None
    df = _numeric(metrics_df, ["accuracy_conditional", "detected_slots"])
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars = ax.bar(df["bbox_height_bin"].astype(str), df["accuracy_conditional"], color="#4c78a8")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("altura de bounding box")
    ax.set_ylabel("accuracy condicionada a deteccion")
    ax.set_title("photos_all bbox-level: confiabilidad segun tamano de rostro")
    ax.grid(axis="y", alpha=0.25)
    labels = [
        "" if pd.isna(acc) else f"{acc:.1%}\nn={int(n)}"
        for acc, n in zip(df["accuracy_conditional"], df["detected_slots"])
    ]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def plot_bbox_size_vs_distance(size_df: pd.DataFrame, out_path: Path) -> Path | None:
    if size_df.empty:
        warnings.warn("No hay datos bbox-size para graficar tamano vs distancia.", stacklevel=2)
        return None
    df = _numeric(size_df, ["bbox_height", "distance_m", "height_m"])
    grouped = (
        df.groupby(["height_m", "distance_m"], dropna=True)["bbox_height"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(["height_m", "distance_m"])
    )
    if grouped.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for height, height_df in grouped.groupby("height_m"):
        ax.plot(
            height_df["distance_m"],
            height_df["mean"],
            marker="o",
            linewidth=1.8,
            label=f"{height:g} m",
        )
    ax.axhline(30, color="#b91c1c", linestyle="--", linewidth=1.3, label="30 px")
    ax.set_xlabel("distancia al sujeto (m)")
    ax.set_ylabel("altura media de bbox (px)")
    ax.set_title("photos_all: tamano de rostro detectado vs distancia")
    ax.grid(alpha=0.25)
    ax.legend(title="altura de dron", ncols=2)
    fig.tight_layout()
    return _save_fig(fig, out_path)


def _select_zoom_examples(size_df: pd.DataFrame, max_examples: int = 4) -> pd.DataFrame:
    if size_df.empty:
        return pd.DataFrame()
    df = size_df.copy()
    selected = []

    candidate_specs = [
        df[df["correct_recognition"] & (df["distance_m"] <= 8)].sort_values("bbox_height", ascending=False),
        df[df["correct_recognition"] & (df["distance_m"] >= 13)].sort_values("bbox_height", ascending=True),
        df[~df["correct_recognition"]].sort_values(["distance_m", "bbox_height"], ascending=[False, True]),
        df[(df["composite_subject_id"].astype(str) == "ijk")].sort_values("bbox_height", ascending=True),
    ]
    seen = set()
    for candidates in candidate_specs:
        for _, row in candidates.iterrows():
            key = (str(row.get("filename", "")), str(row.get("spatial_rank", "")))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            break
        if len(selected) >= max_examples:
            break

    if len(selected) < max_examples:
        remaining = df.sort_values(["distance_m", "bbox_height"], ascending=[False, True])
        for _, row in remaining.iterrows():
            key = (str(row.get("filename", "")), str(row.get("spatial_rank", "")))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= max_examples:
                break

    return pd.DataFrame(selected)


def generate_bbox_zoom_collage(size_df: pd.DataFrame, out_path: Path, max_examples: int = 4) -> Path | None:
    examples = _select_zoom_examples(size_df, max_examples=max_examples)
    if examples.empty:
        warnings.warn("No hay ejemplos para generar collage de zooms.", stacklevel=2)
        return None

    panel_w, panel_h = 960, 500
    cols = 2
    rows = int(np.ceil(len(examples) / cols))
    margin = 26
    title_h = 62
    canvas = Image.new("RGB", (cols * panel_w + margin * 2, rows * panel_h + margin * 2 + title_h), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34, bold=True)
    text_font = _font(22)
    small_font = _font(18)
    draw.text((margin, margin), "Ejemplos con zoom de detecciones en photos_all", font=title_font, fill=(17, 24, 39))

    for idx, row in examples.reset_index(drop=True).iterrows():
        col = idx % cols
        row_idx = idx // cols
        x0 = margin + col * panel_w
        y0 = margin + title_h + row_idx * panel_h
        draw.rounded_rectangle((x0 + 8, y0 + 8, x0 + panel_w - 16, y0 + panel_h - 16), radius=10, fill=(255, 255, 255), outline=(203, 213, 225))

        path = _resolve_image_path(row["path"])
        if not path.exists():
            draw.text((x0 + 30, y0 + 42), f"No se encontro imagen: {path.name}", font=text_font, fill=(185, 28, 28))
            continue

        img = Image.open(path).convert("RGB")
        img = ImageOps.autocontrast(img)
        iw, ih = img.size
        x1, y1, x2, y2 = float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])
        context_box = _center_crop_box(x1, y1, x2, y2, iw, ih, scale=18.0, min_side=520)
        face_box = _center_crop_box(x1, y1, x2, y2, iw, ih, scale=3.2, min_side=90)
        context = img.crop(context_box).resize((360, 300), Image.Resampling.LANCZOS)
        face = img.crop(face_box).resize((230, 230), Image.Resampling.LANCZOS)

        cx1, cy1, _, _ = context_box
        sx = 360 / max(1, context_box[2] - context_box[0])
        sy = 300 / max(1, context_box[3] - context_box[1])
        scaled_box = (
            int((x1 - cx1) * sx),
            int((y1 - cy1) * sy),
            int((x2 - cx1) * sx),
            int((y2 - cy1) * sy),
        )
        context_draw = ImageDraw.Draw(context)
        is_correct = bool(row["correct_recognition"])
        box_color = (22, 163, 74) if is_correct else (220, 38, 38)
        context_draw.rectangle(scaled_box, outline=box_color, width=4)
        face_draw = ImageDraw.Draw(face)
        face_draw.rectangle((2, 2, 228, 228), outline=box_color, width=5)

        canvas.paste(context, (x0 + 30, y0 + 145))
        canvas.paste(face, (x0 + 420, y0 + 150))

        expected = str(row.get("expected_subject", ""))
        predicted = str(row.get("predicted_subject", ""))
        status = "correcto" if is_correct else "error"
        caption = (
            f"{row.get('filename', '')}\n"
            f"esp={expected}, pred={predicted} ({status})\n"
            f"dist={float(row['distance_m']):g} m, altura={float(row['height_m']):g} m, bbox_h={float(row['bbox_height']):.1f} px"
        )
        draw.multiline_text((x0 + 30, y0 + 26), caption, font=small_font, fill=(31, 41, 55), spacing=4)
        draw.text((x0 + 420, y0 + 390), "zoom del rostro detectado", font=text_font, fill=(75, 85, 99))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    return out_path


def run_bbox_size_analysis(
    gallery_name: str = "trio_front",
    labeled_path: Path = V2_RESULTS_DIR / "photos_all_detections_labeled_all_galleries.csv",
    tables_dir: Path = V2_TABLES_DIR,
    figs_dir: Path = V2_FIGS_DIR,
    report_figs_dir: Path = PROJECT_ROOT / "Informe" / "figs",
) -> BBoxSizeAnalysis:
    labeled_df = load_labeled_bbox_detections(labeled_path, gallery_name=gallery_name)
    size_df = add_bbox_size_columns(labeled_df)
    metrics_df = compute_bbox_height_bin_metrics(size_df)

    tables_dir = Path(tables_dir)
    metrics_dir = Path(figs_dir) / "metrics"
    detection_dir = Path(figs_dir) / "detection"
    report_detection_dir = Path(report_figs_dir) / "detection"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)
    report_detection_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / f"photos_all_bbox_size_metrics_{gallery_name}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    figure_paths: list[Path] = []
    for maybe_path in [
        plot_accuracy_by_bbox_height(
            metrics_df,
            metrics_dir / "photos_all_bbox_accuracy_by_bbox_height.png",
        ),
        plot_bbox_size_vs_distance(
            size_df,
            metrics_dir / "photos_all_bbox_size_vs_distance.png",
        ),
        generate_bbox_zoom_collage(
            size_df,
            detection_dir / f"photos_all_bbox_zoom_examples_{gallery_name}.png",
        ),
        generate_bbox_zoom_collage(
            size_df,
            report_detection_dir / "droneface_full_image_detection_zooms.png",
        ),
    ]:
        if maybe_path is not None:
            figure_paths.append(maybe_path)

    return BBoxSizeAnalysis(detections=size_df, metrics=metrics_df, figure_paths=figure_paths)
