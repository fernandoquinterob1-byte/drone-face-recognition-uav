from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import INDEX_DIR, ensure_project_dirs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DRONEFACE_SOURCE_FOLDERS = ["photos_all", "photos_all_faces", "portraits", "trio_cam", "trio_gp"]
DRONEFACE_HEIGHTS = {"0": 1.5, "3": 3.0, "4": 4.0, "5": 5.0, "na": np.nan}
DRONEFACE_SINGLE_SUBJECTS = set("abcdefghijk")
DRONEFACE_UNUSED_SOURCE_FOLDERS = {"photos_all", "trio_cam", "trio_gp"}

DRONEFACE_COLUMNS = [
    "path",
    "filename",
    "dataset",
    "source_folder",
    "subject_id",
    "camera_type",
    "height_id",
    "height_m",
    "image_type",
    "distance_id",
    "distance_m",
    "is_single_subject",
    "split",
    "parsed",
    "parse_error",
]


def find_images(root: Path) -> list[Path]:
    """Return image paths below root using a case-insensitive extension match."""
    root = Path(root)
    if not root.exists():
        warnings.warn(f"DroneFace folder does not exist: {root}", stacklevel=2)
        return []
    if not root.is_dir():
        warnings.warn(f"DroneFace path is not a directory: {root}", stacklevel=2)
        return []

    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        warnings.warn(f"No images found in {root} with extensions {sorted(IMAGE_EXTENSIONS)}", stacklevel=2)
    return paths


def _source_folder(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "other"
    if len(relative.parts) < 2:
        return "other"
    folder = relative.parts[0]
    return folder if folder in DRONEFACE_SOURCE_FOLDERS else "other"


def inspect_droneface_dataset(root: Path, n_examples: int = 30) -> dict:
    """Print and return a DroneFace-specific dataset inspection summary."""
    root = Path(root)
    paths = find_images(root)
    source_counts = {folder: 0 for folder in DRONEFACE_SOURCE_FOLDERS}
    source_counts["other"] = 0

    for path in paths:
        source_counts[_source_folder(path, root)] += 1

    extensions = sorted({path.suffix.lower() for path in paths})
    sample_paths = paths[:n_examples]
    sample_filenames = [path.name for path in sample_paths]
    part_counts = Counter(len(path.stem.split("_")) for path in paths)
    image_type_counts = Counter()
    subject_counts = Counter()

    for path in paths:
        parts = path.stem.split("_")
        if len(parts) >= 5:
            subject_counts[parts[0].lower()] += 1
            image_type_counts[parts[3].lower()] += 1

    info = {
        "root": str(root),
        "exists": root.exists(),
        "n_images": len(paths),
        "source_folder_counts": source_counts,
        "first_paths": [str(path) for path in sample_paths],
        "extensions": extensions,
        "sample_filenames": sample_filenames,
        "name_part_counts": dict(sorted(part_counts.items())),
        "image_type_counts": dict(sorted(image_type_counts.items())),
        "subject_counts": dict(sorted(subject_counts.items())),
    }

    print("\n=== DroneFace ===")
    print(f"Root: {root}")
    print(f"Exists: {info['exists']}")
    print(f"Images found: {info['n_images']}")
    print("Images by main folder:")
    for folder, count in source_counts.items():
        print(f"  {folder}: {count}")
        if folder != "other" and count == 0:
            warnings.warn(f"DroneFace subfolder is empty or missing images: {folder}", stacklevel=2)
    print(f"Extensions: {extensions if extensions else 'none'}")
    print("First paths:")
    for path in sample_paths:
        print(f"  {path}")
    print("Filename examples:")
    for filename in sample_filenames:
        print(f"  {filename}")
    print("Approximate filename patterns by '_' part count:")
    if part_counts:
        for n_parts, count in sorted(part_counts.items()):
            print(f"  {n_parts} parts: {count} files")
    else:
        print("  none")
    print("Image types found:")
    if image_type_counts:
        for image_type, count in sorted(image_type_counts.items()):
            print(f"  {image_type}: {count}")
    else:
        print("  none")

    return info


def _empty_droneface_row(path: Path, root: Path, parse_error: str) -> dict:
    return {
        "path": str(path),
        "filename": path.name,
        "dataset": "DroneFace",
        "source_folder": _source_folder(path, root),
        "subject_id": None,
        "camera_type": None,
        "height_id": None,
        "height_m": np.nan,
        "image_type": None,
        "distance_id": None,
        "distance_m": np.nan,
        "is_single_subject": False,
        "split": "unused",
        "parsed": False,
        "parse_error": parse_error,
    }


def _canonical_image_type(value: str) -> str:
    value_lower = value.lower()
    if value_lower == "porf":
        return "porF"
    if value_lower == "porl":
        return "porL"
    if value_lower == "porr":
        return "porR"
    return value_lower


def _parse_droneface_path(path: Path, root: Path) -> dict:
    source_folder = _source_folder(path, root)
    parts = path.stem.split("_")
    if len(parts) != 5:
        return _empty_droneface_row(path, root, f"Expected 5 filename parts, found {len(parts)}")

    subject_id, camera_type, height_id, image_type_raw, distance_id = [part.lower() for part in parts]
    image_type = _canonical_image_type(image_type_raw)
    parse_errors = []

    if not re.fullmatch(r"[a-z]+", subject_id):
        parse_errors.append(f"invalid subject_id={subject_id}")

    if height_id not in DRONEFACE_HEIGHTS:
        parse_errors.append(f"unknown height_id={height_id}")
        height_m = np.nan
    else:
        height_m = DRONEFACE_HEIGHTS[height_id]

    if distance_id == "na":
        distance_m = np.nan
    elif re.fullmatch(r"\d+", distance_id):
        distance_m = 17 - int(distance_id) / 2
    else:
        parse_errors.append(f"invalid distance_id={distance_id}")
        distance_m = np.nan

    is_single_subject = subject_id in DRONEFACE_SINGLE_SUBJECTS
    parsed = not parse_errors
    parse_error = "; ".join(parse_errors)

    if not parsed or not is_single_subject:
        split = "unused"
    elif source_folder in DRONEFACE_UNUSED_SOURCE_FOLDERS:
        split = "unused"
    elif source_folder == "portraits" or image_type == "porF":
        split = "gallery"
    elif source_folder == "photos_all_faces" or image_type == "ef":
        split = "probe"
    else:
        split = "unused"

    return {
        "path": str(path),
        "filename": path.name,
        "dataset": "DroneFace",
        "source_folder": source_folder,
        "subject_id": subject_id,
        "camera_type": camera_type,
        "height_id": height_id,
        "height_m": height_m,
        "image_type": image_type,
        "distance_id": distance_id,
        "distance_m": distance_m,
        "is_single_subject": is_single_subject,
        "split": split,
        "parsed": parsed,
        "parse_error": parse_error,
    }


def build_droneface_index(root: Path, save: bool = True) -> pd.DataFrame:
    """Build the DroneFace-only protocol index for Prompt 1."""
    ensure_project_dirs()
    root = Path(root)
    paths = find_images(root)
    rows = [_parse_droneface_path(path, root) for path in paths]
    df = pd.DataFrame(rows, columns=DRONEFACE_COLUMNS)

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_path = INDEX_DIR / "droneface_index.csv"
        df.to_csv(index_path, index=False)
        print(f"Saved DroneFace index: {index_path}")

        unparsed = df[df["parsed"] == False] if not df.empty else df
        unparsed_path = INDEX_DIR / "droneface_unparsed.csv"
        if not unparsed.empty:
            unparsed.to_csv(unparsed_path, index=False)
            print(f"Saved DroneFace unparsed files: {unparsed_path}")
        elif unparsed_path.exists():
            unparsed_path.unlink()

    return df


def _read_rgb(path: Path):
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def plot_dataset_samples(paths, title: str, out_path: Path | None = None, max_images: int = 12):
    """Plot a generic grid of image samples and optionally save it."""
    sample_paths = list(paths)[:max_images]
    if not sample_paths:
        warnings.warn(f"No images available to plot for: {title}", stacklevel=2)
        return None

    n_images = len(sample_paths)
    n_cols = min(4, n_images)
    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, path in zip(axes_array, sample_paths):
        ax.axis("off")
        ax.set_title(Path(path).name, fontsize=8)
        image_rgb = _read_rgb(Path(path))
        if image_rgb is None:
            ax.text(0.5, 0.5, "Unreadable image", ha="center", va="center")
            continue
        ax.imshow(image_rgb)

    for ax in axes_array[n_images:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_path}")

    return fig


def _sample_by_folder(index_df: pd.DataFrame, folder: str, n: int) -> list[Path]:
    if index_df.empty or "source_folder" not in index_df.columns:
        return []
    subset = index_df[(index_df["source_folder"] == folder) & (index_df["parsed"] == True)]
    return [Path(path) for path in subset["path"].head(n).tolist()]


def plot_droneface_samples(index_df: pd.DataFrame, out_path: Path, max_images: int = 16):
    """Plot initial DroneFace examples from portraits, cropped faces, and raw frames."""
    if index_df.empty:
        warnings.warn("DroneFace index is empty; sample figure was not generated.", stacklevel=2)
        return None

    groups = [
        ("portraits", _sample_by_folder(index_df, "portraits", max(1, max_images // 3))),
        ("photos_all_faces", _sample_by_folder(index_df, "photos_all_faces", max(1, max_images // 2))),
        ("photos_all", _sample_by_folder(index_df, "photos_all", max(1, max_images // 6))),
    ]
    groups = [(label, paths) for label, paths in groups if paths]
    if not groups:
        warnings.warn("No DroneFace samples found for the expected source folders.", stacklevel=2)
        return None

    total_images = sum(len(paths) for _, paths in groups)
    n_cols = 4
    n_rows = math.ceil(total_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    ax_idx = 0
    for label, paths in groups:
        for path in paths:
            ax = axes_array[ax_idx]
            ax_idx += 1
            ax.axis("off")
            ax.set_title(f"{label}\n{path.name}", fontsize=8)
            image_rgb = _read_rgb(path)
            if image_rgb is None:
                ax.text(0.5, 0.5, "Unreadable image", ha="center", va="center")
                continue
            ax.imshow(image_rgb)

    for ax in axes_array[ax_idx:]:
        ax.axis("off")

    fig.suptitle("DroneFace - muestras iniciales")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    return fig
