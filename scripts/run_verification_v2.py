from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2_verification import (
    V2_RESULTS_DIR,
    build_photos_all_bbox_verification_scores,
    build_photos_all_faces_verification_scores,
    load_gallery_embeddings,
    run_saved_verification_analysis,
    verification_score_paths,
)


FACE_DET_SIZE = (320, 320)
FULL_IMAGE_DET_SIZE = (1280, 1280)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DroneFace V2 verification/AUC scoring.")
    parser.add_argument("--gallery-name", default="trio_front", help="Gallery to evaluate.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute score CSVs even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gallery_name = args.gallery_name
    paths = verification_score_paths(gallery_name)
    needs_scoring = args.overwrite or not paths.faces_scores.exists() or not paths.bbox_scores.exists()
    gallery_embeddings = load_gallery_embeddings(gallery_name) if needs_scoring else None

    if args.overwrite or not paths.faces_scores.exists():
        from src.face_model import FaceRecognizerArcFace

        if gallery_embeddings is None:
            gallery_embeddings = load_gallery_embeddings(gallery_name)
        faces_results_path = V2_RESULTS_DIR / f"photos_all_faces_{gallery_name}.csv"
        faces_results_df = pd.read_csv(faces_results_path)
        model_faces = FaceRecognizerArcFace(model_name="buffalo_l", det_size=FACE_DET_SIZE, allow_cuda=False)
        build_photos_all_faces_verification_scores(
            faces_results_df,
            model_faces,
            gallery_embeddings,
            gallery_name=gallery_name,
            save_path=paths.faces_scores,
        )
        print(f"Saved faces verification scores: {paths.faces_scores}")
    else:
        print(f"Using existing faces verification scores: {paths.faces_scores}")

    if args.overwrite or not paths.bbox_scores.exists():
        from src.face_model import FaceRecognizerArcFace

        if gallery_embeddings is None:
            gallery_embeddings = load_gallery_embeddings(gallery_name)
        labeled_bbox_path = V2_RESULTS_DIR / "photos_all_detections_labeled_all_galleries.csv"
        labeled_bbox_df = pd.read_csv(labeled_bbox_path)
        model_full = FaceRecognizerArcFace(model_name="buffalo_l", det_size=FULL_IMAGE_DET_SIZE, allow_cuda=False)
        build_photos_all_bbox_verification_scores(
            labeled_bbox_df,
            model_full,
            gallery_embeddings,
            gallery_name=gallery_name,
            save_path=paths.bbox_scores,
        )
        print(f"Saved bbox verification scores: {paths.bbox_scores}")
    else:
        print(f"Using existing bbox verification scores: {paths.bbox_scores}")

    analysis = run_saved_verification_analysis(gallery_name)
    print(f"Selected threshold from photos_all_faces: {analysis['threshold']:.6f}")
    print("AUC summary:")
    print(analysis["auc_summary"].to_string(index=False))
    print("Threshold summary:")
    print(analysis["threshold_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
