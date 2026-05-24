from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    ort = None
    _ONNXRUNTIME_IMPORT_ERROR = exc
else:
    _ONNXRUNTIME_IMPORT_ERROR = None

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:
    FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = exc
else:
    _INSIGHTFACE_IMPORT_ERROR = None


def _l2_normalize(embedding) -> np.ndarray | None:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _bbox_area(face) -> float:
    bbox = np.asarray(getattr(face, "bbox", [0, 0, 0, 0]), dtype=np.float32)
    if bbox.size < 4:
        return 0.0
    width = max(0.0, float(bbox[2] - bbox[0]))
    height = max(0.0, float(bbox[3] - bbox[1]))
    return width * height


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=_bbox_area)


def _face_to_embedding_record(face, face_index: int) -> dict:
    bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
    embedding = getattr(face, "embedding", None)
    return {
        "face_index": face_index,
        "embedding": _l2_normalize(embedding),
        "bbox": bbox.tolist() if bbox.size else None,
        "detection_score": float(getattr(face, "det_score", np.nan)),
        "bbox_area": _bbox_area(face),
    }


def _maybe_upscale(image: np.ndarray, min_side_trigger: int = 160, target_min_side: int = 320):
    height, width = image.shape[:2]
    min_side = min(height, width)
    if min_side >= min_side_trigger:
        return image, False
    scale = target_min_side / max(1, min_side)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized, True


class FaceRecognizerArcFace:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list | None = None,
        det_size: tuple[int, int] = (320, 320),
        allow_cuda: bool = False,
    ):
        if FaceAnalysis is None:
            raise ImportError(f"InsightFace could not be imported: {_INSIGHTFACE_IMPORT_ERROR}")

        if providers is None:
            providers = ["CPUExecutionProvider"]
            if allow_cuda:
                if ort is None:
                    warnings.warn(
                        f"onnxruntime could not be imported, using CPU: {_ONNXRUNTIME_IMPORT_ERROR}",
                        stacklevel=2,
                    )
                elif "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    warnings.warn("CUDAExecutionProvider is not available, using CPU.", stacklevel=2)

        self.model_name = model_name
        self.providers = providers
        self.det_size = det_size
        self.using_cuda = "CUDAExecutionProvider" in providers

        try:
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0 if self.using_cuda else -1, det_size=det_size)
        except Exception as exc:
            raise RuntimeError(f"InsightFace initialization failed: {exc}") from exc

    def load_image(self, path):
        path = Path(path)
        image = None
        path_text = str(path)
        if path_text.isascii():
            image = cv2.imread(path_text)
        if image is None and not path_text.isascii():
            try:
                data = np.fromfile(path_text, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            except OSError:
                image = None
        if image is None and not path_text.isascii():
            image = cv2.imread(path_text)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return image

    def detect_faces(self, path_or_image):
        if isinstance(path_or_image, (str, Path)):
            image = self.load_image(path_or_image)
        else:
            image = path_or_image
        return self.app.get(image)

    def extract_single_face_embedding(self, path, allow_upscale: bool = True) -> dict:
        result = {
            "path": str(path),
            "detected": False,
            "embedding": None,
            "bbox": None,
            "detection_score": np.nan,
            "num_faces_detected": 0,
            "upscaled_for_detection": False,
            "image_shape_original": None,
            "image_shape_used": None,
            "error": "",
        }

        try:
            image = self.load_image(path)
        except Exception as exc:
            result["error"] = str(exc)
            return result

        result["image_shape_original"] = tuple(int(v) for v in image.shape)
        image_used = image
        if allow_upscale:
            image_used, upscaled = _maybe_upscale(image)
            result["upscaled_for_detection"] = upscaled
        result["image_shape_used"] = tuple(int(v) for v in image_used.shape)

        try:
            faces = self.detect_faces(image_used)
        except Exception as exc:
            result["error"] = f"InsightFace detection failed: {exc}"
            return result

        result["num_faces_detected"] = len(faces)
        if not faces:
            return result

        face = _largest_face(faces)
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

    def extract_all_face_embeddings(self, path, allow_upscale: bool = True) -> dict:
        result = {
            "path": str(path),
            "detected": False,
            "detections": [],
            "num_faces_detected": 0,
            "upscaled_for_detection": False,
            "image_shape_original": None,
            "image_shape_used": None,
            "error": "",
        }

        try:
            image = self.load_image(path)
        except Exception as exc:
            result["error"] = str(exc)
            return result

        result["image_shape_original"] = tuple(int(v) for v in image.shape)
        image_used = image
        if allow_upscale:
            image_used, upscaled = _maybe_upscale(image)
            result["upscaled_for_detection"] = upscaled
        result["image_shape_used"] = tuple(int(v) for v in image_used.shape)

        try:
            faces = self.detect_faces(image_used)
        except Exception as exc:
            result["error"] = f"InsightFace detection failed: {exc}"
            return result

        faces = sorted(faces, key=_bbox_area, reverse=True)
        result["num_faces_detected"] = len(faces)
        result["detected"] = bool(faces)
        result["detections"] = [_face_to_embedding_record(face, idx) for idx, face in enumerate(faces)]
        return result

    @staticmethod
    def cosine_similarity(embedding_a, embedding_b) -> float:
        if embedding_a is None or embedding_b is None:
            return float("nan")
        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return float("nan")
        return float(np.dot(a, b) / (norm_a * norm_b))
