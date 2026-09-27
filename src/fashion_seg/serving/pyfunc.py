"""MLflow pyfunc wrapper: base64 images in, JSON-friendly instances out.

Request (``POST /invocations`` on ``mlflow models serve``)::

    {"dataframe_records": [{"image": "<base64 jpeg/png>"}], "params": {"min_score": 0.8}}

Response: one entry per input image::

    {"height": 400, "width": 300, "instances": [
        {"class_id": 1, "label": "shirt, blouse", "score": 0.97,
         "box": [y1, x1, y2, x2], "mask_rle": "12 3 40 5 ..."}]}

Masks use the same RLE as the iMaterialist annotations (see ``fashion_seg_core.rle``),
relative to the image after EXIF orientation is applied.
"""

import base64
import io
from typing import Any, Protocol

import mlflow.pyfunc
import numpy as np
import pandas as pd  # MLflow's pyfunc interface: predict() receives a pandas DataFrame
from fashion_seg_core import rle
from fashion_seg_core.labels import load_class_names
from PIL import Image, ImageOps

from fashion_seg.legacy.matterport import Detections

IMAGE_COLUMN = "image"
DEFAULT_MIN_SCORE = 0.7


class Predictor(Protocol):
    """Anything turning an RGB image into detections (legacy TF model, torchvision, ...)."""

    def predict(self, image: np.ndarray) -> Detections:
        """image: [H, W, 3] uint8 RGB."""


def decode_image(payload: str | bytes) -> np.ndarray:
    """Base64 (or raw bytes) image -> [H, W, 3] uint8 RGB, EXIF orientation applied."""
    raw = payload if isinstance(payload, bytes) else base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def encode_image(image: np.ndarray, fmt: str = "PNG") -> str:
    """[H, W, 3] uint8 RGB -> base64 string, the inverse of ``decode_image``."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def format_detections(
    detections: Detections, class_names: list[str], image_shape: tuple[int, ...], min_score: float
) -> dict[str, Any]:
    """Build the JSON response for one image (see the module docstring for the format)."""
    instances = [
        {
            "class_id": int(class_id),
            "label": class_names[class_id],
            "score": round(float(score), 4),
            "box": [int(v) for v in box],
            "mask_rle": rle.encode(detections.masks[:, :, i]),
        }
        for i, (class_id, score, box) in enumerate(
            zip(detections.class_ids, detections.scores, detections.boxes, strict=True)
        )
        if score >= min_score
    ]
    return {"height": int(image_shape[0]), "width": int(image_shape[1]), "instances": instances}


class FashionSegmentationModel(mlflow.pyfunc.PythonModel):  # pylint: disable=abstract-method
    """Artifacts: ``saved_model`` (Matterport SavedModel dir) and ``labels`` (label json)."""

    def __init__(self, predictor: Predictor | None = None, class_names: list[str] | None = None):
        # Arguments allow injecting a fake predictor in tests; MLflow uses load_context.
        self._predictor = predictor
        self._class_names = class_names

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        # Lazy import: TensorFlow is only needed when the real model is loaded.
        # pylint: disable-next=import-outside-toplevel
        from fashion_seg.legacy.matterport import MatterportPredictor

        self._predictor = MatterportPredictor(context.artifacts["saved_model"])
        self._class_names = load_class_names(context.artifacts["labels"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """One result per row of ``model_input[IMAGE_COLUMN]`` (base64 images).

        Only column access is used, so a plain ``{"image": [...]}`` dict works too (tests).
        """
        min_score = float((params or {}).get("min_score", DEFAULT_MIN_SCORE))
        results = []
        for payload in model_input[IMAGE_COLUMN]:
            image = decode_image(payload)
            detections = self._predictor.predict(image)
            results.append(format_detections(detections, self._class_names, image.shape, min_score))
        return results
