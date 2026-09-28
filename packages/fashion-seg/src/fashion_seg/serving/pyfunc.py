"""MLflow pyfunc wrapper: base64 images in, JSON-friendly instances out.

Request (``POST /invocations`` on ``mlflow models serve``), see ``fashion_seg_contract.request``::

    {"dataframe_records": [{"image": "<base64 jpeg/png>"}], "params": {"min_score": 0.8}}

Response: one entry per input image::

    {"height": 400, "width": 300, "instances": [
        {"class_id": 1, "label": "shirt, blouse", "score": 0.97,
         "box": [y1, x1, y2, x2], "mask_rle": "12 3 40 5 ..."}]}

Masks use the same RLE as the iMaterialist annotations (see ``fashion_seg_contract.rle``),
relative to the image after EXIF orientation is applied.
"""

import base64
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd  # MLflow's pyfunc interface: predict() receives a pandas DataFrame
from fashion_seg_contract import request, rle
from fashion_seg_contract.labels import load_class_names
from fashion_seg_contract.schema import Instance, Prediction
from PIL import Image, ImageOps

from fashion_seg.ports import Detections, Predictor


def decode_image(payload: str | bytes) -> np.ndarray:
    """Decode an image, applying its EXIF orientation.

    Parameters
    ----------
    payload : str | bytes
        Base64 string (as sent in requests) or raw encoded bytes.

    Returns
    -------
    np.ndarray
        RGB image, shape ``(height, width, 3)``, uint8.
    """
    raw = payload if isinstance(payload, bytes) else base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def encode_image(image: np.ndarray, fmt: str = "PNG") -> str:
    """Encode an image as base64, the inverse of ``decode_image``.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape ``(height, width, 3)``, uint8.
    fmt : str
        Pillow image format. By default ``"PNG"``.

    Returns
    -------
    str
        The encoded image, base64 (ASCII).
    """
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=fmt)
    return request.encode_image(buffer.getvalue())


def format_detections(
    detections: Detections, class_names: list[str], image_shape: tuple[int, ...], min_score: float
) -> Prediction:
    """Build the response for one image, following the contract.

    Parameters
    ----------
    detections : Detections
        The predictor's output for the image.
    class_names : list[str]
        Class names indexed by model class id.
    image_shape : tuple[int, ...]
        Shape of the input image, ``(height, width, ...)``.
    min_score : float
        Detections below this confidence are dropped.

    Returns
    -------
    Prediction
        The response for the image (see ``fashion_seg_contract.schema``).
    """
    instances: list[Instance] = [
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


# pylint: disable-next=abstract-method  # predict_stream is optional: this model doesn't stream
class FashionSegmentationModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc model: base64 images in, contract predictions out.

    Packaging injects the family's ``load_predictor``; it is pickled with the model (by
    reference: the family's code is bundled in the model), and MLflow calls ``load_context`` to
    load the export (``model`` artifact) and the class names (``labels`` artifact). Tests inject a
    ready predictor instead.

    Parameters
    ----------
    load_predictor : Callable[[Path], Predictor] | None
        The family's ``load_predictor``, called on the export directory. By default ``None``.
    predictor : Predictor | None
        Predictor to use instead of loading the artifacts. By default ``None``.
    class_names : list[str] | None
        Class names to use instead of loading the artifacts. By default ``None``.
    """

    def __init__(
        self,
        load_predictor: Callable[[Path], Predictor] | None = None,
        predictor: Predictor | None = None,
        class_names: list[str] | None = None,
    ):
        self._load_predictor = load_predictor
        self._predictor = predictor
        self._class_names = class_names

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the predictor and the class names from the model's artifacts.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            Gives the local paths of the ``model`` and ``labels`` artifacts.

        Raises
        ------
        RuntimeError
            If the model was packaged without a ``load_predictor``.
        """
        if self._load_predictor is None:
            raise RuntimeError("Packaged without load_predictor: package it with fashion_seg.cli")
        self._predictor = self._load_predictor(Path(context.artifacts["model"]))
        self._class_names = load_class_names(context.artifacts["labels"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Segment the garments in every image of a request.

        Only column access is used, so a plain ``{"image": [...]}`` dict works too (tests).

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext | None
            Unused (the artifacts are loaded in ``load_context``).
        model_input : pd.DataFrame
            One base64 image per row, in the ``image`` column.
        params : dict[str, Any] | None
            ``min_score`` (float, default 0.7) drops low-confidence detections. By default ``None``.

        Returns
        -------
        list[Prediction]
            One response per input row (see ``fashion_seg_contract.schema``).
        """
        min_score = float((params or {}).get(request.MIN_SCORE_PARAM, request.DEFAULT_MIN_SCORE))
        results = []
        for payload in model_input[request.IMAGE_FIELD]:
            image = decode_image(payload)
            detections = self._predictor.predict(image)
            results.append(format_detections(detections, self._class_names, image.shape, min_score))
        return results
