"""The model's response, built from a predictor's detections: pure functions, no MLflow.

Response: one entry per input image (see ``fashion_seg_contract.schema``)::

    {"height": 400, "width": 300, "instances": [
        {"class_id": 1, "label": "shirt, blouse", "score": 0.97,
         "box": [y1, x1, y2, x2], "mask_rle": "12 3 40 5 ..."}]}

Masks use the same RLE as the iMaterialist annotations (see ``fashion_seg_contract.rle``),
relative to the image after EXIF orientation is applied.
"""

import base64
import io

import numpy as np
from fashion_seg_contract import request, rle
from fashion_seg_contract.schema import Instance, Prediction
from PIL import Image, ImageOps

from fashion_seg.ports import Detections


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
