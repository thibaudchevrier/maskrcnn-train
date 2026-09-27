"""Tests of the MLflow serving wrapper, with a fake predictor and, if pulled, the 2021 model."""

import base64
import io
import os
from pathlib import Path

import numpy as np
import pytest
from fashion_seg_contract import rle, schema
from PIL import Image

from fashion_seg.predictors.base import Detections
from fashion_seg.serving.pyfunc import FashionSegmentationModel, decode_image, encode_image

CLASS_NAMES = ["BG", "shirt", "pants"]


class FakePredictor:
    """Predictor returning two fixed detections, sized to the image."""

    def predict(self, image):
        """Return a shirt (top-left) and pants (bottom-right) with fixed scores."""
        h, w = image.shape[:2]
        masks = np.zeros((h, w, 2), dtype=bool)
        masks[:5, :5, 0] = True
        masks[5:, 5:, 1] = True
        return Detections(
            boxes=np.array([[0, 0, 5, 5], [5, 5, h, w]], np.int32),
            class_ids=np.array([1, 2], np.int32),
            scores=np.array([0.95, 0.75], np.float32),
            masks=masks,
        )


def _request(image: np.ndarray) -> dict[str, list[str]]:
    # Same column access as the pandas DataFrame MLflow passes when serving.
    """Build a request with one image, as MLflow passes it (column access only)."""
    return {"image": [encode_image(image)]}


def test_predict_formats_instances_and_filters_by_score():
    """Responses follow the contract, carry labels and decoded masks, and honour min_score."""
    model = FashionSegmentationModel(FakePredictor(), CLASS_NAMES)
    image = np.zeros((20, 10, 3), np.uint8)

    [default] = model.predict(None, _request(image))
    [strict] = model.predict(None, _request(image), params={"min_score": 0.9})

    schema.validate(default)
    assert (default["height"], default["width"]) == (20, 10)
    assert [i["label"] for i in default["instances"]] == ["shirt", "pants"]
    assert [i["label"] for i in strict["instances"]] == ["shirt"]
    mask = rle.decode(default["instances"][0]["mask_rle"], 20, 10)
    assert mask[:5, :5].all() and mask.sum() == 25


def test_decode_image_handles_grayscale_and_raw_bytes():
    """Grayscale images become RGB; raw bytes and base64 are both accepted."""
    buffer = io.BytesIO()
    Image.new("L", (8, 6)).save(buffer, format="JPEG")
    assert decode_image(buffer.getvalue()).shape == (6, 8, 3)
    assert decode_image(base64.b64encode(buffer.getvalue()).decode()).shape == (6, 8, 3)


SAVED_MODEL = Path(os.environ.get("FASHION_SEG_SAVED_MODEL", "deployement"))


@pytest.mark.skipif(not (SAVED_MODEL / "saved_model.pb").exists(), reason="legacy model not pulled")
def test_real_saved_model_runs():
    """The 2021 model (deployement/) runs end to end and follows the contract."""
    # pylint: disable-next=import-outside-toplevel  # TensorFlow loads only when the model is pulled
    from fashion_seg.predictors.matterport import MatterportPredictor

    model = FashionSegmentationModel(MatterportPredictor(SAVED_MODEL), ["BG"] + ["c"] * 46)
    image = np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    [result] = model.predict(None, _request(image), params={"min_score": 0.0})
    schema.validate(result)
    assert (result["height"], result["width"]) == (300, 200)
    for inst in result["instances"]:
        assert 1 <= inst["class_id"] <= 46
