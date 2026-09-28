"""Tests of the MLflow serving wrapper, with a fake predictor (real models: each family's tests)."""

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fashion_seg_contract import rle, schema
from PIL import Image

from fashion_seg.ports import Detections
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
    """Build a request with one image, as MLflow passes it (column access only)."""
    return {"image": [encode_image(image)]}


def test_predict_formats_instances_and_filters_by_score():
    """Responses follow the contract, carry labels and decoded masks, and honour min_score."""
    model = FashionSegmentationModel(predictor=FakePredictor(), class_names=CLASS_NAMES)
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


def test_load_context_uses_the_injected_loader(tmp_path):
    """load_context builds the predictor with the family's loader, on the model artifact."""
    labels = tmp_path / "labels.json"
    labels.write_text('{"categories": [{"id": 0, "name": "shirt"}, {"id": 1, "name": "pants"}]}')
    loaded = []

    def load_predictor(model_dir: Path) -> FakePredictor:
        loaded.append(model_dir)
        return FakePredictor()

    model = FashionSegmentationModel(load_predictor=load_predictor)
    context = SimpleNamespace(artifacts={"model": str(tmp_path / "export"), "labels": str(labels)})
    model.load_context(context)
    [result] = model.predict(None, _request(np.zeros((20, 10, 3), np.uint8)))
    assert loaded == [tmp_path / "export"]
    assert [i["label"] for i in result["instances"]] == ["shirt", "pants"]


def test_load_context_without_loader_fails():
    """A model packaged without a predictor loader fails loudly instead of predicting nothing."""
    with pytest.raises(RuntimeError):
        FashionSegmentationModel().load_context(SimpleNamespace(artifacts={}))
