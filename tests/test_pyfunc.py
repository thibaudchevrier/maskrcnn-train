import base64
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion_seg import rle
from fashion_seg.legacy.matterport import Detections
from fashion_seg.serving.pyfunc import FashionSegmentationModel, decode_image, encode_image

CLASS_NAMES = ["BG", "shirt", "pants"]


class FakePredictor:
    def predict(self, image):
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


def _request(image: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"image": [encode_image(image)]})


def test_predict_formats_instances_and_filters_by_score():
    model = FashionSegmentationModel(FakePredictor(), CLASS_NAMES)
    image = np.zeros((20, 10, 3), np.uint8)

    [default] = model.predict(None, _request(image))
    [strict] = model.predict(None, _request(image), params={"min_score": 0.9})

    assert (default["height"], default["width"]) == (20, 10)
    assert [i["label"] for i in default["instances"]] == ["shirt", "pants"]
    assert [i["label"] for i in strict["instances"]] == ["shirt"]
    mask = rle.decode(default["instances"][0]["mask_rle"], 20, 10)
    assert mask[:5, :5].all() and mask.sum() == 25


def test_decode_image_handles_grayscale_and_raw_bytes():
    buffer = io.BytesIO()
    Image.new("L", (8, 6)).save(buffer, format="JPEG")
    assert decode_image(buffer.getvalue()).shape == (6, 8, 3)
    assert decode_image(base64.b64encode(buffer.getvalue()).decode()).shape == (6, 8, 3)


SAVED_MODEL = Path(os.environ.get("FASHION_SEG_SAVED_MODEL", "deployement"))


@pytest.mark.skipif(not (SAVED_MODEL / "saved_model.pb").exists(), reason="legacy model not pulled")
def test_real_saved_model_runs():
    from fashion_seg.legacy.matterport import MatterportPredictor

    model = FashionSegmentationModel(MatterportPredictor(SAVED_MODEL), ["BG"] + ["c"] * 46)
    image = np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    [result] = model.predict(None, _request(image), params={"min_score": 0.0})
    assert (result["height"], result["width"]) == (300, 200)
    for inst in result["instances"]:
        assert 1 <= inst["class_id"] <= 46
