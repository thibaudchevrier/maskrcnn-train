"""Serving tests of the Matterport family, in families/matterport/serve (TensorFlow 2.18+)."""

import os
from pathlib import Path

import numpy as np
import pytest
from fashion_seg_contract import schema

import fashion_seg_matterport
from fashion_seg.ports import ModelFamily
from fashion_seg.serving.pyfunc import FashionSegmentationModel
from fashion_seg.serving.response import encode_image

# The 2021 model, pulled with `uv run dvc pull deployement.dvc` at the repository root.
SAVED_MODEL = Path(
    os.environ.get("FASHION_SEG_SAVED_MODEL", Path(__file__).parents[4] / "deployement")
)


def test_family_implements_the_port():
    """The package is a ModelFamily whose smoke overrides are valid parameters."""
    assert isinstance(fashion_seg_matterport, ModelFamily)
    assert fashion_seg_matterport.SPEC.name == "matterport"
    config = fashion_seg_matterport.Config
    assert set(config.smoke_overrides) <= set(config.model_fields)


@pytest.mark.skipif(not (SAVED_MODEL / "saved_model.pb").exists(), reason="2021 model not pulled")
def test_real_saved_model_follows_the_contract():
    """The 2021 model runs through the family's predictor loader and follows the contract."""
    model = FashionSegmentationModel(
        predictor=fashion_seg_matterport.load_predictor(SAVED_MODEL),
        class_names=["BG"] + ["c"] * 46,
    )
    image = np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    [result] = model.predict(None, {"image": [encode_image(image)]}, params={"min_score": 0.0})
    schema.validate(result)
    assert (result["height"], result["width"]) == (300, 200)
    for inst in result["instances"]:
        assert 1 <= inst["class_id"] <= 46
