"""Tests of the torchvision predictor and of packaging a torchvision model through the contract."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from fashion_seg_contract import schema

from fashion_seg.package import package
from fashion_seg.predictors.torchvision import TorchvisionPredictor
from fashion_seg.serving.pyfunc import encode_image
from fashion_seg_torchvision.model import build_model


@pytest.fixture
def export_dir(tmp_path):
    """Export a small random 3-class model the way the trainer does."""
    model_dir = tmp_path / "export"
    model_dir.mkdir()
    config = {
        "architecture": "maskrcnn_resnet50_fpn_v2",
        "num_classes": 3,
        "min_size": 64,
        "max_size": 96,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(0)
    torch.save(build_model(3, 64, 96).state_dict(), model_dir / "model.pt")
    return model_dir


def test_predictor_returns_detections_in_image_coordinates(export_dir):
    """Boxes, masks, scores and class ids are consistent with the input image."""
    image = np.random.default_rng(0).integers(0, 255, (50, 40, 3), dtype=np.uint8)
    detections = TorchvisionPredictor(export_dir, device="cpu").predict(image)
    n = len(detections.class_ids)
    assert detections.boxes.shape == (n, 4) and detections.scores.shape == (n,)
    assert detections.masks.shape == (50, 40, n) and detections.masks.dtype == bool
    assert (detections.boxes[:, [0, 2]] <= 50).all() and (detections.boxes[:, [1, 3]] <= 40).all()


def test_packaged_torchvision_model_follows_the_contract(export_dir, tmp_path, monkeypatch):
    """A packaged torchvision model loads its own predictor and answers contract predictions."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"categories": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]}))
    params = {
        "family": "torchvision",
        "model_dir": str(export_dir),
        "label_file": str(labels),
        "output_dir": str(tmp_path / "packaged"),
        "experiment": "test",
        "registered_name": "test-model",
    }

    provenance = package("torchvision_model", params)

    assert provenance["model_family"] == "torchvision"
    requirements = (tmp_path / "packaged" / "requirements.txt").read_text()
    assert "torchvision==" in requirements and "tensorflow" not in requirements
    import mlflow.pyfunc  # pylint: disable=import-outside-toplevel  # after the tracking URI is set

    model = mlflow.pyfunc.load_model(str(tmp_path / "packaged"))
    image = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    [prediction] = model.predict(pd.DataFrame({"image": [encode_image(image)]}))
    schema.validate(prediction)
    assert (prediction["height"], prediction["width"]) == (48, 64)
