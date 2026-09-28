"""Tests of the torchvision predictor and of packaging a torchvision export through the contract."""

import json

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import torch
from fashion_seg_contract import schema

import fashion_seg_torchvision
from fashion_seg.adapters import mlflow_models, mlflow_tracking
from fashion_seg.config import PackagedModel, load_params
from fashion_seg.ports import Infrastructure
from fashion_seg.service.packaging import package
from fashion_seg.serving.response import encode_image
from fashion_seg_testing import write_params
from fashion_seg_torchvision.network import build_model
from fashion_seg_torchvision.predictor import TorchvisionPredictor


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
    data = {"train_csv": "", "train_images": "", "label_file": str(labels), "prepared_dir": ""}
    params = load_params(write_params(tmp_path, data, train={}))
    model = PackagedModel(family="torchvision", source=export_dir, output_dir=tmp_path / "packaged")

    infra = Infrastructure(tracker=mlflow_tracking, repository=mlflow_models)
    provenance = package(fashion_seg_torchvision, "tv", model, params, infra)

    assert provenance["model_family"] == "torchvision"
    requirements = (tmp_path / "packaged" / "requirements.txt").read_text()
    assert "torchvision==" in requirements and "tensorflow" not in requirements
    assert "mlflow==" in requirements and "fashion-seg-contract @ https:" in requirements
    bundled = {p.name for p in (tmp_path / "packaged" / "code").iterdir()}
    assert bundled == {"fashion_seg", "fashion_seg_torchvision"}
    served = mlflow.pyfunc.load_model(str(tmp_path / "packaged"))
    image = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    [prediction] = served.predict(pd.DataFrame({"image": [encode_image(image)]}))
    schema.validate(prediction)
    assert (prediction["height"], prediction["width"]) == (48, 64)
