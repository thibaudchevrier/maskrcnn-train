"""Tests of the YOLO predictor and of packaging a YOLO export."""

import json
import threading

import fashion_seg_yolo
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import torch
from fashion_seg_contract import schema
from fashion_seg_yolo import training
from fashion_seg_yolo.predictor import YoloPredictor, letterbox_content

from fashion_seg.adapters import mlflow_models, mlflow_tracking
from fashion_seg.config import PackagedModel, load_params
from fashion_seg.ports import Infrastructure, TrainingSession
from fashion_seg.service.packaging import package
from fashion_seg.serving.response import encode_image
from fashion_seg_testing import write_params


@pytest.fixture
def export_dir(inputs, config):
    """Export a model trained for one epoch on the tiny dataset, made confident.

    Barely trained, it scores everything near 0: raising its class biases makes it detect,
    so that the masks are exercised.
    """
    session = TrainingSession(lambda metrics, step=None: None, threading.Event())
    training.train(config, inputs, session)
    weights = inputs.export_dir / "model.pt"
    ckpt = torch.load(weights, weights_only=False)
    for branch in ckpt["model"].model[-1].cv3:
        branch[-1].bias.data.fill_(5.0)
    torch.save(ckpt, weights)
    return inputs.export_dir


def test_letterbox_content_follows_ultralytics():
    """The region matches Ultralytics' own: centred, rounded as ``scale_masks`` rounds."""
    assert letterbox_content((480, 640), (300, 500)) == (slice(48, 432), slice(0, 640))
    assert letterbox_content((64, 64), (64, 64)) == (slice(0, 64), slice(0, 64))


def test_large_images_give_full_size_detections(export_dir):
    """A photo larger than imgsz gives boxes and masks at its exact size, best first."""
    predictor = YoloPredictor(export_dir, device="cpu")
    predictor.config["imgsz"] = 64
    image = np.random.default_rng(0).integers(0, 255, (401, 299, 3), dtype=np.uint8)
    detections = predictor.predict(image)
    n = len(detections.class_ids)
    assert n > 0 and detections.masks.shape == (401, 299, n) and detections.boxes.shape == (n, 4)
    assert (detections.boxes[:, [0, 2]] <= 401).all() and (detections.boxes[:, [1, 3]] <= 299).all()
    assert (
        detections.masks.any()
        and ((detections.class_ids >= 1) & (detections.class_ids <= 46)).all()
    )
    assert (np.diff(detections.scores) <= 0).all()


def test_packaged_yolo_model_follows_the_contract(export_dir, tmp_path, monkeypatch):
    """A packaged model bundles its code, pins Ultralytics, and answers the contract."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"categories": [{"id": k, "name": f"c{k}"} for k in range(46)]}))
    data = {"train_csv": "", "train_images": "", "label_file": str(labels), "prepared_dir": ""}
    params = load_params(write_params(tmp_path, data, train={}))
    model = PackagedModel(family="yolo", source=export_dir, output_dir=tmp_path / "packaged")
    infra = Infrastructure(tracker=mlflow_tracking, repository=mlflow_models)

    provenance = package(fashion_seg_yolo, "yolo", model, params, infra)

    assert provenance["model_family"] == "yolo"
    requirements = (tmp_path / "packaged" / "requirements.txt").read_text()
    assert "ultralytics==" in requirements and "tensorflow" not in requirements
    bundled = {p.name for p in (tmp_path / "packaged" / "code").iterdir()}
    assert bundled == {"fashion_seg", "fashion_seg_torch", "fashion_seg_yolo"}
    served = mlflow.pyfunc.load_model(str(tmp_path / "packaged"))
    image = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    [prediction] = served.predict(pd.DataFrame({"image": [encode_image(image)]}))
    schema.validate(prediction)
    assert (prediction["height"], prediction["width"]) == (48, 64)
