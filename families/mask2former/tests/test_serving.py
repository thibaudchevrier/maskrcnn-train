"""Tests of the Mask2Former predictor and of packaging a Mask2Former export."""

import json

import fashion_seg_mask2former
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from fashion_seg_contract import schema
from fashion_seg_mask2former.config import Config
from fashion_seg_mask2former.predictor import Mask2FormerPredictor
from fashion_seg_mask2former.training import export_model

from fashion_seg.adapters import mlflow_models, mlflow_tracking
from fashion_seg.config import PackagedModel, load_params
from fashion_seg.ports import Infrastructure
from fashion_seg.service.packaging import package
from fashion_seg.serving.response import encode_image
from fashion_seg_testing import write_params


@pytest.fixture
def export_dir(tmp_path, tiny_model):
    """Export the tiny random model the way the trainer does."""
    config = Config(
        output_dir=tmp_path,
        init_weights=tmp_path / "missing",
        epochs=1,
        max_size=96,
        batch_size=1,
        learning_rate=0.0001,
        warmup_steps=1,
        num_workers=0,
    )
    export_model(tiny_model, config, tmp_path / "export")
    return tmp_path / "export"


def test_large_images_give_full_size_detections(export_dir):
    """A photo larger than max_size gives boxes and masks at its exact size, best first."""
    image = np.random.default_rng(0).integers(0, 255, (401, 299, 3), dtype=np.uint8)
    detections = Mask2FormerPredictor(export_dir, device="cpu").predict(image)
    n = len(detections.class_ids)
    assert n > 0 and detections.masks.shape == (401, 299, n)
    assert (detections.boxes[:, [0, 2]] <= 401).all() and (detections.boxes[:, [1, 3]] <= 299).all()
    assert ((detections.class_ids >= 1) & (detections.class_ids <= 46)).all()
    assert (np.diff(detections.scores) <= 0).all()


def test_packaged_mask2former_model_follows_the_contract(export_dir, tmp_path, monkeypatch):
    """A packaged model bundles its code, pins transformers, and answers the contract."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"categories": [{"id": k, "name": f"c{k}"} for k in range(46)]}))
    data = {"train_csv": "", "train_images": "", "label_file": str(labels), "prepared_dir": ""}
    params = load_params(write_params(tmp_path, data, train={}))
    model = PackagedModel(family="mask2former", source=export_dir, output_dir=tmp_path / "packaged")
    infra = Infrastructure(tracker=mlflow_tracking, repository=mlflow_models)

    provenance = package(fashion_seg_mask2former, "m2f", model, params, infra)

    assert provenance["model_family"] == "mask2former"
    requirements = (tmp_path / "packaged" / "requirements.txt").read_text()
    assert "transformers==" in requirements and "tensorflow" not in requirements
    bundled = {p.name for p in (tmp_path / "packaged" / "code").iterdir()}
    assert bundled == {"fashion_seg", "fashion_seg_torch", "fashion_seg_mask2former"}
    served = mlflow.pyfunc.load_model(str(tmp_path / "packaged"))
    image = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    [prediction] = served.predict(pd.DataFrame({"image": [encode_image(image)]}))
    schema.validate(prediction)
    assert (prediction["height"], prediction["width"]) == (48, 64)
