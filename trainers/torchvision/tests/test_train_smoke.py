"""End-to-end check of the torchvision trainer on a synthetic prepared dataset (CPU, no DVC)."""

import json

import numpy as np
import polars as pl
import pytest
import yaml
from fashion_seg_contract import rle
from PIL import Image

from fashion_seg_torchvision import train
from fashion_seg_torchvision.data import FashionDataset


def _write_dataset(root):
    """Write five synthetic images, their prepared annotations, a split and labels."""
    image_dir = root / "images"
    image_dir.mkdir()
    rows = []
    for i in range(5):
        image = np.full((120, 160, 3), 70, np.uint8)
        image[20:80, 30:110] = 210
        Image.fromarray(image).save(image_dir / f"img{i}.jpg")
        mask = np.zeros((120, 160), bool)
        mask[20:80, 30:110] = True
        rows.append(
            {
                "image_id": f"img{i}",
                "height": 120,
                "width": 160,
                "class_ids": [5],
                "rles": [rle.encode(mask)],
            }
        )
    prepared = root / "prepared"
    prepared.mkdir()
    pl.DataFrame(rows).write_parquet(prepared / "annotations.parquet")
    (prepared / "split.json").write_text(
        json.dumps({"train": ["img0", "img1", "img2", "img3"], "val": ["img4"]})
    )
    labels = root / "labels.json"
    labels.write_text(json.dumps({"categories": [{"id": k, "name": f"c{k}"} for k in range(46)]}))
    return image_dir, prepared, labels


def test_dataset_downscales_and_maps_classes(tmp_path):
    """Images are downscaled to max_side, masks follow, category k becomes class k + 1."""
    image_dir, prepared, _ = _write_dataset(tmp_path)
    dataset = FashionDataset(pl.read_parquet(prepared / "annotations.parquet"), image_dir, 80)
    image, targets = dataset[0]
    assert tuple(image.shape) == (3, 60, 80)
    assert tuple(targets["masks"].shape) == (1, 60, 80)
    assert targets["labels"].tolist() == [6]
    assert targets["boxes"].tolist() == [[15.0, 10.0, 55.0, 40.0]]


@pytest.fixture
def params_file(tmp_path, monkeypatch):
    """Write a params.yaml for a CPU smoke run and point MLflow at a temporary store."""
    image_dir, prepared, labels = _write_dataset(tmp_path)
    params = {
        "data": {
            "train_images": str(image_dir),
            "prepared_dir": str(prepared),
            "label_file": str(labels),
        },
        "train_torchvision": {
            "experiment": "test",
            "output_dir": str(tmp_path / "out" / "torchvision"),
            "init_weights": str(tmp_path / "missing.pth"),
            "min_size": 800,
            "max_size": 1024,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "warmup_steps": 10,
            "num_workers": 2,
            "log_every": 50,
            "device": "cpu",
            "resume": True,
            "max_train_images": None,
            "max_val_images": None,
        },
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_smoke_training_exports_and_resumes(params_file, caplog):
    """A smoke run exports config.json + model.pt; a second run resumes from the checkpoint."""
    train.main(["--params", "params.yaml", "--smoke"])
    out = params_file / "out" / "torchvision-smoke"
    config = json.loads((out / "model" / "config.json").read_text())
    assert config["num_classes"] == 47 and config["architecture"] == "maskrcnn_resnet50_fpn_v2"
    assert (out / "model" / "model.pt").exists() and (out / "checkpoints" / "last.pt").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())

    caplog.set_level("INFO")
    train.main(["--params", "params.yaml", "--smoke"])
    assert "Resuming after epoch 1" in caplog.text
