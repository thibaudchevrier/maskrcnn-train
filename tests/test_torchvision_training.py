"""End-to-end check of the torchvision family through the CLI, on synthetic data (CPU, no DVC)."""

import json

import polars as pl
import pytest

from fashion_seg.__main__ import main
from fashion_seg.families.torchvision.dataset import FashionDataset
from tests.synthetic import write_dataset, write_params


def test_dataset_downscales_and_maps_classes(tmp_path):
    """Images are downscaled to max_side, masks follow, category k becomes class k + 1."""
    data = write_dataset(tmp_path, n_images=5, height=120, width=160)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    image, targets = FashionDataset(records, data["train_images"], 80)[0]
    assert tuple(image.shape) == (3, 60, 80)
    assert tuple(targets["masks"].shape) == (1, 60, 80)
    assert targets["labels"].tolist() == [6]
    assert targets["boxes"].tolist() == [[16.0, 10.0, 53.0, 40.0]]


@pytest.fixture
def params_file(tmp_path, monkeypatch):
    """Write a params.yaml for a CPU smoke run and point MLflow at a temporary store."""
    data = write_dataset(tmp_path, n_images=5, height=120, width=160)
    train = {
        "torchvision": {
            "output_dir": str(tmp_path / "out" / "torchvision"),
            "init_weights": str(tmp_path / "missing.pth"),
            "epochs": 1,
            "min_size": 800,
            "max_size": 1024,
            "batch_size": 2,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "warmup_steps": 10,
            "num_workers": 2,
            "log_every": 50,
            "device": "cpu",
        }
    }
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    return write_params(tmp_path, data, train)


def test_smoke_training_exports_and_resumes(params_file, caplog):
    """A smoke run exports config.json + model.pt; a second run resumes from the checkpoint."""
    argv = ["--params", str(params_file), "train", "torchvision", "--smoke"]
    main(argv)
    out = params_file.parent / "out" / "torchvision-smoke"
    config = json.loads((out / "model" / "config.json").read_text())
    assert config["num_classes"] == 47 and config["architecture"] == "maskrcnn_resnet50_fpn_v2"
    assert (out / "model" / "model.pt").exists() and (out / "checkpoints" / "last.pt").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())

    caplog.set_level("INFO")
    main(argv)
    assert "Resuming after epoch 1" in caplog.text
