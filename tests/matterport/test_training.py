"""End-to-end check of the Matterport family through the CLI, on synthetic data (no DVC data)."""

import json

import polars as pl

from fashion_seg.__main__ import main
from fashion_seg.families.matterport.dataset import FashionDataset
from tests.synthetic import N_CATEGORIES, write_dataset, write_params


def test_dataset_maps_categories_to_model_classes(tmp_path):
    """Dataset category k becomes model class k + 1, with the decoded mask."""
    data = write_dataset(tmp_path, n_images=4, height=96, width=128)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    names = ["BG"] + [f"c{k}" for k in range(N_CATEGORIES)]
    masks, class_ids = FashionDataset(records, data["train_images"], names).load_mask(0)
    assert masks.shape == (96, 128, 1) and masks.sum() == (64 - 16) * (85 - 25)
    assert class_ids.tolist() == [6]  # category 5 -> model class 6


def test_smoke_training_exports_servable_model(tmp_path, monkeypatch):
    """A smoke run trains, exports config.json + SavedModel (training anchors) and metrics."""
    data = write_dataset(tmp_path, n_images=4, height=96, width=128)
    train = {
        "matterport": {
            "output_dir": str(tmp_path / "out" / "matterport"),
            "init_weights": str(tmp_path / "missing.h5"),
            "epochs": 2,
            "backbone": "resnet101",
            "image_min_dim": 800,
            "image_max_dim": 1024,
            "images_per_gpu": 2,
            "rpn_anchor_scales": [16, 32, 64, 128, 256],
            "train_rois_per_image": 32,
            "learning_rate": 0.001,
            "layers": "all",
            "detection_min_confidence": 0.7,
        }
    }
    params_file = write_params(tmp_path, data, train)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)

    main(["--params", str(params_file), "train", "matterport", "--smoke"])

    out = tmp_path / "out" / "matterport-smoke"
    config = json.loads((out / "model" / "config.json").read_text())
    assert config["NUM_CLASSES"] == 47 and config["RPN_ANCHOR_SCALES"] == [16, 32, 64, 128, 256]
    assert (out / "model" / "saved_model.pb").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())
