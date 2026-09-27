"""End-to-end trainer check on a synthetic prepared dataset (no DVC data needed)."""

import json

import numpy as np
import polars as pl
import yaml
from fashion_seg_core import rle
from fashion_seg_matterport import train
from fashion_seg_matterport.dataset import FashionDataset
from PIL import Image


def _write_dataset(root):
    image_dir = root / "images"
    image_dir.mkdir()
    rows = []
    for i in range(4):
        image = np.full((96, 128, 3), 70, np.uint8)
        image[20:70, 30:90] = 210
        Image.fromarray(image).save(image_dir / f"img{i}.jpg")
        mask = np.zeros((96, 128), bool)
        mask[20:70, 30:90] = True
        rows.append(
            {
                "image_id": f"img{i}",
                "height": 96,
                "width": 128,
                "class_ids": [5],
                "rles": [rle.encode(mask)],
            }
        )
    prepared = root / "prepared"
    prepared.mkdir()
    pl.DataFrame(rows).write_parquet(prepared / "annotations.parquet")
    (prepared / "split.json").write_text(
        json.dumps({"train": ["img0", "img1", "img2"], "val": ["img3"]})
    )
    labels = root / "labels.json"
    labels.write_text(json.dumps({"categories": [{"id": k, "name": f"c{k}"} for k in range(46)]}))
    return image_dir, prepared, labels


def test_dataset_maps_categories_to_model_classes(tmp_path):
    image_dir, prepared, _ = _write_dataset(tmp_path)
    records = pl.read_parquet(prepared / "annotations.parquet")
    names = ["BG"] + [f"c{k}" for k in range(46)]
    dataset = FashionDataset(records, image_dir, names)
    masks, class_ids = dataset.load_mask(0)
    assert masks.shape == (96, 128, 1) and masks.sum() == 50 * 60
    assert class_ids.tolist() == [6]  # category 5 -> model class 6


def test_smoke_training_exports_servable_model(tmp_path, monkeypatch):
    image_dir, prepared, labels = _write_dataset(tmp_path)
    params = {
        "data": {
            "train_images": str(image_dir),
            "prepared_dir": str(prepared),
            "label_file": str(labels),
        },
        "split": {"n_folds": 8, "fold": 0, "seed": 42},
        "train_matterport": {
            "experiment": "test",
            "output_dir": str(tmp_path / "out" / "matterport"),
            "init_weights": str(tmp_path / "missing.h5"),
            "backbone": "resnet101",
            "image_min_dim": 800,
            "image_max_dim": 1024,
            "images_per_gpu": 2,
            "rpn_anchor_scales": [16, 32, 64, 128, 256],
            "train_rois_per_image": 32,
            "learning_rate": 0.001,
            "epochs": 2,
            "layers": "all",
            "steps_per_epoch": None,
            "validation_steps": None,
            "max_train_images": None,
            "max_val_images": None,
            "detection_min_confidence": 0.7,
        },
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)

    train.main(["--params", "params.yaml", "--smoke"])

    out = tmp_path / "out" / "matterport-smoke"
    config = json.loads((out / "model" / "config.json").read_text())
    assert config["NUM_CLASSES"] == 47 and config["RPN_ANCHOR_SCALES"] == [16, 32, 64, 128, 256]
    assert (out / "model" / "saved_model.pb").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())
