"""Synthetic prepared dataset and ``params.yaml`` for end-to-end tests (no DVC data needed)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from fashion_seg_contract import rle
from PIL import Image

N_CATEGORIES = 46


def _garment(height: int, width: int) -> np.ndarray:
    """Build the mask of the garment drawn on every synthetic image."""
    mask = np.zeros((height, width), bool)
    mask[height // 6 : height * 2 // 3, width // 5 : width * 2 // 3] = True
    return mask


def write_dataset(root: Path, n_images: int, height: int, width: int) -> dict[str, str]:
    """Write images with one garment (category 5), their annotations, a split and labels.

    The last image is the validation split. Returns the ``data`` section of ``params.yaml``.
    """
    image_dir = root / "images"
    image_dir.mkdir()
    mask = _garment(height, width)
    image = np.where(mask[..., None], 210, 70).astype(np.uint8).repeat(3, axis=2)
    rows = []
    for i in range(n_images):
        Image.fromarray(image).save(image_dir / f"img{i}.jpg")
        rows.append(
            {
                "image_id": f"img{i}",
                "height": height,
                "width": width,
                "class_ids": [5],
                "rles": [rle.encode(mask)],
            }
        )
    prepared = root / "prepared"
    prepared.mkdir()
    pl.DataFrame(rows).write_parquet(prepared / "annotations.parquet")
    ids = [row["image_id"] for row in rows]
    (prepared / "split.json").write_text(json.dumps({"train": ids[:-1], "val": ids[-1:]}))
    labels = root / "labels.json"
    categories = [{"id": k, "name": f"c{k}"} for k in range(N_CATEGORIES)]
    labels.write_text(json.dumps({"categories": categories}))
    return {
        "train_csv": str(root / "train.csv"),
        "train_images": str(image_dir),
        "label_file": str(labels),
        "prepared_dir": str(prepared),
    }


def write_params(root: Path, data: dict[str, str], train: dict[str, dict[str, Any]]) -> Path:
    """Write a complete ``params.yaml`` with the given data and training sections."""
    params = {
        "data": data,
        "split": {"n_folds": 8, "fold": 0, "seed": 42},
        "tracking": {
            "training_experiment": "test-training",
            "packaging_experiment": "test-packaging",
            "evaluation_experiment": "test-evaluation",
            "registered_name": "test-model",
        },
        "train": train,
        "models": {},
        "evaluate": {"split": "val", "max_images": None, "min_score": 0.0, "output_dir": "metrics"},
    }
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(params))
    return path
