"""Test helpers shared by the library and the families: a synthetic dataset and its ``params.yaml``.

End-to-end tests (training smoke runs, packaging) use them instead of the DVC data.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from fashion_seg_contract import rle
from PIL import Image

N_CATEGORIES = 46  # iMaterialist categories; model classes add the background


def garment_mask(height: int, width: int) -> np.ndarray:
    """Build the mask of the garment drawn on every synthetic image.

    Parameters
    ----------
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    np.ndarray
        Boolean mask: a rectangle from ``(height // 6, width // 5)`` to
        ``(2 * height // 3, 2 * width // 3)``, excluded.

    Examples
    --------
    >>> int(garment_mask(6, 5).sum())
    6
    """
    mask = np.zeros((height, width), bool)
    mask[height // 6 : height * 2 // 3, width // 5 : width * 2 // 3] = True
    return mask


def write_dataset(root: Path, n_images: int, height: int, width: int) -> dict[str, str]:
    """Write images with one garment (category 5), their annotations, a split and labels.

    The last image is the validation split, the others the training split.

    Parameters
    ----------
    root : Path
        Directory to write to.
    n_images : int
        Number of images (at least 2).
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    dict[str, str]
        The ``data`` section of ``params.yaml`` pointing at the files.
    """
    image_dir = root / "images"
    image_dir.mkdir()
    mask = garment_mask(height, width)
    image = np.where(mask[..., None], 210, 70).astype(np.uint8).repeat(3, axis=2)
    ids = [f"img{i}" for i in range(n_images)]
    for image_id in ids:
        Image.fromarray(image).save(image_dir / f"{image_id}.jpg")
    prepared = root / "prepared"
    prepared.mkdir()
    pl.DataFrame(
        {
            "image_id": ids,
            "height": [height] * n_images,
            "width": [width] * n_images,
            "class_ids": [[5]] * n_images,
            "rles": [[rle.encode(mask)]] * n_images,
        }
    ).write_parquet(prepared / "annotations.parquet")
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


def write_params(
    root: Path,
    data: dict[str, str],
    train: dict[str, dict[str, Any]],
    models: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Write a complete ``params.yaml`` with test experiments.

    Parameters
    ----------
    root : Path
        Directory to write to.
    data : dict[str, str]
        The ``data`` section (see ``write_dataset``).
    train : dict[str, dict[str, Any]]
        The ``train`` section.
    models : dict[str, dict[str, str]] | None
        The ``models`` section. By default ``None``: no served model.

    Returns
    -------
    Path
        Path of the written file.
    """
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
        "models": models or {},
        "evaluate": {"split": "val", "max_images": None, "min_score": 0.0, "output_dir": "metrics"},
    }
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(params))
    return path
