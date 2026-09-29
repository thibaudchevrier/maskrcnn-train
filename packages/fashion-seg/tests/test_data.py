"""Tests of the data modules: per-image annotations and the frozen split."""

import numpy as np
import polars as pl
import pytest
from fashion_seg_contract import rle
from PIL import Image

from fashion_seg.data import annotations, files, images
from fashion_seg.data.split import split_image_ids


def test_group_by_image_sorts_and_collects_masks():
    """Masks are grouped per image, images sorted by id, masks kept in file order."""
    masks = pl.DataFrame(
        {
            "ImageId": ["b", "a", "b"],
            "EncodedPixels": ["1 2", "3 4", "5 6"],
            "Height": [10, 20, 10],
            "Width": [11, 21, 11],
            "ClassId": [3, 1, 7],
        }
    )
    grouped = annotations.group_by_image(masks)
    assert grouped["image_id"].to_list() == ["a", "b"]
    row = grouped.row(1, named=True)
    assert (row["height"], row["width"]) == (10, 11)
    assert list(row["class_ids"]) == [3, 7] and list(row["rles"]) == ["1 2", "5 6"]


def test_split_is_deterministic_disjoint_and_complete():
    """The split is reproducible, and train and val partition the images."""
    ids = [f"{i:04d}" for i in range(100)]
    first = split_image_ids(ids, n_folds=8, fold=0, seed=42)
    assert first == split_image_ids(ids, n_folds=8, fold=0, seed=42)
    assert not set(first["train"]) & set(first["val"])
    assert sorted(first["train"] + first["val"]) == ids
    assert len(first["val"]) == 13  # 100 = 4 * 13 + 4 * 12, the first folds are larger


def test_split_rejects_unknown_fold():
    """Asking for a fold beyond n_folds fails."""
    with pytest.raises(ValueError):
        split_image_ids(["a", "b", "c"], n_folds=3, fold=3, seed=0)


def test_load_restricts_to_ids_in_order(tmp_path):
    """Loading a subset keeps the requested order and rejects unknown ids."""
    frame = pl.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "height": [1, 2, 3],
            "width": [1, 2, 3],
            "class_ids": [[0], [1], [2]],
            "rles": [["1 1"], ["1 1"], ["1 1"]],
        }
    )
    frame.write_parquet(tmp_path / files.ANNOTATIONS_FILE)
    loaded = files.load_annotations(tmp_path, ["c", "a"])
    assert loaded["image_id"].to_list() == ["c", "a"]
    with pytest.raises(KeyError):
        files.load_annotations(tmp_path, ["zz"])


def test_select_ids_keeps_local_images_then_limits(tmp_path):
    """Only images on disk are kept when asked, then the first ``limit``, in split order."""
    for image_id in ("b", "c", "d"):
        (tmp_path / f"{image_id}.jpg").touch()
    ids = ["a", "b", "c", "d"]
    assert files.select_ids(ids, tmp_path, limit=2, local_only=True) == ["b", "c"]
    assert files.select_ids(ids, tmp_path) == ids


def test_load_example_downscales_image_and_masks_together(tmp_path):
    """The image and its masks are downscaled to max_side; categories become model classes."""
    Image.new("RGB", (200, 100), (10, 20, 30)).save(tmp_path / "a.jpg")
    mask = np.zeros((100, 200), bool)
    mask[20:60, 40:120] = True
    row = {
        "image_id": "a",
        "height": 100,
        "width": 200,
        "class_ids": [4],
        "rles": [rle.encode(mask)],
    }
    example = images.load_example(row, tmp_path, max_side=50)
    assert example.image.shape == (25, 50, 3)
    assert example.labels == [5]
    [small] = example.masks
    assert small.shape == (25, 50) and small.sum() == 10 * 20
