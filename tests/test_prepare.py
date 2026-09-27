"""Tests of the prepare stage: per-image annotations and the frozen split."""

import polars as pl
import pytest

from fashion_seg.prepare import split_image_ids
from fashion_seg_core import annotations


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
    frame.write_parquet(tmp_path / "annotations.parquet")
    loaded = annotations.load(tmp_path / "annotations.parquet", ["c", "a"])
    assert loaded["image_id"].to_list() == ["c", "a"]
    with pytest.raises(KeyError):
        annotations.load(tmp_path / "annotations.parquet", ["zz"])
