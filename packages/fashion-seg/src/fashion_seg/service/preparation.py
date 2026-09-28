"""Prepare step: per-image annotations and the frozen train/val split shared by every model.

Run through DVC: ``uv run dvc repro --single-item prepare``.
"""

import json

import polars as pl

from fashion_seg.config import DataConfig, SplitConfig
from fashion_seg.data import annotations
from fashion_seg.data.split import split_image_ids


def prepare(data: DataConfig, split: SplitConfig) -> dict[str, int]:
    """Write ``annotations.parquet`` and ``split.json`` in the prepared directory.

    Parameters
    ----------
    data : DataConfig
        Dataset files: reads ``train_csv``, writes to ``prepared_dir``.
    split : SplitConfig
        The fold to keep for validation.

    Returns
    -------
    dict[str, int]
        Number of ``images``, ``masks``, ``train`` and ``val`` images.
    """
    data.prepared_dir.mkdir(parents=True, exist_ok=True)
    masks = pl.read_csv(
        data.train_csv,
        columns=["ImageId", "EncodedPixels", "Height", "Width", "ClassId"],
        schema_overrides={"ImageId": pl.String, "ClassId": pl.Int64},
    )
    per_image = annotations.group_by_image(masks)
    per_image.write_parquet(data.prepared_dir / annotations.ANNOTATIONS_FILE)

    ids = per_image["image_id"].to_list()
    result = split_image_ids(ids, split.n_folds, split.fold, split.seed)
    (data.prepared_dir / annotations.SPLIT_FILE).write_text(
        json.dumps({"params": split.model_dump(), **result}) + "\n", encoding="utf-8"
    )
    return {
        "images": len(per_image),
        "masks": len(masks),
        "train": len(result["train"]),
        "val": len(result["val"]),
    }
