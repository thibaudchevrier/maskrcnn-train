"""Per-image annotations: the compact format written by the ``prepare`` stage.

``train.csv`` has one row per mask (333k rows, 1.4 GB). ``prepare`` groups it into one row per
image, sorted by ``image_id``, and stores it as Parquet so every trainer loads it in seconds:

    image_id: str, height: int, width: int,
    class_ids: list[int]  (dataset category ids, 0-based; model class = category + 1),
    rles: list[str]       (one RLE mask per class id, see ``fashion_seg_core.rle``)
"""

import json
from pathlib import Path

import polars as pl

COLUMNS = ["image_id", "height", "width", "class_ids", "rles"]


def group_by_image(masks: pl.DataFrame) -> pl.DataFrame:
    """``train.csv`` rows (one per mask) -> one row per image, sorted by ``image_id``.

    Masks keep their file order within each image.
    """
    return (
        masks.group_by("ImageId", maintain_order=True)
        .agg(
            pl.col("Height").first().alias("height"),
            pl.col("Width").first().alias("width"),
            pl.col("ClassId").alias("class_ids"),
            pl.col("EncodedPixels").alias("rles"),
        )
        .rename({"ImageId": "image_id"})
        .with_columns(pl.col("image_id").cast(pl.String))
        .sort("image_id")
        .select(COLUMNS)
    )


def load(path: str | Path, image_ids: list[str] | None = None) -> pl.DataFrame:
    """Read the prepared annotations, optionally restricted to ``image_ids`` (in that order)."""
    frame = pl.read_parquet(path)
    if image_ids is None:
        return frame
    order = pl.DataFrame({"image_id": image_ids})
    selected = order.join(frame, on="image_id", how="left", maintain_order="left")
    missing = selected.filter(pl.col("height").is_null())["image_id"].to_list()
    if missing:
        raise KeyError(f"{len(missing)} image ids not in {path}, e.g. {missing[:3]}")
    return selected


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Read ``split.json``: ``{"train": [...], "val": [...]}`` image ids."""
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"train": split["train"], "val": split["val"]}
