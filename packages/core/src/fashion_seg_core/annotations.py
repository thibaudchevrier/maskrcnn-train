"""Per-image annotations: the compact format written by the ``prepare`` stage.

``train.csv`` has one row per mask (333k rows, 1.4 GB). ``prepare`` groups it into one row per
image, sorted by ``image_id``, and stores it as Parquet so every trainer loads it in seconds:

    image_id: str, height: int, width: int,
    class_ids: list[int]  (dataset category ids, 0-based; model class = category + 1),
    rles: list[str]       (one RLE mask per class id, see ``fashion_seg_core.rle``)
"""

import json
from pathlib import Path

import pandas as pd

COLUMNS = ["image_id", "height", "width", "class_ids", "rles"]


def group_by_image(masks: pd.DataFrame) -> pd.DataFrame:
    """``train.csv`` rows (one per mask) -> one row per image, sorted by ``image_id``."""
    grouped = (
        masks.groupby("ImageId", sort=True)
        .agg(
            height=("Height", "first"),
            width=("Width", "first"),
            class_ids=("ClassId", list),
            rles=("EncodedPixels", list),
        )
        .reset_index()
        .rename(columns={"ImageId": "image_id"})
    )
    grouped["image_id"] = grouped["image_id"].astype(str)
    return grouped[COLUMNS]


def load(path: str | Path, image_ids: list[str] | None = None) -> pd.DataFrame:
    """Read the prepared annotations, optionally restricted to ``image_ids`` (in that order)."""
    frame = pd.read_parquet(path)
    if image_ids is None:
        return frame
    return frame.set_index("image_id").loc[image_ids].reset_index()


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Read ``split.json``: ``{"train": [...], "val": [...]}`` image ids."""
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"train": split["train"], "val": split["val"]}
