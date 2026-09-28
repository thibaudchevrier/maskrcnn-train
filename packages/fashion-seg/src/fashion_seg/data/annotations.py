"""Per-image annotations: the compact format written by the ``prepare`` stage.

``train.csv`` has one row per mask (333k rows, 1.4 GB). ``prepare`` groups it into one row per
image, sorted by ``image_id``, and stores it as Parquet so every trainer loads it in seconds:

    image_id: str, height: int, width: int,
    class_ids: list[int]  (dataset category ids, 0-based; model class = category + 1),
    rles: list[str]       (one RLE mask per class id, see ``fashion_seg_contract.rle``)

Every model trains and is evaluated on these files: ``ANNOTATIONS_FILE`` and ``SPLIT_FILE`` in the
prepared directory.
"""

import json
from pathlib import Path

import polars as pl

COLUMNS = ["image_id", "height", "width", "class_ids", "rles"]
ANNOTATIONS_FILE = "annotations.parquet"
SPLIT_FILE = "split.json"


def group_by_image(masks: pl.DataFrame) -> pl.DataFrame:
    """Group ``train.csv`` rows (one per mask) into one row per image.

    Masks keep their file order within each image.

    Parameters
    ----------
    masks : pl.DataFrame
        ``train.csv`` columns ``ImageId``, ``EncodedPixels``, ``Height``, ``Width``, ``ClassId``.

    Returns
    -------
    pl.DataFrame
        One row per image with the ``COLUMNS`` of this module, sorted by ``image_id``.
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
    """Read the prepared annotations, optionally restricted to some images.

    Parameters
    ----------
    path : str | Path
        Path of ``annotations.parquet``.
    image_ids : list[str] | None
        Images to keep, in the order to return them. By default ``None``: every image.

    Returns
    -------
    pl.DataFrame
        One row per image with the ``COLUMNS`` of this module.

    Raises
    ------
    KeyError
        If some of ``image_ids`` are not in the annotations.
    """
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
    """Read the frozen train/val split.

    Parameters
    ----------
    path : str | Path
        Path of ``split.json``.

    Returns
    -------
    dict[str, list[str]]
        Image ids under ``"train"`` and ``"val"``.
    """
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"train": split["train"], "val": split["val"]}


def select_ids(
    ids: list[str], image_dir: Path, limit: int | None = None, local_only: bool = False
) -> list[str]:
    """Pick images of a split: optionally only those on disk, then the first ``limit``.

    Parameters
    ----------
    ids : list[str]
        Image ids of the split, in order.
    image_dir : Path
        Directory of the ``<image_id>.jpg`` files.
    limit : int | None
        Maximum number of images; ``None`` for all. By default ``None``.
    local_only : bool
        Keep only the images present in ``image_dir`` (partial pulls). By default ``False``.

    Returns
    -------
    list[str]
        The selected image ids, in split order.

    Examples
    --------
    >>> select_ids(["a", "b", "c"], Path("."), limit=2)
    ['a', 'b']
    """
    if local_only:
        ids = [i for i in ids if (image_dir / f"{i}.jpg").exists()]
    return ids[:limit] if limit else ids
