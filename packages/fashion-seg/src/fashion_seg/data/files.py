"""The dataset's files: ``train.csv``, the prepared annotations and split, the images.

Every model trains and is evaluated on the prepared directory's ``ANNOTATIONS_FILE`` and
``SPLIT_FILE``, written by the ``prepare`` stage.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl

from fashion_seg.data import annotations

ANNOTATIONS_FILE = "annotations.parquet"
SPLIT_FILE = "split.json"


def read_masks(train_csv: Path) -> pl.DataFrame:
    """Read ``train.csv``: one row per mask.

    Parameters
    ----------
    train_csv : Path
        iMaterialist ``train.csv``.

    Returns
    -------
    pl.DataFrame
        Columns ``ImageId``, ``EncodedPixels``, ``Height``, ``Width``, ``ClassId``.
    """
    return pl.read_csv(
        train_csv,
        columns=["ImageId", "EncodedPixels", "Height", "Width", "ClassId"],
        schema_overrides={"ImageId": pl.String, "ClassId": pl.Int64},
    )


def write_prepared(
    prepared_dir: Path,
    per_image: pl.DataFrame,
    split: dict[str, list[str]],
    split_params: dict[str, Any],
) -> None:
    """Write the prepared annotations and split.

    Parameters
    ----------
    prepared_dir : Path
        Output directory, created if needed.
    per_image : pl.DataFrame
        Per-image annotations (``annotations.group_by_image``).
    split : dict[str, list[str]]
        Image ids under ``"train"`` and ``"val"``.
    split_params : dict[str, Any]
        Parameters of the split, recorded with it.
    """
    prepared_dir.mkdir(parents=True, exist_ok=True)
    per_image.write_parquet(prepared_dir / ANNOTATIONS_FILE)
    (prepared_dir / SPLIT_FILE).write_text(
        json.dumps({"params": split_params, **split}) + "\n", encoding="utf-8"
    )


def load_annotations(prepared_dir: Path, image_ids: list[str] | None = None) -> pl.DataFrame:
    """Read the prepared annotations, optionally restricted to some images.

    Parameters
    ----------
    prepared_dir : Path
        The prepared directory.
    image_ids : list[str] | None
        Images to keep, in the order to return them. By default ``None``: every image.

    Returns
    -------
    pl.DataFrame
        One row per image with the ``COLUMNS`` of ``annotations``. Unknown ``image_ids`` raise
        ``annotations.select``'s ``KeyError``.
    """
    frame = pl.read_parquet(prepared_dir / ANNOTATIONS_FILE)
    return frame if image_ids is None else annotations.select(frame, image_ids)


def load_split(prepared_dir: Path) -> dict[str, list[str]]:
    """Read the frozen train/val split.

    Parameters
    ----------
    prepared_dir : Path
        The prepared directory.

    Returns
    -------
    dict[str, list[str]]
        Image ids under ``"train"`` and ``"val"``.
    """
    split = json.loads((Path(prepared_dir) / SPLIT_FILE).read_text(encoding="utf-8"))
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
