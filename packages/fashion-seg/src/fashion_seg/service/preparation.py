"""Prepare step: per-image annotations and the frozen train/val split shared by every model.

Run through DVC: ``uv run dvc repro --single-item prepare``.
"""

from fashion_seg.config import DataConfig, SplitConfig
from fashion_seg.data import annotations, files
from fashion_seg.data.split import split_image_ids


def prepare(data: DataConfig, split: SplitConfig) -> dict[str, int]:
    """Write the prepared annotations and split from ``train.csv``.

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
    masks = files.read_masks(data.train_csv)
    per_image = annotations.group_by_image(masks)
    ids = per_image["image_id"].to_list()
    result = split_image_ids(ids, split.n_folds, split.fold, split.seed)
    files.write_prepared(data.prepared_dir, per_image, result, split.model_dump())
    return {
        "images": len(per_image),
        "masks": len(masks),
        "train": len(result["train"]),
        "val": len(result["val"]),
    }
