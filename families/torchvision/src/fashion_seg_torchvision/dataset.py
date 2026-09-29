"""Prepared iMaterialist annotations as a torchvision detection dataset."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from fashion_seg.data.images import load_example
from fashion_seg_torchvision.network import to_tensor


class FashionDataset(Dataset):
    """Images and instance targets, downscaled so the long side is at most ``max_side``.

    Images and masks come from ``fashion_seg.data.images.load_example`` (fast JPEG downscaling);
    the model resizes further as configured.

    Parameters
    ----------
    records : pl.DataFrame
        Prepared annotations (see ``fashion_seg.data.annotations``) of the images to include.
    image_dir : str | Path
        Directory of the ``<image_id>.jpg`` files.
    max_side : int
        Largest width or height after downscaling.

    Attributes
    ----------
    records : pl.DataFrame
        The annotations.
    image_dir : Path
        Directory of the images.
    max_side : int
        Largest width or height after downscaling.
    """

    records: pl.DataFrame
    image_dir: Path
    max_side: int

    def __init__(self, records: pl.DataFrame, image_dir: str | Path, max_side: int):
        self.records = records
        self.image_dir = Path(image_dir)
        self.max_side = max_side

    def __len__(self) -> int:
        """Count the images.

        Returns
        -------
        int
            Number of images.
        """
        return self.records.height

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load one image and its targets.

        Parameters
        ----------
        index : int
            Row of ``records``.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            ``[3, H, W]`` float image in [0, 1], and the targets torchvision expects: ``boxes``
            ``[N, (x1, y1, x2, y2)]``, ``labels`` ``[N]`` (category + 1) and ``masks``
            ``[N, H, W]`` uint8. Instances that vanish when downscaled are dropped.
        """
        example = load_example(self.records.row(index, named=True), self.image_dir, self.max_side)
        return to_tensor(example.image), build_targets(example.masks, example.labels)


def build_targets(masks: list[np.ndarray], labels: list[int]) -> dict[str, torch.Tensor]:
    """Build torchvision targets from masks, dropping empty ones.

    Parameters
    ----------
    masks : list[np.ndarray]
        ``[H, W]`` masks of 0 and 1, one per instance.
    labels : list[int]
        Model class id of each mask.

    Returns
    -------
    dict[str, torch.Tensor]
        ``boxes`` ``[N, (x1, y1, x2, y2)]`` float32, ``labels`` ``[N]`` int64, ``masks``
        ``[N, H, W]`` uint8.

    Examples
    --------
    >>> m = np.zeros((4, 4), np.uint8); m[1:3, 0:2] = 1
    >>> build_targets([m, np.zeros((4, 4), np.uint8)], [5, 6])["boxes"].tolist()
    [[0.0, 1.0, 2.0, 3.0]]
    """
    boxes, kept_masks, kept_labels = [], [], []
    for mask, label in zip(masks, labels, strict=True):
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        kept_masks.append(mask)
        kept_labels.append(label)
    height, width = masks[0].shape if masks else (0, 0)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(kept_labels, dtype=torch.int64),
        "masks": torch.from_numpy(np.stack(kept_masks))
        if kept_masks
        else torch.zeros((0, height, width), dtype=torch.uint8),
    }


def collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Keep images and targets as lists (they have different sizes).

    Parameters
    ----------
    batch : list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
        ``(image, targets)`` pairs.

    Returns
    -------
    tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]
        The images and the targets.
    """
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
