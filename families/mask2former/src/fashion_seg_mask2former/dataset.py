"""Prepared iMaterialist annotations as Mask2Former training batches.

Mask2Former takes a list of binary masks and class indices per image, so overlapping instances
(a sleeve on a top) are kept as they are. Its class indices are the dataset categories (model
class id - 1).
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from fashion_seg.data.images import load_example
from fashion_seg_mask2former.network import to_pixels


class FashionDataset(Dataset):
    """Images and their instances, downscaled so the long side is at most ``max_side``.

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one image and its instances.

        Parameters
        ----------
        index : int
            Row of ``records``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``[3, H, W]`` normalized image, ``[N, H, W]`` float masks and ``[N]`` class indices
            (dataset categories). Instances that vanish when downscaled are dropped.
        """
        example = load_example(self.records.row(index, named=True), self.image_dir, self.max_side)
        height, width = example.image.shape[:2]
        kept = [
            (m, label) for m, label in zip(example.masks, example.labels, strict=True) if m.any()
        ]
        masks = (
            torch.from_numpy(np.stack([m for m, _ in kept])).float()
            if kept
            else torch.zeros((0, height, width))
        )
        labels = torch.tensor([label - 1 for _, label in kept], dtype=torch.long)
        return to_pixels(example.image), masks, labels


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Pad images and masks to the batch's largest size.

    Parameters
    ----------
    batch : list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ``(image, masks, labels)`` items.

    Returns
    -------
    dict[str, torch.Tensor | list[torch.Tensor]]
        ``pixel_values`` ``[B, 3, H, W]``, ``pixel_mask`` ``[B, H, W]`` (1 on real pixels),
        ``mask_labels`` (one ``[N_i, H, W]`` per image) and ``class_labels`` (one ``[N_i]``).

    Examples
    --------
    >>> item = (torch.ones(3, 2, 4), torch.ones(1, 2, 4), torch.tensor([5]))
    >>> other = (torch.ones(3, 3, 2), torch.zeros(0, 3, 2), torch.zeros(0, dtype=torch.long))
    >>> out = collate([item, other])
    >>> tuple(out["pixel_values"].shape), out["pixel_mask"][1].sum().item()
    ((2, 3, 3, 4), 6)
    """
    height = max(image.shape[1] for image, _, _ in batch)
    width = max(image.shape[2] for image, _, _ in batch)
    pixel_values = torch.zeros((len(batch), 3, height, width))
    pixel_mask = torch.zeros((len(batch), height, width), dtype=torch.long)
    mask_labels = []
    for i, (image, masks, _) in enumerate(batch):
        h, w = image.shape[1:]
        pixel_values[i, :, :h, :w] = image
        pixel_mask[i, :h, :w] = 1
        padded = torch.zeros((masks.shape[0], height, width))
        padded[:, :h, :w] = masks
        mask_labels.append(padded)
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": [labels for _, _, labels in batch],
    }
