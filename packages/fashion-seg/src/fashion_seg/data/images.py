"""Training examples read from disk: an image and its instance masks, downscaled for a model.

Shared by every family's dataset. iMaterialist photos are large (often 3,000-5,000 px):
decoding the JPEG at a reduced scale and resizing the masks before building targets keeps data
loading fast.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fashion_seg_contract import rle
from PIL import Image


@dataclass(frozen=True)
class Example:
    """One training image and its instances, at the training size.

    Attributes
    ----------
    image : np.ndarray
        RGB image, ``(height, width, 3)`` uint8.
    masks : list[np.ndarray]
        One ``(height, width)`` uint8 mask of 0 and 1 per instance (may be empty after
        downscaling a tiny instance).
    labels : list[int]
        Model class id of each mask (dataset category + 1, 0 being the background).
    """

    image: np.ndarray
    masks: list[np.ndarray]
    labels: list[int]


def scaled_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    """Size of an image downscaled so its long side is at most ``max_side`` (never upscaled).

    Parameters
    ----------
    height : int
        Original height.
    width : int
        Original width.
    max_side : int
        Largest allowed width or height.

    Returns
    -------
    tuple[int, int]
        ``(width, height)``, as Pillow expects it.

    Examples
    --------
    >>> scaled_size(3000, 2000, max_side=1024), scaled_size(300, 200, max_side=1024)
    ((683, 1024), (200, 300))
    """
    scale = min(1.0, max_side / max(height, width))
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_example(row: dict[str, Any], image_dir: Path, max_side: int) -> Example:
    """Load an image and decode its masks, downscaled to ``max_side``.

    Parameters
    ----------
    row : dict[str, Any]
        A row of the prepared annotations (see ``fashion_seg.data.annotations``).
    image_dir : Path
        Directory of the ``<image_id>.jpg`` files.
    max_side : int
        Largest width or height after downscaling.

    Returns
    -------
    Example
        The image, its masks (nearest-neighbour resized) and their model class ids.
    """
    height, width = int(row["height"]), int(row["width"])
    size = scaled_size(height, width, max_side)
    with Image.open(Path(image_dir) / f"{row['image_id']}.jpg") as img:
        img.draft("RGB", size)  # JPEG decoded at a reduced scale: much faster
        image = np.asarray(img.convert("RGB").resize(size, Image.Resampling.BILINEAR))
    masks = [
        np.asarray(
            Image.fromarray(rle.decode(r, height, width).astype(np.uint8)).resize(
                size, Image.Resampling.NEAREST
            )
        )
        for r in row["rles"]
    ]
    return Example(image, masks, [int(c) + 1 for c in row["class_ids"]])
