"""Images and masks between their full size and a model's working size.

- Training: ``load_example`` reads an image and its masks downscaled for a model. iMaterialist
  photos are large (often 3,000-5,000 px): decoding the JPEG at a reduced scale and resizing the
  masks before building targets keeps data loading fast.
- Serving: ``full_size_masks`` brings masks predicted on a shrunk image back to its full size,
  box by box, as booleans (a full-resolution float mask per instance would take gigabytes).

Shared by every family's dataset and predictor.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fashion_seg_contract import rle
from PIL import Image

MASK_THRESHOLD = 0.5


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


def full_size_masks(probs: np.ndarray, boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bring masks predicted on a shrunk image back to the image's size, box by box.

    Only each box's region is resized (bilinear on the probabilities, then thresholded), and the
    result is boolean: memory stays proportional to the masks, not to ``N`` float images.

    Parameters
    ----------
    probs : np.ndarray
        ``[N, h, w]`` mask probabilities on the shrunk image.
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` int32 boxes in the full image, ``(y2, x2)`` excluded.
    size : tuple[int, int]
        ``(height, width)`` of the full image.

    Returns
    -------
    np.ndarray
        ``[height, width, N]`` bool masks.

    Examples
    --------
    >>> probs = np.zeros((1, 10, 10), np.float32); probs[0, 2:6, 4:8] = 1.0
    >>> masks = full_size_masks(probs, np.array([[4, 8, 12, 16]]), size=(20, 20))
    >>> masks.shape, int(masks.sum())
    ((20, 20, 1), 64)
    """
    masks = np.zeros((*size, probs.shape[0]), dtype=bool)
    for i, (y1, x1, y2, x2) in enumerate(boxes):
        if y2 <= y1 or x2 <= x1:
            continue
        rows, cols = shrunk_region((y1, x1, y2, x2), probs.shape[1:], size)
        region = Image.fromarray(np.ascontiguousarray(probs[i, rows, cols], np.float32))
        resized = np.asarray(region.resize((x2 - x1, y2 - y1), Image.Resampling.BILINEAR))
        masks[y1:y2, x1:x2, i] = resized > MASK_THRESHOLD
    return masks


def shrunk_region(
    box: tuple[int, int, int, int], shrunk: tuple[int, int], size: tuple[int, int]
) -> tuple[slice, slice]:
    """Find the region of the shrunk image covering a box of the full image.

    Parameters
    ----------
    box : tuple[int, int, int, int]
        ``(y1, x1, y2, x2)`` in the full image.
    shrunk : tuple[int, int]
        ``(height, width)`` of the shrunk image.
    size : tuple[int, int]
        ``(height, width)`` of the full image.

    Returns
    -------
    tuple[slice, slice]
        Rows and columns of the shrunk image, at least one pixel each.

    Examples
    --------
    >>> shrunk_region((4, 8, 12, 16), shrunk=(10, 10), size=(20, 20))
    (slice(2, 6, None), slice(4, 8, None))
    """
    (y1, x1, y2, x2), (small_h, small_w), (height, width) = box, shrunk, size
    top, left = math.floor(y1 * small_h / height), math.floor(x1 * small_w / width)
    bottom = min(small_h, max(top + 1, math.ceil(y2 * small_h / height)))
    right = min(small_w, max(left + 1, math.ceil(x2 * small_w / width)))
    return slice(top, bottom), slice(left, right)
