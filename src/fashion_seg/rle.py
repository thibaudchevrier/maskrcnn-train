"""Run-length encoding used by the iMaterialist annotations.

Format: space-separated ``start length`` pairs, 1-indexed starts, pixels
enumerated in column-major (Fortran) order, i.e. top-to-bottom then left-to-right.
The same format is used for predicted masks so training data and predictions
can be decoded by the same function.
"""

import numpy as np


def decode(rle: str, height: int, width: int) -> np.ndarray:
    """Decode an RLE string into a boolean mask of shape ``(height, width)``."""
    flat = np.zeros(height * width, dtype=bool)
    values = np.array(rle.split(), dtype=np.int64)
    for start, length in zip(values[::2] - 1, values[1::2], strict=True):
        flat[start : start + length] = True
    return flat.reshape((height, width), order="F")


def encode(mask: np.ndarray) -> str:
    """Encode a 2D boolean mask into an RLE string."""
    flat = np.asarray(mask, dtype=bool).flatten(order="F")
    # Pad so that runs touching the borders produce a transition.
    padded = np.concatenate([[False], flat, [False]])
    transitions = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    starts, ends = transitions[::2], transitions[1::2]
    return " ".join(f"{s} {e - s}" for s, e in zip(starts, ends, strict=True))
