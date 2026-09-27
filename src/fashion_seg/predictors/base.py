"""The interface every model family implements to be served: image in, ``Detections`` out."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detections:
    """Detections for one image, in original image pixel coordinates."""

    boxes: np.ndarray  # [N, (y1, x1, y2, x2)] int32, (y2, x2) excluded
    class_ids: np.ndarray  # [N] int32, 0 is the background
    scores: np.ndarray  # [N] float32
    masks: np.ndarray  # [H, W, N] bool


class Predictor(Protocol):
    """Anything turning an RGB image into detections (legacy TF model, torchvision, ...)."""

    def predict(self, image: np.ndarray) -> Detections:
        """image: [H, W, 3] uint8 RGB."""
