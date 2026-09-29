"""Mask2Former exports (Hugging Face model + ``fashion_seg.json``) as a serving ``Predictor``.

Each of the model's queries proposes one instance: a class and a mask. Its score is the class
probability times the mask's confidence, as in Mask2Former's instance post-processing, but masks
are kept per instance (they may overlap, like a sleeve on a top) instead of one label per pixel.
Large photos are shrunk to ``max_size`` first, and masks come back to full size on the CPU, box
by box.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fashion_seg_torch.device import pick_device
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation

from fashion_seg.data.images import MASK_THRESHOLD, full_size_masks, scaled_size
from fashion_seg.ports import Detections
from fashion_seg_mask2former.network import to_pixels

EXPORT_CONFIG = "fashion_seg.json"


class Mask2FormerPredictor:
    """Runs a model exported by ``training`` on single RGB images.

    Parameters
    ----------
    model_dir : str | Path
        Export directory.
    device : str
        ``"auto"`` (CUDA, then Apple GPU, then CPU), ``"cuda"``, ``"mps"`` or ``"cpu"``.
        By default ``"auto"``.

    Attributes
    ----------
    config : dict[str, Any]
        The export's ``fashion_seg.json``.
    device : torch.device
        The device the model runs on.
    """

    config: dict[str, Any]
    device: torch.device

    def __init__(self, model_dir: str | Path, device: str = "auto"):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / EXPORT_CONFIG).read_text(encoding="utf-8"))
        self.device = pick_device(device)
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
        self._model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Detections:
        """Detect the garments in one image.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape ``(height, width, 3)``, uint8.

        Returns
        -------
        Detections
            The detections, in the image's pixel coordinates, best first.
        """
        height, width = image.shape[:2]
        size = scaled_size(height, width, self.config["max_size"])
        if size != (width, height):
            image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))
        outputs = self._model(pixel_values=to_pixels(image)[None].to(self.device))
        probs = torch.nn.functional.interpolate(
            outputs.masks_queries_logits, size=(size[1], size[0]), mode="bilinear"
        )[0].sigmoid()
        scores, classes = outputs.class_queries_logits[0].softmax(-1)[:, :-1].max(-1)
        binary = probs > MASK_THRESHOLD
        area = binary.flatten(1).sum(1)
        scores = scores * (probs * binary).flatten(1).sum(1) / area.clamp(min=1)
        keep = torch.nonzero(area > 0).flatten()
        keep = keep[scores[keep].argsort(descending=True)]
        probs = probs[keep].cpu().numpy()
        boxes = boxes_from_masks(binary[keep].cpu().numpy(), (height, width))
        return Detections(
            boxes=boxes,
            class_ids=(classes[keep] + 1).cpu().numpy().astype(np.int32),
            scores=scores[keep].cpu().numpy().astype(np.float32),
            masks=full_size_masks(probs, boxes, (height, width)),
        )


def boxes_from_masks(masks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Find each mask's box, scaled from the shrunk image to the full one.

    Parameters
    ----------
    masks : np.ndarray
        ``[N, h, w]`` bool masks on the shrunk image, none empty.
    size : tuple[int, int]
        ``(height, width)`` of the full image.

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` int32 in the full image, ``(y2, x2)`` excluded.

    Examples
    --------
    >>> m = np.zeros((1, 10, 10), bool); m[0, 2:6, 4:8] = True
    >>> boxes_from_masks(m, (20, 20)).tolist()
    [[4, 8, 12, 16]]
    """
    if masks.shape[0] == 0:
        return np.zeros((0, 4), np.int32)
    small_h, small_w = masks.shape[1:]
    rows, cols = masks.any(axis=2), masks.any(axis=1)
    y1 = rows.argmax(axis=1)
    y2 = small_h - rows[:, ::-1].argmax(axis=1)
    x1 = cols.argmax(axis=1)
    x2 = small_w - cols[:, ::-1].argmax(axis=1)
    height, width = size
    scale_y, scale_x = height / small_h, width / small_w
    return np.stack(
        [
            np.floor(y1 * scale_y).clip(0, height),
            np.floor(x1 * scale_x).clip(0, width),
            np.ceil(y2 * scale_y).clip(0, height),
            np.ceil(x2 * scale_x).clip(0, width),
        ],
        axis=1,
    ).astype(np.int32)
