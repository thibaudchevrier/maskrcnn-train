"""torchvision Mask R-CNN exports (``config.json`` + ``model.pt``) as a serving ``Predictor``.

Large photos are shrunk to the model's ``max_size`` before the network, and each mask is brought
back to the photo's size on the CPU, inside its box only: torchvision's own post-processing would
paste every mask at full resolution in float32 on the device (gigabytes for a 20-megapixel photo).
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fashion_seg_torch.device import pick_device
from PIL import Image

from fashion_seg.ports import Detections
from fashion_seg_torchvision.network import build_model, to_tensor

MASK_THRESHOLD = 0.5


class TorchvisionPredictor:
    """Runs a model exported by ``training`` on single RGB images.

    Parameters
    ----------
    model_dir : str | Path
        Export directory (``config.json`` + ``model.pt``).
    device : str
        ``"auto"`` (CUDA, then Apple GPU, then CPU), ``"cuda"``, ``"mps"`` or ``"cpu"``.
        By default ``"auto"``.

    Attributes
    ----------
    config : dict[str, Any]
        The exported configuration.
    device : torch.device
        The device the model runs on.
    """

    config: dict[str, Any]
    device: torch.device

    def __init__(self, model_dir: str | Path, device: str = "auto"):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.device = pick_device(device)
        self._model = build_model(
            self.config["num_classes"], self.config["min_size"], self.config["max_size"]
        )
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
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
            The detections, in the image's pixel coordinates.
        """
        height, width = image.shape[:2]
        scale = min(1.0, self.config["max_size"] / max(height, width))
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))
        output = self._model([to_tensor(image).to(self.device)])[0]
        boxes = to_yxyx(output["boxes"].cpu().numpy() / scale, height, width)
        return Detections(
            boxes=boxes,
            class_ids=output["labels"].cpu().numpy().astype(np.int32),
            scores=output["scores"].cpu().numpy().astype(np.float32),
            masks=full_size_masks(output["masks"][:, 0].cpu().numpy(), boxes, (height, width)),
        )


def to_yxyx(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert torchvision boxes to the contract's integer boxes, inside the image.

    Parameters
    ----------
    boxes : np.ndarray
        ``[N, (x1, y1, x2, y2)]`` float, in image pixels.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    np.ndarray
        ``[N, (y1, x1, y2, x2)]`` int32, ``(y2, x2)`` excluded.

    Examples
    --------
    >>> to_yxyx(np.array([[1.5, 2.2, 9.9, 30.0]]), height=20, width=8).tolist()
    [[2, 1, 20, 8]]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), np.int32)
    return np.stack(
        [
            np.floor(boxes[:, 1]).clip(0, height),
            np.floor(boxes[:, 0]).clip(0, width),
            np.ceil(boxes[:, 3]).clip(0, height),
            np.ceil(boxes[:, 2]).clip(0, width),
        ],
        axis=1,
    ).astype(np.int32)


def full_size_masks(probs: np.ndarray, boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bring masks predicted on a shrunk image back to the image's size, box by box.

    Only each box's region is resized (bilinear on the probabilities, then thresholded), and the
    result is boolean: memory stays proportional to the masks, not to ``N`` float images.

    Parameters
    ----------
    probs : np.ndarray
        ``[N, h, w]`` mask probabilities on the shrunk image.
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` int32 boxes in the full image (see ``to_yxyx``).
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
