"""YOLO exports (``model.pt`` + ``fashion_seg.json``) as a serving ``Predictor``.

Ultralytics letterboxes the image to ``imgsz`` and predicts masks on that grid; they come back to
full size on the CPU, box by box, from the region the image covers (without the padding).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from fashion_seg_torch.device import pick_device
from ultralytics import YOLO

from fashion_seg.data.images import full_size_masks, to_yxyx
from fashion_seg.ports import Detections

EXPORT_CONFIG = "fashion_seg.json"
EXPORT_WEIGHTS = "model.pt"
# Keep low-confidence detections, as the other families do: mAP needs the whole ranking.
MIN_SCORE = 0.001
MAX_DETECTIONS = 100


class YoloPredictor:
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
    device : str
        The device the model runs on.
    """

    config: dict[str, Any]
    device: str

    def __init__(self, model_dir: str | Path, device: str = "auto"):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / EXPORT_CONFIG).read_text(encoding="utf-8"))
        self.device = pick_device(device).type
        self._model = YOLO(model_dir / EXPORT_WEIGHTS, task="segment")

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
        [result] = self._model.predict(
            np.ascontiguousarray(image[..., ::-1]),  # Ultralytics expects OpenCV's BGR
            imgsz=self.config["imgsz"],
            conf=MIN_SCORE,
            max_det=MAX_DETECTIONS,
            device="0" if self.device == "cuda" else self.device,
            verbose=False,
        )
        if result.masks is None:
            return Detections(
                boxes=np.zeros((0, 4), np.int32),
                class_ids=np.zeros(0, np.int32),
                scores=np.zeros(0, np.float32),
                masks=np.zeros((height, width, 0), bool),
            )
        boxes = to_yxyx(result.boxes.xyxy.cpu().numpy(), height, width)
        masks = result.masks.data.cpu().numpy().astype(np.float32)
        rows, cols = letterbox_content(masks.shape[1:], (height, width))
        return Detections(
            boxes=boxes,
            class_ids=result.boxes.cls.cpu().numpy().astype(np.int32) + 1,
            scores=result.boxes.conf.cpu().numpy().astype(np.float32),
            masks=full_size_masks(masks[:, rows, cols], boxes, (height, width)),
        )


def letterbox_content(grid: tuple[int, int], size: tuple[int, int]) -> tuple[slice, slice]:
    """Find where the image lies in Ultralytics' letterboxed grid (centred, padded around).

    Parameters
    ----------
    grid : tuple[int, int]
        ``(height, width)`` of the letterboxed grid.
    size : tuple[int, int]
        ``(height, width)`` of the image.

    Returns
    -------
    tuple[slice, slice]
        Rows and columns of the grid covered by the image (as in Ultralytics'
        ``ops.scale_masks``).

    Examples
    --------
    >>> letterbox_content((640, 640), (1000, 500))
    (slice(0, 640, None), slice(160, 480, None))
    """
    (grid_h, grid_w), (height, width) = grid, size
    gain = min(grid_h / height, grid_w / width)
    content_h, content_w = round(height * gain), round(width * gain)
    top, left = round((grid_h - content_h) / 2 - 0.1), round((grid_w - content_w) / 2 - 0.1)
    return slice(top, top + content_h), slice(left, left + content_w)
