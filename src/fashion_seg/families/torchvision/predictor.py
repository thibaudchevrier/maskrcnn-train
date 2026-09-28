"""torchvision Mask R-CNN exports (``config.json`` + ``model.pt``) as a serving ``Predictor``."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fashion_seg.families.torchvision.network import build_model, pick_device, to_tensor
from fashion_seg.ports import Detections

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
        output = self._model([to_tensor(image).to(self.device)])[0]
        boxes = output["boxes"].cpu().numpy()  # (x1, y1, x2, y2), float
        height, width = image.shape[:2]
        return Detections(
            boxes=np.stack(
                [
                    np.floor(boxes[:, 1]).clip(0, height),
                    np.floor(boxes[:, 0]).clip(0, width),
                    np.ceil(boxes[:, 3]).clip(0, height),
                    np.ceil(boxes[:, 2]).clip(0, width),
                ],
                axis=1,
            ).astype(np.int32)
            if len(boxes)
            else np.zeros((0, 4), np.int32),
            class_ids=output["labels"].cpu().numpy().astype(np.int32),
            scores=output["scores"].cpu().numpy().astype(np.float32),
            masks=(output["masks"][:, 0] > MASK_THRESHOLD).cpu().numpy().transpose(1, 2, 0)
            if len(boxes)
            else np.zeros((height, width, 0), dtype=bool),
        )
