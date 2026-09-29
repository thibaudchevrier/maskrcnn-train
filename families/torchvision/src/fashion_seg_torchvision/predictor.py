"""torchvision Mask R-CNN exports (``config.json`` + ``model.pt``) as a serving ``Predictor``.

Large photos are shrunk to the model's ``max_size`` before the network, and each mask is brought
back to the photo's size on the CPU, inside its box only: torchvision's own post-processing would
paste every mask at full resolution in float32 on the device (gigabytes for a 20-megapixel photo).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fashion_seg_torch.device import pick_device
from PIL import Image

from fashion_seg.data.images import full_size_masks, to_yxyx
from fashion_seg.ports import Detections
from fashion_seg_torchvision.network import build_model, to_tensor


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
