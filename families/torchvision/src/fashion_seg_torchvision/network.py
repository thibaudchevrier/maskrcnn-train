"""Build Mask R-CNN v2 for the fashion classes; image helper.

Shared by training and serving: only torch, torchvision and numpy.
"""

from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ARCHITECTURE = "maskrcnn_resnet50_fpn_v2"
COCO_CLASSES = 91  # torchvision's COCO head size, background included


def build_model(
    num_classes: int, min_size: int, max_size: int, coco_weights: Path | None = None
) -> MaskRCNN:
    """Build Mask R-CNN v2 with box and mask heads for ``num_classes`` classes.

    Parameters
    ----------
    num_classes : int
        Number of classes, background included (47 for iMaterialist).
    min_size : int
        Images are resized so their short side is ``min_size``...
    max_size : int
        ...without the long side exceeding ``max_size``.
    coco_weights : Path | None
        torchvision's COCO checkpoint to start from; the class-specific heads are then replaced.
        ``None`` builds random weights (to load a trained ``state_dict`` afterwards).
        By default ``None``.

    Returns
    -------
    MaskRCNN
        The model, on the CPU.
    """
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        num_classes=COCO_CLASSES if coco_weights else num_classes,
        min_size=min_size,
        max_size=max_size,
    )
    if coco_weights:
        model.load_state_dict(torch.load(coco_weights, map_location="cpu", weights_only=True))
        box_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
        mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an RGB image to the float tensor torchvision models expect.

    Parameters
    ----------
    image : np.ndarray
        ``[H, W, 3]`` uint8 image.

    Returns
    -------
    torch.Tensor
        ``[3, H, W]`` float32 image in [0, 1].
    """
    # np.array copies: arrays from PIL are read-only, which torch.from_numpy doesn't support
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
