"""Build Mask2Former (Swin-Tiny backbone) for the fashion classes; image normalization.

Shared by training and serving: only torch, transformers and numpy.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig

ARCHITECTURE = "mask2former-swin-tiny"
# ImageNet statistics, as Mask2Former's image processor normalizes.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def swin_tiny_config(labels: list[str]) -> Mask2FormerConfig:
    """Describe Mask2Former with a Swin-Tiny backbone, like the COCO instance checkpoint.

    Parameters
    ----------
    labels : list[str]
        Class names, in the model's class order (dataset categories).

    Returns
    -------
    Mask2FormerConfig
        The configuration.
    """
    # pylint: disable-next=unexpected-keyword-arg  # transformers configs take fields as keyword arguments, built at run time
    backbone = SwinConfig(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    # pylint: disable-next=unexpected-keyword-arg  # transformers configs take fields as keyword arguments, built at run time
    return Mask2FormerConfig(
        backbone_config=backbone,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )


def build_model(
    labels: list[str],
    pretrained: Path | None = None,
    config: Mask2FormerConfig | None = None,
) -> Mask2FormerForUniversalSegmentation:
    """Build Mask2Former for the given classes.

    Parameters
    ----------
    labels : list[str]
        Class names, in the model's class order (dataset categories, background excluded).
    pretrained : Path | None
        A Hugging Face checkpoint directory (``config.json`` + weights) to start from; its class
        head is replaced to fit ``labels``. ``None`` builds random weights. By default ``None``.
    config : Mask2FormerConfig | None
        Architecture of the random model (tests use a tiny one). By default ``None``: Swin-Tiny.

    Returns
    -------
    Mask2FormerForUniversalSegmentation
        The model, on the CPU.
    """
    if pretrained is not None:
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained,
            id2label=dict(enumerate(labels)),
            label2id={label: i for i, label in enumerate(labels)},
            ignore_mismatched_sizes=True,  # the class head: COCO's 80 classes -> ours
        )
    return Mask2FormerForUniversalSegmentation(config or swin_tiny_config(labels))


def to_pixels(image: np.ndarray) -> torch.Tensor:
    """Convert an RGB image to the normalized tensor Mask2Former expects.

    Parameters
    ----------
    image : np.ndarray
        ``[H, W, 3]`` uint8 image.

    Returns
    -------
    torch.Tensor
        ``[3, H, W]`` float32, normalized with ``IMAGE_MEAN`` and ``IMAGE_STD``.

    Examples
    --------
    >>> to_pixels(np.zeros((2, 3, 3), np.uint8)).shape
    torch.Size([3, 2, 3])
    """
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (pixels - mean) / std
