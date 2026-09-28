"""Training parameters of the matterport family (``params.yaml:train.matterport``)."""

from typing import Any, ClassVar, Literal

from fashion_seg.config import TrainConfig


class Config(TrainConfig):
    """Training parameters (``params.yaml:train.matterport``), besides ``TrainConfig``'s.

    Defaults in ``params.yaml`` follow the 2021 notebook.

    Attributes
    ----------
    backbone : Literal["resnet50", "resnet101"]
        ResNet backbone.
    image_min_dim : int
        Images are resized so their short side is at least ``image_min_dim``...
    image_max_dim : int
        ...and their long side at most ``image_max_dim``.
    images_per_gpu : int
        Batch size.
    rpn_anchor_scales : tuple[int, ...]
        Anchor side lengths, one per feature pyramid level.
    train_rois_per_image : int
        Regions of interest sampled per image for the heads.
    learning_rate : float
        SGD learning rate.
    layers : Literal["heads", "3+", "4+", "5+", "all"]
        Layers to train.
    steps_per_epoch : int | None
        ``None``: one pass over the training images.
    validation_steps : int | None
        ``None``: one pass over the validation images.
    detection_min_confidence : float
        Detections below this confidence are dropped inside the exported model.
    smoke_overrides : ClassVar[dict[str, Any]]
        Parameters replaced for a smoke run.
    """

    backbone: Literal["resnet50", "resnet101"]
    image_min_dim: int
    image_max_dim: int
    images_per_gpu: int
    rpn_anchor_scales: tuple[int, ...]
    train_rois_per_image: int
    learning_rate: float
    layers: Literal["heads", "3+", "4+", "5+", "all"]
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    detection_min_confidence: float
    smoke_overrides: ClassVar[dict[str, Any]] = {
        "backbone": "resnet50",
        "image_min_dim": 256,
        "image_max_dim": 256,
        "images_per_gpu": 1,
        "train_rois_per_image": 16,
        "epochs": 1,
        "steps_per_epoch": 2,
        "validation_steps": 1,
        "max_train_images": 4,
        "max_val_images": 2,
        "log_every": 1,
    }
