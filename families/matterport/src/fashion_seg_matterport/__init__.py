"""Matterport Mask R-CNN (the 2021 model's architecture): a ``fashion_seg.ports.ModelFamily``.

Two uv environments run it through ``python -m fashion_seg_matterport`` (``__main__``):

- ``families/matterport``: training, on Python 3.11 with TensorFlow 2.15
  (``maskrcnn-matterport[train]``);
- ``families/matterport/serve``: packaging and evaluation, on Python 3.12 with TensorFlow 2.18+,
  which runs the exported SavedModel (and the 2021 one) as fashion-serving does.

This module declares the family and imports TensorFlow only when it trains or predicts. The model
logic is in:

- ``dataset``: prepared annotations as a Matterport ``utils.Dataset`` (training env);
- ``training``: configuration, training and export (training env);
- ``predictor``: runs an export for serving (serving env).
"""

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from fashion_seg.config import TrainConfig
from fashion_seg.ports import (
    FamilySpec,
    MetricLogger,
    Predictor,
    ServingRequirements,
    TrainInputs,
    TrainResult,
)

SPEC = FamilySpec(
    name="matterport",
    serving=ServingRequirements(
        pinned=("tensorflow",),
        released=("maskrcnn-matterport",),
    ),
)

# Export settings logged when a model is packaged.
DESCRIBED = ("BACKBONE", "NUM_CLASSES", "IMAGE_MAX_DIM", "RPN_ANCHOR_SCALES")


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
    }


def train(config: Config, inputs: TrainInputs, log_metrics: MetricLogger) -> TrainResult:
    """Train Matterport Mask R-CNN and export it (see ``training.train``).

    Parameters
    ----------
    config : Config
        Training parameters.
    inputs : TrainInputs
        Data, and where to write.
    log_metrics : MetricLogger
        Records every epoch's losses.

    Returns
    -------
    TrainResult
        Last epoch's losses, and run tags.
    """
    # pylint: disable-next=import-outside-toplevel  # TensorFlow 2.15: training environment only
    from fashion_seg_matterport import training

    return training.train(config, inputs, log_metrics)


def load_predictor(model_dir: Path) -> Predictor:
    """Load an export (``config.json`` + TF SavedModel) for serving.

    Parameters
    ----------
    model_dir : Path
        Export directory.

    Returns
    -------
    Predictor
        The loaded predictor.
    """
    # pylint: disable-next=import-outside-toplevel  # TensorFlow 2.18+: serving environment
    from fashion_seg_matterport.predictor import MatterportPredictor

    return MatterportPredictor(model_dir)


def describe(model_dir: Path) -> dict[str, Any]:
    """Read the main settings of the export's ``config.json``, logged when it is packaged.

    Parameters
    ----------
    model_dir : Path
        Export directory.

    Returns
    -------
    dict[str, Any]
        Backbone, number of classes, image size and anchor scales.
    """
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return {key: config[key] for key in DESCRIBED}
