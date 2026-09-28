"""Train Matterport Mask R-CNN (TensorFlow 2.15, ``families/matterport``) and export it for serving.

Writes to the directories the service gives (``TrainInputs``):

- ``export_dir``: inference model as ``config.json`` + TF SavedModel, what ``predictor`` serves;
- ``checkpoint_dir``: Keras checkpoints of every epoch.
"""

import shutil
from importlib.metadata import version
from pathlib import Path

import keras  # Keras 2.15, the same package as tf.keras in TensorFlow 2.15
import tensorflow as tf
from mrcnn import config as mconfig
from mrcnn import model as modellib

from fashion_seg.ports import MetricLogger, TrainInputs, TrainResult
from fashion_seg_matterport.config import Config
from fashion_seg_matterport.dataset import FashionDataset

# Layers re-initialised when starting from COCO: they depend on the number of classes.
HEAD_LAYERS = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]


class EpochLogger(keras.callbacks.Callback):
    """Keras callback recording every epoch's metrics (losses, validation losses).

    Parameters
    ----------
    log_metrics : MetricLogger
        Records the metrics.

    Attributes
    ----------
    last_logs : dict[str, float]
        Metrics of the last finished epoch.
    """

    last_logs: dict[str, float]

    def __init__(self, log_metrics: MetricLogger) -> None:
        super().__init__()
        self._log_metrics = log_metrics
        self.last_logs = {}

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Record the epoch's metrics.

        Parameters
        ----------
        epoch : int
            0-based epoch index (recorded as step ``epoch + 1``).
        logs : dict[str, float] | None
            Metrics computed by Keras for the epoch. By default ``None``.
        """
        self.last_logs = {key: float(value) for key, value in (logs or {}).items()}
        self._log_metrics(self.last_logs, step=epoch + 1)


def build_config(
    config: Config,
    num_classes: int,
    batch_steps: tuple[int, int],
    *,
    inference: bool = False,
) -> mconfig.Config:
    """Build the Matterport ``Config`` from the training parameters.

    Inference uses the training anchors, unlike the 2021 export (see the README's roadmap).

    Parameters
    ----------
    config : Config
        Training parameters.
    num_classes : int
        Number of classes, background included.
    batch_steps : tuple[int, int]
        Training and validation steps per epoch.
    inference : bool
        Build the inference configuration (one image per batch). By default ``False``.

    Returns
    -------
    mconfig.Config
        The configuration.
    """
    steps_per_epoch, validation_steps = batch_steps
    attrs = {
        "NAME": "fashion",
        "NUM_CLASSES": num_classes,
        "GPU_COUNT": 1,
        "IMAGES_PER_GPU": 1 if inference else config.images_per_gpu,
        "BACKBONE": config.backbone,
        "IMAGE_MIN_DIM": config.image_min_dim,
        "IMAGE_MAX_DIM": config.image_max_dim,
        "RPN_ANCHOR_SCALES": config.rpn_anchor_scales,
        "TRAIN_ROIS_PER_IMAGE": config.train_rois_per_image,
        "LEARNING_RATE": config.learning_rate,
        "STEPS_PER_EPOCH": steps_per_epoch,
        "VALIDATION_STEPS": validation_steps,
        "DETECTION_MIN_CONFIDENCE": config.detection_min_confidence,
    }
    return type("FashionConfig", (mconfig.Config,), attrs)()


def export_inference_model(
    config: Config, num_classes: int, checkpoint: str, inputs: TrainInputs
) -> None:
    """Export a checkpoint as an inference model (``config.json`` + SavedModel).

    Parameters
    ----------
    config : Config
        Training parameters.
    num_classes : int
        Number of classes, background included.
    checkpoint : str
        Path of the ``.h5`` weights to export.
    inputs : TrainInputs
        Gives the export directory to (re)create.
    """
    inference = modellib.MaskRCNN(
        mode="inference",
        config=build_config(config, num_classes, (1, 1), inference=True),
        model_dir=str(inputs.checkpoint_dir),
    )
    inference.load_weights(checkpoint, by_name=True)
    if inputs.export_dir.exists():
        shutil.rmtree(inputs.export_dir)
    inputs.export_dir.mkdir(parents=True)
    inference.save(str(inputs.export_dir))


def train(config: Config, inputs: TrainInputs, log_metrics: MetricLogger) -> TrainResult:
    """Train from the COCO weights (if present), then export the last checkpoint.

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
        Last epoch's losses; maskrcnn-matterport and TensorFlow versions.
    """
    datasets = [
        FashionDataset(records, inputs.image_dir, inputs.class_names)
        for records in (inputs.train, inputs.val)
    ]
    batch, num_classes = config.images_per_gpu, len(inputs.class_names)
    steps = (
        config.steps_per_epoch or max(1, len(datasets[0].image_ids) // batch),
        config.validation_steps or max(1, len(datasets[1].image_ids) // batch),
    )
    inputs.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = modellib.MaskRCNN(
        mode="training",
        config=build_config(config, num_classes, steps),
        model_dir=str(inputs.checkpoint_dir),
    )
    if Path(config.init_weights).exists():
        model.load_weights(str(config.init_weights), by_name=True, exclude=HEAD_LAYERS)

    epoch_logger = EpochLogger(log_metrics)
    model.train(
        datasets[0],
        datasets[1],
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        layers=config.layers,
        augmentation=None,
        custom_callbacks=[epoch_logger],
    )
    export_inference_model(config, num_classes, model.find_last(), inputs)
    return TrainResult(
        metrics=epoch_logger.last_logs,
        tags={
            "maskrcnn_matterport_version": version("maskrcnn-matterport"),
            "tensorflow_version": tf.__version__,
        },
    )
