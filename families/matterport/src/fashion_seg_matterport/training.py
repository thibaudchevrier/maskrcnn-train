"""Train Matterport Mask R-CNN (TensorFlow 2.15, ``families/matterport``) and export it for serving.

Writes to the directories the service gives (``TrainInputs``):

- ``export_dir``: inference model as ``config.json`` + TF SavedModel, what ``predictor`` serves;
- ``checkpoint_dir``: ``last.h5`` (weights) and ``last.json`` (position), every
  ``checkpoint_every`` steps and at each epoch's end, to resume; the fork's own per-epoch files.

The fork's ``MaskRCNN.train`` runs Keras ``fit``, extended with a callback (``TrainingControl``)
that logs, checkpoints and stops. A resumed run first finishes the interrupted epoch (its
remaining steps), then continues. Two limits, both from the fork: the rest of that epoch draws a
new shuffle (the fork shuffles with numpy's global generator, seeded with ``config.seed`` at each
start), and the SGD momentum restarts (the fork saves weights only; it rebuilds in a few steps).
"""

import json
import logging
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import keras  # Keras 2.15, the same package as tf.keras in TensorFlow 2.15
import numpy as np
import tensorflow as tf
from mrcnn import config as mconfig
from mrcnn import model as modellib

from fashion_seg.ports import TrainingSession, TrainInputs, TrainResult
from fashion_seg_matterport.config import Config
from fashion_seg_matterport.dataset import FashionDataset

logger = logging.getLogger(__name__)

# Layers re-initialised when starting from COCO: they depend on the number of classes.
HEAD_LAYERS = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]


class TrainingStopped(Exception):
    """Raised by ``TrainingControl`` to leave Keras ``fit`` at once, after a checkpoint."""


@dataclass(frozen=True)
class Checkpoint:
    """Where a run's resumable state is: weights and position.

    Attributes
    ----------
    weights : Path
        Keras weights (HDF5).
    position : Path
        JSON: ``epoch`` (epochs completed) and ``global_step`` (steps completed).
    """

    weights: Path
    position: Path

    @classmethod
    def in_dir(cls, directory: Path) -> "Checkpoint":
        """Name the checkpoint files of a directory.

        Parameters
        ----------
        directory : Path
            The run's checkpoint directory.

        Returns
        -------
        Checkpoint
            ``last.h5`` and ``last.json`` in it.
        """
        return cls(directory / "last.h5", directory / "last.json")

    def read_position(self) -> tuple[int, int] | None:
        """Read where the saved run was.

        Returns
        -------
        tuple[int, int] | None
            Epochs and steps completed, or ``None`` if there is no checkpoint.
        """
        if not (self.weights.exists() and self.position.exists()):
            return None
        position = json.loads(self.position.read_text(encoding="utf-8"))
        return position["epoch"], position["global_step"]


class TrainingControl(keras.callbacks.Callback):
    """Keras callback: logs the losses, checkpoints, and stops when the session asks.

    Parameters
    ----------
    session : TrainingSession
        Records the metrics; tells when to stop.
    config : Config
        Training parameters (``log_every``, ``checkpoint_every``).
    checkpoint : Checkpoint
        Where to save.
    position : tuple[int, int]
        Epochs and steps completed before this run.

    Attributes
    ----------
    epoch : int
        Epochs completed.
    global_step : int
        Steps completed.
    last_logs : dict[str, float]
        Metrics of the last completed epoch.
    """

    epoch: int
    global_step: int
    last_logs: dict[str, float]

    def __init__(
        self,
        session: TrainingSession,
        config: Config,
        checkpoint: Checkpoint,
        position: tuple[int, int],
    ) -> None:
        super().__init__()
        self._session, self._config, self._checkpoint = session, config, checkpoint
        self.epoch, self.global_step = position
        self.last_logs = {}

    def save(self) -> None:
        """Write the weights, then the position (so the position never runs ahead of them)."""
        partial = self._checkpoint.weights.with_suffix(".partial.h5")
        self.model.save_weights(str(partial))
        partial.replace(self._checkpoint.weights)
        self._checkpoint.position.write_text(
            json.dumps({"epoch": self.epoch, "global_step": self.global_step}), encoding="utf-8"
        )

    def on_train_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
        """Count the step; log, checkpoint or stop.

        Parameters
        ----------
        batch : int
            Step index within the epoch.
        logs : dict[str, float] | None
            Losses averaged since the epoch started, computed by Keras. By default ``None``.

        Raises
        ------
        TrainingStopped
            When the session asks to stop, after saving a checkpoint.
        """
        self.global_step += 1
        if self.global_step % self._config.log_every == 0:
            losses = {f"train_{k}": float(v) for k, v in (logs or {}).items() if "loss" in k}
            self._session.log_metrics(losses, step=self.global_step)
        if self._session.stop.is_set():
            self.save()
            raise TrainingStopped(self.global_step)
        if self.global_step % self._config.checkpoint_every == 0:
            self.save()

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Record the epoch's losses and validation losses, and checkpoint.

        Parameters
        ----------
        epoch : int
            0-based epoch index.
        logs : dict[str, float] | None
            Metrics computed by Keras for the epoch. By default ``None``.
        """
        self.epoch = epoch + 1
        self.last_logs = {key: float(value) for key, value in (logs or {}).items()}
        self._session.log_metrics({**self.last_logs, "epoch": self.epoch}, step=self.global_step)
        self.save()


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


def fit(
    model: modellib.MaskRCNN, datasets: list, config: Config, callback: TrainingControl
) -> None:
    """Run the remaining epochs, starting with the rest of an interrupted one.

    Parameters
    ----------
    model : modellib.MaskRCNN
        The model, in training mode.
    datasets : list
        Training and validation ``FashionDataset``.
    config : Config
        Training parameters.
    callback : TrainingControl
        Logs, checkpoints and stops; knows where the run is.
    """
    steps_per_epoch = model.config.STEPS_PER_EPOCH
    kwargs = {
        "learning_rate": config.learning_rate,
        "layers": config.layers,
        "augmentation": None,
        "custom_callbacks": [callback],
    }
    done = callback.global_step - callback.epoch * steps_per_epoch
    if 0 < done < steps_per_epoch:  # finish the interrupted epoch: its remaining steps
        model.epoch = callback.epoch
        model.config.STEPS_PER_EPOCH = steps_per_epoch - done
        model.train(*datasets, epochs=callback.epoch + 1, **kwargs)
        model.config.STEPS_PER_EPOCH = steps_per_epoch
    elif done >= steps_per_epoch:  # stopped on the epoch's last step: only its validation is lost
        callback.epoch += 1
    if callback.epoch < config.epochs:
        model.epoch = callback.epoch
        model.train(*datasets, epochs=config.epochs, **kwargs)


def train(config: Config, inputs: TrainInputs, session: TrainingSession) -> TrainResult:
    """Train from the last checkpoint (or the COCO weights), then export the trained model.

    Parameters
    ----------
    config : Config
        Training parameters.
    inputs : TrainInputs
        Data, and where to write.
    session : TrainingSession
        Records the losses; tells when to stop.

    Returns
    -------
    TrainResult
        Last epoch's losses; maskrcnn-matterport and TensorFlow versions. If stopped early:
        ``stopped_at``, the checkpoint saved and no export.
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
    checkpoint = Checkpoint.in_dir(inputs.checkpoint_dir)
    position = checkpoint.read_position() if config.resume else None
    if position is not None:
        model.load_weights(str(checkpoint.weights), by_name=True)
        logger.info("Resuming at epoch %d, step %d", *position)
    elif Path(config.init_weights).exists():
        model.load_weights(str(config.init_weights), by_name=True, exclude=HEAD_LAYERS)
    np.random.seed(config.seed)  # the fork's shuffle and ROI sampling use numpy's generator

    callback = TrainingControl(session, config, checkpoint, position or (0, 0))
    tags = {
        "maskrcnn_matterport_version": version("maskrcnn-matterport"),
        "tensorflow_version": tf.__version__,
    }
    try:
        fit(model, datasets, config, callback)
    except TrainingStopped:
        return TrainResult(metrics=callback.last_logs, tags=tags, stopped_at=callback.global_step)
    export_inference_model(config, num_classes, str(checkpoint.weights), inputs)
    return TrainResult(metrics=callback.last_logs, tags=tags)
