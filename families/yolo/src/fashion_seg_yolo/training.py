"""Fine-tune YOLO segmentation from COCO with Ultralytics' trainer, then export it for serving.

Ultralytics runs its own loop (augmentation, loss, EMA weights, validation with mask mAP);
callbacks plug the pipeline's conventions into it:

- losses and learning rate logged every ``log_every`` batches, validation metrics each epoch;
- a checkpoint every ``checkpoint_every`` batches besides Ultralytics' own at each epoch's end;
- a stop request saves a checkpoint and ends the run (``TrainResult.stopped_at``).

Ultralytics' data order cannot be replayed from the middle of an epoch: a checkpoint taken
mid-epoch resumes at the start of that epoch, with the weights and optimizer state reached (its
images are seen again). Steps count batches. Writes to the directories the service gives:

- ``checkpoint_dir / "dataset"``: the images and polygons in Ultralytics' format (``dataset``);
- ``checkpoint_dir / "runs/train"``: Ultralytics' run (``weights/last.pt`` resumes it);
- ``export_dir``: ``model.pt`` (the last epoch's EMA weights) and ``fashion_seg.json``.
"""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import ultralytics
from fashion_seg_torch.device import pick_device
from ultralytics.models.yolo.segment import SegmentationTrainer

from fashion_seg.ports import TrainingSession, TrainInputs, TrainResult
from fashion_seg_yolo.config import Config
from fashion_seg_yolo.dataset import write_dataset
from fashion_seg_yolo.isolation import disable_integrations

EXPORT_CONFIG = "fashion_seg.json"
EXPORT_WEIGHTS = "model.pt"


class Stopped(Exception):
    """Raised from a callback to end Ultralytics' run after a stop request (checkpoint saved)."""


class FashionTrainer(SegmentationTrainer):
    """Ultralytics' segmentation trainer, resuming from a checkpoint taken in the first epoch.

    Ultralytics refuses to resume a checkpoint of epoch -1, which is what a checkpoint taken
    during the first epoch is (see ``Callbacks.checkpoint``).
    """

    def resume_training(self, ckpt: dict[str, Any] | None) -> None:
        """Restore the optimizer, EMA and best fitness, and start at the checkpoint's next epoch.

        Parameters
        ----------
        ckpt : dict[str, Any] | None
            The checkpoint Ultralytics loaded, ``None`` for a new run.
        """
        if ckpt is None or not self.resume:
            return
        self._load_checkpoint_state(ckpt)  # pylint: disable=protected-access  # Ultralytics' own restore
        self.start_epoch = ckpt["epoch"] + 1


def metric_name(key: str) -> str:
    """Turn an Ultralytics metric name into a valid MLflow one.

    Parameters
    ----------
    key : str
        Ultralytics' name.

    Returns
    -------
    str
        The name without the characters MLflow rejects.

    Examples
    --------
    >>> metric_name("metrics/mAP50-95(M)")
    'metrics/mAP50-95_M'
    """
    return re.sub(r"[^\w\-./ ]+", "_", key).strip("_")


def epoch_metrics(trainer: SegmentationTrainer) -> dict[str, float]:
    """Collect the metrics of the epoch that just ended: training losses and validation.

    Parameters
    ----------
    trainer : SegmentationTrainer
        The trainer, after validation.

    Returns
    -------
    dict[str, float]
        ``train/<loss>``, ``val/<loss>``, ``metrics/<metric>``, and the totals ``train_loss``
        and ``val_loss`` (comparable with the other families' losses).
    """
    train = trainer.label_loss_items(trainer.tloss or {}, prefix="train")
    metrics = {metric_name(k): float(v) for k, v in {**train, **trainer.metrics}.items()}
    metrics["train_loss"] = sum(v for k, v in metrics.items() if k.startswith("train/"))
    metrics["val_loss"] = sum(v for k, v in metrics.items() if k.startswith("val/"))
    return metrics


@dataclass
class Callbacks:
    """Ultralytics callbacks logging to the session, checkpointing and stopping.

    Attributes
    ----------
    config : Config
        Training parameters (``log_every``, ``checkpoint_every``).
    session : TrainingSession
        Records the metrics; tells when to stop.
    step : int
        Batches trained, from the start of the run.
    metrics : dict[str, float] | None
        Metrics of the last completed epoch, if any.
    """

    config: Config
    session: TrainingSession
    step: int = 0
    metrics: dict[str, float] | None = None

    def register(self, trainer: SegmentationTrainer) -> None:
        """Attach the callbacks to a trainer.

        Parameters
        ----------
        trainer : SegmentationTrainer
            The trainer.
        """
        trainer.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        trainer.add_callback("on_train_batch_end", self.on_train_batch_end)
        trainer.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def on_train_epoch_start(self, trainer: SegmentationTrainer) -> None:
        """Position the step counter at the epoch's start.

        Parameters
        ----------
        trainer : SegmentationTrainer
            The trainer.
        """
        self.step = trainer.epoch * len(trainer.train_loader)

    def on_train_batch_end(self, trainer: SegmentationTrainer) -> None:
        """Log, checkpoint, or stop after a batch.

        Parameters
        ----------
        trainer : SegmentationTrainer
            The trainer.

        Raises
        ------
        Stopped
            When a stop was requested (checkpoint saved).
        """
        self.step += 1
        if self.step % self.config.log_every == 0 and trainer.tloss:
            losses = trainer.label_loss_items(trainer.tloss, prefix="train")
            metrics = {metric_name(k): float(v) for k, v in losses.items()}
            metrics["train_loss"] = sum(metrics.values())
            metrics["learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
            self.session.log_metrics(metrics, step=self.step)
        if self.session.stop.is_set():
            self.checkpoint(trainer)
            raise Stopped(self.step)
        if self.step % self.config.checkpoint_every == 0:
            self.checkpoint(trainer)

    def on_fit_epoch_end(self, trainer: SegmentationTrainer) -> None:
        """Log the epoch's metrics (not the final re-validation of ``best.pt``).

        Parameters
        ----------
        trainer : SegmentationTrainer
            The trainer.
        """
        if trainer.epoch >= trainer.epochs:  # Ultralytics' final validation of best.pt
            return
        self.metrics = epoch_metrics(trainer)
        self.session.log_metrics({**self.metrics, "epoch": trainer.epoch + 1}, step=self.step)

    @staticmethod
    def checkpoint(trainer: SegmentationTrainer) -> None:
        """Save ``last.pt`` in the middle of an epoch, to resume at that epoch's start.

        Ultralytics resumes after the checkpoint's epoch: saving it as the previous one makes
        the interrupted epoch start over, with the weights reached.

        Parameters
        ----------
        trainer : SegmentationTrainer
            The trainer.
        """
        epoch = trainer.epoch
        trainer.epoch = epoch - 1
        try:
            trainer.save_model()
        finally:
            trainer.epoch = epoch


def device_name(name: str) -> str:
    """Resolve the configured device into Ultralytics' spelling.

    Parameters
    ----------
    name : str
        ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Returns
    -------
    str
        ``"0"`` (first CUDA GPU), ``"mps"`` or ``"cpu"``.

    Examples
    --------
    >>> device_name("cpu")
    'cpu'
    """
    device = pick_device(name).type
    return "0" if device == "cuda" else device


def final_metrics(last: Path) -> dict[str, float] | None:
    """Read the metrics of a completed run (Ultralytics strips the optimizer at the end).

    Parameters
    ----------
    last : Path
        Ultralytics' ``weights/last.pt``.

    Returns
    -------
    dict[str, float] | None
        The last epoch's validation metrics, with their total ``val_loss``; ``None`` if the run
        did not complete.
    """
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    if ckpt.get("optimizer") is not None:
        return None
    metrics = {metric_name(k): float(v) for k, v in ckpt["train_metrics"].items()}
    metrics["val_loss"] = sum(v for k, v in metrics.items() if k.startswith("val/"))
    return metrics


def overrides(config: Config, data: Path, run_dir: Path, last: Path) -> dict[str, Any]:
    """Build Ultralytics' training arguments: a new run, or the resumption of ``last``.

    Parameters
    ----------
    config : Config
        Training parameters.
    data : Path
        The dataset's ``data.yaml``.
    run_dir : Path
        Ultralytics' run directory.
    last : Path
        Its checkpoint.

    Returns
    -------
    dict[str, Any]
        Arguments of ``FashionTrainer``. On resume, Ultralytics restores the others from the
        checkpoint.
    """
    common = {
        "data": str(data),
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "device": device_name(config.device),
        "workers": config.num_workers,
        "plots": False,
    }
    if config.resume and last.exists():
        return {**common, "model": str(last), "resume": str(last)}
    pretrained = config.init_weights.exists()
    return {
        **common,
        "model": str(config.init_weights) if pretrained else f"{config.model}.yaml",
        "pretrained": pretrained,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch_size,
        "lr0": config.learning_rate,
        "optimizer": config.optimizer,
        "seed": config.seed,
    }


def export_model(last: Path, config: Config, num_labels: int, export_dir: Path) -> None:
    """Write the trained model for serving: ``model.pt`` and ``fashion_seg.json``.

    Parameters
    ----------
    last : Path
        Ultralytics' final ``last.pt`` (EMA weights, optimizer stripped).
    config : Config
        Training parameters (architecture, image size).
    num_labels : int
        Number of classes.
    export_dir : Path
        Directory to (re)create with the export.
    """
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    shutil.copyfile(last, export_dir / EXPORT_WEIGHTS)
    export_config = {
        "architecture": config.model,
        "imgsz": config.imgsz,
        "num_labels": num_labels,
        "ultralytics_version": ultralytics.__version__,
    }
    (export_dir / EXPORT_CONFIG).write_text(
        json.dumps(export_config, indent=2) + "\n", encoding="utf-8"
    )


def train(config: Config, inputs: TrainInputs, session: TrainingSession) -> TrainResult:
    """Train (resuming from the last checkpoint if asked), then export.

    Parameters
    ----------
    config : Config
        Training parameters.
    inputs : TrainInputs
        Data, and where to write.
    session : TrainingSession
        Records the metrics; tells when to stop.

    Returns
    -------
    TrainResult
        Metrics of the last epoch; architecture, framework versions and device. If stopped
        early: ``stopped_at``, the checkpoint saved and no export.
    """
    data = write_dataset(inputs, config.imgsz, config.num_workers)
    run_dir = inputs.checkpoint_dir / "runs" / "train"
    last = run_dir / "weights" / "last.pt"
    tags = {
        "architecture": config.model,
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "device": pick_device(config.device).type,
    }
    callbacks = Callbacks(config, session)
    if config.resume and last.exists():
        callbacks.metrics = final_metrics(last)  # nothing left to train if the run completed
    if callbacks.metrics is None:
        disable_integrations()  # before the trainer adds its callbacks
        trainer = FashionTrainer(overrides=overrides(config, data, run_dir, last))
        callbacks.register(trainer)
        try:
            trainer.train()
        except Stopped:
            return TrainResult(
                metrics=callbacks.metrics or {}, tags=tags, stopped_at=callbacks.step
            )
    export_model(last, config, len(inputs.class_names) - 1, inputs.export_dir)
    return TrainResult(metrics=callbacks.metrics or {}, tags=tags)
