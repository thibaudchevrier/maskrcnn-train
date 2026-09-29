"""Fine-tune Mask R-CNN v2 from COCO with the shared training loop, then export it for serving.

The loop (``fashion_seg_torch.loop.fit``) logs, checkpoints, validates, stops and resumes; this
module gives it the model, the optimizer and a ``Task`` (how to compute Mask R-CNN's losses).
Writes to the directories the service gives (``TrainInputs``):

- ``export_dir``: ``config.json`` + ``model.pt`` (``state_dict``), what ``predictor`` serves;
- ``checkpoint_dir / "last.pt"``: model, optimizer, schedule and position, every
  ``checkpoint_every`` steps and at the end of each epoch, to resume exactly.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torchvision
from fashion_seg_torch.data import make_loaders
from fashion_seg_torch.device import pick_device
from fashion_seg_torch.loop import Loop, Task, TrainingState, fit, resume, warmup_cosine
from torchvision.models.detection import MaskRCNN

from fashion_seg.ports import TrainingSession, TrainInputs, TrainResult
from fashion_seg_torchvision.config import Config
from fashion_seg_torchvision.dataset import FashionDataset, collate
from fashion_seg_torchvision.network import ARCHITECTURE, build_model


def losses(model: torch.nn.Module, batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    """Compute Mask R-CNN's losses on a batch.

    Parameters
    ----------
    model : torch.nn.Module
        The model, on ``device``.
    batch : Any
        ``(images, targets)`` as ``dataset.collate`` builds them.
    device : torch.device
        Compute device.

    Returns
    -------
    dict[str, torch.Tensor]
        torchvision's five losses (classifier, box, mask, objectness, RPN box).
    """
    images, targets = batch
    return model(
        [image.to(device) for image in images],
        [{k: v.to(device) for k, v in target.items()} for target in targets],
    )


def validation_mode(model: torch.nn.Module) -> None:
    """Prepare Mask R-CNN to compute validation losses.

    torchvision only returns losses in training mode: batch normalization layers are switched
    to evaluation so validation images don't change their statistics.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):  # the v2 heads use BatchNorm2d
            module.eval()


TASK = Task(losses=losses, validation_mode=validation_mode)


def init_state(config: Config, num_classes: int, steps_per_epoch: int) -> TrainingState:
    """Build the model (from COCO if the weights are there), optimizer and schedule.

    Parameters
    ----------
    config : Config
        Training parameters.
    num_classes : int
        Number of classes, background included.
    steps_per_epoch : int
        Optimizer steps per epoch.

    Returns
    -------
    TrainingState
        The initial state, on the configured device.
    """
    device = pick_device(config.device)
    model = build_model(
        num_classes,
        config.min_size,
        config.max_size,
        coco_weights=config.init_weights if config.init_weights.exists() else None,
    ).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = warmup_cosine(optimizer, config.warmup_steps, config.epochs * steps_per_epoch)
    return TrainingState(model, optimizer, scheduler, device)


def export_model(model: MaskRCNN, config: Config, num_classes: int, export_dir: Path) -> None:
    """Write the trained model for serving: ``config.json`` + ``model.pt``.

    Parameters
    ----------
    model : MaskRCNN
        The trained model.
    config : Config
        Training parameters (image sizes).
    num_classes : int
        Number of classes, background included.
    export_dir : Path
        Directory to (re)create with the export.
    """
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    export_config = {
        "architecture": ARCHITECTURE,
        "num_classes": num_classes,
        "min_size": config.min_size,
        "max_size": config.max_size,
        "torchvision_version": torchvision.__version__,
    }
    (export_dir / "config.json").write_text(
        json.dumps(export_config, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, export_dir / "model.pt")


def train(config: Config, inputs: TrainInputs, session: TrainingSession) -> TrainResult:
    """Train (resuming from the last checkpoint if asked), validating each epoch, then export.

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
        Validation losses of the last epoch; architecture, framework versions and device. If
        stopped early: ``stopped_at``, the checkpoint saved and no export.
    """
    datasets = (
        FashionDataset(inputs.train, inputs.image_dir, config.max_size),
        FashionDataset(inputs.val, inputs.image_dir, config.max_size),
    )
    loaders = make_loaders(datasets, collate, config.batch_size, config.num_workers, config.seed)
    steps_per_epoch = math.ceil(inputs.train.height / config.batch_size)
    state = init_state(config, len(inputs.class_names), steps_per_epoch)
    checkpoint = inputs.checkpoint_dir / "last.pt"
    if config.resume:
        resume(state, checkpoint)

    loop = Loop(
        session.log_metrics, config.log_every, checkpoint, config.checkpoint_every, session.stop
    )
    stopped_at = fit(state, TASK, loaders, loop, config.epochs)
    tags = {
        "architecture": ARCHITECTURE,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": state.device.type,
    }
    if stopped_at is not None:
        return TrainResult(metrics=state.val_losses, tags=tags, stopped_at=stopped_at)
    export_model(state.model, config, len(inputs.class_names), inputs.export_dir)
    return TrainResult(metrics=state.val_losses, tags=tags)
