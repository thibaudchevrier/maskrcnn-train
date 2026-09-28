"""Fine-tune Mask R-CNN v2 from COCO: data loaders, training loop, checkpoints, export.

Writes to the directories the service gives (``TrainInputs``):

- ``export_dir``: ``config.json`` + ``model.pt`` (``state_dict``), what ``predictor`` serves;
- ``checkpoint_dir / "last.pt"``: model, optimizer and schedule after each epoch, to resume.
"""

import json
import logging
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN

from fashion_seg.ports import MetricLogger, TrainInputs, TrainResult
from fashion_seg_torchvision.config import Config
from fashion_seg_torchvision.dataset import FashionDataset, collate
from fashion_seg_torchvision.network import ARCHITECTURE, build_model, pick_device

logger = logging.getLogger(__name__)


def build_loaders(config: Config, inputs: TrainInputs) -> tuple[DataLoader, DataLoader]:
    """Build the training (shuffled) and validation loaders.

    Parameters
    ----------
    config : Config
        Training parameters.
    inputs : TrainInputs
        Training and validation annotations, and the image directory.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training loader and validation loader.
    """
    loaders = []
    for records, shuffle in ((inputs.train, True), (inputs.val, False)):
        loaders.append(
            DataLoader(
                FashionDataset(records, inputs.image_dir, config.max_size),
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                persistent_workers=config.num_workers > 0,
                collate_fn=collate,
            )
        )
    return loaders[0], loaders[1]


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier: linear warm-up, then cosine decay to zero.

    Parameters
    ----------
    step : int
        Optimizer step, from 0.
    warmup_steps : int
        Length of the warm-up.
    total_steps : int
        Total number of steps of the run.

    Returns
    -------
    float
        Factor applied to the base learning rate.

    Examples
    --------
    >>> [round(lr_factor(s, 2, 10), 2) for s in (0, 1, 2, 10)]
    [0.5, 1.0, 1.0, 0.0]
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def to_device(
    images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]], device: torch.device
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Move a batch to the device.

    Parameters
    ----------
    images : list[torch.Tensor]
        Images of the batch.
    targets : list[dict[str, torch.Tensor]]
        Targets of the batch.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]
        The batch on the device.
    """
    return (
        [image.to(device) for image in images],
        [{k: v.to(device) for k, v in t.items()} for t in targets],
    )


@dataclass
class TrainingState:
    """Everything a training run updates, and checkpoints after each epoch.

    Attributes
    ----------
    model : MaskRCNN
        The model, on ``device``.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Per-step learning-rate schedule.
    device : torch.device
        Compute device.
    epoch : int
        Epochs completed.
    global_step : int
        Optimizer steps completed.
    val_losses : dict[str, float]
        Validation losses after the last completed epoch.
    """

    model: MaskRCNN
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: torch.device
    epoch: int = 0
    global_step: int = 0
    val_losses: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Write a checkpoint that ``restore`` can resume from.

        Parameters
        ----------
        path : Path
            Checkpoint file; its directory is created if needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "val_losses": self.val_losses,
            },
            path,
        )

    def restore(self, path: Path) -> None:
        """Resume from a checkpoint written by ``save``.

        Parameters
        ----------
        path : Path
            Checkpoint file.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch, self.global_step = state["epoch"], state["global_step"]
        self.val_losses = state.get("val_losses", {})


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
    total_steps = config.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, config.warmup_steps, total_steps)
    )
    return TrainingState(model, optimizer, scheduler, device)


def train_one_epoch(
    state: TrainingState, loader: DataLoader, log_every: int, log_metrics: MetricLogger
) -> None:
    """Train for one epoch, logging the average losses every ``log_every`` steps.

    Parameters
    ----------
    state : TrainingState
        Model, optimizer, schedule and counters; updated in place.
    loader : DataLoader
        Training loader.
    log_every : int
        Steps between two logs.
    log_metrics : MetricLogger
        Records the losses and the learning rate.
    """
    state.model.train()
    running: dict[str, float] = defaultdict(float)
    for images, targets in loader:
        images, targets = to_device(images, targets, state.device)
        losses = state.model(images, targets)
        loss = sum(losses.values())
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()
        state.scheduler.step()
        state.global_step += 1
        for name, value in {**losses, "loss": loss}.items():
            running[name] += value.detach().item()
        if state.global_step % log_every == 0:
            metrics = {f"train_{k}": v / log_every for k, v in running.items()}
            metrics["learning_rate"] = state.scheduler.get_last_lr()[0]
            log_metrics(metrics, step=state.global_step)
            logger.info("step %d: loss %.4f", state.global_step, metrics["train_loss"])
            running.clear()
    state.epoch += 1


@torch.no_grad()
def validate(model: MaskRCNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute the average validation losses.

    torchvision only returns losses in training mode: batch normalization layers are switched
    to evaluation so validation images don't change their statistics.

    Parameters
    ----------
    model : MaskRCNN
        The model, on ``device``.
    loader : DataLoader
        Validation loader.
    device : torch.device
        Compute device.

    Returns
    -------
    dict[str, float]
        ``val_<loss>`` averaged over the batches.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):  # the v2 heads use BatchNorm2d
            module.eval()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for images, targets in loader:
        images, targets = to_device(images, targets, device)
        losses = model(images, targets)
        for name, value in {**losses, "loss": sum(losses.values())}.items():
            totals[f"val_{name}"] += value.item()
        batches += 1
    return {k: v / max(1, batches) for k, v in totals.items()}


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


def train(config: Config, inputs: TrainInputs, log_metrics: MetricLogger) -> TrainResult:
    """Train (resuming from the last checkpoint if asked), validating each epoch, then export.

    Parameters
    ----------
    config : Config
        Training parameters.
    inputs : TrainInputs
        Data, and where to write.
    log_metrics : MetricLogger
        Records the losses during training.

    Returns
    -------
    TrainResult
        Validation losses of the last epoch; architecture, framework versions and device.
    """
    train_loader, val_loader = build_loaders(config, inputs)
    state = init_state(config, len(inputs.class_names), len(train_loader))
    checkpoint = inputs.checkpoint_dir / "last.pt"
    if config.resume and checkpoint.exists():
        state.restore(checkpoint)
        logger.info("Resuming after epoch %d (step %d)", state.epoch, state.global_step)

    while state.epoch < config.epochs:
        train_one_epoch(state, train_loader, config.log_every, log_metrics)
        state.val_losses = validate(state.model, val_loader, state.device)
        log_metrics({**state.val_losses, "epoch": state.epoch}, step=state.global_step)
        state.save(checkpoint)
        logger.info("epoch %d: val loss %.4f", state.epoch, state.val_losses["val_loss"])
    export_model(state.model, config, len(inputs.class_names), inputs.export_dir)
    return TrainResult(
        metrics=state.val_losses,
        tags={
            "architecture": ARCHITECTURE,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": state.device.type,
        },
    )
