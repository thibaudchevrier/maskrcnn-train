"""Fine-tune Mask R-CNN v2 from COCO: data loaders, training loop, checkpoints, export.

Writes to the directories the service gives (``TrainInputs``):

- ``export_dir``: ``config.json`` + ``model.pt`` (``state_dict``), what ``predictor`` serves;
- ``checkpoint_dir / "last.pt"``: model, optimizer, schedule and position, every
  ``checkpoint_every`` steps and at the end of each epoch, to resume.

A run can stop at any step and resume there exactly: each epoch's shuffle order is reproducible
(``fashion_seg.progress.epoch_order``), so a resumed run skips the images the stopped one had
trained on. When the session's stop signal is set, the current step finishes, a checkpoint is
saved and ``train`` returns where it stopped.
"""

import json
import logging
import math
import shutil
import signal
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Sampler
from torchvision.models.detection import MaskRCNN

from fashion_seg.ports import MetricLogger, StopSignal, TrainingSession, TrainInputs, TrainResult
from fashion_seg.progress import epoch_order
from fashion_seg_torchvision.config import Config
from fashion_seg_torchvision.dataset import FashionDataset, collate
from fashion_seg_torchvision.network import ARCHITECTURE, build_model, pick_device

logger = logging.getLogger(__name__)


class EpochSampler(Sampler[int]):
    """Shuffle the images the same way for a given epoch, from any position in it.

    Parameters
    ----------
    size : int
        Number of images.
    seed : int
        Training seed (see ``fashion_seg.progress.epoch_order``).

    Attributes
    ----------
    size : int
        Number of images.
    seed : int
        Shuffle seed.
    epoch : int
        Epoch whose order is produced.
    start : int
        Position in that order to start from (images already trained on are skipped).
    """

    size: int
    seed: int
    epoch: int
    start: int

    def __init__(self, size: int, seed: int):
        super().__init__()
        self.size, self.seed = size, seed
        self.epoch = self.start = 0

    def set_position(self, epoch: int, start: int) -> None:
        """Select the epoch and the position to iterate from.

        Parameters
        ----------
        epoch : int
            Epoch, from 0.
        start : int
            Number of images of the epoch to skip.
        """
        self.epoch, self.start = epoch, start

    def __iter__(self) -> Iterator[int]:
        """Iterate over the epoch's shuffled image indices, from ``start``.

        Returns
        -------
        Iterator[int]
            Dataset indices.
        """
        return iter(epoch_order(self.size, self.seed, self.epoch)[self.start :])

    def __len__(self) -> int:
        """Count the images left in the epoch.

        Returns
        -------
        int
            ``size - start``.
        """
        return max(0, self.size - self.start)


def _ignore_interrupts(worker_id: int) -> None:  # pylint: disable=unused-argument  # torch's hook
    """Let data-loading workers ignore Ctrl+C: the main process stops the run cleanly.

    Parameters
    ----------
    worker_id : int
        Worker index, given by torch.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_loaders(config: Config, inputs: TrainInputs) -> tuple[DataLoader, DataLoader]:
    """Build the training loader (``EpochSampler`` order) and the validation loader.

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
    for records, sampler in (
        (inputs.train, EpochSampler(inputs.train.height, config.seed)),
        (inputs.val, None),
    ):
        loaders.append(
            DataLoader(
                FashionDataset(records, inputs.image_dir, config.max_size),
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=config.num_workers,
                persistent_workers=config.num_workers > 0,
                collate_fn=collate,
                worker_init_fn=_ignore_interrupts,
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
    """Everything a training run updates, and checkpoints.

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
            Checkpoint file; its directory is created if needed. Written to a temporary file
            first, so an interruption never leaves a corrupt checkpoint.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "val_losses": self.val_losses,
            },
            partial,
        )
        partial.replace(path)

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


@dataclass(frozen=True)
class Loop:
    """How the training loop reports, checkpoints and stops.

    Attributes
    ----------
    log_metrics : MetricLogger
        Records the losses and the learning rate.
    log_every : int
        Steps between two logs.
    checkpoint : Path
        Checkpoint file.
    checkpoint_every : int
        Steps between two checkpoints.
    stop : StopSignal
        Set when the run must stop.
    """

    log_metrics: MetricLogger
    log_every: int
    checkpoint: Path
    checkpoint_every: int
    stop: StopSignal


def train_step(state: TrainingState, images: list, targets: list) -> dict[str, float]:
    """Run one optimizer step.

    Parameters
    ----------
    state : TrainingState
        Model, optimizer, schedule and counters; updated in place.
    images : list
        Images of the batch.
    targets : list
        Targets of the batch.

    Returns
    -------
    dict[str, float]
        The step's losses, including their sum ``loss``.
    """
    images, targets = to_device(images, targets, state.device)
    losses = state.model(images, targets)
    loss = sum(losses.values())
    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    state.scheduler.step()
    state.global_step += 1
    return {name: value.detach().item() for name, value in {**losses, "loss": loss}.items()}


def train_one_epoch(state: TrainingState, loader: DataLoader, loop: Loop) -> bool:
    """Train until the end of the epoch, or until a stop is requested.

    Logs the average losses every ``log_every`` steps and saves a checkpoint every
    ``checkpoint_every`` steps, and when stopping.

    Parameters
    ----------
    state : TrainingState
        Model, optimizer, schedule and counters; updated in place.
    loader : DataLoader
        Training loader, positioned where the epoch resumes.
    loop : Loop
        Logging, checkpointing and stopping.

    Returns
    -------
    bool
        ``True`` if the epoch completed, ``False`` if it stopped early (checkpoint saved).
    """
    state.model.train()
    running: dict[str, float] = defaultdict(float)
    for images, targets in loader:
        for name, value in train_step(state, images, targets).items():
            running[name] += value
        if state.global_step % loop.log_every == 0:
            metrics = {f"train_{k}": v / loop.log_every for k, v in running.items()}
            metrics["learning_rate"] = state.scheduler.get_last_lr()[0]
            loop.log_metrics(metrics, step=state.global_step)
            logger.info("step %d: loss %.4f", state.global_step, metrics["train_loss"])
            running.clear()
        if loop.stop.is_set():
            state.save(loop.checkpoint)
            return False
        if state.global_step % loop.checkpoint_every == 0:
            state.save(loop.checkpoint)
    state.epoch += 1
    return True


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
    train_loader, val_loader = build_loaders(config, inputs)
    steps_per_epoch = math.ceil(inputs.train.height / config.batch_size)
    state = init_state(config, len(inputs.class_names), steps_per_epoch)
    checkpoint = inputs.checkpoint_dir / "last.pt"
    if config.resume and checkpoint.exists():
        state.restore(checkpoint)
        logger.info("Resuming at epoch %d, step %d", state.epoch, state.global_step)
    tags = {
        "architecture": ARCHITECTURE,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": state.device.type,
    }

    loop = Loop(
        session.log_metrics, config.log_every, checkpoint, config.checkpoint_every, session.stop
    )
    while state.epoch < config.epochs:
        # Steps already done in this epoch (all of them if a stop came on its last step: the
        # epoch then completes at once and is validated).
        done = state.global_step - state.epoch * steps_per_epoch
        train_loader.sampler.set_position(state.epoch, done * config.batch_size)
        if not train_one_epoch(state, train_loader, loop):
            return TrainResult(metrics=state.val_losses, tags=tags, stopped_at=state.global_step)
        state.val_losses = validate(state.model, val_loader, state.device)
        session.log_metrics({**state.val_losses, "epoch": state.epoch}, step=state.global_step)
        state.save(checkpoint)
        logger.info("epoch %d: val loss %.4f", state.epoch, state.val_losses["val_loss"])
    export_model(state.model, config, len(inputs.class_names), inputs.export_dir)
    return TrainResult(metrics=state.val_losses, tags=tags)
