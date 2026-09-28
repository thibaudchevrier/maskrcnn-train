"""The training loop: steps, logs, checkpoints, validation, stop and resume.

``fit`` trains a model for a family, which describes its batches with a ``Task``. It checkpoints
(``TrainingState``) every ``checkpoint_every`` steps and at each epoch's end, and returns early
(after a checkpoint) when the session's stop signal is set; a later ``fit`` on the restored state
resumes exactly: each epoch's order is reproducible (``fashion_seg_torch.data.EpochSampler``).
"""

import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fashion_seg.ports import MetricLogger, StopSignal

logger = logging.getLogger(__name__)

LossFunction = Callable[[torch.nn.Module, Any, torch.device], dict[str, torch.Tensor]]


def eval_mode(model: torch.nn.Module) -> None:
    """Put a model in evaluation mode (the default validation mode).

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    """
    model.eval()


@dataclass(frozen=True)
class Task:
    """What the loop needs to know about a family's model and batches.

    Attributes
    ----------
    losses : LossFunction
        Moves a batch to the device, runs the model and returns its named losses (scalars).
    validation_mode : Callable[[torch.nn.Module], None]
        Prepares the model to compute validation losses. By default ``eval_mode``.
    """

    losses: LossFunction
    validation_mode: Callable[[torch.nn.Module], None] = eval_mode


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


def warmup_cosine(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the per-step schedule of ``lr_factor``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer.
    warmup_steps : int
        Length of the warm-up.
    total_steps : int
        Total number of steps of the run.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        The schedule, stepped after each optimizer step.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup_steps, total_steps)
    )


@dataclass
class TrainingState:
    """Everything a training run updates, and checkpoints.

    Attributes
    ----------
    model : torch.nn.Module
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

    model: torch.nn.Module
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


@dataclass(frozen=True)
class Loop:
    """How the loop reports, checkpoints and stops.

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


def train_step(state: TrainingState, task: Task, batch: Any) -> dict[str, float]:
    """Run one optimizer step.

    Parameters
    ----------
    state : TrainingState
        Model, optimizer, schedule and counters; updated in place.
    task : Task
        Computes the batch's losses.
    batch : Any
        A batch, as the family's loader yields it.

    Returns
    -------
    dict[str, float]
        The step's losses, including their sum ``loss``.
    """
    losses = task.losses(state.model, batch, state.device)
    loss = sum(losses.values())
    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    state.scheduler.step()
    state.global_step += 1
    return {name: value.detach().item() for name, value in {**losses, "loss": loss}.items()}


def train_one_epoch(state: TrainingState, task: Task, loader: DataLoader, loop: Loop) -> bool:
    """Train until the end of the epoch, or until a stop is requested.

    Logs the average losses every ``log_every`` steps and saves a checkpoint every
    ``checkpoint_every`` steps, and when stopping.

    Parameters
    ----------
    state : TrainingState
        Model, optimizer, schedule and counters; updated in place.
    task : Task
        Computes the batches' losses.
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
    for batch in loader:
        for name, value in train_step(state, task, batch).items():
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
def validate(state: TrainingState, task: Task, loader: DataLoader) -> dict[str, float]:
    """Compute the average validation losses.

    Parameters
    ----------
    state : TrainingState
        The model and its device.
    task : Task
        Computes the batches' losses, and sets the validation mode.
    loader : DataLoader
        Validation loader.

    Returns
    -------
    dict[str, float]
        ``val_<loss>`` averaged over the batches.
    """
    task.validation_mode(state.model)
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for batch in loader:
        losses = task.losses(state.model, batch, state.device)
        for name, value in {**losses, "loss": sum(losses.values())}.items():
            totals[f"val_{name}"] += value.item()
        batches += 1
    return {k: v / max(1, batches) for k, v in totals.items()}


def fit(
    state: TrainingState,
    task: Task,
    loaders: tuple[DataLoader, DataLoader],
    loop: Loop,
    epochs: int,
) -> int | None:
    """Train up to ``epochs`` epochs from the state's position, validating after each.

    Parameters
    ----------
    state : TrainingState
        Where the run is (restored from a checkpoint to resume); updated in place.
    task : Task
        Computes the losses.
    loaders : tuple[DataLoader, DataLoader]
        Training loader (with an ``EpochSampler``) and validation loader.
    loop : Loop
        Logging, checkpointing and stopping.
    epochs : int
        Total number of epochs of the run.

    Returns
    -------
    int | None
        The step it stopped at (checkpoint saved), or ``None`` if all epochs completed.
    """
    train_loader, val_loader = loaders
    steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
    while state.epoch < epochs:
        # Steps already done in this epoch (all of them if a stop came on its last step: the
        # epoch then completes at once and is validated).
        done = state.global_step - state.epoch * steps_per_epoch
        train_loader.sampler.set_position(state.epoch, done * train_loader.batch_size)
        if not train_one_epoch(state, task, train_loader, loop):
            return state.global_step
        state.val_losses = validate(state, task, val_loader)
        loop.log_metrics({**state.val_losses, "epoch": state.epoch}, step=state.global_step)
        state.save(loop.checkpoint)
        logger.info("epoch %d: val loss %.4f", state.epoch, state.val_losses["val_loss"])
    return None


def resume(state: TrainingState, checkpoint: Path) -> None:
    """Restore the state from its checkpoint, if there is one.

    Parameters
    ----------
    state : TrainingState
        The freshly built state; updated in place.
    checkpoint : Path
        Checkpoint file.
    """
    if checkpoint.exists():
        state.restore(checkpoint)
        logger.info("Resuming at epoch %d, step %d", state.epoch, state.global_step)
