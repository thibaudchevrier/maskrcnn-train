"""Fine-tune Mask2Former from COCO with the shared training loop, then export it for serving.

The loop (``fashion_seg_torch.loop.fit``) logs, checkpoints, validates, stops and resumes; this
module gives it the model, the optimizer and a ``Task`` (how to compute Mask2Former's losses).
Writes to the directories the service gives (``TrainInputs``):

- ``export_dir``: the Hugging Face model (``config.json`` + ``model.safetensors``) and
  ``fashion_seg.json``, what ``predictor`` serves;
- ``checkpoint_dir / "last.pt"``: model, optimizer, schedule and position, to resume exactly.
"""

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import torch
import transformers
from fashion_seg_torch.data import make_loaders
from fashion_seg_torch.device import pick_device
from fashion_seg_torch.loop import Loop, Task, TrainingState, fit, resume, warmup_cosine
from transformers import Mask2FormerForUniversalSegmentation

from fashion_seg.ports import TrainingSession, TrainInputs, TrainResult
from fashion_seg_mask2former.config import Config
from fashion_seg_mask2former.dataset import FashionDataset, collate
from fashion_seg_mask2former.network import ARCHITECTURE, build_model

# Parameters of the Swin backbone: trained at backbone_lr_factor x the learning rate.
BACKBONE = "pixel_level_module.encoder"
EXPORT_CONFIG = "fashion_seg.json"


def group_losses(losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sum each loss over the decoder layers (``loss_dice_3`` goes into ``loss_dice``).

    Parameters
    ----------
    losses : dict[str, torch.Tensor]
        Weighted losses of the final and auxiliary predictions, from ``get_loss_dict``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``loss_cross_entropy`` (classes), ``loss_mask`` and ``loss_dice`` (masks); their sum is
        Mask2Former's loss.

    Examples
    --------
    >>> grouped = group_losses({"loss_dice": torch.tensor(1.0), "loss_dice_0": torch.tensor(2.0)})
    >>> {k: v.item() for k, v in grouped.items()}
    {'loss_dice': 3.0}
    """
    grouped: dict[str, torch.Tensor] = {}
    for name, value in losses.items():
        key = re.sub(r"_\d+$", "", name)
        grouped[key] = grouped[key] + value if key in grouped else value
    return grouped


def losses(model: torch.nn.Module, batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    """Compute Mask2Former's losses on a batch, by component.

    Parameters
    ----------
    model : torch.nn.Module
        The model, on ``device``.
    batch : Any
        A batch, as ``dataset.collate`` builds it.
    device : torch.device
        Compute device.

    Returns
    -------
    dict[str, torch.Tensor]
        See ``group_losses``.
    """
    mask_labels = [masks.to(device) for masks in batch["mask_labels"]]
    class_labels = [labels.to(device) for labels in batch["class_labels"]]
    outputs = model(
        pixel_values=batch["pixel_values"].to(device),
        pixel_mask=batch["pixel_mask"].to(device),
        output_auxiliary_logits=True,
    )
    return group_losses(
        model.get_loss_dict(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            mask_labels,
            class_labels,
            outputs.auxiliary_logits,
        )
    )


TASK = Task(losses=losses)


def init_state(config: Config, labels: list[str], steps_per_epoch: int) -> TrainingState:
    """Build the model (from COCO if the weights are there), optimizer and schedule.

    Parameters
    ----------
    config : Config
        Training parameters.
    labels : list[str]
        Class names, in the model's class order.
    steps_per_epoch : int
        Optimizer steps per epoch.

    Returns
    -------
    TrainingState
        The initial state, on the configured device.
    """
    device = pick_device(config.device)
    pretrained = config.init_weights if config.init_weights.exists() else None
    model = build_model(labels, pretrained).to(device)
    backbone = [p for n, p in model.named_parameters() if BACKBONE in n and p.requires_grad]
    others = [p for n, p in model.named_parameters() if BACKBONE not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.learning_rate * config.backbone_lr_factor},
            {"params": others, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = warmup_cosine(optimizer, config.warmup_steps, config.epochs * steps_per_epoch)
    return TrainingState(model, optimizer, scheduler, device)


def export_model(
    model: Mask2FormerForUniversalSegmentation, config: Config, export_dir: Path
) -> None:
    """Write the trained model for serving: the Hugging Face model and ``fashion_seg.json``.

    Parameters
    ----------
    model : Mask2FormerForUniversalSegmentation
        The trained model.
    config : Config
        Training parameters (image size).
    export_dir : Path
        Directory to (re)create with the export.
    """
    if export_dir.exists():
        shutil.rmtree(export_dir)
    model.save_pretrained(export_dir)
    export_config = {
        "architecture": ARCHITECTURE,
        "max_size": config.max_size,
        "num_labels": model.config.num_labels,
        "transformers_version": transformers.__version__,
    }
    (export_dir / EXPORT_CONFIG).write_text(
        json.dumps(export_config, indent=2) + "\n", encoding="utf-8"
    )


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
    state = init_state(config, inputs.class_names[1:], steps_per_epoch)
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
        "transformers_version": transformers.__version__,
        "device": state.device.type,
    }
    if stopped_at is not None:
        return TrainResult(metrics=state.val_losses, tags=tags, stopped_at=stopped_at)
    export_model(state.model, config, inputs.export_dir)
    return TrainResult(metrics=state.val_losses, tags=tags)
