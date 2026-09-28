"""``train_torchvision`` stage: fine-tune Mask R-CNN v2 from COCO, log to MLflow, export.

Outputs (``train_torchvision.output_dir``):

- ``model/``: ``config.json`` + ``model.pt`` (``state_dict``), served by
  ``fashion_seg.predictors.torchvision.TorchvisionPredictor``;
- ``checkpoints/last.pt``: model, optimizer and schedule after each epoch, to resume;
- ``metrics.json``: validation losses of the last epoch (``dvc metrics show``).

Run through DVC (``uv run dvc repro --single-item train_torchvision``) or, for a quick check on
the few images you have pulled: ``make train-torchvision-smoke``.
"""

import argparse
import json
import logging
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import torch
import torchvision
import yaml
from fashion_seg_contract.labels import load_class_names
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN

from fashion_seg_core import annotations
from fashion_seg_core.tracking import setup_experiment
from fashion_seg_torchvision.data import FashionDataset, collate
from fashion_seg_torchvision.model import ARCHITECTURE, build_model, pick_device

logger = logging.getLogger(__name__)

# Tiny, fast settings for `--smoke`: checks the whole chain, not model quality.
SMOKE_OVERRIDES = {
    "min_size": 256,
    "max_size": 320,
    "batch_size": 2,
    "epochs": 1,
    "warmup_steps": 1,
    "num_workers": 0,
    "log_every": 1,
    "max_train_images": 4,
    "max_val_images": 2,
}


def load_params(params_file: str, smoke: bool) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Read the ``data`` and ``train_torchvision`` sections, applying the smoke overrides.

    Parameters
    ----------
    params_file : str
        Path of ``params.yaml``.
    smoke : bool
        Replace the training parameters with ``SMOKE_OVERRIDES``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], Path]
        The ``data`` and ``train_torchvision`` sections, and the output directory
        (``<output_dir>-smoke`` in smoke mode).
    """
    all_params = yaml.safe_load(Path(params_file).read_text(encoding="utf-8"))
    params = dict(all_params["train_torchvision"])
    output_dir = Path(params["output_dir"])
    if smoke:
        params.update(SMOKE_OVERRIDES)
        output_dir = output_dir.with_name(output_dir.name + "-smoke")
    return all_params["data"], params, output_dir


def select_ids(ids: list[str], limit: int | None, image_dir: Path, local_only: bool) -> list[str]:
    """Apply ``max_*_images``; in smoke mode keep only images present on disk.

    Parameters
    ----------
    ids : list[str]
        Image ids of the split.
    limit : int | None
        Maximum number of images, ``None`` for all.
    image_dir : Path
        Directory of the ``<image_id>.jpg`` files.
    local_only : bool
        Keep only the images that exist in ``image_dir``.

    Returns
    -------
    list[str]
        The selected image ids, in split order.
    """
    if local_only:
        ids = [i for i in ids if (image_dir / f"{i}.jpg").exists()]
    return ids[:limit] if limit else ids


def build_loaders(
    data: dict[str, Any], params: dict[str, Any], smoke: bool
) -> tuple[DataLoader, DataLoader]:
    """Build the training and validation loaders from the prepared annotations and the split.

    Parameters
    ----------
    data : dict[str, Any]
        The ``data`` section of ``params.yaml``.
    params : dict[str, Any]
        The ``train_torchvision`` section of ``params.yaml``.
    smoke : bool
        Keep only the images present on disk.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training loader (shuffled) and validation loader.

    Raises
    ------
    SystemExit
        If no training or validation image is available.
    """
    image_dir = Path(data["train_images"])
    split = annotations.load_split(Path(data["prepared_dir"]) / "split.json")
    train_ids = select_ids(split["train"], params["max_train_images"], image_dir, smoke)
    val_ids = select_ids(split["val"], params["max_val_images"], image_dir, smoke)
    if not train_ids or not val_ids:
        raise SystemExit(
            f"No training/validation images found in {image_dir}: run `uv run dvc pull data.dvc` "
            "(or `make pull-sample` for --smoke)."
        )
    prepared = Path(data["prepared_dir"]) / "annotations.parquet"
    loaders = []
    for ids, shuffle in ((train_ids, True), (val_ids, False)):
        dataset = FashionDataset(annotations.load(prepared, ids), image_dir, params["max_size"])
        loaders.append(
            DataLoader(
                dataset,
                batch_size=params["batch_size"],
                shuffle=shuffle,
                num_workers=params["num_workers"],
                persistent_workers=params["num_workers"] > 0,
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
    """

    model: MaskRCNN
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: torch.device
    epoch: int = 0
    global_step: int = 0

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


def train_one_epoch(state: TrainingState, loader: DataLoader, log_every: int) -> None:
    """Train for one epoch, logging the average losses every ``log_every`` steps.

    Parameters
    ----------
    state : TrainingState
        Model, optimizer, schedule and counters; updated in place.
    loader : DataLoader
        Training loader.
    log_every : int
        Steps between two MLflow logs.
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
            mlflow.log_metrics(metrics, step=state.global_step)
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


def export_model(
    model: MaskRCNN, params: dict[str, Any], num_classes: int, export_dir: Path
) -> None:
    """Write the trained model for serving: ``config.json`` + ``model.pt``.

    Parameters
    ----------
    model : MaskRCNN
        The trained model.
    params : dict[str, Any]
        The ``train_torchvision`` section of ``params.yaml``.
    num_classes : int
        Number of classes, background included.
    export_dir : Path
        Directory to (re)create with the export.
    """
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    config = {
        "architecture": ARCHITECTURE,
        "num_classes": num_classes,
        "min_size": params["min_size"],
        "max_size": params["max_size"],
        "torchvision_version": torchvision.__version__,
    }
    (export_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, export_dir / "model.pt")


def run(
    params: dict[str, Any],
    loaders: tuple[DataLoader, DataLoader],
    num_classes: int,
    output_dir: Path,
    smoke: bool,
) -> dict[str, float]:
    """Train in the active MLflow run (resuming from the last checkpoint if asked), then export.

    Parameters
    ----------
    params : dict[str, Any]
        The ``train_torchvision`` section of ``params.yaml``.
    loaders : tuple[DataLoader, DataLoader]
        Training and validation loaders.
    num_classes : int
        Number of classes, background included.
    output_dir : Path
        Where to write checkpoints, the exported model and the metrics.
    smoke : bool
        Allow training without the COCO weights.

    Returns
    -------
    dict[str, float]
        Validation losses of the last epoch.

    Raises
    ------
    SystemExit
        If the COCO weights are missing outside smoke mode.
    """
    train_loader, val_loader = loaders
    device = pick_device(params["device"])
    init_weights = Path(params["init_weights"])
    if not init_weights.exists() and not smoke:
        raise SystemExit(f"{init_weights} missing: run `uv run dvc pull weights/`")
    model = build_model(
        num_classes,
        params["min_size"],
        params["max_size"],
        coco_weights=init_weights if init_weights.exists() else None,
    ).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=params["learning_rate"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
    )
    total_steps = params["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, params["warmup_steps"], total_steps)
    )
    state = TrainingState(model, optimizer, scheduler, device)
    checkpoint = output_dir / "checkpoints" / "last.pt"
    if params["resume"] and checkpoint.exists():
        state.restore(checkpoint)
        logger.info("Resuming after epoch %d (step %d)", state.epoch, state.global_step)
    mlflow.set_tag("device", device.type)

    val_losses: dict[str, float] = {}
    while state.epoch < params["epochs"]:
        train_one_epoch(state, train_loader, params["log_every"])
        val_losses = validate(model, val_loader, device)
        mlflow.log_metrics({**val_losses, "epoch": state.epoch}, step=state.global_step)
        state.save(checkpoint)
        logger.info("epoch %d: val loss %.4f", state.epoch, val_losses["val_loss"])
    export_model(model, params, num_classes, output_dir / "model")
    return val_losses


def main(argv: list[str] | None = None) -> None:
    """Train, log to MLflow, and export the model and the final validation losses.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments, ``sys.argv[1:]`` when ``None``. By default ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--smoke", action="store_true", help="tiny run on locally pulled images")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data, params, output_dir = load_params(args.params, args.smoke)
    loaders = build_loaders(data, params, args.smoke)
    num_classes = len(load_class_names(data["label_file"]))

    setup_experiment(params["experiment"])
    with mlflow.start_run(run_name="torchvision-smoke" if args.smoke else "torchvision"):
        mlflow.set_tags(
            {
                "model_family": "torchvision",
                "architecture": ARCHITECTURE,
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "smoke": str(args.smoke).lower(),
            }
        )
        mlflow.log_params(
            {
                **params,
                "n_train_images": len(loaders[0].dataset),
                "n_val_images": len(loaders[1].dataset),
            }
        )
        val_losses = run(params, loaders, num_classes, output_dir, args.smoke)
        (output_dir / "metrics.json").write_text(
            json.dumps(val_losses, indent=2) + "\n", encoding="utf-8"
        )
        if not args.smoke:
            mlflow.log_artifacts(str(output_dir / "model"), artifact_path="model")
    logger.info("Model exported to %s", output_dir / "model")


if __name__ == "__main__":
    main()
