"""torchvision Mask R-CNN v2, fine-tuned from COCO: a ``fashion_seg.ports.ModelFamily``.

Its own uv project (``families/torchvision``: Python 3.12, PyTorch) trains, packages and evaluates
it through ``python -m fashion_seg_torchvision`` (``__main__``). This module declares the family
(spec, training parameters) and imports PyTorch only when it trains or predicts. The model logic
is in:

- ``config``: its training parameters;
- ``network``: builds the network, shared by training and serving;
- ``dataset``: prepared annotations as a torch dataset;
- ``training``: the training loop, checkpoints and export;
- ``predictor``: runs an export for serving.
"""

import json
from pathlib import Path
from typing import Any

from fashion_seg.ports import (
    FamilySpec,
    MetricLogger,
    Predictor,
    ServingRequirements,
    TrainInputs,
    TrainResult,
)
from fashion_seg_torchvision.config import Config

SPEC = FamilySpec(
    name="torchvision",
    serving=ServingRequirements(
        pinned=("torch", "torchvision"),
        # CPU-only PyTorch wheels on Linux, as in families/torchvision/pyproject.toml (the
        # default ones bundle CUDA).
        pip_options=("--extra-index-url https://download.pytorch.org/whl/cpu",),
    ),
)


def train(config: Config, inputs: TrainInputs, log_metrics: MetricLogger) -> TrainResult:
    """Fine-tune Mask R-CNN v2 and export it (see ``training.train``).

    Parameters
    ----------
    config : Config
        Training parameters.
    inputs : TrainInputs
        Data, and where to write.
    log_metrics : MetricLogger
        Records metrics during training.

    Returns
    -------
    TrainResult
        Last validation losses, and run tags.
    """
    # pylint: disable-next=import-outside-toplevel  # PyTorch loads only when training
    from fashion_seg_torchvision import training

    return training.train(config, inputs, log_metrics)


def load_predictor(model_dir: Path) -> Predictor:
    """Load an export (``config.json`` + ``model.pt``) for serving.

    Parameters
    ----------
    model_dir : Path
        Export directory.

    Returns
    -------
    Predictor
        The loaded predictor.
    """
    # pylint: disable-next=import-outside-toplevel  # PyTorch loads only when serving
    from fashion_seg_torchvision.predictor import TorchvisionPredictor

    return TorchvisionPredictor(model_dir)


def describe(model_dir: Path) -> dict[str, Any]:
    """Read the export's configuration, logged when it is packaged.

    Parameters
    ----------
    model_dir : Path
        Export directory.

    Returns
    -------
    dict[str, Any]
        ``config.json``: architecture, number of classes, image sizes, torchvision version.
    """
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
