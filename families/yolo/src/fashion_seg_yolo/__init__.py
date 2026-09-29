"""YOLO11 segmentation (Ultralytics), fine-tuned from COCO: a ``fashion_seg.ports.ModelFamily``.

A one-stage detector predicting boxes, classes and mask coefficients in a single pass: fast, and
light enough for a laptop. Its own uv project (``families/yolo``: Python 3.12, PyTorch,
Ultralytics) trains, packages and evaluates it through ``python -m fashion_seg_yolo``
(``__main__``). This module declares the family and imports Ultralytics only when it trains or
predicts. The model logic is in:

- ``config``: its training parameters;
- ``isolation``: keeps Ultralytics' settings, analytics and integrations out of the way;
- ``dataset``: prepared annotations as an Ultralytics dataset (images and polygons);
- ``training``: Ultralytics' trainer, with the pipeline's logging, checkpoints and stop;
- ``predictor``: runs an export for serving.

Ultralytics is AGPL-3.0: a service running it over a network must publish its source.
"""

import json
from pathlib import Path
from typing import Any

from fashion_seg.ports import (
    FamilySpec,
    Predictor,
    ServingRequirements,
    TrainingSession,
    TrainInputs,
    TrainResult,
)
from fashion_seg_yolo import isolation
from fashion_seg_yolo.config import Config

isolation.environment()  # before anything imports Ultralytics

SPEC = FamilySpec(
    name="yolo",
    serving=ServingRequirements(
        pinned=("torch", "ultralytics"),
        # CPU-only PyTorch wheels on Linux, as in families/yolo/pyproject.toml (the default ones
        # bundle CUDA).
        pip_options=("--extra-index-url https://download.pytorch.org/whl/cpu",),
    ),
    bundles=("fashion_seg_torch",),  # the predictor picks its device with it
)


def train(config: Config, inputs: TrainInputs, session: TrainingSession) -> TrainResult:
    """Fine-tune YOLO and export it (see ``training.train``).

    Parameters
    ----------
    config : Config
        Training parameters.
    inputs : TrainInputs
        Data, and where to write.
    session : TrainingSession
        Records metrics; tells when to stop.

    Returns
    -------
    TrainResult
        Last epoch's metrics and run tags, or where it stopped.
    """
    # pylint: disable-next=import-outside-toplevel  # Ultralytics loads only when training
    from fashion_seg_yolo import training

    return training.train(config, inputs, session)


def load_predictor(model_dir: Path) -> Predictor:
    """Load an export for serving.

    Parameters
    ----------
    model_dir : Path
        Export directory.

    Returns
    -------
    Predictor
        The loaded predictor.
    """
    # pylint: disable-next=import-outside-toplevel  # Ultralytics loads only when serving
    from fashion_seg_yolo.predictor import YoloPredictor

    return YoloPredictor(model_dir)


def describe(model_dir: Path) -> dict[str, Any]:
    """Read the export's settings, logged when it is packaged.

    Parameters
    ----------
    model_dir : Path
        Export directory.

    Returns
    -------
    dict[str, Any]
        ``fashion_seg.json``: architecture, image size, number of classes, Ultralytics version.
    """
    return json.loads((model_dir / "fashion_seg.json").read_text(encoding="utf-8"))
