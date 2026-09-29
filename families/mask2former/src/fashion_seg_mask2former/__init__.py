"""Mask2Former (Swin-Tiny), fine-tuned from COCO: a ``fashion_seg.ports.ModelFamily``.

A transformer that predicts each garment as a query (a class and a mask), without anchors or
boxes. Its own uv project (``families/mask2former``: Python 3.12, PyTorch, Hugging Face
``transformers``) trains, packages and evaluates it through ``python -m fashion_seg_mask2former``
(``__main__``). This module declares the family and imports PyTorch only when it trains or
predicts. The model logic is in:

- ``config``: its training parameters;
- ``network``: builds the model, shared by training and serving;
- ``dataset``: prepared annotations as Mask2Former batches;
- ``training``: its losses and optimizer, with the shared loop (``fashion_seg_torch``);
- ``predictor``: runs an export for serving.
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
from fashion_seg_mask2former.config import Config

SPEC = FamilySpec(
    name="mask2former",
    serving=ServingRequirements(
        pinned=("torch", "transformers"),
        # CPU-only PyTorch wheels on Linux, as in families/mask2former/pyproject.toml (the
        # default ones bundle CUDA).
        pip_options=("--extra-index-url https://download.pytorch.org/whl/cpu",),
    ),
    bundles=("fashion_seg_torch",),  # the predictor picks its device with it
)


def train(config: Config, inputs: TrainInputs, session: TrainingSession) -> TrainResult:
    """Fine-tune Mask2Former and export it (see ``training.train``).

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
        Last validation losses and run tags, or where it stopped.
    """
    # pylint: disable-next=import-outside-toplevel  # PyTorch loads only when training
    from fashion_seg_mask2former import training

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
    # pylint: disable-next=import-outside-toplevel  # PyTorch loads only when serving
    from fashion_seg_mask2former.predictor import Mask2FormerPredictor

    return Mask2FormerPredictor(model_dir)


def describe(model_dir: Path) -> dict[str, Any]:
    """Read the export's settings, logged when it is packaged.

    Parameters
    ----------
    model_dir : Path
        Export directory.

    Returns
    -------
    dict[str, Any]
        ``fashion_seg.json``: architecture, image size, number of classes, transformers version.
    """
    return json.loads((model_dir / "fashion_seg.json").read_text(encoding="utf-8"))
