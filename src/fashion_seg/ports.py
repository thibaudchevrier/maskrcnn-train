"""Ports: the interfaces between the workflow (``fashion_seg.service``) and the model families.

The service only knows these types. Each family of ``fashion_seg.families`` implements
``ModelFamily`` by duck typing: a module with the right attributes, no base class to inherit.
Nothing here imports a deep-learning framework (nor polars, used for type hints only), so every
environment can load it, including the serving image of a packaged model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from fashion_seg.config import Frozen, TrainConfig

if TYPE_CHECKING:  # the serving image has no polars; only training passes DataFrames
    import polars as pl


@dataclass(frozen=True)
class Detections:
    """Detections for one image, in original image pixel coordinates.

    Attributes
    ----------
    boxes : np.ndarray
        ``[N, (y1, x1, y2, x2)]`` int32, ``(y2, x2)`` excluded.
    class_ids : np.ndarray
        ``[N]`` int32 model class ids; 0 is the background.
    scores : np.ndarray
        ``[N]`` float32 confidences.
    masks : np.ndarray
        ``[height, width, N]`` bool instance masks.
    """

    boxes: np.ndarray  # [N, (y1, x1, y2, x2)] int32, (y2, x2) excluded
    class_ids: np.ndarray  # [N] int32, 0 is the background
    scores: np.ndarray  # [N] float32
    masks: np.ndarray  # [H, W, N] bool


class Predictor(Protocol):
    """Anything turning an RGB image into detections: what a family serves."""

    def predict(self, image: np.ndarray) -> Detections:
        """Detect the garments in one image.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape ``(height, width, 3)``, uint8.

        Returns
        -------
        Detections
            The detections, in the image's pixel coordinates.
        """


class MetricLogger(Protocol):
    """Records training metrics; ``mlflow.log_metrics`` in the pipeline, a fake in tests."""

    def __call__(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record metrics at a step.

        Parameters
        ----------
        metrics : dict[str, float]
            Metric values, by name.
        step : int | None
            Training step (or epoch). By default ``None``.
        """


@dataclass(frozen=True)
class TrainInputs:
    """What the service gives a family to train on, and where to write.

    Attributes
    ----------
    train : pl.DataFrame
        Prepared annotations of the training images (see ``fashion_seg.data.annotations``).
    val : pl.DataFrame
        Prepared annotations of the validation images.
    image_dir : Path
        Directory of the ``<image_id>.jpg`` files.
    class_names : list[str]
        Class names indexed by model class id (0 = background).
    export_dir : Path
        Where to write the servable export (read back by the family's ``load_predictor``).
    checkpoint_dir : Path
        Where to write checkpoints.
    """

    train: pl.DataFrame
    val: pl.DataFrame
    image_dir: Path
    class_names: list[str]
    export_dir: Path
    checkpoint_dir: Path


@dataclass(frozen=True)
class TrainResult:
    """What a family returns after training.

    Attributes
    ----------
    metrics : dict[str, float]
        Final metrics (e.g. last validation losses), written to ``metrics.json``.
    tags : dict[str, str]
        Run tags (framework versions, device...).
    """

    metrics: dict[str, float]
    tags: dict[str, str]


class Runtime(Frozen):
    """Python environment a step runs in.

    Attributes
    ----------
    python : str
        Python ``major.minor`` version, e.g. ``"3.12"``.
    project : str | None
        Directory of the environment's uv project, relative to the repository root; ``None``
        for the root environment. By default ``None``.
    """

    python: str
    project: str | None = None


class ServingRequirements(Frozen):
    """Runtime requirements of a packaged model: only its own framework.

    Attributes
    ----------
    pinned : tuple[str, ...]
        Installed packages, pinned to their installed version.
    released : tuple[str, ...]
        Released packages, pinned to the wheel URL they were installed from. By default ``()``.
    pip_options : tuple[str, ...]
        Extra ``requirements.txt`` lines, e.g. an index URL. By default ``()``.
    """

    pinned: tuple[str, ...]
    released: tuple[str, ...] = ()
    pip_options: tuple[str, ...] = ()


class FamilySpec(Frozen):
    """Static description of a model family.

    Attributes
    ----------
    name : str
        Family name, the module name in ``fashion_seg.families``.
    train_runtime : Runtime
        Environment training runs in.
    serving : ServingRequirements
        Requirements of its packaged models.
    """

    name: str
    train_runtime: Runtime
    serving: ServingRequirements


@runtime_checkable
class ModelFamily(Protocol):
    """A model family: a module of ``fashion_seg.families`` holding all of its model logic.

    The module imports its framework lazily (inside ``train`` and ``load_predictor``), so it
    loads in every environment. See ``fashion_seg.registry`` to get one by name.

    Attributes
    ----------
    SPEC : FamilySpec
        Name, training environment and serving requirements.
    Config : type[TrainConfig]
        Its training parameters, validating ``params.yaml:train.<name>``.
    """

    SPEC: FamilySpec
    Config: type[TrainConfig]

    def train(
        self, config: TrainConfig, inputs: TrainInputs, log_metrics: MetricLogger
    ) -> TrainResult:
        """Train a model and write its servable export to ``inputs.export_dir``.

        Parameters
        ----------
        config : TrainConfig
            Training parameters, an instance of the family's ``Config``.
        inputs : TrainInputs
            Data, and where to write.
        log_metrics : MetricLogger
            Records metrics during training.

        Returns
        -------
        TrainResult
            Final metrics and run tags.
        """

    def load_predictor(self, model_dir: Path) -> Predictor:
        """Load an export for serving.

        Parameters
        ----------
        model_dir : Path
            Export directory, as written by ``train``.

        Returns
        -------
        Predictor
            The loaded predictor.
        """

    def describe(self, model_dir: Path) -> dict[str, Any]:
        """Read the settings of an export worth logging when it is packaged.

        Parameters
        ----------
        model_dir : Path
            Export directory.

        Returns
        -------
        dict[str, Any]
            Settings, by name.
        """
