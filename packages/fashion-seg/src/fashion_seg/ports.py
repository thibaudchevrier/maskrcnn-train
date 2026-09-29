"""Ports: the interfaces between the workflow (``fashion_seg.service``) and what it drives.

The service only knows these types:

- ``ModelFamily``: a model family (``families/<name>``, its own uv project), implemented by
  duck typing: a module with the right attributes, no base class to inherit;
- ``Tracker`` and ``ModelRepository``: experiment tracking and the model registry, implemented
  with MLflow by ``fashion_seg.adapters``.

``fashion_seg.cli.main`` (called by each family's entrypoint) wires them together. Nothing here
imports a deep-learning framework nor MLflow (nor polars, used for type hints only), so every
environment can load it, including the serving image of a packaged model.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from fashion_seg.config import Frozen, TrainConfig

if TYPE_CHECKING:  # the serving image has no polars; only training passes DataFrames
    import polars as pl
    from fashion_seg_contract.schema import Prediction


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
    stopped_at : int | None
        Step at which a stop request ended the training (checkpoint saved, no export); ``None``
        when it completed. By default ``None``.
    """

    metrics: dict[str, float]
    tags: dict[str, str]
    stopped_at: int | None = None


class StopSignal(Protocol):
    """Tells a training whether it must stop (e.g. after Ctrl+C); checked between steps."""

    def is_set(self) -> bool:
        """Tell whether a stop was requested.

        Returns
        -------
        bool
            ``True`` once a stop is requested.
        """


@dataclass(frozen=True)
class TrainingSession:
    """What a family's training reports to, and listens to.

    Attributes
    ----------
    log_metrics : MetricLogger
        Records metrics during training.
    stop : StopSignal
        Checked between steps: when set, the family saves a checkpoint and returns a
        ``TrainResult`` with ``stopped_at`` (the next run resumes there).
    """

    log_metrics: MetricLogger
    stop: StopSignal


class ServingRequirements(Frozen):
    """Runtime requirements of a family's packaged models: its framework.

    The serving wrapper's own requirements are added when packaging.

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
        Family name, the key of its ``params.yaml:train`` section.
    serving : ServingRequirements
        Requirements of its packaged models.
    bundles : tuple[str, ...]
        Other first-party packages its serving code imports (e.g. ``fashion_seg_torch``):
        bundled with its packaged models, with ``fashion_seg`` and its own package.
        By default ``()``.
    """

    name: str
    serving: ServingRequirements
    bundles: tuple[str, ...] = ()


@runtime_checkable
class ModelFamily(Protocol):
    """A model family: the package ``fashion_seg_<name>``, holding all of its model logic.

    It imports its framework lazily (inside ``train`` and ``load_predictor``): a family may
    train and serve in two environments (Matterport), each having only one of them.

    Attributes
    ----------
    SPEC : FamilySpec
        Name and serving requirements.
    Config : type[TrainConfig]
        Its training parameters, validating ``params.yaml:train.<name>``.
    """

    SPEC: FamilySpec
    Config: type[TrainConfig]

    def train(
        self, config: TrainConfig, inputs: TrainInputs, session: TrainingSession
    ) -> TrainResult:
        """Train a model and write its servable export to ``inputs.export_dir``.

        Resumes from the last checkpoint in ``inputs.checkpoint_dir`` when ``config.resume``;
        checkpoints every ``config.checkpoint_every`` steps; stops early (after a checkpoint) when
        ``session.stop`` is set.

        Parameters
        ----------
        config : TrainConfig
            Training parameters, an instance of the family's ``Config``.
        inputs : TrainInputs
            Data, and where to write.
        session : TrainingSession
            Records metrics; tells when to stop.

        Returns
        -------
        TrainResult
            Final metrics and run tags, or where it stopped.
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


@runtime_checkable
class Tracker(Protocol):
    """Experiment tracking: runs, with their parameters, metrics, tags and artifacts.

    Logging calls apply to the run opened by ``run``.
    """

    def run(self, experiment: str, name: str, tags: dict[str, str]) -> AbstractContextManager[str]:
        """Open a run for the duration of a ``with`` block.

        Parameters
        ----------
        experiment : str
            Experiment of the run, created if needed.
        name : str
            Run name.
        tags : dict[str, str]
            Tags set when the run starts.

        Returns
        -------
        AbstractContextManager[str]
            Context manager giving the run id.
        """

    def set_tags(self, tags: dict[str, str]) -> None:
        """Tag the current run.

        Parameters
        ----------
        tags : dict[str, str]
            Tags, by name.
        """

    def log_params(self, params: dict[str, Any]) -> None:
        """Record parameters of the current run.

        Parameters
        ----------
        params : dict[str, Any]
            Parameters, by name.
        """

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record metrics of the current run (a ``MetricLogger``).

        Parameters
        ----------
        metrics : dict[str, float]
            Metric values, by name.
        step : int | None
            Training step (or epoch). By default ``None``.
        """

    def log_artifacts(self, local_dir: Path, artifact_path: str) -> None:
        """Attach a directory to the current run.

        Parameters
        ----------
        local_dir : Path
            Directory to upload.
        artifact_path : str
            Its path within the run's artifacts.
        """

    def log_table(self, table: pl.DataFrame, file_name: str) -> None:
        """Attach a table to the current run, as a CSV file.

        Parameters
        ----------
        table : pl.DataFrame
            The table.
        file_name : str
            Name of the CSV file among the run's artifacts.
        """


@dataclass(frozen=True)
class ModelPackage:
    """A model export to serve through the contract, and what serving it needs.

    Attributes
    ----------
    family : str
        Model family name.
    load_predictor : Callable[[Path], Predictor]
        The family's ``load_predictor``, called on the export when the model is loaded.
    export_dir : Path
        The export (trained model files).
    label_file : Path
        ``label_descriptions.json`` (class names).
    code_dirs : list[Path]
        Python packages to bundle with the model.
    requirements : list[str]
        pip requirement lines of the serving environment.
    """

    family: str
    load_predictor: Callable[[Path], Predictor]
    export_dir: Path
    label_file: Path
    code_dirs: list[Path]
    requirements: list[str]


@runtime_checkable
class ModelRepository(Protocol):
    """Where packaged models are published, versioned, and loaded as they are served."""

    def publish(
        self, package: ModelPackage, registered_name: str, output_dir: Path
    ) -> dict[str, Any]:
        """Package a model in the current tracking run, register a new version, save a copy.

        Parameters
        ----------
        package : ModelPackage
            What to package.
        registered_name : str
            Registered model receiving the new version.
        output_dir : Path
            Where to save the packaged model (replaced if present), with its provenance.

        Returns
        -------
        dict[str, Any]
            Provenance: tracking run, model URI, family, registered name and version.
        """

    def provenance(self, model_dir: Path) -> dict[str, Any]:
        """Read the provenance saved with a packaged model.

        Parameters
        ----------
        model_dir : Path
            A packaged model, as saved by ``publish``.

        Returns
        -------
        dict[str, Any]
            Its provenance.
        """

    def load(self, model_dir: Path) -> Callable[[bytes, float], Prediction]:
        """Load a packaged model exactly as it is served.

        Parameters
        ----------
        model_dir : Path
            A packaged model, as saved by ``publish``.

        Returns
        -------
        Callable[[bytes, float], Prediction]
            Predicts one encoded image (JPEG or PNG) with a ``min_score``.
        """

    def tag_version(self, registered_name: str, version: str, tags: dict[str, str]) -> None:
        """Tag a registered model version (e.g. with its evaluation scores).

        Parameters
        ----------
        registered_name : str
            Registered model.
        version : str
            Its version.
        tags : dict[str, str]
            Tags, by name.
        """


@dataclass(frozen=True)
class Infrastructure:
    """The infrastructure a workflow step drives, built by the composition root.

    Attributes
    ----------
    tracker : Tracker
        Experiment tracking.
    repository : ModelRepository
        Packaged models: publish, load, tag.
    stop : StopSignal
        Set when the process must stop (``adapters.signals``); never set by default.
    """

    tracker: Tracker
    repository: ModelRepository
    stop: StopSignal = field(default_factory=threading.Event)
