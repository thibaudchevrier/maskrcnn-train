"""Pipeline parameters: ``params.yaml``, validated with pydantic.

Each section has its model. The training section of a model family is validated by the family's
own ``Config`` (a ``TrainConfig`` subclass, see ``fashion_seg.ports.ModelFamily``): this module
only knows the fields every family shares.

Unknown keys are errors, so a typo in ``params.yaml`` fails at load time instead of being ignored.
"""

from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Immutable pydantic model rejecting unknown fields; base of every configuration.

    Attributes
    ----------
    model_config : ClassVar[ConfigDict]
        pydantic settings: frozen, extra fields forbidden, ``model_`` field names allowed.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", protected_namespaces=()
    )


class DataConfig(Frozen):
    """Dataset files (``data`` section).

    Attributes
    ----------
    train_csv : Path
        iMaterialist ``train.csv``, one row per mask.
    train_images : Path
        Directory of the ``<image_id>.jpg`` files.
    label_file : Path
        ``label_descriptions.json`` (class names).
    prepared_dir : Path
        Output of the ``prepare`` stage: ``annotations.parquet`` and ``split.json``.
    """

    train_csv: Path
    train_images: Path
    label_file: Path
    prepared_dir: Path


class SplitConfig(Frozen):
    """Frozen train/validation split (``split`` section): one fold of a shuffled K-fold.

    Attributes
    ----------
    n_folds : int
        Number of folds.
    fold : int
        Index of the fold kept for validation.
    seed : int
        Shuffle seed.
    """

    n_folds: int
    fold: int
    seed: int


class TrackingConfig(Frozen):
    """MLflow experiments and registered model name (``tracking`` section).

    Attributes
    ----------
    training_experiment : str
        Experiment of the training runs.
    packaging_experiment : str
        Experiment of the packaging runs.
    evaluation_experiment : str
        Experiment of the evaluation runs.
    registered_name : str
        Registered model receiving every packaged model as a new version.
    """

    training_experiment: str
    packaging_experiment: str
    evaluation_experiment: str
    registered_name: str


class TrainConfig(Frozen):
    """Training parameters every model family has (``train.<family>`` section).

    A family subclasses it with its own parameters and smoke overrides. Every family checkpoints
    every ``checkpoint_every`` steps and resumes from its last checkpoint (``resume``), so a
    training can be stopped (Ctrl+C, SIGTERM) and continued later.

    Attributes
    ----------
    output_dir : Path
        Receives ``model/`` (the export), ``checkpoints/`` and ``metrics.json``.
    init_weights : Path
        Pre-trained weights to start from (e.g. COCO).
    epochs : int
        Number of epochs.
    max_train_images : int | None
        Use only the first training images of the split; ``None`` for all.
    max_val_images : int | None
        Use only the first validation images of the split; ``None`` for all.
    resume : bool
        Continue from the last checkpoint in ``output_dir/checkpoints`` if there is one.
        By default ``True``.
    checkpoint_every : int
        Steps between two checkpoints: the most a stop, a crash or a power cut can lose.
        By default 500.
    log_every : int
        Steps between two logs of the training losses. By default 50.
    seed : int
        Seed of the training images' shuffle order. By default 0.
    smoke_overrides : ClassVar[dict[str, Any]]
        Parameters replaced for a smoke run (``--smoke``): tiny and fast, checks the whole chain,
        not model quality.
    """

    output_dir: Path
    init_weights: Path
    epochs: int
    max_train_images: int | None = None
    max_val_images: int | None = None
    resume: bool = True
    checkpoint_every: int = 500
    log_every: int = 50
    seed: int = 0
    smoke_overrides: ClassVar[dict[str, Any]] = {}


class PackagedModel(Frozen):
    """A served model: which export to package, and where (``models.<name>`` section).

    Attributes
    ----------
    family : str
        Model family (``SPEC.name``): only that family's entrypoint packages and evaluates it.
    source : Path
        The export to package (the output of a training, or the 2021 model).
    output_dir : Path
        Where the packaged MLflow model is written (tracked by DVC).
    """

    family: str
    source: Path
    output_dir: Path


class EvaluateConfig(Frozen):
    """Evaluation of the packaged models (``evaluate`` section).

    Attributes
    ----------
    split : Literal["train", "val"]
        Split to evaluate on.
    max_images : int | None
        Evaluate only the first images of the split; ``None`` for all.
    min_score : float
        ``min_score`` request parameter.
    output_dir : Path
        Directory of the ``evaluate-<model>.json`` metrics files.
    """

    split: Literal["train", "val"]
    max_images: int | None
    min_score: float
    output_dir: Path


class Params(Frozen):
    """The whole ``params.yaml``.

    Attributes
    ----------
    data : DataConfig
        Dataset files.
    split : SplitConfig
        Frozen train/validation split.
    tracking : TrackingConfig
        MLflow experiments and registered model.
    train : dict[str, dict[str, Any]]
        Training parameters per family; each family validates its own.
    models : dict[str, PackagedModel]
        Served models, by name.
    evaluate : EvaluateConfig
        Evaluation settings.
    """

    data: DataConfig
    split: SplitConfig
    tracking: TrackingConfig
    train: dict[str, dict[str, Any]]
    models: dict[str, PackagedModel]
    evaluate: EvaluateConfig


def load_params(path: str | Path) -> Params:
    """Read and validate ``params.yaml``.

    Parameters
    ----------
    path : str | Path
        Path of the file.

    Returns
    -------
    Params
        The validated parameters.
    """
    return Params.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
