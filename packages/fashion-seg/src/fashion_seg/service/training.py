"""Train step: train any model family on the prepared data and split, tracked in MLflow.

The family does the model work (``ModelFamily.train``); this step does the rest, identically for
every family: select the images, log parameters and tags, record the metrics, write
``metrics.json`` and log the export. Outputs, in ``output_dir`` (``<output_dir>-smoke`` for a
smoke run):

- ``model/``: the servable export (DVC-tracked);
- ``checkpoints/``: the family's checkpoints (git- and DVC-ignored);
- ``metrics.json``: the final metrics (``dvc metrics show``).
"""

import json
from pathlib import Path

import mlflow
from fashion_seg_contract.labels import load_class_names

from fashion_seg.config import DataConfig, Params, TrainConfig
from fashion_seg.data import annotations
from fashion_seg.ports import ModelFamily, TrainInputs, TrainResult
from fashion_seg.tracking import setup_experiment


def smoke_config(config: TrainConfig) -> TrainConfig:
    """Apply the family's smoke overrides: a tiny, fast run checking the whole chain.

    Parameters
    ----------
    config : TrainConfig
        Training parameters.

    Returns
    -------
    TrainConfig
        The same parameters with the overrides, validated again.
    """
    return type(config).model_validate({**config.model_dump(), **config.smoke_overrides})


def output_dir_of(config: TrainConfig, smoke: bool) -> Path:
    """Where a run writes: ``output_dir``, or ``<output_dir>-smoke`` for a smoke run.

    Parameters
    ----------
    config : TrainConfig
        Training parameters.
    smoke : bool
        Whether this is a smoke run.

    Returns
    -------
    Path
        The output directory.
    """
    return (
        config.output_dir.with_name(config.output_dir.name + "-smoke")
        if smoke
        else config.output_dir
    )


def load_inputs(
    config: TrainConfig, data: DataConfig, output_dir: Path, local_only: bool
) -> TrainInputs:
    """Load the training and validation images of the frozen split.

    Parameters
    ----------
    config : TrainConfig
        Training parameters (``max_train_images``, ``max_val_images``).
    data : DataConfig
        Dataset files.
    output_dir : Path
        Where the run writes.
    local_only : bool
        Keep only the images present on disk (partial pull, smoke runs).

    Returns
    -------
    TrainInputs
        Annotations, image directory, class names and output directories.

    Raises
    ------
    SystemExit
        If no training or validation image is available.
    """
    split = annotations.load_split(data.prepared_dir / annotations.SPLIT_FILE)
    ids = {
        "train": annotations.select_ids(
            split["train"], data.train_images, config.max_train_images, local_only
        ),
        "val": annotations.select_ids(
            split["val"], data.train_images, config.max_val_images, local_only
        ),
    }
    if not ids["train"] or not ids["val"]:
        raise SystemExit(
            f"No training/validation images in {data.train_images}: run "
            "`uv run dvc pull data.dvc` (or `make pull-sample` for a smoke run)."
        )
    annotations_file = data.prepared_dir / annotations.ANNOTATIONS_FILE
    return TrainInputs(
        train=annotations.load(annotations_file, ids["train"]),
        val=annotations.load(annotations_file, ids["val"]),
        image_dir=data.train_images,
        class_names=load_class_names(data.label_file),
        export_dir=output_dir / "model",
        checkpoint_dir=output_dir / "checkpoints",
    )


def train(
    family: ModelFamily, config: TrainConfig, params: Params, *, smoke: bool = False
) -> TrainResult:
    """Train a family's model in an MLflow run, then write ``metrics.json``.

    Parameters
    ----------
    family : ModelFamily
        The model family.
    config : TrainConfig
        Its training parameters (an instance of ``family.Config``).
    params : Params
        The pipeline parameters (data, split, tracking).
    smoke : bool
        Smoke run: the family's smoke overrides, images on disk only, COCO weights optional,
        export not logged. By default ``False``.

    Returns
    -------
    TrainResult
        The family's final metrics and tags.

    Raises
    ------
    SystemExit
        If the initial weights are missing (outside smoke runs).
    """
    if smoke:
        config = smoke_config(config)
    elif not config.init_weights.exists():
        raise SystemExit(f"{config.init_weights} missing: run `uv run dvc pull weights/`")
    output_dir = output_dir_of(config, smoke)
    inputs = load_inputs(config, params.data, output_dir, local_only=smoke)

    name = family.SPEC.name
    setup_experiment(params.tracking.training_experiment)
    with mlflow.start_run(run_name=f"{name}-smoke" if smoke else name):
        mlflow.set_tags({"model_family": name, "smoke": str(smoke).lower()})
        mlflow.log_params(
            {
                **config.model_dump(mode="json"),
                **{f"split.{k}": v for k, v in params.split.model_dump().items()},
                "n_train_images": inputs.train.height,
                "n_val_images": inputs.val.height,
            }
        )
        result = family.train(config, inputs, mlflow.log_metrics)
        mlflow.set_tags(result.tags)
        (output_dir / "metrics.json").write_text(
            json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8"
        )
        if not smoke:
            mlflow.log_artifacts(str(inputs.export_dir), artifact_path="model")
    return result
