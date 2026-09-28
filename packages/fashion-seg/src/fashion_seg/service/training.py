"""Train step: train any model family on the prepared data and split, tracked by a ``Tracker``.

The family does the model work (``ModelFamily.train``); this step does the rest, identically for
every family: select the images, open a tracked run, record parameters, metrics and tags, and log
the export. The family writes, in ``output_dir_of(config, smoke)``:

- ``model/``: the servable export (DVC-tracked);
- ``checkpoints/``: its checkpoints (git- and DVC-ignored).
"""

from pathlib import Path

from fashion_seg_contract.labels import load_class_names

from fashion_seg.config import DataConfig, Params, TrainConfig
from fashion_seg.data import files
from fashion_seg.ports import Infrastructure, ModelFamily, TrainingSession, TrainInputs, TrainResult


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
    split = files.load_split(data.prepared_dir)
    ids = {
        "train": files.select_ids(
            split["train"], data.train_images, config.max_train_images, local_only
        ),
        "val": files.select_ids(split["val"], data.train_images, config.max_val_images, local_only),
    }
    if not ids["train"] or not ids["val"]:
        raise SystemExit(
            f"No training/validation images in {data.train_images}: run "
            "`uv run dvc pull data.dvc` (or `make pull-sample` for a smoke run)."
        )
    return TrainInputs(
        train=files.load_annotations(data.prepared_dir, ids["train"]),
        val=files.load_annotations(data.prepared_dir, ids["val"]),
        image_dir=data.train_images,
        class_names=load_class_names(data.label_file),
        export_dir=output_dir / "model",
        checkpoint_dir=output_dir / "checkpoints",
    )


def train(
    family: ModelFamily,
    config: TrainConfig,
    params: Params,
    infra: Infrastructure,
    *,
    smoke: bool = False,
) -> TrainResult:
    """Train a family's model in a tracked run; it can stop early and resume later.

    Parameters
    ----------
    family : ModelFamily
        The model family.
    config : TrainConfig
        Its training parameters (an instance of ``family.Config``).
    params : Params
        The pipeline parameters (data, split, tracking).
    infra : Infrastructure
        The tracker records the run (parameters, the family's metrics, tags, the export); the
        stop signal ends the training early, after a checkpoint.
    smoke : bool
        Smoke run: the family's smoke overrides, images on disk only, COCO weights optional,
        export not logged. By default ``False``.

    Returns
    -------
    TrainResult
        The family's final metrics and tags, or ``stopped_at`` if it stopped early.

    Raises
    ------
    SystemExit
        If the initial weights are missing (outside smoke runs).
    """
    if smoke:
        config = smoke_config(config)
    elif not config.init_weights.exists():
        raise SystemExit(f"{config.init_weights} missing: run `uv run dvc pull weights/`")
    inputs = load_inputs(config, params.data, output_dir_of(config, smoke), local_only=smoke)

    name = family.SPEC.name
    tags = {"model_family": name, "smoke": str(smoke).lower()}
    tracker = infra.tracker
    with tracker.run(params.tracking.training_experiment, f"{name}-smoke" if smoke else name, tags):
        tracker.log_params(
            {
                **config.model_dump(mode="json"),
                **{f"split.{k}": v for k, v in params.split.model_dump().items()},
                "n_train_images": inputs.train.height,
                "n_val_images": inputs.val.height,
            }
        )
        result = family.train(config, inputs, TrainingSession(tracker.log_metrics, infra.stop))
        tracker.set_tags(result.tags)
        if result.stopped_at is not None:
            tracker.set_tags({"stopped_at_step": str(result.stopped_at)})
        elif not smoke:
            tracker.log_artifacts(inputs.export_dir, "model")
    return result
