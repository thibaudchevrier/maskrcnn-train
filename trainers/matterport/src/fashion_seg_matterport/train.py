"""``train_matterport`` stage: train Matterport Mask R-CNN, log to MLflow, export for serving.

Outputs (``train_matterport.output_dir``):
- ``model/``: inference model as ``config.json`` + TF SavedModel, the format that
  ``fashion_seg.predictors.matterport.MatterportPredictor`` serves;
- ``metrics.json``: final-epoch losses (``dvc metrics show``).

Run through DVC (``uv run dvc repro --single-item train_matterport``) or, for a quick check on
the few images you have pulled: ``make train-matterport-smoke``.
"""

import argparse
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

import keras  # Keras 2.15, the same package as tf.keras in TensorFlow 2.15
import mlflow
import tensorflow as tf
import yaml
from fashion_seg_contract.labels import load_class_names
from mrcnn import config as mconfig
from mrcnn import model as modellib

from fashion_seg_core import annotations
from fashion_seg_core.tracking import setup_experiment
from fashion_seg_matterport.dataset import FashionDataset

# Layers re-initialised when starting from COCO: they depend on the number of classes.
HEAD_LAYERS = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]

# Tiny, fast settings for `--smoke`: checks the whole chain, not model quality.
SMOKE_OVERRIDES = {
    "backbone": "resnet50",
    "image_min_dim": 256,
    "image_max_dim": 256,
    "images_per_gpu": 1,
    "train_rois_per_image": 16,
    "epochs": 1,
    "steps_per_epoch": 2,
    "validation_steps": 1,
    "max_train_images": 4,
    "max_val_images": 2,
}


class MlflowEpochLogger(keras.callbacks.Callback):
    """Keras callback logging every epoch metric (losses, validation losses) to MLflow.

    Attributes
    ----------
    last_logs : dict[str, float]
        Metrics of the last finished epoch.
    """

    last_logs: dict[str, float]

    def __init__(self) -> None:
        super().__init__()
        self.last_logs = {}

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Log the epoch's metrics to the active MLflow run.

        Parameters
        ----------
        epoch : int
            0-based epoch index (logged as step ``epoch + 1``).
        logs : dict[str, float] | None
            Metrics computed by Keras for the epoch. By default ``None``.
        """
        self.last_logs = {key: float(value) for key, value in (logs or {}).items()}
        mlflow.log_metrics(self.last_logs, step=epoch + 1)


def build_config(
    params: dict[str, Any],
    num_classes: int,
    batch_steps: tuple[int, int],
    *,
    inference: bool = False,
) -> mconfig.Config:
    """Build the Matterport ``Config`` from the ``train_matterport`` parameters.

    Inference uses the training anchors, unlike the 2021 export (see the README's roadmap).

    Parameters
    ----------
    params : dict[str, Any]
        The ``train_matterport`` section of ``params.yaml``.
    num_classes : int
        Number of classes, background included.
    batch_steps : tuple[int, int]
        Training and validation steps per epoch.
    inference : bool
        Build the inference configuration (one image per batch). By default ``False``.

    Returns
    -------
    mconfig.Config
        The configuration.
    """
    steps_per_epoch, validation_steps = batch_steps
    attrs = {
        "NAME": "fashion",
        "NUM_CLASSES": num_classes,
        "GPU_COUNT": 1,
        "IMAGES_PER_GPU": 1 if inference else params["images_per_gpu"],
        "BACKBONE": params["backbone"],
        "IMAGE_MIN_DIM": params["image_min_dim"],
        "IMAGE_MAX_DIM": params["image_max_dim"],
        "RPN_ANCHOR_SCALES": tuple(params["rpn_anchor_scales"]),
        "TRAIN_ROIS_PER_IMAGE": params["train_rois_per_image"],
        "LEARNING_RATE": params["learning_rate"],
        "STEPS_PER_EPOCH": steps_per_epoch,
        "VALIDATION_STEPS": validation_steps,
        "DETECTION_MIN_CONFIDENCE": params["detection_min_confidence"],
    }
    return type("FashionConfig", (mconfig.Config,), attrs)()


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


def load_params(
    params_file: str, smoke: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Read the parameters, applying the smoke overrides if requested.

    Parameters
    ----------
    params_file : str
        Path of ``params.yaml``.
    smoke : bool
        Replace the training parameters with ``SMOKE_OVERRIDES``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]
        The ``data``, ``split`` and ``train_matterport`` sections, and the output directory
        (``<output_dir>-smoke`` in smoke mode).
    """
    all_params = yaml.safe_load(Path(params_file).read_text(encoding="utf-8"))
    params = dict(all_params["train_matterport"])
    output_dir = Path(params["output_dir"])
    if smoke:
        params.update(SMOKE_OVERRIDES)
        output_dir = output_dir.with_name(output_dir.name + "-smoke")
    return all_params["data"], all_params["split"], params, output_dir


def build_datasets(
    data: dict[str, Any], params: dict[str, Any], smoke: bool
) -> tuple[FashionDataset, FashionDataset, list[str]]:
    """Build the train and validation datasets from the prepared annotations and the split.

    Parameters
    ----------
    data : dict[str, Any]
        The ``data`` section of ``params.yaml``.
    params : dict[str, Any]
        The ``train_matterport`` section of ``params.yaml``.
    smoke : bool
        Keep only the images present on disk.

    Returns
    -------
    tuple[FashionDataset, FashionDataset, list[str]]
        Training set, validation set, and class names indexed by model class id.

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
    class_names = load_class_names(data["label_file"])
    return (
        FashionDataset(annotations.load(prepared, train_ids), image_dir, class_names),
        FashionDataset(annotations.load(prepared, val_ids), image_dir, class_names),
        class_names,
    )


def train_and_export(
    params: dict[str, Any],
    datasets: tuple[FashionDataset, FashionDataset],
    num_classes: int,
    output_dir: Path,
    smoke: bool,
) -> Path:
    """Train in the active MLflow run, then write ``model/`` and ``metrics.json``.

    Parameters
    ----------
    params : dict[str, Any]
        The ``train_matterport`` section of ``params.yaml``.
    datasets : tuple[FashionDataset, FashionDataset]
        Training and validation sets.
    num_classes : int
        Number of classes, background included.
    output_dir : Path
        Where to write checkpoints, the exported model and the metrics.
    smoke : bool
        Allow training without the COCO weights.

    Returns
    -------
    Path
        The export directory (``config.json`` + SavedModel).

    Raises
    ------
    SystemExit
        If the COCO weights are missing outside smoke mode.
    """
    train_set, val_set = datasets
    batch = params["images_per_gpu"]
    steps = (
        params["steps_per_epoch"] or max(1, len(train_set.image_ids) // batch),
        params["validation_steps"] or max(1, len(val_set.image_ids) // batch),
    )
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    model = modellib.MaskRCNN(
        mode="training", config=build_config(params, num_classes, steps), model_dir=str(checkpoints)
    )
    init_weights = Path(params["init_weights"])
    if init_weights.exists():
        model.load_weights(str(init_weights), by_name=True, exclude=HEAD_LAYERS)
    elif not smoke:
        raise SystemExit(f"{init_weights} missing: run `uv run dvc pull weights/`")

    logger = MlflowEpochLogger()
    model.train(
        train_set,
        val_set,
        learning_rate=params["learning_rate"],
        epochs=params["epochs"],
        layers=params["layers"],
        augmentation=None,
        custom_callbacks=[logger],
    )
    export_dir = output_dir / "model"
    _export_inference_model(params, num_classes, model.find_last(), export_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(logger.last_logs, indent=2) + "\n", encoding="utf-8"
    )
    return export_dir


def main(argv: list[str] | None = None) -> None:
    """Train, log to MLflow, and export the inference model and the final metrics.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments, ``sys.argv[1:]`` when ``None``. By default ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--smoke", action="store_true", help="tiny run on locally pulled images")
    args = parser.parse_args(argv)

    data, split_params, params, output_dir = load_params(args.params, args.smoke)
    train_set, val_set, class_names = build_datasets(data, params, args.smoke)

    setup_experiment(params["experiment"])
    with mlflow.start_run(run_name="matterport-smoke" if args.smoke else "matterport"):
        mlflow.set_tags(
            {
                "model_family": "matterport",
                "maskrcnn_matterport_version": version("maskrcnn-matterport"),
                "tensorflow_version": tf.__version__,
                "smoke": str(args.smoke).lower(),
            }
        )
        mlflow.log_params(
            {
                **params,
                **{f"split.{k}": v for k, v in split_params.items()},
                "n_train_images": len(train_set.image_ids),
                "n_val_images": len(val_set.image_ids),
            }
        )

        export_dir = train_and_export(
            params, (train_set, val_set), len(class_names), output_dir, args.smoke
        )
        if not args.smoke:
            mlflow.log_artifacts(str(export_dir), artifact_path="model")
        print(f"Trained {params['epochs']} epoch(s); inference model exported to {export_dir}")


def _export_inference_model(
    params: dict[str, Any], num_classes: int, checkpoint: str, export_dir: Path
) -> None:
    """Export a checkpoint as an inference model (``config.json`` + SavedModel).

    Parameters
    ----------
    params : dict[str, Any]
        The ``train_matterport`` section of ``params.yaml``.
    num_classes : int
        Number of classes, background included.
    checkpoint : str
        Path of the ``.h5`` weights to export.
    export_dir : Path
        Directory to (re)create with the export.
    """
    config = build_config(params, num_classes, (1, 1), inference=True)
    inference = modellib.MaskRCNN(
        mode="inference", config=config, model_dir=str(export_dir.parent / "checkpoints")
    )
    inference.load_weights(checkpoint, by_name=True)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    inference.save(str(export_dir))


if __name__ == "__main__":
    main()
