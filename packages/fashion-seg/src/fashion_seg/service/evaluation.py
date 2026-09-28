"""Evaluate step: score a packaged model on the frozen split (COCO mask and box mAP).

The model is loaded exactly as it is served (MLflow pyfunc), so the score describes the deployed
artifact, whatever its family. Results go to:

- MLflow: a run in the evaluation experiment (metrics, per-class AP table), and ``<split>_*``
  tags on the registered model version (whole split only);
- ``metrics/evaluate-<model>.json``, for ``dvc metrics show`` / ``dvc metrics diff``.

Run through DVC (``uv run dvc repro --single-item evaluate@legacy``) on the whole split, or quickly
on a subset with ``make evaluate-quick MODEL=legacy``.
"""

import json
import logging
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
import polars as pl
from fashion_seg_contract import request
from fashion_seg_contract.labels import load_class_names
from fashion_seg_contract.schema import Prediction

from fashion_seg import scoring
from fashion_seg.config import DataConfig, EvaluateConfig, PackagedModel, Params
from fashion_seg.data import annotations
from fashion_seg.tracking import setup_experiment

logger = logging.getLogger(__name__)

# Metrics copied to the registered model version as tags.
VERSION_TAGS = ("mask_map", "mask_ap50", "box_map", "n_images")


def predict_images(
    model: mlflow.pyfunc.PyFuncModel, paths: list[Path], min_score: float
) -> Iterator[Prediction]:
    """Run the packaged model on images, one request per image, as serving does.

    Parameters
    ----------
    model : mlflow.pyfunc.PyFuncModel
        The packaged model.
    paths : list[Path]
        Image files.
    min_score : float
        ``min_score`` request parameter.

    Yields
    ------
    Prediction
        The model's response for each image, in order.
    """
    for path in paths:
        rows = pd.DataFrame({request.IMAGE_FIELD: [request.encode_image(path.read_bytes())]})
        yield model.predict(rows, params={request.MIN_SCORE_PARAM: min_score})[0]


def slug(label: str) -> str:
    """Turn a class label into an MLflow metric name fragment.

    Parameters
    ----------
    label : str
        Class label, e.g. ``"shirt, blouse"``.

    Returns
    -------
    str
        Lower-case, underscores only, e.g. ``"shirt_blouse"``.

    Examples
    --------
    >>> slug("top, t-shirt, sweatshirt")
    'top_t_shirt_sweatshirt'
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def select_images(data: DataConfig, config: EvaluateConfig) -> pl.DataFrame:
    """Pick the split's images to evaluate: the first ``max_images`` that are on disk.

    Parameters
    ----------
    data : DataConfig
        Dataset files.
    config : EvaluateConfig
        Evaluation settings (split, ``max_images``).

    Returns
    -------
    pl.DataFrame
        Prepared annotations of the selected images, in split order.

    Raises
    ------
    SystemExit
        If none of the split's images is on disk.
    """
    split = annotations.load_split(data.prepared_dir / annotations.SPLIT_FILE)[config.split]
    present = annotations.select_ids(split, data.train_images, local_only=True)
    if len(present) < len(split):
        logger.warning(
            "%d of %d %s images are not pulled", len(split) - len(present), len(split), config.split
        )
    selected = annotations.select_ids(present, data.train_images, config.max_images)
    if not selected:
        raise SystemExit(f"No {config.split} image in {data.train_images}: run `make pull-val`.")
    return annotations.load(data.prepared_dir / annotations.ANNOTATIONS_FILE, selected)


def log_run(
    experiment: str,
    run_name: str,
    run_params: dict[str, Any],
    metrics: dict[str, float],
    per_class: pl.DataFrame,
) -> None:
    """Log an evaluation run: parameters, metrics, per-class AP and the per-class table.

    Parameters
    ----------
    experiment : str
        MLflow experiment.
    run_name : str
        Run name.
    run_params : dict[str, Any]
        Run parameters; ``registered_name``, ``registered_version`` and ``model_family`` are
        logged as tags.
    metrics : dict[str, float]
        Metrics from ``scoring.score``.
    per_class : pl.DataFrame
        Per-class table from ``scoring.score``.
    """
    tag_keys = ("registered_name", "registered_version", "model_family")
    setup_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({k: str(run_params[k]) for k in tag_keys})
        mlflow.log_params({k: v for k, v in run_params.items() if k not in tag_keys})
        mlflow.log_metrics(metrics)
        for row in per_class.iter_rows(named=True):
            if row["mask_ap"] == row["mask_ap"]:  # skip NaN: no ground truth for the class
                mlflow.log_metric(f"mask_ap/{slug(row['label'])}", row["mask_ap"])
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "per_class.csv"
            per_class.write_csv(table)
            mlflow.log_artifact(str(table))


def tag_version(provenance: dict[str, Any], split: str, metrics: dict[str, float]) -> None:
    """Copy the main metrics to the registered model version as ``<split>_<metric>`` tags.

    Parameters
    ----------
    provenance : dict[str, Any]
        The packaged model's ``provenance.json`` (registered name and version).
    split : str
        Evaluated split.
    metrics : dict[str, float]
        Metrics from ``scoring.score``.
    """
    client = mlflow.MlflowClient()
    name, version = provenance["registered_name"], str(provenance["registered_version"])
    for key in VERSION_TAGS:
        client.set_model_version_tag(name, version, f"{split}_{key}", metrics[key])


def evaluate(
    name: str,
    model: PackagedModel,
    params: Params,
    *,
    max_images: int | None = None,
    output: Path | None = None,
) -> dict[str, float]:
    """Score a packaged model, log the results and write the metrics file.

    Parameters
    ----------
    name : str
        Name of the served model (``params.yaml:models.<name>``).
    model : PackagedModel
        The served model (its ``output_dir`` holds the packaged model).
    params : Params
        The pipeline parameters (data, evaluation settings, tracking).
    max_images : int | None
        Override ``evaluate.max_images``; a subset score is not copied to the registry.
        By default ``None``.
    output : Path | None
        Metrics file. By default ``None``: ``<evaluate.output_dir>/evaluate-<name>.json``.

    Returns
    -------
    dict[str, float]
        The metrics.
    """
    config = params.evaluate
    if max_images is not None:
        config = config.model_copy(update={"max_images": max_images})
    provenance = json.loads((model.output_dir / "provenance.json").read_text(encoding="utf-8"))
    records = select_images(params.data, config)
    logger.info("Evaluating %s on %d images", model.output_dir, records.height)

    pyfunc_model = mlflow.pyfunc.load_model(str(model.output_dir))
    paths = [params.data.train_images / f"{i}.jpg" for i in records["image_id"]]
    start = time.monotonic()
    metrics, per_class = scoring.score(
        records,
        predict_images(pyfunc_model, paths, config.min_score),
        load_class_names(params.data.label_file),
    )
    metrics["seconds_per_image"] = (time.monotonic() - start) / max(1, records.height)

    run_params = {
        **provenance,
        "model": name,
        "model_dir": str(model.output_dir),
        **config.model_dump(mode="json", exclude={"output_dir"}),
    }
    run_name = f"{name}-v{provenance['registered_version']}-{config.split}"
    log_run(params.tracking.evaluation_experiment, run_name, run_params, metrics, per_class)
    if not config.max_images:  # a subset score must not look like the version's score
        tag_version(provenance, config.split, metrics)

    output = output or config.output_dir / f"evaluate-{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
