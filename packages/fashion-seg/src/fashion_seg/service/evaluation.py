"""Evaluate step: score a packaged model on the frozen split (COCO mask and box mAP).

The ``ModelRepository`` loads the model exactly as it is served, so the score describes the
deployed artifact, whatever its family. The results are recorded by the ``Tracker`` (a run with
the metrics and the per-class AP table) and, for a whole-split score, as ``<split>_*`` tags on the
registered model version. The command line writes them to ``metrics/evaluate-<model>.json`` too,
for ``dvc metrics show`` / ``dvc metrics diff``.

Run through DVC (``uv run dvc repro --single-item evaluate_legacy``) on the whole split, or quickly
on a subset with ``make evaluate-quick MODEL=legacy``.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import polars as pl
from fashion_seg_contract.labels import load_class_names
from fashion_seg_contract.schema import Prediction

from fashion_seg import scoring
from fashion_seg.config import DataConfig, EvaluateConfig, PackagedModel, Params
from fashion_seg.data import files
from fashion_seg.ports import Infrastructure, Tracker

logger = logging.getLogger(__name__)

# Metrics copied to the registered model version as tags.
VERSION_TAGS = ("mask_map", "mask_ap50", "box_map", "n_images")


def predict_images(
    predict: Callable[[bytes, float], Prediction], paths: list[Path], min_score: float
) -> Iterator[Prediction]:
    """Run the packaged model on image files, one request per image, as serving does.

    Parameters
    ----------
    predict : Callable[[bytes, float], Prediction]
        The loaded model (``ModelRepository.load``).
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
        yield predict(path.read_bytes(), min_score)


def slug(label: str) -> str:
    """Turn a class label into a metric name fragment.

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
    split = files.load_split(data.prepared_dir)[config.split]
    present = files.select_ids(split, data.train_images, local_only=True)
    if len(present) < len(split):
        logger.warning(
            "%d of %d %s images are not pulled", len(split) - len(present), len(split), config.split
        )
    selected = files.select_ids(present, data.train_images, config.max_images)
    if not selected:
        raise SystemExit(f"No {config.split} image in {data.train_images}: run `make pull-val`.")
    return files.load_annotations(data.prepared_dir, selected)


def log_scores(tracker: Tracker, metrics: dict[str, float], per_class: pl.DataFrame) -> None:
    """Record the scores in the current run: metrics, per-class mask AP, the per-class table.

    Parameters
    ----------
    tracker : Tracker
        Records the run.
    metrics : dict[str, float]
        Metrics from ``scoring.score``.
    per_class : pl.DataFrame
        Per-class table from ``scoring.score``.
    """
    tracker.log_metrics(metrics)
    tracker.log_metrics(
        {
            f"mask_ap/{slug(row['label'])}": row["mask_ap"]
            for row in per_class.iter_rows(named=True)
            if row["mask_ap"] == row["mask_ap"]  # skip NaN: no ground truth for the class
        }
    )
    tracker.log_table(per_class, "per_class.csv")


def score_model(
    predict: Callable[[bytes, float], Prediction],
    records: pl.DataFrame,
    data: DataConfig,
    min_score: float,
) -> tuple[dict[str, float], pl.DataFrame]:
    """Predict the selected images and score the predictions.

    Parameters
    ----------
    predict : Callable[[bytes, float], Prediction]
        The loaded model (``ModelRepository.load``).
    records : pl.DataFrame
        Prepared annotations of the selected images.
    data : DataConfig
        Dataset files (images, labels).
    min_score : float
        ``min_score`` request parameter.

    Returns
    -------
    tuple[dict[str, float], pl.DataFrame]
        Metrics from ``scoring.score`` plus ``seconds_per_image``, and the per-class table.
    """
    paths = [data.train_images / f"{i}.jpg" for i in records["image_id"]]
    start = time.monotonic()
    metrics, per_class = scoring.score(
        records, predict_images(predict, paths, min_score), load_class_names(data.label_file)
    )
    metrics["seconds_per_image"] = (time.monotonic() - start) / max(1, records.height)
    return metrics, per_class


def run_details(
    name: str, model: PackagedModel, provenance: dict[str, Any], config: EvaluateConfig
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Name, tags and parameters of the evaluation run.

    Parameters
    ----------
    name : str
        Name of the served model.
    model : PackagedModel
        The served model.
    provenance : dict[str, Any]
        Its provenance (registered name and version, family, model URI).
    config : EvaluateConfig
        Evaluation settings.

    Returns
    -------
    tuple[str, dict[str, str], dict[str, Any]]
        Run name (``<model>-v<version>-<split>``), tags and parameters.
    """
    version = str(provenance["registered_version"])
    tags = {
        "registered_name": provenance["registered_name"],
        "registered_version": version,
        "model_family": provenance["model_family"],
    }
    run_params = {
        "model": name,
        "model_dir": str(model.output_dir),
        "model_uri": provenance["model_uri"],
        **config.model_dump(mode="json", exclude={"output_dir"}),
    }
    return f"{name}-v{version}-{config.split}", tags, run_params


def evaluate(
    name: str,
    model: PackagedModel,
    params: Params,
    infra: Infrastructure,
    max_images: int | None = None,
) -> dict[str, float]:
    """Score a packaged model and record the results.

    Parameters
    ----------
    name : str
        Name of the served model (``params.yaml:models.<name>``).
    model : PackagedModel
        The served model (its ``output_dir`` holds the packaged model).
    params : Params
        The pipeline parameters (data, evaluation settings, tracking).
    infra : Infrastructure
        The repository loads the model and tags its version; the tracker records the run.
    max_images : int | None
        Override ``evaluate.max_images``; a subset score is not copied to the registry.
        By default ``None``.

    Returns
    -------
    dict[str, float]
        The metrics.
    """
    config = params.evaluate
    if max_images is not None:
        config = config.model_copy(update={"max_images": max_images})
    provenance = infra.repository.provenance(model.output_dir)
    records = select_images(params.data, config)
    logger.info("Evaluating %s on %d images", model.output_dir, records.height)
    metrics, per_class = score_model(
        infra.repository.load(model.output_dir), records, params.data, config.min_score
    )

    run_name, tags, run_params = run_details(name, model, provenance, config)
    with infra.tracker.run(params.tracking.evaluation_experiment, run_name, tags):
        infra.tracker.log_params(run_params)
        log_scores(infra.tracker, metrics, per_class)
    if not config.max_images:  # a subset score must not look like the version's score
        infra.repository.tag_version(
            tags["registered_name"],
            tags["registered_version"],
            {f"{config.split}_{key}": str(metrics[key]) for key in VERSION_TAGS},
        )
    return metrics
