"""MLflow implementation of ``fashion_seg.ports.Tracker``: this module is the tracker.

The tracking store is ``$MLFLOW_TRACKING_URI``, local SQLite (``mlflow.db``) by default.
"""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import polars as pl

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
ARTIFACT_ROOT = Path("mlartifacts")


def setup_experiment(name: str) -> str:
    """Point MLflow at the tracking store and select an experiment, creating it if needed.

    The store is ``$MLFLOW_TRACKING_URI``, local SQLite (``mlflow.db``) by default. With the local
    default, run artifacts go to ``./mlartifacts/<name>`` so they don't mix with the archived,
    DVC-tracked ``./mlruns`` folder.

    Parameters
    ----------
    name : str
        Experiment name, e.g. ``fashion-seg-training``.

    Returns
    -------
    str
        The experiment id.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(name) is None:
        artifact_location = (
            (ARTIFACT_ROOT / name).absolute().as_uri() if uri.startswith("sqlite") else None
        )
        mlflow.create_experiment(name, artifact_location=artifact_location)
    return mlflow.set_experiment(name).experiment_id


@contextmanager
def run(experiment: str, name: str, tags: dict[str, str]) -> Iterator[str]:
    """Open an MLflow run for the duration of a ``with`` block.

    Parameters
    ----------
    experiment : str
        Experiment of the run, created if needed.
    name : str
        Run name.
    tags : dict[str, str]
        Tags set when the run starts.

    Yields
    ------
    str
        The run id.
    """
    setup_experiment(experiment)
    with mlflow.start_run(run_name=name, tags=tags) as active:
        yield active.info.run_id


def set_tags(tags: dict[str, str]) -> None:
    """Tag the active run.

    Parameters
    ----------
    tags : dict[str, str]
        Tags, by name.
    """
    mlflow.set_tags(tags)


def log_params(params: dict[str, Any]) -> None:
    """Record parameters of the active run.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters, by name.
    """
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Record metrics of the active run.

    Parameters
    ----------
    metrics : dict[str, float]
        Metric values, by name.
    step : int | None
        Training step (or epoch). By default ``None``.
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifacts(local_dir: Path, artifact_path: str) -> None:
    """Attach a directory to the active run.

    Parameters
    ----------
    local_dir : Path
        Directory to upload.
    artifact_path : str
        Its path within the run's artifacts.
    """
    mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)


def log_table(table: pl.DataFrame, file_name: str) -> None:
    """Attach a table to the active run, as a CSV file.

    Parameters
    ----------
    table : pl.DataFrame
        The table.
    file_name : str
        Name of the CSV file among the run's artifacts.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / file_name
        table.write_csv(path)
        mlflow.log_artifact(str(path))
