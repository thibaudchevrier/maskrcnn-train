"""MLflow tracking setup shared by every workflow step."""

import os
from pathlib import Path

import mlflow

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
