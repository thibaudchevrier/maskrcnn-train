"""MLflow tracking setup shared by training and packaging scripts."""

import os
from pathlib import Path

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
ARTIFACT_ROOT = Path("mlartifacts")


def setup_experiment(name: str) -> str:
    """Point MLflow at ``$MLFLOW_TRACKING_URI`` (local SQLite by default) and select ``name``.

    With the local default, run artifacts go to ``./mlartifacts`` so they don't
    mix with the legacy DVC-tracked ``./mlruns`` folder.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(name) is None:
        artifact_location = (
            (ARTIFACT_ROOT / name).absolute().as_uri() if uri.startswith("sqlite") else None
        )
        mlflow.create_experiment(name, artifact_location=artifact_location)
    return mlflow.set_experiment(name).experiment_id
