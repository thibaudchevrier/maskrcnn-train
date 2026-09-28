"""Tests of the MLflow tracker adapter against a temporary tracking store.

The model repository adapter is tested by each family, on a real export (``test_serving.py``).
"""

import mlflow
import polars as pl

from fashion_seg.adapters import mlflow_models, mlflow_tracking
from fashion_seg.ports import ModelRepository, Tracker


def test_adapters_implement_the_ports():
    """The adapter modules are a Tracker and a ModelRepository."""
    assert isinstance(mlflow_tracking, Tracker)
    assert isinstance(mlflow_models, ModelRepository)


def test_tracker_records_a_run(tmp_path, monkeypatch):
    """A run gets its tags, parameters, metrics and table, readable back from MLflow."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)

    with mlflow_tracking.run("exp", "name", {"a": "1"}) as run_id:
        mlflow_tracking.set_tags({"b": "2"})
        mlflow_tracking.log_params({"p": 3})
        mlflow_tracking.log_metrics({"m": 0.5}, step=1)
        mlflow_tracking.log_table(pl.DataFrame({"x": [1, 2]}), "table.csv")

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "name"
    assert {"a": "1", "b": "2"}.items() <= run.data.tags.items()
    assert run.data.params == {"p": "3"} and run.data.metrics == {"m": 0.5}
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)] == ["table.csv"]
