"""Adapters: the infrastructure behind the ports, here MLflow.

Each module implements a Protocol of ``fashion_seg.ports`` with plain functions (duck typing):
``mlflow_tracking`` is a ``Tracker``, ``mlflow_models`` a ``ModelRepository``. Only
``fashion_seg.cli`` (the composition root) imports them.
"""
