"""Tests of the YOLO family's training: port, isolation, dataset, and runs that stop and resume."""

import json
import os
import threading
from pathlib import Path

import fashion_seg_yolo
from fashion_seg_yolo import isolation, training
from fashion_seg_yolo.dataset import write_dataset as write_yolo_dataset

from fashion_seg.cli import main
from fashion_seg.ports import ModelFamily, TrainingSession
from fashion_seg_testing import write_dataset, write_params


def test_family_implements_the_port():
    """The package is a ModelFamily whose smoke overrides are valid parameters."""
    assert isinstance(fashion_seg_yolo, ModelFamily)
    assert fashion_seg_yolo.SPEC.name == "yolo"
    config = fashion_seg_yolo.Config
    assert set(config.smoke_overrides) <= set(config.model_fields)


def test_ultralytics_is_kept_private_and_offline():
    """Its settings live outside the user's home; analytics and integrations are off."""
    assert os.environ["YOLO_OFFLINE"] == "1"
    isolation.disable_integrations()
    from ultralytics.utils import SETTINGS  # pylint: disable=import-outside-toplevel

    assert SETTINGS.file.parent == Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics"
    assert not SETTINGS["sync"] and not SETTINGS["mlflow"] and not SETTINGS["dvc"]


def test_dataset_is_written_once_as_polygons(inputs):
    """Images are downscaled; each garment becomes a normalized polygon of its category."""
    yaml_path = write_yolo_dataset(inputs, imgsz=80)

    root = inputs.checkpoint_dir / "dataset"
    assert len(list((root / "images" / "train").iterdir())) == 4
    label = sorted((root / "labels" / "train").iterdir())[0]
    [line] = label.read_text().splitlines()
    category, *points = line.split()
    assert category == "5" and len(points) >= 6
    assert all(0.0 <= float(p) <= 1.0 for p in points)
    assert "c0" in yaml_path.read_text() and "BG" not in yaml_path.read_text()
    written = label.stat().st_mtime_ns
    write_yolo_dataset(inputs, imgsz=80)
    assert label.stat().st_mtime_ns == written  # kept, not rewritten


def test_a_stopped_run_resumes_at_its_epoch_start(inputs, config):
    """A stop saves a checkpoint and returns; the next run completes the epoch and exports."""
    stop = threading.Event()

    def stop_after_one_batch(metrics, step=None):  # pylint: disable=unused-argument  # MetricLogger
        """Ask to stop, like Ctrl+C, once the first batch is logged."""
        if step == 1:
            stop.set()

    stopped = training.train(config, inputs, TrainingSession(stop_after_one_batch, stop))
    assert stopped.stopped_at == 1
    assert (inputs.checkpoint_dir / "runs" / "train" / "weights" / "last.pt").exists()
    assert not inputs.export_dir.exists()

    steps = []

    def record(metrics, step=None):
        """Record the steps of the training losses."""
        if "learning_rate" in metrics:
            steps.append(step)

    done = training.train(config, inputs, TrainingSession(record, threading.Event()))
    assert done.stopped_at is None and steps == [1, 2]  # the epoch started over
    assert "val_loss" in done.metrics and "metrics/mAP50-95_M" in done.metrics
    assert (inputs.export_dir / "model.pt").exists()

    again = training.train(config, inputs, TrainingSession(record, threading.Event()))
    assert steps == [1, 2] and again.metrics.keys() >= {"val_loss"}  # nothing left to train


def test_a_new_run_ignores_the_checkpoint(inputs, config):
    """With resume off, a completed run is trained again from scratch."""
    training.train(
        config, inputs, TrainingSession(lambda metrics, step=None: None, threading.Event())
    )
    steps = []

    def record(metrics, step=None):
        """Record the steps of the training losses."""
        if "learning_rate" in metrics:
            steps.append(step)

    config = config.model_copy(update={"resume": False})
    training.train(config, inputs, TrainingSession(record, threading.Event()))
    assert steps == [1, 2]


def test_smoke_training_from_the_command_line(tmp_path, monkeypatch):
    """``train --smoke`` exports a model and writes its metrics."""
    data = write_dataset(tmp_path, n_images=6, height=120, width=160)
    train = {
        "yolo": {
            "output_dir": str(tmp_path / "out" / "yolo"),
            "init_weights": str(tmp_path / "missing.pt"),
            "epochs": 1,
            "model": "yolo11n-seg",
            "imgsz": 640,
            "batch_size": 16,
            "learning_rate": 0.01,
            "num_workers": 0,
            "device": "cpu",
        }
    }
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    main(
        fashion_seg_yolo, ["--params", str(write_params(tmp_path, data, train)), "train", "--smoke"]
    )
    out = tmp_path / "out" / "yolo-smoke"
    assert json.loads((out / "model" / "fashion_seg.json").read_text())["num_labels"] == 46
    assert (out / "model" / "model.pt").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())
