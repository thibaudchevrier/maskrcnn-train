"""End-to-end check of Matterport training through the family's CLI, on synthetic data (no DVC)."""

import json
import threading

import polars as pl

import fashion_seg_matterport
from fashion_seg.cli import main
from fashion_seg.ports import TrainingSession, TrainInputs
from fashion_seg_matterport import training
from fashion_seg_matterport.config import Config
from fashion_seg_matterport.dataset import FashionDataset
from fashion_seg_testing import N_CATEGORIES, write_dataset, write_params


def test_dataset_maps_categories_to_model_classes(tmp_path):
    """Dataset category k becomes model class k + 1, with the decoded mask."""
    data = write_dataset(tmp_path, n_images=4, height=96, width=128)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    names = ["BG"] + [f"c{k}" for k in range(N_CATEGORIES)]
    masks, class_ids = FashionDataset(records, data["train_images"], names).load_mask(0)
    assert masks.shape == (96, 128, 1) and masks.sum() == (64 - 16) * (85 - 25)
    assert class_ids.tolist() == [6]  # category 5 -> model class 6


def test_smoke_training_exports_servable_model(tmp_path, monkeypatch):
    """A smoke run trains, exports config.json + SavedModel (training anchors) and metrics."""
    data = write_dataset(tmp_path, n_images=4, height=96, width=128)
    train = {
        "matterport": {
            "output_dir": str(tmp_path / "out" / "matterport"),
            "init_weights": str(tmp_path / "missing.h5"),
            "epochs": 2,
            "backbone": "resnet101",
            "image_min_dim": 800,
            "image_max_dim": 1024,
            "images_per_gpu": 2,
            "rpn_anchor_scales": [16, 32, 64, 128, 256],
            "train_rois_per_image": 32,
            "learning_rate": 0.001,
            "layers": "all",
            "detection_min_confidence": 0.7,
        }
    }
    params_file = write_params(tmp_path, data, train)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)

    main(fashion_seg_matterport, ["--params", str(params_file), "train", "--smoke"])

    out = tmp_path / "out" / "matterport-smoke"
    config = json.loads((out / "model" / "config.json").read_text())
    assert config["NUM_CLASSES"] == 47 and config["RPN_ANCHOR_SCALES"] == [16, 32, 64, 128, 256]
    assert (out / "model" / "saved_model.pb").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())


def test_a_stopped_run_resumes_and_completes(tmp_path):
    """A stop saves weights and position; the next run finishes the epoch, then exports."""
    write_dataset(tmp_path, n_images=4, height=96, width=128)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    inputs = TrainInputs(
        train=records[:3],
        val=records[3:],
        image_dir=tmp_path / "images",
        class_names=["BG"] + [f"c{k}" for k in range(N_CATEGORIES)],
        export_dir=tmp_path / "model",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    config = Config.model_validate(
        {
            "output_dir": tmp_path,
            "init_weights": tmp_path / "missing.h5",
            "epochs": 1,
            "backbone": "resnet50",
            "image_min_dim": 128,
            "image_max_dim": 128,
            "images_per_gpu": 1,
            "rpn_anchor_scales": [8, 16, 32, 64, 128],
            "train_rois_per_image": 16,
            "learning_rate": 0.001,
            "layers": "heads",
            "steps_per_epoch": 3,
            "validation_steps": 1,
            "detection_min_confidence": 0.7,
            "log_every": 1,
        }
    )
    stop = threading.Event()

    def stop_at_step_one(metrics, step=None):  # pylint: disable=unused-argument  # MetricLogger
        """Ask to stop, like Ctrl+C, once step 1 is logged."""
        if step == 1:
            stop.set()

    stopped = training.train(config, inputs, TrainingSession(stop_at_step_one, stop))
    assert stopped.stopped_at == 1
    assert json.loads((tmp_path / "checkpoints" / "last.json").read_text()) == {
        "epoch": 0,
        "global_step": 1,
    }
    assert not (tmp_path / "model").exists()

    steps = []

    def record(metrics, step=None):
        """Record the steps of the training losses."""
        if "train_loss" in metrics:
            steps.append(step)

    result = training.train(config, inputs, TrainingSession(record, threading.Event()))
    assert steps == [2, 3]  # the epoch's 3 steps: 1 before the stop, 2 after
    assert "val_loss" in result.metrics and (tmp_path / "model" / "saved_model.pb").exists()
