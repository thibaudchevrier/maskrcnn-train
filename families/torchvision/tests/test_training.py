"""End-to-end check of the torchvision family through its CLI, on synthetic data (CPU, no DVC)."""

import json
import threading

import polars as pl
import pytest

import fashion_seg_torchvision
from fashion_seg.cli import main
from fashion_seg.ports import ModelFamily, TrainingSession, TrainInputs
from fashion_seg_testing import write_dataset, write_params
from fashion_seg_torchvision import training
from fashion_seg_torchvision.config import Config
from fashion_seg_torchvision.dataset import FashionDataset


def test_family_implements_the_port():
    """The package is a ModelFamily whose smoke overrides are valid parameters."""
    assert isinstance(fashion_seg_torchvision, ModelFamily)
    assert fashion_seg_torchvision.SPEC.name == "torchvision"
    config = fashion_seg_torchvision.Config
    assert set(config.smoke_overrides) <= set(config.model_fields)


def test_dataset_downscales_and_maps_classes(tmp_path):
    """Images are downscaled to max_side, masks follow, category k becomes class k + 1."""
    data = write_dataset(tmp_path, n_images=5, height=120, width=160)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    image, targets = FashionDataset(records, data["train_images"], 80)[0]
    assert tuple(image.shape) == (3, 60, 80)
    assert tuple(targets["masks"].shape) == (1, 60, 80)
    assert targets["labels"].tolist() == [6]
    assert targets["boxes"].tolist() == [[16.0, 10.0, 53.0, 40.0]]


@pytest.fixture
def params_file(tmp_path, monkeypatch):
    """Write a params.yaml for a CPU smoke run and point MLflow at a temporary store."""
    data = write_dataset(tmp_path, n_images=5, height=120, width=160)
    train = {
        "torchvision": {
            "output_dir": str(tmp_path / "out" / "torchvision"),
            "init_weights": str(tmp_path / "missing.pth"),
            "epochs": 1,
            "min_size": 800,
            "max_size": 1024,
            "batch_size": 2,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "warmup_steps": 10,
            "num_workers": 2,
            "log_every": 50,
            "device": "cpu",
        }
    }
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    return write_params(tmp_path, data, train)


def test_smoke_training_exports_and_resumes(params_file, caplog):
    """A smoke run exports config.json + model.pt; a second run resumes from the checkpoint."""
    argv = ["--params", str(params_file), "train", "--smoke"]
    main(fashion_seg_torchvision, argv)
    out = params_file.parent / "out" / "torchvision-smoke"
    config = json.loads((out / "model" / "config.json").read_text())
    assert config["num_classes"] == 47 and config["architecture"] == "maskrcnn_resnet50_fpn_v2"
    assert (out / "model" / "model.pt").exists() and (out / "checkpoints" / "last.pt").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())

    caplog.set_level("INFO")
    main(fashion_seg_torchvision, argv)
    assert "Resuming at epoch 1" in caplog.text
    # nothing left to train: the metrics are the checkpoint's, not empty
    assert "val_loss" in json.loads((out / "metrics.json").read_text())


def test_sampler_order_is_reproducible_and_resumable():
    """An epoch's order is the same every time, and resuming skips the images already seen."""
    sampler = training.EpochSampler(10, seed=3)
    sampler.set_position(epoch=1, start=0)
    full = list(sampler)
    sampler.set_position(epoch=1, start=4)
    assert list(sampler) == full[4:] and len(sampler) == 6
    sampler.set_position(epoch=2, start=0)
    assert sorted(sampler) == list(range(10)) and list(sampler) != full


def test_a_stopped_run_resumes_mid_epoch(tmp_path):
    """A stop saves a checkpoint and returns; the next run continues at the following step."""
    write_dataset(tmp_path, n_images=5, height=120, width=160)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    inputs = TrainInputs(
        train=records[:4],
        val=records[4:],
        image_dir=tmp_path / "images",
        class_names=["BG"] + [f"c{k}" for k in range(46)],
        export_dir=tmp_path / "model",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    config = Config(
        output_dir=tmp_path,
        init_weights=tmp_path / "missing.pth",
        epochs=1,
        min_size=64,
        max_size=96,
        batch_size=1,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0,
        warmup_steps=1,
        num_workers=0,
        log_every=1,
        checkpoint_every=100,
        device="cpu",
    )

    stop = threading.Event()

    def stop_at_step_two(metrics, step=None):  # pylint: disable=unused-argument  # MetricLogger
        """Ask to stop, like Ctrl+C, once step 2 is logged."""
        if step == 2:
            stop.set()

    stopped = training.train(config, inputs, TrainingSession(stop_at_step_two, stop))
    assert stopped.stopped_at == 2
    assert (tmp_path / "checkpoints" / "last.pt").exists()
    assert not (tmp_path / "model").exists()

    steps = []

    def record(metrics, step=None):
        """Record the steps of the training losses."""
        if "train_loss" in metrics:
            steps.append(step)

    result = training.train(config, inputs, TrainingSession(record, threading.Event()))
    assert steps == [3, 4]  # the 4 images of the epoch: 2 before the stop, 2 after
    assert "val_loss" in result.metrics and (tmp_path / "model" / "model.pt").exists()
