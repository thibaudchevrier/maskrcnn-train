"""Fixtures of the YOLO tests: tiny datasets and training settings (fast on a CPU)."""

import polars as pl
import pytest
from fashion_seg_yolo.config import Config

from fashion_seg.ports import TrainInputs
from fashion_seg_testing import N_CATEGORIES, write_dataset

CLASS_NAMES = ["BG"] + [f"c{k}" for k in range(N_CATEGORIES)]


@pytest.fixture
def inputs(tmp_path):
    """Give four training images and one validation image, each with one garment of category 5."""
    write_dataset(tmp_path, n_images=5, height=120, width=160)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    return TrainInputs(
        train=records[:4],
        val=records[4:],
        image_dir=tmp_path / "images",
        class_names=CLASS_NAMES,
        export_dir=tmp_path / "model",
        checkpoint_dir=tmp_path / "checkpoints",
    )


@pytest.fixture
def config(tmp_path):
    """Train a random YOLO11n-seg at 64 px, on the CPU: two batches per epoch."""
    return Config(
        output_dir=tmp_path,
        init_weights=tmp_path / "missing.pt",
        epochs=1,
        model="yolo11n-seg",
        imgsz=64,
        batch_size=2,
        learning_rate=0.01,
        num_workers=0,
        log_every=1,
        checkpoint_every=100,
        device="cpu",
    )
