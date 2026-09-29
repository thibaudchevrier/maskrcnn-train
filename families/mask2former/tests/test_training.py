"""Tests of the Mask2Former family's training: port, dataset, losses, and a smoke run."""

import json

import fashion_seg_mask2former
import polars as pl
import pytest
import torch
from fashion_seg_mask2former import training
from fashion_seg_mask2former.dataset import FashionDataset, collate

from fashion_seg.cli import main
from fashion_seg.ports import ModelFamily
from fashion_seg_testing import write_dataset, write_params


def test_family_implements_the_port():
    """The package is a ModelFamily whose smoke overrides are valid parameters."""
    assert isinstance(fashion_seg_mask2former, ModelFamily)
    assert fashion_seg_mask2former.SPEC.name == "mask2former"
    config = fashion_seg_mask2former.Config
    assert set(config.smoke_overrides) <= set(config.model_fields)


def test_dataset_gives_masks_and_categories(tmp_path):
    """Images are downscaled with their masks; class indices are the dataset categories."""
    data = write_dataset(tmp_path, n_images=3, height=120, width=160)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    image, masks, labels = FashionDataset(records, data["train_images"], 80)[0]
    assert tuple(image.shape) == (3, 60, 80) and tuple(masks.shape) == (1, 60, 80)
    assert labels.tolist() == [5]  # category 5, i.e. model class 6


def test_loss_components_add_up_to_mask2formers_loss(tmp_path, tiny_model):
    """The logged components (summed over decoder layers) are exactly the trained loss."""
    data = write_dataset(tmp_path, n_images=3, height=64, width=96)
    records = pl.read_parquet(tmp_path / "prepared" / "annotations.parquet")
    dataset = FashionDataset(records, data["train_images"], 96)
    batch = collate([dataset[0], dataset[1]])
    torch.manual_seed(1)
    components = training.losses(tiny_model, batch, torch.device("cpu"))
    torch.manual_seed(1)
    reference = tiny_model(
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        mask_labels=batch["mask_labels"],
        class_labels=batch["class_labels"],
    ).loss
    assert set(components) == {"loss_cross_entropy", "loss_mask", "loss_dice"}
    assert sum(components.values()).item() == pytest.approx(reference.item(), rel=1e-5)


def test_smoke_training_exports_and_resumes(tmp_path, monkeypatch, caplog):
    """A smoke run exports the model; a second run resumes from the checkpoint."""
    data = write_dataset(tmp_path, n_images=5, height=120, width=160)
    train = {
        "mask2former": {
            "output_dir": str(tmp_path / "out" / "mask2former"),
            "init_weights": str(tmp_path / "missing"),
            "epochs": 1,
            "max_size": 128,
            "batch_size": 2,
            "learning_rate": 0.0001,
            "warmup_steps": 1,
            "num_workers": 0,
            "device": "cpu",
        }
    }
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    argv = ["--params", str(write_params(tmp_path, data, train)), "train", "--smoke"]
    main(fashion_seg_mask2former, argv)
    out = tmp_path / "out" / "mask2former-smoke"
    assert json.loads((out / "model" / "fashion_seg.json").read_text())["num_labels"] == 46
    assert (out / "model" / "model.safetensors").exists()
    assert "val_loss" in json.loads((out / "metrics.json").read_text())

    caplog.set_level("INFO")
    main(fashion_seg_mask2former, argv)
    assert "Resuming at epoch 1" in caplog.text
