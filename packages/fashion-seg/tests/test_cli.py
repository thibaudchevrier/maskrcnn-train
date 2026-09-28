"""Tests of the generic command line with an injected fake family (no framework needed)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import yaml
from pydantic import ValidationError

from fashion_seg.cli import main
from fashion_seg.config import TrainConfig
from fashion_seg.ports import FamilySpec, ServingRequirements, TrainInputs, TrainResult
from fashion_seg.service.training import output_dir_of, smoke_config
from fashion_seg_testing import write_dataset, write_params


class FakeConfig(TrainConfig):
    """Training parameters of the fake family.

    Attributes
    ----------
    width : int
        A family-specific parameter.
    smoke_overrides : ClassVar[dict[str, Any]]
        Parameters replaced for a smoke run.
    """

    width: int
    smoke_overrides: ClassVar[dict[str, Any]] = {"epochs": 1}


def _fake_family(calls: list) -> SimpleNamespace:
    """Build a family recording its calls: a duck-typed ModelFamily, like a family module."""

    def train(config: FakeConfig, inputs: TrainInputs, session) -> TrainResult:
        calls.append((config, inputs))
        session.log_metrics({"loss": 1.0}, step=1)
        if config.width == 99:  # stands for a stop request during the training
            return TrainResult(metrics={}, tags={}, stopped_at=1)
        inputs.export_dir.mkdir(parents=True)
        return TrainResult(metrics={"val_loss": 0.5}, tags={"device": "cpu"})

    return SimpleNamespace(
        SPEC=FamilySpec(name="fake", serving=ServingRequirements(pinned=("numpy",))),
        Config=FakeConfig,
        train=train,
        load_predictor=lambda model_dir: None,
        describe=lambda model_dir: {},
    )


@pytest.fixture
def params_file(tmp_path, monkeypatch):
    """Write a synthetic dataset and params.yaml; point MLflow at a temporary store."""
    data = write_dataset(tmp_path, n_images=3, height=12, width=10)
    train = {
        "fake": {
            "output_dir": str(tmp_path / "out" / "fake"),
            "init_weights": str(tmp_path / "missing"),
            "epochs": 5,
            "width": 64,
        }
    }
    models = {"other": {"family": "torchvision", "source": "x", "output_dir": "y"}}
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    return write_params(tmp_path, data, train, models)


def test_train_runs_the_injected_family_through_the_service(params_file):
    """Train validates the family's config, gives it the split, and writes metrics.json."""
    calls = []
    main(_fake_family(calls), ["--params", str(params_file), "train", "--smoke"])

    [(config, inputs)] = calls
    assert config.epochs == 1  # smoke overrides
    assert inputs.train.height == 2 and inputs.val.height == 1
    assert len(inputs.class_names) == 47
    out = params_file.parent / "out" / "fake-smoke"
    assert inputs.export_dir == out / "model"
    assert json.loads((out / "metrics.json").read_text()) == {"val_loss": 0.5}


def test_real_training_needs_the_initial_weights(params_file):
    """Outside smoke runs, missing initial weights stop the run before training."""
    with pytest.raises(SystemExit, match="missing"):
        main(_fake_family([]), ["--params", str(params_file), "train"])


def test_a_family_only_packages_its_own_models(params_file):
    """Packaging another family's model is refused: it needs that family's environment."""
    with pytest.raises(SystemExit, match="not a fake model"):
        main(_fake_family([]), ["--params", str(params_file), "package", "other"])


def test_main_rejects_objects_that_are_not_families():
    """Injecting something that isn't a ModelFamily fails with a clear error."""
    with pytest.raises(TypeError):
        main(SimpleNamespace(SPEC=None), ["train"])


def test_configs_reject_unknown_parameters(tmp_path):
    """A typo in params.yaml fails at load time instead of being ignored."""
    valid = {"output_dir": tmp_path, "init_weights": tmp_path, "epochs": 1, "width": 3}
    assert FakeConfig.model_validate(valid).max_train_images is None
    with pytest.raises(ValidationError):
        FakeConfig.model_validate({**valid, "epoch": 2})


def test_smoke_runs_write_next_to_the_real_output(tmp_path):
    """Smoke runs apply the overrides and use their own directory."""
    config = FakeConfig(output_dir=tmp_path / "fake", init_weights=Path("w"), epochs=9, width=3)
    smoke = smoke_config(config)
    assert (smoke.epochs, smoke.width) == (1, 3)  # overridden, kept
    assert output_dir_of(smoke, smoke=True) == tmp_path / "fake-smoke"


def test_a_stopped_training_exits_without_metrics(params_file):
    """A family that stopped early makes train exit non-zero, asking to resume, no metrics.json."""
    params = yaml.safe_load(params_file.read_text())
    params["train"]["fake"]["width"] = 99
    params_file.write_text(yaml.safe_dump(params))
    with pytest.raises(SystemExit, match="Stopped at step 1"):
        main(_fake_family([]), ["--params", str(params_file), "train", "--smoke"])
    assert not (params_file.parent / "out" / "fake-smoke" / "metrics.json").exists()
