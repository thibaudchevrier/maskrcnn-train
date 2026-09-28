"""Tests of the architecture: families implement the port, modules respect the dependency rules."""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from fashion_seg import registry, runtime
from fashion_seg.config import TrainConfig
from fashion_seg.ports import ModelFamily, Runtime
from fashion_seg.service.training import output_dir_of, smoke_config

PACKAGE = Path(__file__).parents[1] / "src" / "fashion_seg"

# Which fashion_seg modules each layer may import (inner layers know nothing of outer ones).
ALLOWED = {
    "config": set(),
    "data": set(),
    "scoring": set(),
    "tracking": set(),
    "ports": {"config"},
    "runtime": {"ports"},
    "registry": {"families", "ports"},
    "families": {"config", "ports", "families"},
    "serving": {"registry", "ports"},
    "service": {"config", "data", "ports", "scoring", "serving", "tracking"},
}
# MLflow is the workflow's business (and the serving wrapper's, tracking's): families record
# metrics through a MetricLogger, data and scoring know nothing of tracking.
MLFLOW_LAYERS = {"service", "serving", "tracking"}


def _imports(path: Path) -> set[str]:
    """Names of the modules a file imports (``fashion_seg.x.y`` kept whole)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            base = node.module
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _layer(path: Path) -> str:
    """Name the layer of a module: its first component under ``fashion_seg``."""
    return path.relative_to(PACKAGE).parts[0].removesuffix(".py")


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.name))
def test_modules_respect_the_dependency_rules(path):
    """Each module imports only the layers it may depend on; only __main__ depends on all."""
    layer = _layer(path)
    if layer in ("__init__", "__main__"):
        return
    imported = {
        name.split(".")[1] for name in _imports(path) if name.startswith("fashion_seg.")
    } - {layer}
    assert imported <= ALLOWED[layer], f"{layer} imports {imported - ALLOWED[layer]}"
    if layer not in MLFLOW_LAYERS:
        assert "mlflow" not in {name.split(".")[0] for name in _imports(path)}


@pytest.mark.parametrize("name", registry.names())
def test_every_family_implements_the_port(name):
    """Each family module is a ModelFamily whose name, config and smoke overrides are valid."""
    family = registry.get_family(name)
    assert isinstance(family, ModelFamily)
    assert family.SPEC.name == name
    assert issubclass(family.Config, TrainConfig)
    assert set(family.Config.smoke_overrides) <= set(family.Config.model_fields)


def test_unknown_family_is_rejected():
    """Asking for a family that does not exist names the available ones."""
    with pytest.raises(ValueError, match="torchvision"):
        registry.get_family("nope")


def test_configs_reject_unknown_parameters(tmp_path):
    """A typo in params.yaml fails at load time instead of being ignored."""
    config = registry.get_family("torchvision").Config
    valid = {
        "output_dir": tmp_path,
        "init_weights": tmp_path / "w.pth",
        "epochs": 1,
        "min_size": 8,
        "max_size": 8,
        "batch_size": 1,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "num_workers": 0,
        "log_every": 1,
    }
    assert config.model_validate(valid).device == "auto"
    with pytest.raises(ValidationError):
        config.model_validate({**valid, "epoch": 2})


def test_smoke_runs_use_the_overrides_and_their_own_directory(tmp_path):
    """Smoke runs apply the family's overrides and write next to the real output."""
    config = registry.get_family("matterport").Config.model_validate(
        {
            "output_dir": tmp_path / "matterport",
            "init_weights": tmp_path / "w.h5",
            "epochs": 30,
            "backbone": "resnet101",
            "image_min_dim": 800,
            "image_max_dim": 1024,
            "images_per_gpu": 2,
            "rpn_anchor_scales": [16, 32],
            "train_rois_per_image": 32,
            "learning_rate": 0.001,
            "layers": "all",
            "detection_min_confidence": 0.7,
        }
    )
    smoke = smoke_config(config)
    assert (smoke.epochs, smoke.backbone, smoke.output_dir) == (1, "resnet50", config.output_dir)
    assert output_dir_of(smoke, smoke=True) == tmp_path / "matterport-smoke"


def test_runtime_switches_environment_only_when_needed():
    """The current Python is active; another one runs through uv, the root one cannot."""
    current = registry.get_family("torchvision").SPEC.train_runtime
    assert runtime.is_active(current)
    assert runtime.command(Runtime(python="3.11", project="envs/x"), ["train", "x"])[-2:] == [
        "train",
        "x",
    ]
    with pytest.raises(RuntimeError):
        runtime.command(Runtime(python="3.10"), [])
