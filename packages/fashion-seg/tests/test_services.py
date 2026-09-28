"""Tests of the workflow steps with in-memory infrastructure: no MLflow, no model files."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fashion_seg_contract import rle

from fashion_seg.config import PackagedModel, TrainConfig, load_params
from fashion_seg.ports import (
    FamilySpec,
    Infrastructure,
    ModelRepository,
    ServingRequirements,
    Tracker,
    TrainResult,
)
from fashion_seg.service import evaluation, packaging, training
from fashion_seg_testing import garment_mask, write_dataset, write_params


class FakeTracker:
    """Tracker recording every run in memory."""

    def __init__(self):
        """Start with no run."""
        self.runs = []

    @contextmanager
    def run(self, experiment, name, tags):
        """Open a recorded run."""
        self.runs.append(
            {
                "experiment": experiment,
                "name": name,
                "tags": dict(tags),
                "params": {},
                "metrics": {},
                "artifacts": [],
                "tables": [],
            }
        )
        yield f"run-{len(self.runs)}"

    def set_tags(self, tags):
        """Record tags."""
        self.runs[-1]["tags"].update(tags)

    def log_params(self, params):
        """Record parameters."""
        self.runs[-1]["params"].update(params)

    # pylint: disable-next=unused-argument  # the port's signature
    def log_metrics(self, metrics, step=None):
        """Record metrics."""
        self.runs[-1]["metrics"].update(metrics)

    def log_artifacts(self, local_dir, artifact_path):
        """Record an artifact directory."""
        self.runs[-1]["artifacts"].append((Path(local_dir), artifact_path))

    def log_table(self, table, file_name):
        """Record a table's name and size."""
        self.runs[-1]["tables"].append((file_name, table.height))


class FakeRepository:
    """Repository whose model predicts the synthetic dataset's garment perfectly."""

    def __init__(self, height, width):
        """Serve images of this size."""
        self.size = (height, width)
        self.published = []
        self.tags = {}

    def publish(self, package, registered_name, output_dir):
        """Record the package and return its provenance."""
        self.published.append((package, registered_name, output_dir))
        return {"registered_name": registered_name, "registered_version": 7}

    # pylint: disable-next=unused-argument  # the port's signature
    def provenance(self, model_dir):
        """Return a fixed provenance."""
        return {
            "registered_name": "test-model",
            "registered_version": 7,
            "model_family": "fake",
            "model_uri": "models:/x",
        }

    # pylint: disable-next=unused-argument  # the port's signature
    def load(self, model_dir):
        """Return a model answering the ground truth (category 5, model class 6)."""
        height, width = self.size
        mask = garment_mask(height, width)
        ys, xs = np.nonzero(mask)
        instance = {
            "class_id": 6,
            "label": "c5",
            "score": 0.9,
            "box": [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1],
            "mask_rle": rle.encode(mask),
        }
        return lambda image, min_score: {"height": height, "width": width, "instances": [instance]}

    def tag_version(self, registered_name, version, tags):
        """Record version tags."""
        self.tags[(registered_name, version)] = tags


@pytest.fixture
def params(tmp_path, monkeypatch):
    """Parameters of a synthetic dataset (2 train, 1 val images), with one served model."""
    monkeypatch.chdir(tmp_path)
    data = write_dataset(tmp_path, n_images=3, height=12, width=10)
    models = {"m": {"family": "fake", "source": "export", "output_dir": "packaged"}}
    return load_params(write_params(tmp_path, data, train={}, models=models))


def test_fakes_implement_the_ports():
    """The in-memory fakes satisfy the Protocols, like the MLflow adapters."""
    assert isinstance(FakeTracker(), Tracker)
    assert isinstance(FakeRepository(1, 1), ModelRepository)


def test_evaluation_scores_logs_and_tags_the_whole_split(params):
    """A perfect model scores 1; the run is recorded and the registered version tagged."""
    tracker, repository = FakeTracker(), FakeRepository(12, 10)
    model = params.models["m"]
    metrics = evaluation.evaluate("m", model, params, Infrastructure(tracker, repository))

    assert metrics["mask_map"] == pytest.approx(1.0)
    [run] = tracker.runs
    assert run["name"] == "m-v7-val" and run["tags"]["registered_version"] == "7"
    assert run["metrics"]["mask_ap/c5"] == pytest.approx(1.0)
    assert run["tables"] == [("per_class.csv", 46)]
    assert repository.tags[("test-model", "7")]["val_mask_map"] == str(metrics["mask_map"])


def test_evaluation_of_a_subset_does_not_tag_the_version(params):
    """A subset score must not look like the version's score in the registry."""
    repository = FakeRepository(12, 10)
    infra = Infrastructure(FakeTracker(), repository)
    evaluation.evaluate("m", params.models["m"], params, infra, max_images=1)
    assert not repository.tags


def test_training_records_the_run_and_the_family_metrics(params, tmp_path):
    """The family's metrics go through the tracker; tags, parameters and export are recorded."""
    config = TrainConfig(output_dir=tmp_path / "out", init_weights=tmp_path, epochs=1)

    # pylint: disable-next=unused-argument  # the family's signature
    def train(cfg, inputs, log_metrics):
        """Report one training loss and the final validation loss."""
        log_metrics({"loss": 0.5}, step=1)
        return TrainResult(metrics={"val_loss": 0.4}, tags={"device": "cpu"})

    family = SimpleNamespace(SPEC=SimpleNamespace(name="fake"), train=train)
    tracker = FakeTracker()
    result = training.train(family, config, params, tracker)

    [run] = tracker.runs
    assert result.metrics == {"val_loss": 0.4}
    assert run["metrics"] == {"loss": 0.5} and run["tags"]["device"] == "cpu"
    assert run["params"]["n_train_images"] == 2
    assert run["artifacts"] == [(tmp_path / "out" / "model", "model")]


def test_packaging_publishes_the_wrapper_and_family_requirements(params):
    """The package pins the wrapper's requirements with the family's, and bundles both codes."""
    family = SimpleNamespace(
        SPEC=FamilySpec(name="fake", serving=ServingRequirements(pinned=("numpy",))),
        load_predictor=np.asarray,  # any importable function: its package is bundled
        describe=lambda model_dir: {"layers": 3},
    )
    tracker, repository = FakeTracker(), FakeRepository(1, 1)
    model = PackagedModel(family="fake", source=Path("export"), output_dir=Path("packaged"))
    provenance = packaging.package(family, "m", model, params, Infrastructure(tracker, repository))

    [(package, name, output_dir)] = repository.published
    assert provenance["registered_version"] == 7 and name == "test-model"
    assert output_dir == Path("packaged")
    assert any(r.startswith("mlflow==") for r in package.requirements)
    assert any(r.startswith("fashion-seg-contract @ ") for r in package.requirements)
    assert [p.name for p in package.code_dirs] == ["fashion_seg", "numpy"]
    assert tracker.runs[0]["params"] == {"model.layers": 3}
