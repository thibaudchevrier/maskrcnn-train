"""Package step: wrap any family's export as a self-contained MLflow pyfunc model, and register it.

For the served model ``params.yaml:models.<name>``:

- logs a run in MLflow and registers a new version of the model;
- saves the same model to ``output_dir`` so DVC can version and push it; fashion-serving imports
  it with ``dvc import``.

Every model is served by ``FashionSegmentationModel`` (the contract), with the family's predictor.
It bundles the ``fashion_seg`` code and pins its runtime requirements to the versions installed
when packaging (released packages by wheel URL), so a serving environment only needs
``pip install -r requirements.txt``. Each family ships only its own framework.

Run through DVC: ``uv run dvc repro --single-item package@legacy`` (or ``package@torchvision``).
"""

import json
import shutil
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

import fashion_seg
from fashion_seg.config import PackagedModel, Params
from fashion_seg.ports import ModelFamily, ServingRequirements
from fashion_seg.serving.pyfunc import (
    DEFAULT_MIN_SCORE,
    IMAGE_COLUMN,
    FashionSegmentationModel,
    encode_image,
)
from fashion_seg.tracking import setup_experiment

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", IMAGE_COLUMN)]),
    params=ParamSchema([ParamSpec("min_score", "double", DEFAULT_MIN_SCORE)]),
)


def pip_requirements(requirements: ServingRequirements) -> list[str]:
    """List a family's runtime requirements, pinned to what is installed.

    Parameters
    ----------
    requirements : ServingRequirements
        The family's serving requirements.

    Returns
    -------
    list[str]
        pip requirement lines: options, pinned versions, released wheels by URL.
    """
    pinned = [f"{pkg}=={version(pkg)}" for pkg in requirements.pinned]
    released = [f"{pkg} @ {_installed_from(pkg)}" for pkg in requirements.released]
    return [*requirements.pip_options, *pinned, *released]


def _installed_from(name: str) -> str:
    """Find the URL a package was installed from.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    str
        The URL recorded in the package's ``direct_url.json``.

    Raises
    ------
    RuntimeError
        If the package was not installed from a URL.
    """
    direct_url = distribution(name).read_text("direct_url.json")
    if not direct_url:
        raise RuntimeError(f"{name} was not installed from a URL: check [tool.uv.sources]")
    return json.loads(direct_url)["url"]


def input_example() -> dict[str, list[str]]:
    """Build a small random image request, used by MLflow to validate the model and document it.

    Returns
    -------
    dict[str, list[str]]
        One base64 PNG under the ``image`` column.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 48, 3), dtype=np.uint8)
    return {IMAGE_COLUMN: [encode_image(image)]}


def package(family: ModelFamily, name: str, model: PackagedModel, params: Params) -> dict[str, Any]:
    """Log, register and save a family's export as an MLflow pyfunc model.

    Parameters
    ----------
    family : ModelFamily
        The export's model family.
    name : str
        Name of the served model (``params.yaml:models.<name>``), in the MLflow run name.
    model : PackagedModel
        The export to package, and where to write the packaged model.
    params : Params
        The pipeline parameters (label file, tracking).

    Returns
    -------
    dict[str, Any]
        Provenance of the packaged model (MLflow run, model URI, family, registered name and
        version), also written to ``provenance.json``.
    """
    model_kwargs = {
        "python_model": FashionSegmentationModel(),
        "artifacts": {"model": str(model.source), "labels": str(params.data.label_file)},
        "model_config": {"predictor": family.SPEC.name},
        "code_paths": [str(Path(fashion_seg.__file__).parent)],
        "pip_requirements": pip_requirements(family.SPEC.serving),
        "signature": SIGNATURE,
        "input_example": input_example(),
    }
    registered_name = params.tracking.registered_name
    setup_experiment(params.tracking.packaging_experiment)
    with mlflow.start_run(run_name=f"package-{name}") as run:
        mlflow.set_tags({"model_family": family.SPEC.name, "packaged_model": name})
        mlflow.log_params({f"model.{k}": v for k, v in family.describe(model.source).items()})
        info = mlflow.pyfunc.log_model(
            name="model", registered_model_name=registered_name, **model_kwargs
        )
    if model.output_dir.exists():
        shutil.rmtree(model.output_dir)
    mlflow.pyfunc.save_model(path=str(model.output_dir), **model_kwargs)
    provenance = {
        "mlflow_run_id": run.info.run_id,
        "model_uri": info.model_uri,
        "model_family": family.SPEC.name,
        "registered_name": registered_name,
        "registered_version": info.registered_model_version,
    }
    (model.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance
