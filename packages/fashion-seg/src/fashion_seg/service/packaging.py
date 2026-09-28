"""Package step: wrap any family's export as a self-contained MLflow pyfunc model, and register it.

For the served model ``params.yaml:models.<name>``:

- logs a run in MLflow and registers a new version of the model;
- saves the same model to ``output_dir`` so DVC can version and push it; fashion-serving imports
  it with ``dvc import``.

Every model is served by ``FashionSegmentationModel`` (the contract), with the family's predictor.
It bundles the code of ``fashion_seg`` and of the family, and pins its runtime requirements to the
versions installed in the family's environment (released packages by wheel URL), so a serving
environment only needs ``pip install -r requirements.txt``. Each family ships only its framework.

Run through DVC in the family's serving environment: ``uv run dvc repro --single-item
package_legacy`` (or ``package_torchvision``).
"""

import importlib
import json
import shutil
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from fashion_seg_contract import request
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

import fashion_seg
from fashion_seg.config import PackagedModel, Params
from fashion_seg.ports import ModelFamily, ServingRequirements
from fashion_seg.serving.pyfunc import FashionSegmentationModel, encode_image
from fashion_seg.tracking import setup_experiment

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", request.IMAGE_FIELD)]),
    params=ParamSchema([ParamSpec(request.MIN_SCORE_PARAM, "double", request.DEFAULT_MIN_SCORE)]),
)

# What the serving wrapper itself needs, whatever the family: added to each family's own.
WRAPPER_REQUIREMENTS = ServingRequirements(
    pinned=("mlflow", "pydantic", "numpy", "pandas", "pillow"),
    released=("fashion-seg-contract",),
)


def pip_requirements(requirements: ServingRequirements) -> list[str]:
    """List a packaged model's runtime requirements, pinned to what is installed.

    Parameters
    ----------
    requirements : ServingRequirements
        The family's serving requirements (its framework); the wrapper's are added.

    Returns
    -------
    list[str]
        pip requirement lines: options, pinned versions, released wheels by URL.
    """
    pinned = [*WRAPPER_REQUIREMENTS.pinned, *requirements.pinned]
    released = [*WRAPPER_REQUIREMENTS.released, *requirements.released]
    return [
        *requirements.pip_options,
        *(f"{pkg}=={version(pkg)}" for pkg in pinned),
        *(f"{pkg} @ {_installed_from(pkg)}" for pkg in released),
    ]


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


def code_paths(family: ModelFamily) -> list[str]:
    """Locate the code bundled in the model: ``fashion_seg`` and the family's package.

    Parameters
    ----------
    family : ModelFamily
        The model family.

    Returns
    -------
    list[str]
        The two package directories.
    """
    family_package = importlib.import_module(family.load_predictor.__module__.split(".")[0])
    return [str(Path(module.__file__).parent) for module in (fashion_seg, family_package)]


def input_example() -> dict[str, list[str]]:
    """Build a small random image request, used by MLflow to validate the model and document it.

    Returns
    -------
    dict[str, list[str]]
        One base64 PNG under the ``image`` column.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 48, 3), dtype=np.uint8)
    return {request.IMAGE_FIELD: [encode_image(image)]}


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
        "python_model": FashionSegmentationModel(load_predictor=family.load_predictor),
        "artifacts": {"model": str(model.source), "labels": str(params.data.label_file)},
        "code_paths": code_paths(family),
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
