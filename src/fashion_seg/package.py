"""Package a model export as a self-contained MLflow pyfunc model, served through the contract.

``python -m fashion_seg.package <section>`` reads ``params.yaml:<section>`` (``family``,
``model_dir``, ``output_dir``, ``label_file``, ``experiment``, ``registered_name``), then:

- logs a run in MLflow and registers a new version of the model,
- saves the same model to ``output_dir`` so DVC can version and push it; fashion-serving imports
  it with ``dvc import``.

The model bundles our serving code and pins its runtime requirements to the versions installed
when packaging (released packages by wheel URL), so a serving environment only needs
``pip install -r requirements.txt``. Each family ships only its own framework.

Run through DVC: ``uv run dvc repro --single-item package_legacy`` (or ``package_torchvision``).
"""

import argparse
import json
import shutil
from importlib.metadata import distribution, version
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
import numpy as np
import yaml
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

import fashion_seg
import fashion_seg_torchvision
from fashion_seg.serving.pyfunc import (
    DEFAULT_MIN_SCORE,
    IMAGE_COLUMN,
    FashionSegmentationModel,
    encode_image,
)
from fashion_seg_core.tracking import setup_experiment

# Per family: installed packages pinned by version, released packages pinned by wheel URL,
# extra pip options, and our code bundled in the model.
FAMILIES: dict[str, dict[str, Any]] = {
    "matterport": {
        "pinned": ("mlflow", "tensorflow", "numpy", "pandas", "pillow"),
        "released": ("fashion-seg-contract", "maskrcnn-matterport"),
        "pip_options": (),
        "code": (fashion_seg,),
    },
    "torchvision": {
        "pinned": ("mlflow", "torch", "torchvision", "numpy", "pandas", "pillow"),
        "released": ("fashion-seg-contract",),
        # CPU-only PyTorch wheels on Linux, as in pyproject.toml (the default ones bundle CUDA).
        "pip_options": ("--extra-index-url https://download.pytorch.org/whl/cpu",),
        "code": (fashion_seg, fashion_seg_torchvision),
    },
}

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", IMAGE_COLUMN)]),
    params=ParamSchema([ParamSpec("min_score", "double", DEFAULT_MIN_SCORE)]),
)


def pip_requirements(family: str) -> list[str]:
    """List a family's runtime requirements, pinned to what is installed.

    Parameters
    ----------
    family : str
        Model family, a key of ``FAMILIES``.

    Returns
    -------
    list[str]
        pip requirement lines: options, pinned versions, released wheels by URL.
    """
    spec = FAMILIES[family]
    pinned = [f"{pkg}=={version(pkg)}" for pkg in spec["pinned"]]
    released = [f"{pkg} @ {_installed_from(pkg)}" for pkg in spec["released"]]
    return [*spec["pip_options"], *pinned, *released]


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


def code_paths(modules: tuple[ModuleType, ...]) -> list[str]:
    """Locate the source directories of the packages bundled in the model.

    Parameters
    ----------
    modules : tuple[ModuleType, ...]
        Packages to bundle.

    Returns
    -------
    list[str]
        Their directories.
    """
    return [str(Path(module.__file__).parent) for module in modules]


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


def export_config(family: str, model_dir: Path) -> dict[str, Any]:
    """Read the settings worth logging from a model export.

    Parameters
    ----------
    family : str
        Model family.
    model_dir : Path
        The export directory.

    Returns
    -------
    dict[str, Any]
        Settings, prefixed with ``model.``.
    """
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if family == "matterport":
        keys = ("BACKBONE", "NUM_CLASSES", "IMAGE_MAX_DIM", "RPN_ANCHOR_SCALES")
        config = {k: config[k] for k in keys}
    return {f"model.{k}": v for k, v in config.items()}


def package(section: str, params: dict[str, Any]) -> dict[str, Any]:
    """Log, register and save a model export as an MLflow pyfunc model.

    Parameters
    ----------
    section : str
        Name of the ``params.yaml`` section, used as the MLflow run name.
    params : dict[str, Any]
        That section: ``family``, ``model_dir``, ``output_dir``, ``label_file``, ``experiment``,
        ``registered_name``.

    Returns
    -------
    dict[str, Any]
        Provenance of the packaged model (MLflow run, model URI, registered name and version).
    """
    family, model_dir = params["family"], Path(params["model_dir"])
    model_kwargs = {
        "python_model": FashionSegmentationModel(),
        "artifacts": {"model": str(model_dir), "labels": params["label_file"]},
        "model_config": {"predictor": family},
        "code_paths": code_paths(FAMILIES[family]["code"]),
        "pip_requirements": pip_requirements(family),
        "signature": SIGNATURE,
        "input_example": input_example(),
    }
    setup_experiment(params["experiment"])
    with mlflow.start_run(run_name=f"package-{section}") as run:
        mlflow.set_tags({"model_family": family, "package_section": section})
        mlflow.log_params(export_config(family, model_dir))
        info = mlflow.pyfunc.log_model(
            name="model", registered_model_name=params["registered_name"], **model_kwargs
        )
    output_dir = Path(params["output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    mlflow.pyfunc.save_model(path=str(output_dir), **model_kwargs)
    provenance = {
        "mlflow_run_id": run.info.run_id,
        "model_uri": info.model_uri,
        "model_family": family,
        "registered_name": params["registered_name"],
        "registered_version": info.registered_model_version,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main(argv: list[str] | None = None) -> None:
    """Package the model configured in a ``params.yaml`` section.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments, ``sys.argv[1:]`` when ``None``. By default ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("section", help="params.yaml section, e.g. legacy_model")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args(argv)
    params = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))[args.section]
    provenance = package(args.section, params)
    print(
        f"Registered {provenance['registered_name']} v{provenance['registered_version']} "
        f"({provenance['model_family']}); saved to {params['output_dir']}"
    )


if __name__ == "__main__":
    main()
