"""Package step: publish any family's export as a model served through the contract.

For the served model ``params.yaml:models.<name>``, in a tracked run: describe the export, then
publish it to the ``ModelRepository`` (a new registered version, and a copy in ``output_dir``
that DVC versions and fashion-serving imports).

A packaged model bundles the code of ``fashion_seg`` and of the family, and pins its runtime
requirements to the versions installed in the family's environment (released packages by wheel
URL), so a serving environment only needs ``pip install -r requirements.txt``. Each family ships
only its own framework.

Run through DVC in the family's serving environment: ``uv run dvc repro --single-item
package_legacy`` (or ``package_torchvision``).
"""

import importlib
import json
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any

import fashion_seg
from fashion_seg.config import PackagedModel, Params
from fashion_seg.ports import Infrastructure, ModelFamily, ModelPackage, ServingRequirements

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


def code_dirs(family: ModelFamily) -> list[Path]:
    """Locate the code bundled in the model: ``fashion_seg``, the family's package, its bundles.

    Parameters
    ----------
    family : ModelFamily
        The model family.

    Returns
    -------
    list[Path]
        The package directories.
    """
    names = [family.load_predictor.__module__.split(".")[0], *family.SPEC.bundles]
    packages = [fashion_seg, *(importlib.import_module(name) for name in names)]
    return [Path(module.__file__).parent for module in packages]


def package(
    family: ModelFamily,
    name: str,
    model: PackagedModel,
    params: Params,
    infra: Infrastructure,
) -> dict[str, Any]:
    """Publish a family's export in a tracked run.

    Parameters
    ----------
    family : ModelFamily
        The export's model family.
    name : str
        Name of the served model (``params.yaml:models.<name>``), in the run name.
    model : PackagedModel
        The export to package, and where to write the packaged model.
    params : Params
        The pipeline parameters (label file, tracking).
    infra : Infrastructure
        The tracker records the run; the repository publishes the model.

    Returns
    -------
    dict[str, Any]
        Provenance of the packaged model (tracking run, model URI, family, registered name and
        version).
    """
    to_publish = ModelPackage(
        family=family.SPEC.name,
        load_predictor=family.load_predictor,
        export_dir=model.source,
        label_file=params.data.label_file,
        code_dirs=code_dirs(family),
        requirements=pip_requirements(family.SPEC.serving),
    )
    tags = {"model_family": family.SPEC.name, "packaged_model": name}
    with infra.tracker.run(params.tracking.packaging_experiment, f"package-{name}", tags):
        infra.tracker.log_params(
            {f"model.{k}": v for k, v in family.describe(model.source).items()}
        )
        return infra.repository.publish(
            to_publish, params.tracking.registered_name, model.output_dir
        )
