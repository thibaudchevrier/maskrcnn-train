"""MLflow implementation of ``fashion_seg.ports.ModelRepository``: this module is the repository.

A packaged model is an MLflow pyfunc model served by
``fashion_seg.serving.pyfunc.FashionSegmentationModel`` (the contract): it is logged in the active
run, registered as a new version, and saved to a directory (versioned by DVC, imported by
fashion-serving) with a ``provenance.json``.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fashion_seg_contract import request
from fashion_seg_contract.schema import Prediction
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

from fashion_seg.ports import ModelPackage
from fashion_seg.serving.pyfunc import FashionSegmentationModel
from fashion_seg.serving.response import encode_image

PROVENANCE_FILE = "provenance.json"

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", request.IMAGE_FIELD)]),
    params=ParamSchema([ParamSpec(request.MIN_SCORE_PARAM, "double", request.DEFAULT_MIN_SCORE)]),
)


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


def _model_kwargs(package: ModelPackage) -> dict[str, Any]:
    """Arguments of ``mlflow.pyfunc.log_model`` / ``save_model`` for a package.

    Parameters
    ----------
    package : ModelPackage
        What to package.

    Returns
    -------
    dict[str, Any]
        The pyfunc model, its artifacts, code, requirements, signature and input example.
    """
    return {
        "python_model": FashionSegmentationModel(load_predictor=package.load_predictor),
        "artifacts": {"model": str(package.export_dir), "labels": str(package.label_file)},
        "code_paths": [str(path) for path in package.code_dirs],
        "pip_requirements": package.requirements,
        "signature": SIGNATURE,
        "input_example": input_example(),
    }


def publish(package: ModelPackage, registered_name: str, output_dir: Path) -> dict[str, Any]:
    """Log the model in the active run, register a new version, save a copy with its provenance.

    Parameters
    ----------
    package : ModelPackage
        What to package.
    registered_name : str
        Registered model receiving the new version.
    output_dir : Path
        Where to save the packaged model (replaced if present).

    Returns
    -------
    dict[str, Any]
        Provenance: MLflow run, model URI, family, registered name and version.

    Raises
    ------
    RuntimeError
        If no run is active (open one with the tracker).
    """
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("publish() logs the model in the active run: open one with the tracker")
    kwargs = _model_kwargs(package)
    info = mlflow.pyfunc.log_model(name="model", registered_model_name=registered_name, **kwargs)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    mlflow.pyfunc.save_model(path=str(output_dir), **kwargs)
    record = {
        "mlflow_run_id": active.info.run_id,
        "model_uri": info.model_uri,
        "model_family": package.family,
        "registered_name": registered_name,
        "registered_version": info.registered_model_version,
    }
    (output_dir / PROVENANCE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def provenance(model_dir: Path) -> dict[str, Any]:
    """Read the provenance saved with a packaged model.

    Parameters
    ----------
    model_dir : Path
        A packaged model, as saved by ``publish``.

    Returns
    -------
    dict[str, Any]
        Its provenance.
    """
    return json.loads((model_dir / PROVENANCE_FILE).read_text(encoding="utf-8"))


def load(model_dir: Path) -> Callable[[bytes, float], Prediction]:
    """Load a packaged model exactly as it is served (MLflow pyfunc).

    Parameters
    ----------
    model_dir : Path
        A packaged model, as saved by ``publish``.

    Returns
    -------
    Callable[[bytes, float], Prediction]
        Predicts one encoded image (JPEG or PNG) with a ``min_score``, one request per image as
        serving does.
    """
    model = mlflow.pyfunc.load_model(str(model_dir))

    def predict(image: bytes, min_score: float) -> Prediction:
        """Predict one encoded image, as a one-row request.

        Parameters
        ----------
        image : bytes
            Encoded image (JPEG or PNG).
        min_score : float
            ``min_score`` request parameter.

        Returns
        -------
        Prediction
            The model's response for the image.
        """
        rows = pd.DataFrame({request.IMAGE_FIELD: [request.encode_image(image)]})
        return model.predict(rows, params={request.MIN_SCORE_PARAM: min_score})[0]

    return predict


def tag_version(registered_name: str, version: str, tags: dict[str, str]) -> None:
    """Tag a registered model version.

    Parameters
    ----------
    registered_name : str
        Registered model.
    version : str
        Its version.
    tags : dict[str, str]
        Tags, by name.
    """
    client = mlflow.MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(registered_name, version, key, value)
