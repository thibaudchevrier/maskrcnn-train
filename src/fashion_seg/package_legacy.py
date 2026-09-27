"""Package the 2021 Matterport SavedModel as an MLflow pyfunc model.

- logs a run in MLflow and registers a new version of the model,
- saves the same model to ``models/<name>`` so DVC can version and push it,
  which is what the serving repo pulls with ``dvc import``.

Run through DVC: ``uv run dvc repro package_legacy``.
"""

import json
import shutil
from importlib.metadata import version
from pathlib import Path

import fashion_seg_core
import mlflow
import numpy as np
import yaml
from fashion_seg_core.tracking import setup_experiment
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

import fashion_seg
from fashion_seg.serving.pyfunc import (
    DEFAULT_MIN_SCORE,
    IMAGE_COLUMN,
    FashionSegmentationModel,
    encode_image,
)

# The model ships its own copy of the code it imports at load time.
CODE_PATHS = [str(Path(pkg.__file__).parent) for pkg in (fashion_seg, fashion_seg_core)]
LOGGED_CONFIG_KEYS = ("BACKBONE", "NUM_CLASSES", "IMAGE_MAX_DIM", "DETECTION_MIN_CONFIDENCE")

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", IMAGE_COLUMN)]),
    params=ParamSchema([ParamSpec("min_score", "double", DEFAULT_MIN_SCORE)]),
)


def pip_requirements() -> list[str]:
    """Runtime requirements of the model, pinned to the versions used to package it."""
    return [
        f"{pkg}=={version(pkg)}"
        for pkg in ("mlflow", "tensorflow", "numpy", "pandas", "scikit-image", "pillow")
    ]


def input_example() -> dict[str, list[str]]:
    """Small random image used by MLflow to validate the model and document the input."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 48, 3), dtype=np.uint8)
    return {IMAGE_COLUMN: [encode_image(image)]}


def main() -> None:
    """Log, register and export the legacy model as configured in ``params.yaml``."""
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["legacy_model"]
    saved_model = Path(params["saved_model_dir"])
    labels = Path(params["label_file"])
    output_dir = Path(params["output_dir"])

    model_kwargs = {
        "python_model": FashionSegmentationModel(),
        "artifacts": {"saved_model": str(saved_model), "labels": str(labels)},
        "code_paths": CODE_PATHS,
        "pip_requirements": pip_requirements(),
        "signature": SIGNATURE,
        "input_example": input_example(),
    }

    setup_experiment(params["experiment"])
    with mlflow.start_run(run_name="legacy-matterport-2021") as run:
        mlflow.set_tags({"framework": "matterport-maskrcnn-tf2", "origin": "legacy-2021"})
        model_config = json.loads((saved_model / "config.json").read_text(encoding="utf-8"))
        mlflow.log_params({f"model.{k}": model_config[k] for k in LOGGED_CONFIG_KEYS})
        info = mlflow.pyfunc.log_model(
            name="model", registered_model_name=params["registered_name"], **model_kwargs
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    mlflow.pyfunc.save_model(path=str(output_dir), **model_kwargs)
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "mlflow_run_id": run.info.run_id,
                "model_uri": info.model_uri,
                "registered_name": params["registered_name"],
                "registered_version": info.registered_model_version,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Registered {params['registered_name']} v{info.registered_model_version}; "
        f"saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
