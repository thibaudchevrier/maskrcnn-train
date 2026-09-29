"""Keep Ultralytics to itself: private settings, offline, no integrations.

By default Ultralytics keeps its settings in the user's home (shared by every project), sends
usage analytics, and hooks MLflow, ClearML, Comet, DVC, Ray Tune... into every training (its
MLflow hook would start and end runs of its own). ``environment`` must run before the first
``import ultralytics``: the package's ``__init__`` calls it, so any import of this family's
modules does. ``disable_integrations`` runs before a trainer is built.
"""

import os
import tempfile
from pathlib import Path

# Ultralytics' settings, and the fonts it may fetch: private, outside the user's home.
CONFIG_DIR = Path(tempfile.gettempdir()) / "fashion_seg_yolo"

INTEGRATIONS = (
    "clearml",
    "comet",
    "dvc",
    "hub",
    "mlflow",
    "neptune",
    "raytune",
    "tensorboard",
    "wandb",
)


def environment() -> None:
    """Point Ultralytics at a private settings folder, offline (no analytics, no update checks).

    An already set ``YOLO_CONFIG_DIR`` is kept. The folder must exist: Ultralytics otherwise
    falls back to ``/tmp/Ultralytics`` or the working directory.
    """
    config_dir = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR)))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_OFFLINE"] = "1"


def disable_integrations() -> None:
    """Turn off Ultralytics' analytics and its experiment-tracker callbacks.

    The pipeline logs to MLflow itself (``training.Callbacks``).
    """
    # pylint: disable-next=import-outside-toplevel  # after environment(), and only when training
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"sync": False, **{name: False for name in INTEGRATIONS if name in SETTINGS}})
