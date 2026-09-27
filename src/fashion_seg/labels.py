"""Category names from the iMaterialist ``label_descriptions.json``."""

import json
from pathlib import Path

BACKGROUND = "BG"


def load_class_names(label_file: str | Path) -> list[str]:
    """Return model class names indexed by model class id.

    The model reserves id 0 for the background, so dataset category ``k`` is
    model class ``k + 1``.
    """
    categories = json.loads(Path(label_file).read_text())["categories"]
    ordered = sorted(categories, key=lambda c: c["id"])
    return [BACKGROUND] + [c["name"] for c in ordered]
