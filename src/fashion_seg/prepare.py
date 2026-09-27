"""``prepare`` stage: per-image annotations + the frozen train/val split shared by every model.

The split reproduces the 2021 notebook: images sorted by id, ``KFold(n_folds, shuffle=True,
random_state=seed)``, keep fold ``fold`` as validation. With the defaults (8 folds, fold 0, seed
42) the 2021 model is evaluated on the images it was validated on, not ones it was trained on.

Run through DVC: ``uv run dvc repro --single-item prepare``.
"""

import json
from pathlib import Path

import pandas as pd
import yaml
from fashion_seg_core import annotations
from sklearn.model_selection import KFold


def split_image_ids(image_ids: list[str], n_folds: int, fold: int, seed: int) -> dict[str, list]:
    """Train/val image ids for one fold; ``image_ids`` must already be sorted."""
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(image_ids)
    for index, (train_idx, val_idx) in enumerate(folds):
        if index == fold:
            return {
                "train": [image_ids[i] for i in train_idx],
                "val": [image_ids[i] for i in val_idx],
            }
    raise ValueError(f"fold {fold} out of range for {n_folds} folds")


def main() -> None:
    """Write ``prepared/annotations.parquet`` and ``prepared/split.json``."""
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    paths, split = params["data"], params["split"]
    output_dir = Path(paths["prepared_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = pd.read_csv(
        paths["train_csv"], usecols=["ImageId", "EncodedPixels", "Height", "Width", "ClassId"]
    )
    per_image = annotations.group_by_image(masks)
    per_image.to_parquet(output_dir / "annotations.parquet", index=False)

    ids = per_image["image_id"].tolist()
    result = split_image_ids(ids, split["n_folds"], split["fold"], split["seed"])
    (output_dir / "split.json").write_text(
        json.dumps({"params": split, **result}) + "\n", encoding="utf-8"
    )
    print(
        f"{len(per_image)} images, {len(masks)} masks -> "
        f"{len(result['train'])} train / {len(result['val'])} val"
    )


if __name__ == "__main__":
    main()
