"""The prepared annotations in Ultralytics' format: images, polygon labels and ``data.yaml``.

Ultralytics trains on image files with a text file of polygons per image. The conversion writes,
once, each image downscaled to the training size and each mask as its largest outline
(normalized ``class x1 y1 x2 y2 ...``, class = dataset category). Files already written are kept,
so an interrupted conversion resumes. Masks too small to outline after downscaling are dropped.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from fashion_seg.data.images import load_example
from fashion_seg.ports import TrainInputs


def polygon(mask: np.ndarray) -> list[float] | None:
    """Outline a mask as a normalized polygon (its largest part).

    Parameters
    ----------
    mask : np.ndarray
        ``(height, width)`` mask of 0 and 1.

    Returns
    -------
    list[float] | None
        ``[x1, y1, x2, y2, ...]`` in [0, 1], or ``None`` if there is no outline of 3+ points.

    Examples
    --------
    >>> m = np.zeros((10, 20), np.uint8); m[2:8, 5:15] = 1
    >>> [round(v, 2) for v in polygon(m)]
    [0.25, 0.2, 0.25, 0.7, 0.7, 0.7, 0.7, 0.2]
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(points) < 3:
        return None
    height, width = mask.shape
    return (points / [width, height]).flatten().tolist()


def convert(row: dict, image_dir: Path, root: Path, split: str, imgsz: int) -> None:
    """Write one image and its label file, unless both exist already.

    Parameters
    ----------
    row : dict
        A row of the prepared annotations.
    image_dir : Path
        Directory of the original ``<image_id>.jpg`` files.
    root : Path
        Root of the converted dataset.
    split : str
        ``"train"`` or ``"val"``.
    imgsz : int
        Long side of the written image.
    """
    image_path = root / "images" / split / f"{row['image_id']}.jpg"
    label_path = root / "labels" / split / f"{row['image_id']}.txt"
    if image_path.exists() and label_path.exists():
        return
    example = load_example(row, image_dir, imgsz)
    lines = []
    for mask, label in zip(example.masks, example.labels, strict=True):
        points = polygon(mask)
        if points is not None:
            lines.append(" ".join([str(label - 1), *(f"{v:.6f}" for v in points)]))
    Image.fromarray(example.image).save(image_path, quality=95)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset(inputs: TrainInputs, imgsz: int, workers: int = 0) -> Path:
    """Convert the training and validation images to Ultralytics' format, with ``data.yaml``.

    Parameters
    ----------
    inputs : TrainInputs
        The splits, the images, the class names; written under ``checkpoint_dir / "dataset"``.
    imgsz : int
        Long side of the written images.
    workers : int
        Parallel conversion processes; 0 converts in this process. By default 0.

    Returns
    -------
    Path
        The ``data.yaml`` to train on.
    """
    root = inputs.checkpoint_dir / "dataset"
    for split, records in (("train", inputs.train), ("val", inputs.val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        job = partial(convert, image_dir=inputs.image_dir, root=root, split=split, imgsz=imgsz)
        rows = list(records.iter_rows(named=True))
        if workers > 0:
            with ProcessPoolExecutor(workers) as pool:
                list(pool.map(job, rows, chunksize=64))
        else:
            for row in rows:
                job(row)
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(inputs.class_names[1:])),  # YOLO classes: no background
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root / "data.yaml"
