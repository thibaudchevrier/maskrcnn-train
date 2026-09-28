"""``evaluate`` stage: score the packaged model on the validation split (COCO mask and box mAP).

The model is loaded exactly as it is served (MLflow pyfunc), so the score describes the deployed
artifact, whatever its framework. Results go to:

- MLflow: a run in the ``fashion-seg-evaluation`` experiment (metrics, per-class AP table), and
  ``val_*`` tags on the registered model version;
- ``metrics/evaluate.json``, for ``dvc metrics show`` / ``dvc metrics diff``.

Run through DVC (``uv run dvc repro --single-item evaluate``) on the whole split, or quickly on a
subset with ``make evaluate-quick``.
"""

import argparse
import base64
import json
import logging
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import polars as pl
import yaml
from fashion_seg_contract import rle
from fashion_seg_contract.labels import load_class_names
from fashion_seg_contract.schema import Prediction
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from fashion_seg_core import annotations
from fashion_seg_core.tracking import setup_experiment

logger = logging.getLogger(__name__)

# COCOeval.stats indices of the reported summary metrics
SUMMARY = {"map": 0, "ap50": 1, "ap75": 2, "ar100": 8}


def to_coco_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean mask as a COCO (compressed) RLE.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask of shape ``(height, width)``.

    Returns
    -------
    dict[str, Any]
        ``{"size": [height, width], "counts": bytes}``, as pycocotools expects.
    """
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))


def box_to_xywh(box: list[int]) -> list[float]:
    """Convert a contract box to a COCO box.

    Parameters
    ----------
    box : list[int]
        ``[y1, x1, y2, x2]`` in pixels, ``(y2, x2)`` excluded.

    Returns
    -------
    list[float]
        ``[x, y, width, height]``.

    Examples
    --------
    >>> box_to_xywh([10, 20, 30, 60])
    [20.0, 10.0, 40.0, 20.0]
    """
    y1, x1, y2, x2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def ground_truth(records: pl.DataFrame, class_names: list[str]) -> dict[str, Any]:
    """Build a COCO ground-truth dataset from the prepared annotations.

    Parameters
    ----------
    records : pl.DataFrame
        Prepared annotations (see ``fashion_seg_core.annotations``), one row per image; the row
        index is the COCO image id.
    class_names : list[str]
        Class names indexed by model class id (0 = background).

    Returns
    -------
    dict[str, Any]
        COCO ``images``, ``annotations`` (RLE masks, model class ids) and ``categories``.
    """
    images, anns = [], []
    for image_index, row in enumerate(records.iter_rows(named=True)):
        height, width = int(row["height"]), int(row["width"])
        images.append({"id": image_index, "height": height, "width": width})
        for category, mask_rle in zip(row["class_ids"], row["rles"], strict=True):
            segmentation = to_coco_rle(rle.decode(mask_rle, height, width))
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": image_index,
                    "category_id": int(category) + 1,  # dataset category k = model class k + 1
                    "segmentation": segmentation,
                    "area": float(coco_mask.area(segmentation)),
                    "bbox": coco_mask.toBbox(segmentation).tolist(),
                    "iscrowd": 0,
                }
            )
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names) if i > 0]
    return {"images": images, "annotations": anns, "categories": categories}


def detections(image_index: int, prediction: Prediction) -> tuple[list[dict], list[dict]]:
    """Convert one prediction into COCO segmentation and box results.

    Parameters
    ----------
    image_index : int
        COCO image id.
    prediction : Prediction
        The model's response for the image.

    Returns
    -------
    tuple[list[dict], list[dict]]
        Mask results and box results, one per instance.
    """
    masks, boxes = [], []
    height, width = prediction["height"], prediction["width"]
    for inst in prediction["instances"]:
        common = {"image_id": image_index, "category_id": inst["class_id"], "score": inst["score"]}
        segmentation = to_coco_rle(rle.decode(inst["mask_rle"], height, width))
        masks.append({**common, "segmentation": segmentation})
        boxes.append({**common, "bbox": box_to_xywh(inst["box"])})
    return masks, boxes


def coco_evaluate(
    gt: dict[str, Any], results: list[dict], iou_type: str
) -> tuple[dict[str, float], np.ndarray]:
    """Run the COCO evaluation.

    Parameters
    ----------
    gt : dict[str, Any]
        COCO ground truth (see ``ground_truth``).
    results : list[dict]
        COCO results of the matching type.
    iou_type : str
        ``"segm"`` (masks) or ``"bbox"`` (boxes).

    Returns
    -------
    tuple[dict[str, float], np.ndarray]
        Summary metrics (``map``, ``ap50``, ``ap75``, ``ar100``), and the per-class AP averaged
        over the IoU thresholds (NaN for classes without ground truth).
    """
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    n_classes = len(gt["categories"])
    if not results:
        return dict.fromkeys(SUMMARY, 0.0), np.full(n_classes, np.nan)
    coco_eval = COCOeval(coco_gt, coco_gt.loadRes(results), iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    summary = {name: float(coco_eval.stats[i]) for name, i in SUMMARY.items()}
    # precision: [iou thresholds, recall, class, area range, max dets]; -1 = no ground truth
    precision = coco_eval.eval["precision"][:, :, :, 0, -1]
    per_class = np.array(
        [
            float(np.mean(p[p > -1])) if (p > -1).any() else np.nan
            for p in np.moveaxis(precision, 2, 0)
        ]
    )
    return summary, per_class


def predict_images(
    model: mlflow.pyfunc.PyFuncModel, paths: list[Path], min_score: float
) -> Iterator[Prediction]:
    """Run the packaged model on images, one request per image, as serving does.

    Parameters
    ----------
    model : mlflow.pyfunc.PyFuncModel
        The packaged model.
    paths : list[Path]
        Image files.
    min_score : float
        ``min_score`` request parameter.

    Yields
    ------
    Prediction
        The model's response for each image, in order.
    """
    for path in paths:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        request = pd.DataFrame({"image": [payload]})
        yield model.predict(request, params={"min_score": min_score})[0]


def slug(label: str) -> str:
    """Turn a class label into an MLflow metric name fragment.

    Parameters
    ----------
    label : str
        Class label, e.g. ``"shirt, blouse"``.

    Returns
    -------
    str
        Lower-case, underscores only, e.g. ``"shirt_blouse"``.

    Examples
    --------
    >>> slug("top, t-shirt, sweatshirt")
    'top_t_shirt_sweatshirt'
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load_params(params_file: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the ``data`` and ``evaluate`` sections of ``params.yaml``.

    Parameters
    ----------
    params_file : str
        Path of ``params.yaml``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The ``data`` and ``evaluate`` sections.
    """
    params = yaml.safe_load(Path(params_file).read_text(encoding="utf-8"))
    return params["data"], params["evaluate"]


def select_images(data: dict[str, Any], params: dict[str, Any]) -> pl.DataFrame:
    """Pick the split's images to evaluate: the first ``max_images`` that are on disk.

    Parameters
    ----------
    data : dict[str, Any]
        The ``data`` section of ``params.yaml``.
    params : dict[str, Any]
        The ``evaluate`` section of ``params.yaml``.

    Returns
    -------
    pl.DataFrame
        Prepared annotations of the selected images, in split order.

    Raises
    ------
    SystemExit
        If none of the split's images is on disk.
    """
    split = annotations.load_split(Path(data["prepared_dir"]) / "split.json")[params["split"]]
    image_dir = Path(data["train_images"])
    present = [i for i in split if (image_dir / f"{i}.jpg").exists()]
    if len(present) < len(split):
        logger.warning(
            "%d of %d %s images are not pulled",
            len(split) - len(present),
            len(split),
            params["split"],
        )
    selected = present[: params["max_images"]] if params["max_images"] else present
    if not selected:
        raise SystemExit(f"No {params['split']} image in {image_dir}: run `make pull-val`.")
    return annotations.load(Path(data["prepared_dir"]) / "annotations.parquet", selected)


def score(
    model: mlflow.pyfunc.PyFuncModel,
    records: pl.DataFrame,
    image_dir: Path,
    class_names: list[str],
    min_score: float,
) -> tuple[dict[str, float], pl.DataFrame]:
    """Predict every selected image and compute the metrics.

    Images whose prediction size differs from the annotation (e.g. EXIF rotation) are skipped
    and counted in ``n_skipped``.

    Parameters
    ----------
    model : mlflow.pyfunc.PyFuncModel
        The packaged model.
    records : pl.DataFrame
        Prepared annotations of the images to evaluate.
    image_dir : Path
        Directory of the ``<image_id>.jpg`` files.
    class_names : list[str]
        Class names indexed by model class id.
    min_score : float
        ``min_score`` request parameter.

    Returns
    -------
    tuple[dict[str, float], pl.DataFrame]
        Metrics, and the per-class table (``class_id``, ``label``, ``n_gt``, ``mask_ap``,
        ``box_ap``).
    """
    paths = [image_dir / f"{i}.jpg" for i in records["image_id"]]
    mask_results, box_results, kept, skipped = [], [], [], 0
    start = time.monotonic()
    for index, (row, prediction) in enumerate(
        zip(records.iter_rows(named=True), predict_images(model, paths, min_score), strict=True)
    ):
        if (prediction["height"], prediction["width"]) != (row["height"], row["width"]):
            skipped += 1
            continue
        masks, boxes = detections(len(kept), prediction)
        mask_results += masks
        box_results += boxes
        kept.append(index)
        if (index + 1) % 50 == 0:
            logger.info("%d/%d images", index + 1, len(paths))
    seconds = time.monotonic() - start
    gt = ground_truth(records[kept], class_names)
    mask_summary, mask_per_class = coco_evaluate(gt, mask_results, "segm")
    box_summary, box_per_class = coco_evaluate(gt, box_results, "bbox")
    metrics = {
        **{f"mask_{k}": v for k, v in mask_summary.items()},
        **{f"box_{k}": v for k, v in box_summary.items()},
        "n_images": float(len(kept)),
        "n_skipped": float(skipped),
        "seconds_per_image": seconds / max(1, len(paths)),
    }
    n_gt = np.bincount([a["category_id"] for a in gt["annotations"]], minlength=len(class_names))
    per_class = pl.DataFrame(
        {
            "class_id": list(range(1, len(class_names))),
            "label": class_names[1:],
            "n_gt": n_gt[1:].tolist(),
            "mask_ap": mask_per_class.tolist(),
            "box_ap": box_per_class.tolist(),
        }
    )
    return metrics, per_class


def log_to_mlflow(
    params: dict[str, Any],
    provenance: dict[str, Any],
    metrics: dict[str, float],
    per_class: pl.DataFrame,
) -> None:
    """Log the evaluation run; on the whole split, also tag the registered model version.

    Parameters
    ----------
    params : dict[str, Any]
        The ``evaluate`` section of ``params.yaml``.
    provenance : dict[str, Any]
        The packaged model's ``provenance.json`` (registered name and version).
    metrics : dict[str, float]
        Metrics returned by ``score``.
    per_class : pl.DataFrame
        Per-class table returned by ``score``.
    """
    name, version = provenance["registered_name"], str(provenance["registered_version"])
    setup_experiment(params["experiment"])
    with mlflow.start_run(run_name=f"{name}-v{version}-{params['split']}"):
        mlflow.set_tags({"registered_name": name, "registered_version": version})
        mlflow.log_params({**params, "model_uri": provenance["model_uri"]})
        mlflow.log_metrics(metrics)
        for row in per_class.iter_rows(named=True):
            if row["mask_ap"] == row["mask_ap"]:  # skip NaN: no ground truth for the class
                mlflow.log_metric(f"mask_ap/{slug(row['label'])}", row["mask_ap"])
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "per_class.csv"
            per_class.write_csv(table)
            mlflow.log_artifact(str(table))
    if params["max_images"]:
        return  # a subset score must not look like the model version's score in the registry
    client = mlflow.MlflowClient()
    for key in ("mask_map", "mask_ap50", "box_map", "n_images"):
        client.set_model_version_tag(name, version, f"{params['split']}_{key}", metrics[key])


def main(argv: list[str] | None = None) -> None:
    """Evaluate the packaged model and log the results.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments, ``sys.argv[1:]`` when ``None``. By default ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--max-images", type=int, help="override evaluate.max_images")
    parser.add_argument("--output", help="override evaluate.output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data, params = load_params(args.params)
    if args.max_images is not None:
        params["max_images"] = args.max_images
    model_dir = Path(params["model_dir"])
    provenance = json.loads((model_dir / "provenance.json").read_text(encoding="utf-8"))
    records = select_images(data, params)
    logger.info("Evaluating %s on %d images", model_dir, len(records))

    model = mlflow.pyfunc.load_model(str(model_dir))
    metrics, per_class = score(
        model,
        records,
        Path(data["train_images"]),
        load_class_names(data["label_file"]),
        params["min_score"],
    )
    log_to_mlflow(params, provenance, metrics, per_class)
    output = Path(args.output or params["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "mask mAP %.4f | box mAP %.4f | %d images",
        metrics["mask_map"],
        metrics["box_map"],
        metrics["n_images"],
    )


if __name__ == "__main__":
    main()
