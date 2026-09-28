"""COCO mask and box mAP of contract predictions against the prepared annotations.

Pure scoring, independent of models and tracking: predictions in, metrics out (pycocotools).
"""

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl
from fashion_seg_contract import rle
from fashion_seg_contract.schema import Prediction
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        Prepared annotations (see ``fashion_seg.data.annotations``), one row per image; the row
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


def collect(
    records: pl.DataFrame, predictions: Iterable[Prediction]
) -> tuple[list[dict], list[dict], list[int]]:
    """Convert the predictions to COCO results, skipping images whose size doesn't match.

    Parameters
    ----------
    records : pl.DataFrame
        Prepared annotations of the evaluated images.
    predictions : Iterable[Prediction]
        The model's response for each image of ``records``, in the same order (consumed lazily).

    Returns
    -------
    tuple[list[dict], list[dict], list[int]]
        Mask results, box results, and the rows of ``records`` they belong to (the COCO image
        ids are positions in that list).
    """
    mask_results, box_results, kept = [], [], []
    for index, (row, prediction) in enumerate(
        zip(records.iter_rows(named=True), predictions, strict=True)
    ):
        if (index + 1) % 50 == 0:
            logger.info("%d/%d images", index + 1, records.height)
        if (prediction["height"], prediction["width"]) != (row["height"], row["width"]):
            continue  # e.g. EXIF rotation
        masks, boxes = detections(len(kept), prediction)
        mask_results += masks
        box_results += boxes
        kept.append(index)
    return mask_results, box_results, kept


def score(
    records: pl.DataFrame, predictions: Iterable[Prediction], class_names: list[str]
) -> tuple[dict[str, float], pl.DataFrame]:
    """Compute the metrics of the predictions of the images of ``records``.

    Images whose prediction size differs from the annotation (e.g. EXIF rotation) are skipped
    and counted in ``n_skipped``.

    Parameters
    ----------
    records : pl.DataFrame
        Prepared annotations of the evaluated images.
    predictions : Iterable[Prediction]
        The model's response for each image of ``records``, in the same order (consumed lazily).
    class_names : list[str]
        Class names indexed by model class id.

    Returns
    -------
    tuple[dict[str, float], pl.DataFrame]
        Metrics (``mask_map``, ``box_map``... ``n_images``, ``n_skipped``), and the per-class table
        (``class_id``, ``label``, ``n_gt``, ``mask_ap``, ``box_ap``).
    """
    mask_results, box_results, kept = collect(records, predictions)
    gt = ground_truth(records[kept], class_names)
    mask_summary, mask_per_class = coco_evaluate(gt, mask_results, "segm")
    box_summary, box_per_class = coco_evaluate(gt, box_results, "bbox")
    metrics = {
        **{f"mask_{k}": v for k, v in mask_summary.items()},
        **{f"box_{k}": v for k, v in box_summary.items()},
        "n_images": float(len(kept)),
        "n_skipped": float(records.height - len(kept)),
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
