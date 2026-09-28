"""Tests of the evaluate stage: COCO scores of contract predictions vs prepared annotations."""

import numpy as np
import polars as pl
import pytest
from fashion_seg_contract import rle
from PIL import Image

from fashion_seg.evaluate import score

CLASS_NAMES = ["BG", "shirt", "pants"]


def _mask(height: int, width: int, top: int, left: int, size: int) -> np.ndarray:
    """Build a square boolean mask."""
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


def _records() -> pl.DataFrame:
    """Two images with one shirt (category 0) and one pants (category 1) each."""
    rows = []
    for image_id in ("a", "b"):
        rows.append(
            {
                "image_id": image_id,
                "height": 40,
                "width": 30,
                "class_ids": [0, 1],
                "rles": [
                    rle.encode(_mask(40, 30, 2, 2, 10)),
                    rle.encode(_mask(40, 30, 20, 10, 15)),
                ],
            }
        )
    return pl.DataFrame(rows)


class FakeModel:
    """Stand-in for the packaged model: returns prepared responses in order."""

    def __init__(self, responses: list[dict]):
        self.responses = iter(responses)

    # pylint: disable-next=unused-argument  # same signature as PyFuncModel.predict
    def predict(self, request, params=None):
        """Return the next prepared response, ignoring the image."""
        return [next(self.responses)]


def _perfect(records: pl.DataFrame) -> list[dict]:
    """Responses reproducing the ground truth exactly, with score 0.9."""
    responses = []
    for row in records.iter_rows(named=True):
        instances = []
        for category, mask_rle in zip(row["class_ids"], row["rles"], strict=True):
            mask = rle.decode(mask_rle, row["height"], row["width"])
            ys, xs = np.nonzero(mask)
            box = [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1]
            instances.append(
                {
                    "class_id": category + 1,
                    "label": CLASS_NAMES[category + 1],
                    "score": 0.9,
                    "box": box,
                    "mask_rle": mask_rle,
                }
            )
        responses.append({"height": row["height"], "width": row["width"], "instances": instances})
    return responses


@pytest.fixture
def image_dir(tmp_path):
    """Write the images the evaluation reads (their content is ignored by the fake model)."""
    for image_id in ("a", "b"):
        Image.new("RGB", (30, 40)).save(tmp_path / f"{image_id}.jpg")
    return tmp_path


def test_perfect_predictions_score_one(image_dir):
    """Predictions equal to the ground truth give mask and box mAP of 1."""
    records = _records()
    metrics, per_class = score(FakeModel(_perfect(records)), records, image_dir, CLASS_NAMES, 0.0)
    assert metrics["mask_map"] == pytest.approx(1.0)
    assert metrics["box_map"] == pytest.approx(1.0)
    assert metrics["n_images"] == 2 and metrics["n_skipped"] == 0
    assert per_class["n_gt"].to_list() == [2, 2]
    assert per_class["mask_ap"].to_list() == pytest.approx([1.0, 1.0])


def test_no_predictions_score_zero(image_dir):
    """Empty responses give a mAP of 0."""
    records = _records()
    empty = [{"height": 40, "width": 30, "instances": []}] * 2
    metrics, _ = score(FakeModel(empty), records, image_dir, CLASS_NAMES, 0.0)
    assert metrics["mask_map"] == 0.0 and metrics["box_map"] == 0.0


def test_size_mismatch_is_skipped(image_dir):
    """An image whose prediction size differs from its annotation is skipped and counted."""
    records = _records()
    responses = _perfect(records)
    responses[1] = {"height": 30, "width": 40, "instances": []}  # e.g. rotated by EXIF
    metrics, _ = score(FakeModel(responses), records, image_dir, CLASS_NAMES, 0.0)
    assert metrics["n_images"] == 1 and metrics["n_skipped"] == 1
    assert metrics["mask_map"] == pytest.approx(1.0)
