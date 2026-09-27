import numpy as np
import pytest

from fashion_seg.legacy.matterport import (
    MatterportConfig,
    denorm_boxes,
    norm_boxes,
    pyramid_anchors,
    resize_square,
    unmold_detections,
)

CONFIG = MatterportConfig(
    num_classes=47,
    image_min_dim=800,
    image_max_dim=1024,
    image_min_scale=0,
    image_resize_mode="square",
    mean_pixel=(123.7, 116.8, 103.9),
    backbone_strides=(4, 8, 16, 32, 64),
    rpn_anchor_scales=(32, 64, 128, 256, 512),
    rpn_anchor_ratios=(0.5, 1, 2),
    rpn_anchor_stride=1,
    detection_min_confidence=0.7,
)


def test_resize_square_keeps_aspect_ratio_and_pads():
    image = np.full((400, 200, 3), 255, dtype=np.uint8)
    molded, window, scale = resize_square(image, 800, 1024, 0)
    assert molded.shape == (1024, 1024, 3)
    assert scale == pytest.approx(2.56)  # capped by max_dim: 400 * 2.56 = 1024
    y1, x1, y2, x2 = window
    assert (y2 - y1, x2 - x1) == (1024, 512)
    assert molded[:, :x1].max() == 0 and molded[y1:y2, x1:x2].min() > 0


def test_box_normalization_roundtrip():
    boxes = np.array([[10, 20, 110, 220], [0, 0, 1024, 1024]])
    np.testing.assert_array_equal(
        denorm_boxes(norm_boxes(boxes, (1024, 1024)), (1024, 1024)), boxes
    )


def test_pyramid_anchor_count():
    anchors = pyramid_anchors(CONFIG, (1024, 1024))
    expected = sum(3 * (1024 // s) ** 2 for s in CONFIG.backbone_strides)
    assert anchors.shape == (expected, 4)


def test_unmold_detections_maps_back_to_original_image():
    original, molded_shape = (400, 200, 3), (1024, 1024, 3)
    _, window, _ = resize_square(np.zeros(original, np.uint8), 800, 1024, 0)
    # One detection covering the full real image, class 5, then zero padding.
    y1, x1, y2, x2 = norm_boxes(np.array(window), molded_shape[:2])
    detections = np.zeros((100, 6), np.float32)
    detections[0] = [y1, x1, y2, x2, 5, 0.9]
    masks = np.zeros((100, 28, 28, 47), np.float32)
    masks[0, :, :, 5] = 1.0

    result = unmold_detections(detections, masks, original, molded_shape, window)

    np.testing.assert_array_equal(result.boxes, [[0, 0, 400, 200]])
    np.testing.assert_array_equal(result.class_ids, [5])
    assert result.masks.shape == (400, 200, 1) and result.masks.all()


def test_unmold_detections_without_detections():
    result = unmold_detections(
        np.zeros((100, 6), np.float32),
        np.zeros((100, 28, 28, 47), np.float32),
        (50, 60, 3),
        (1024, 1024, 3),
        (0, 0, 1024, 1024),
    )
    assert result.boxes.shape == (0, 4) and result.masks.shape == (50, 60, 0)
