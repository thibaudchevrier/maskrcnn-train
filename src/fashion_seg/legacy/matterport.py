"""Numpy pre/post-processing of the Matterport Mask R-CNN, ported from the original repo.

The exported SavedModel only contains the network graph: resizing, anchor
generation, image meta and mask un-molding happen outside of it. This module
reimplements those steps without depending on the Matterport training code.
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import skimage.transform


@dataclass(frozen=True)
class MatterportConfig:
    """Subset of the Matterport ``Config`` needed at inference time."""

    num_classes: int
    image_min_dim: int
    image_max_dim: int
    image_min_scale: float
    image_resize_mode: str
    mean_pixel: tuple[float, float, float]
    backbone_strides: tuple[int, ...]
    rpn_anchor_scales: tuple[int, ...]
    rpn_anchor_ratios: tuple[float, ...]
    rpn_anchor_stride: int
    detection_min_confidence: float

    @classmethod
    def from_json(cls, path: str | Path) -> "MatterportConfig":
        raw = json.loads(Path(path).read_text())
        if raw["IMAGE_RESIZE_MODE"] != "square":
            raise ValueError(f"Unsupported IMAGE_RESIZE_MODE: {raw['IMAGE_RESIZE_MODE']}")
        return cls(
            num_classes=raw["NUM_CLASSES"],
            image_min_dim=raw["IMAGE_MIN_DIM"],
            image_max_dim=raw["IMAGE_MAX_DIM"],
            image_min_scale=raw["IMAGE_MIN_SCALE"],
            image_resize_mode=raw["IMAGE_RESIZE_MODE"],
            mean_pixel=tuple(raw["MEAN_PIXEL"]),
            backbone_strides=tuple(raw["BACKBONE_STRIDES"]),
            rpn_anchor_scales=tuple(raw["RPN_ANCHOR_SCALES"]),
            rpn_anchor_ratios=tuple(raw["RPN_ANCHOR_RATIOS"]),
            rpn_anchor_stride=raw["RPN_ANCHOR_STRIDE"],
            detection_min_confidence=raw["DETECTION_MIN_CONFIDENCE"],
        )


@dataclass(frozen=True)
class Detections:
    """Detections for one image, in original image pixel coordinates."""

    boxes: np.ndarray  # [N, (y1, x1, y2, x2)] int32, (y2, x2) excluded
    class_ids: np.ndarray  # [N] int32, 0 is background
    scores: np.ndarray  # [N] float32
    masks: np.ndarray  # [H, W, N] bool


# --------------------------------------------------------------------------- boxes


def norm_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pixel to normalized coordinates. In pixels (y2, x2) is outside the box."""
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide(boxes - shift, scale).astype(np.float32)


def denorm_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Normalized to pixel coordinates."""
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


# ---------------------------------------------------------------- pre-processing


def resize_square(
    image: np.ndarray, min_dim: int, max_dim: int, min_scale: float
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Resize keeping aspect ratio, then zero-pad to ``max_dim x max_dim``.

    Returns the padded image, the window (y1, x1, y2, x2) holding the real
    image, and the scale factor applied.
    """
    h, w = image.shape[:2]
    scale = max(1.0, min_dim / min(h, w)) if min_dim else 1.0
    if min_scale and scale < min_scale:
        scale = min_scale
    if round(max(h, w) * scale) > max_dim:
        scale = max_dim / max(h, w)

    if scale != 1:
        image = skimage.transform.resize(
            image,
            (round(h * scale), round(w * scale)),
            order=1,
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        )

    h, w = image.shape[:2]
    top, left = (max_dim - h) // 2, (max_dim - w) // 2
    padding = [(top, max_dim - h - top), (left, max_dim - w - left), (0, 0)]
    image = np.pad(image, padding, mode="constant", constant_values=0)
    return image, (top, left, h + top, w + left), scale


def compose_image_meta(
    original_shape: tuple[int, ...],
    molded_shape: tuple[int, ...],
    window: tuple[int, int, int, int],
    scale: float,
    num_classes: int,
) -> np.ndarray:
    """Pack image attributes in the 1D layout expected by the network."""
    return np.array(
        [0, *original_shape, *molded_shape, *window, scale, *np.zeros(num_classes)],
        dtype=np.float32,
    )


def _anchors_for_level(
    scale: int, ratios: tuple[float, ...], shape: tuple[int, int], stride: int, anchor_stride: int
) -> np.ndarray:
    scales, ratios_ = np.meshgrid(np.array([scale]), np.array(ratios))
    scales, ratios_ = scales.flatten(), ratios_.flatten()
    heights = scales / np.sqrt(ratios_)
    widths = scales * np.sqrt(ratios_)

    shifts_y = np.arange(0, shape[0], anchor_stride) * stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    return np.concatenate([centers - 0.5 * sizes, centers + 0.5 * sizes], axis=1)


@lru_cache(maxsize=4)
def pyramid_anchors(config: MatterportConfig, image_shape: tuple[int, int]) -> np.ndarray:
    """Normalized anchors [N, (y1, x1, y2, x2)] for every FPN level."""
    backbone_shapes = [
        (math.ceil(image_shape[0] / s), math.ceil(image_shape[1] / s))
        for s in config.backbone_strides
    ]
    anchors = np.concatenate(
        [
            _anchors_for_level(
                scale, config.rpn_anchor_ratios, shape, stride, config.rpn_anchor_stride
            )
            for scale, shape, stride in zip(
                config.rpn_anchor_scales, backbone_shapes, config.backbone_strides, strict=True
            )
        ]
    )
    return norm_boxes(anchors, image_shape)


# --------------------------------------------------------------- post-processing


def unmold_mask(mask: np.ndarray, box: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Fit a small float mask (e.g. 28x28) into its box on a full-size canvas."""
    y1, x1, y2, x2 = box
    resized = skimage.transform.resize(
        mask, (y2 - y1, x2 - x1), order=1, mode="constant", anti_aliasing=False
    )
    full = np.zeros(image_shape, dtype=bool)
    full[y1:y2, x1:x2] = resized >= 0.5
    return full


def unmold_detections(
    detections: np.ndarray,
    mrcnn_mask: np.ndarray,
    original_shape: tuple[int, ...],
    molded_shape: tuple[int, ...],
    window: tuple[int, int, int, int],
) -> Detections:
    """Convert raw network outputs for one image to original image coordinates.

    detections: [max_instances, (y1, x1, y2, x2, class_id, score)] normalized coords,
        zero-padded after the last detection.
    mrcnn_mask: [max_instances, h, w, num_classes]
    """
    zero_ix = np.flatnonzero(detections[:, 4] == 0)
    n = zero_ix[0] if zero_ix.size else detections.shape[0]

    boxes = detections[:n, :4]
    class_ids = detections[:n, 4].astype(np.int32)
    scores = detections[:n, 5].astype(np.float32)
    masks = mrcnn_mask[np.arange(n), :, :, class_ids]

    # Normalized coords in the molded image -> normalized coords in the window
    # -> pixel coords in the original image.
    wy1, wx1, wy2, wx2 = norm_boxes(np.array(window), molded_shape[:2])
    shift = np.array([wy1, wx1, wy1, wx1])
    scale = np.array([wy2 - wy1, wx2 - wx1, wy2 - wy1, wx2 - wx1])
    boxes = denorm_boxes((boxes - shift) / scale, original_shape[:2])

    keep = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0
    boxes, class_ids, scores, masks = boxes[keep], class_ids[keep], scores[keep], masks[keep]

    height, width = original_shape[:2]
    full_masks = (
        np.stack(
            [unmold_mask(m, b, (height, width)) for m, b in zip(masks, boxes, strict=True)], -1
        )
        if len(boxes)
        else np.zeros((height, width, 0), dtype=bool)
    )
    return Detections(boxes=boxes, class_ids=class_ids, scores=scores, masks=full_masks)


# --------------------------------------------------------------------- predictor


class MatterportPredictor:
    """Runs the exported SavedModel on single RGB images."""

    def __init__(self, model_dir: str | Path):
        import tensorflow as tf  # imported lazily: heavy, and not needed for tests of numpy code

        model_dir = Path(model_dir)
        self.config = MatterportConfig.from_json(model_dir / "config.json")
        self._model = tf.saved_model.load(str(model_dir))
        self._infer = self._model.signatures["serving_default"]
        self._tf = tf

    def predict(self, image: np.ndarray) -> Detections:
        """image: [H, W, 3] uint8 RGB."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image of shape [H, W, 3], got {image.shape}")
        cfg = self.config
        molded, window, scale = resize_square(
            image, cfg.image_min_dim, cfg.image_max_dim, cfg.image_min_scale
        )
        molded = molded.astype(np.float32) - np.array(cfg.mean_pixel, dtype=np.float32)
        meta = compose_image_meta(image.shape, molded.shape, window, scale, cfg.num_classes)
        anchors = pyramid_anchors(cfg, molded.shape[:2])

        outputs = self._infer(
            input_image=self._tf.constant(molded[np.newaxis]),
            input_image_meta=self._tf.constant(meta[np.newaxis]),
            input_anchors=self._tf.constant(anchors[np.newaxis]),
        )
        return unmold_detections(
            outputs["mrcnn_detection"].numpy()[0],
            outputs["mrcnn_mask"].numpy()[0],
            image.shape,
            molded.shape,
            window,
        )
