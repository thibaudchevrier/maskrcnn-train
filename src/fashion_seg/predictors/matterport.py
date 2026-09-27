"""Matterport Mask R-CNN exports (``config.json`` + TF SavedModel) as a serving ``Predictor``.

Serves both the 2021 model (``deployement/``) and models trained by ``trainers/matterport``. The
pre/post-processing is maskrcnn-matterport's own (``mrcnn.serving``), so this only adapts types.
"""

from pathlib import Path

import numpy as np
from mrcnn.serving import SavedModelPredictor

from fashion_seg.predictors.base import Detections


class MatterportPredictor:
    """Runs an exported Matterport model on single RGB images."""

    def __init__(self, model_dir: str | Path):
        self._model = SavedModelPredictor(str(model_dir))
        self.config = self._model.config

    def predict(self, image: np.ndarray) -> Detections:
        """image: [H, W, 3] uint8 RGB."""
        result = self._model.detect(image)
        return Detections(
            boxes=result["rois"].astype(np.int32),
            class_ids=result["class_ids"].astype(np.int32),
            scores=result["scores"].astype(np.float32),
            masks=result["masks"].astype(bool),
        )
