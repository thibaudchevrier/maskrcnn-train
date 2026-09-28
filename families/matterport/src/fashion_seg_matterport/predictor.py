"""Matterport Mask R-CNN exports (``config.json`` + TF SavedModel) as a serving ``Predictor``.

Serves both the 2021 model (``deployement/``) and models trained by ``training``. The
pre/post-processing is maskrcnn-matterport's own (``mrcnn.serving``), so this only adapts types.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mrcnn.serving import SavedModelPredictor

from fashion_seg.ports import Detections


class MatterportPredictor:
    """Runs an exported Matterport model (``config.json`` + TF SavedModel) on single RGB images.

    Parameters
    ----------
    model_dir : str | Path
        Export directory, as written by ``MaskRCNN.save`` (e.g. ``deployement/``).

    Attributes
    ----------
    config : SimpleNamespace
        The exported model configuration (``config.json``).
    """

    config: SimpleNamespace

    def __init__(self, model_dir: str | Path):
        self._model = SavedModelPredictor(str(model_dir))
        self.config = self._model.config

    def predict(self, image: np.ndarray) -> Detections:
        """Detect the garments in one image.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape ``(height, width, 3)``, uint8.

        Returns
        -------
        Detections
            The detections, in the image's pixel coordinates.
        """
        result = self._model.detect(image)
        return Detections(
            boxes=result["rois"].astype(np.int32),
            class_ids=result["class_ids"].astype(np.int32),
            scores=result["scores"].astype(np.float32),
            masks=result["masks"].astype(bool),
        )
