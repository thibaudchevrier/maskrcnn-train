"""MLflow pyfunc wrapper serving any family through the contract: base64 images in, predictions out.

Request (``POST /invocations`` on ``mlflow models serve``), see ``fashion_seg_contract.request``::

    {"dataframe_records": [{"image": "<base64 jpeg/png>"}], "params": {"min_score": 0.8}}

The response is built by ``fashion_seg.serving.response``.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd  # MLflow's pyfunc interface: predict() receives a pandas DataFrame
from fashion_seg_contract import request
from fashion_seg_contract.labels import load_class_names
from fashion_seg_contract.schema import Prediction

from fashion_seg.ports import Predictor
from fashion_seg.serving.response import decode_image, format_detections


# pylint: disable-next=abstract-method  # predict_stream is optional: this model doesn't stream
class FashionSegmentationModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc model: base64 images in, contract predictions out.

    Packaging injects the family's ``load_predictor``; it is pickled with the model (by
    reference: the family's code is bundled in the model), and MLflow calls ``load_context`` to
    load the export (``model`` artifact) and the class names (``labels`` artifact). Tests inject a
    ready predictor instead.

    Parameters
    ----------
    load_predictor : Callable[[Path], Predictor] | None
        The family's ``load_predictor``, called on the export directory. By default ``None``.
    predictor : Predictor | None
        Predictor to use instead of loading the artifacts. By default ``None``.
    class_names : list[str] | None
        Class names to use instead of loading the artifacts. By default ``None``.
    """

    def __init__(
        self,
        load_predictor: Callable[[Path], Predictor] | None = None,
        predictor: Predictor | None = None,
        class_names: list[str] | None = None,
    ):
        self._load_predictor = load_predictor
        self._predictor = predictor
        self._class_names = class_names

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the predictor and the class names from the model's artifacts.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            Gives the local paths of the ``model`` and ``labels`` artifacts.

        Raises
        ------
        RuntimeError
            If the model was packaged without a ``load_predictor``.
        """
        if self._load_predictor is None:
            raise RuntimeError("Packaged without load_predictor: package it with fashion_seg.cli")
        self._predictor = self._load_predictor(Path(context.artifacts["model"]))
        self._class_names = load_class_names(context.artifacts["labels"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Segment the garments in every image of a request.

        Only column access is used, so a plain ``{"image": [...]}`` dict works too (tests).

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext | None
            Unused (the artifacts are loaded in ``load_context``).
        model_input : pd.DataFrame
            One base64 image per row, in the ``image`` column.
        params : dict[str, Any] | None
            ``min_score`` (float, default 0.7) drops low-confidence detections. By default ``None``.

        Returns
        -------
        list[Prediction]
            One response per input row (see ``fashion_seg_contract.schema``).
        """
        min_score = float((params or {}).get(request.MIN_SCORE_PARAM, request.DEFAULT_MIN_SCORE))
        results = []
        for payload in model_input[request.IMAGE_FIELD]:
            image = decode_image(payload)
            detections = self._predictor.predict(image)
            results.append(format_detections(detections, self._class_names, image.shape, min_score))
        return results
