"""Training parameters of the yolo family (``params.yaml:train.yolo``)."""

from typing import Any, ClassVar, Literal

from fashion_seg.config import TrainConfig


class Config(TrainConfig):
    """Training parameters (``params.yaml:train.yolo``), besides ``TrainConfig``'s.

    Attributes
    ----------
    model : str
        Architecture, an Ultralytics model name (``yolo11s-seg``); used when ``init_weights`` is
        missing (random weights: tests, smoke runs).
    imgsz : int
        Training and inference size: images are letterboxed to ``imgsz`` x ``imgsz``.
    batch_size : int
        Images per optimizer step.
    learning_rate : float
        Initial learning rate (Ultralytics' ``lr0``).
    optimizer : Literal["auto", "SGD", "AdamW"]
        Ultralytics optimizer; ``"auto"`` chooses from the run length. By default ``"auto"``.
    num_workers : int
        Data loading processes (also used to convert the dataset).
    device : Literal["auto", "cuda", "mps", "cpu"]
        ``"auto"``: CUDA, then Apple GPU (MPS), then CPU. By default ``"auto"``.
    smoke_overrides : ClassVar[dict[str, Any]]
        Parameters replaced for a smoke run.
    """

    model: str
    imgsz: int
    batch_size: int
    learning_rate: float
    optimizer: Literal["auto", "SGD", "AdamW"] = "auto"
    num_workers: int
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    smoke_overrides: ClassVar[dict[str, Any]] = {
        "imgsz": 128,
        "batch_size": 2,
        "epochs": 1,
        "num_workers": 0,
        "log_every": 1,
        "max_train_images": 4,
        "max_val_images": 2,
    }
