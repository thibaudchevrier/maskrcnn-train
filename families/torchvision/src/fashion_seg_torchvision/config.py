"""Training parameters of the torchvision family (``params.yaml:train.torchvision``)."""

from typing import Any, ClassVar, Literal

from fashion_seg.config import TrainConfig


class Config(TrainConfig):
    """Training parameters (``params.yaml:train.torchvision``), besides ``TrainConfig``'s.

    Attributes
    ----------
    min_size : int
        Images are resized so their short side is ``min_size``...
    max_size : int
        ...without the long side exceeding ``max_size``.
    batch_size : int
        Images per optimizer step.
    learning_rate : float
        Peak learning rate (SGD).
    momentum : float
        SGD momentum.
    weight_decay : float
        SGD weight decay.
    warmup_steps : int
        Steps of linear warm-up, before the cosine decay.
    num_workers : int
        Data loading processes.
    log_every : int
        Steps between two logs of the training losses.
    device : Literal["auto", "cuda", "mps", "cpu"]
        ``"auto"``: CUDA, then Apple GPU (MPS), then CPU. By default ``"auto"``.
    resume : bool
        Continue from the last checkpoint if there is one. By default ``True``.
    smoke_overrides : ClassVar[dict[str, Any]]
        Parameters replaced for a smoke run.
    """

    min_size: int
    max_size: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    warmup_steps: int
    num_workers: int
    log_every: int
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    resume: bool = True
    smoke_overrides: ClassVar[dict[str, Any]] = {
        "min_size": 256,
        "max_size": 320,
        "batch_size": 2,
        "epochs": 1,
        "warmup_steps": 1,
        "num_workers": 0,
        "log_every": 1,
        "max_train_images": 4,
        "max_val_images": 2,
    }
