"""Training parameters of the mask2former family (``params.yaml:train.mask2former``)."""

from typing import Any, ClassVar, Literal

from fashion_seg.config import TrainConfig


class Config(TrainConfig):
    """Training parameters (``params.yaml:train.mask2former``), besides ``TrainConfig``'s.

    Defaults in ``params.yaml`` follow the Mask2Former recipe (AdamW, weight decay 0.05, backbone
    at a tenth of the learning rate), scaled to a small batch.

    Attributes
    ----------
    max_size : int
        Images are downscaled so their long side is at most ``max_size``.
    batch_size : int
        Images per optimizer step.
    learning_rate : float
        Peak learning rate (AdamW) of the decoder and heads.
    backbone_lr_factor : float
        Multiplier of the learning rate for the Swin backbone. By default 0.1.
    weight_decay : float
        AdamW weight decay. By default 0.05.
    warmup_steps : int
        Steps of linear warm-up, before the cosine decay.
    num_workers : int
        Data loading processes.
    device : Literal["auto", "cuda", "mps", "cpu"]
        ``"auto"``: CUDA, then Apple GPU (MPS), then CPU. By default ``"auto"``.
    smoke_overrides : ClassVar[dict[str, Any]]
        Parameters replaced for a smoke run.
    """

    max_size: int
    batch_size: int
    learning_rate: float
    backbone_lr_factor: float = 0.1
    weight_decay: float = 0.05
    warmup_steps: int
    num_workers: int
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    smoke_overrides: ClassVar[dict[str, Any]] = {
        "max_size": 128,
        "batch_size": 2,
        "epochs": 1,
        "warmup_steps": 1,
        "num_workers": 0,
        "log_every": 1,
        "max_train_images": 4,
        "max_val_images": 2,
    }
