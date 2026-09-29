"""Tests of the shared loop on a toy model: reproducible order, exact stop and resume."""

import threading

import torch
from fashion_seg_torch.data import EpochSampler, make_loaders
from fashion_seg_torch.loop import Loop, Task, TrainingState, fit, resume, warmup_cosine
from torch.utils.data import TensorDataset


def test_sampler_order_is_reproducible_and_resumable():
    """An epoch's order is the same every time, and resuming skips the images already seen."""
    sampler = EpochSampler(10, seed=3)
    sampler.set_position(epoch=1, start=0)
    full = list(sampler)
    sampler.set_position(epoch=1, start=4)
    assert list(sampler) == full[4:] and len(sampler) == 6
    sampler.set_position(epoch=2, start=0)
    assert sorted(sampler) == list(range(10)) and list(sampler) != full


def _squared_error(model, batch, device):
    """Toy task: regress y from x."""
    x, y = (t.to(device) for t in batch)
    return {"mse": ((model(x) - y) ** 2).mean()}


TOY = Task(losses=_squared_error)


def _state():
    """Build the same small linear model, SGD and warm-up + cosine schedule every time."""
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return TrainingState(model, optimizer, warmup_cosine(optimizer, 2, 12), torch.device("cpu"))


def _loaders():
    """12 training samples in batches of 2 (6 steps per epoch), 4 validation samples."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(16, 3, generator=generator)
    y = x.sum(dim=1, keepdim=True)
    datasets = (TensorDataset(x[:12], y[:12]), TensorDataset(x[12:], y[12:]))
    return make_loaders(datasets, torch.utils.data.default_collate, 2, 0, seed=5)


def _collect(steps):
    """Build a metric logger recording the steps of the training losses."""

    # pylint: disable-next=unused-argument  # the MetricLogger signature
    def log(metrics, step=None):
        """Record the step of a training log."""
        if "train_loss" in metrics:
            steps.append(step)

    return log


def test_a_stopped_and_resumed_run_ends_like_an_uninterrupted_one(tmp_path):
    """Stopping mid-epoch then resuming gives exactly the same weights as never stopping."""
    straight = _state()
    fit(
        straight,
        TOY,
        _loaders(),
        Loop(_collect([]), 1, tmp_path / "a.pt", 100, threading.Event()),
        2,
    )

    stop, steps = threading.Event(), []

    def stop_at_step_four(metrics, step=None):
        """Log, and ask to stop once step 4 (mid-epoch) is logged."""
        _collect(steps)(metrics, step)
        if step == 4:
            stop.set()

    first = _state()
    loop = Loop(stop_at_step_four, 1, tmp_path / "b.pt", 100, stop)
    assert fit(first, TOY, _loaders(), loop, 2) == 4

    resumed = _state()
    resume(resumed, tmp_path / "b.pt")
    loop = Loop(_collect(steps), 1, tmp_path / "b.pt", 100, threading.Event())
    assert fit(resumed, TOY, _loaders(), loop, 2) is None

    assert steps == list(range(1, 13))  # each step once: 4 before the stop, 8 after
    assert resumed.epoch == 2 and "val_loss" in resumed.val_losses
    for a, b in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.allclose(a, b)
