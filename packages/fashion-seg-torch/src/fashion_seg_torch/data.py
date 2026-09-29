"""Training data order and loaders: reproducible per epoch, resumable mid-epoch."""

import signal
from collections.abc import Callable, Iterator
from typing import Any

from torch.utils.data import DataLoader, Dataset, Sampler

from fashion_seg.progress import epoch_order


class EpochSampler(Sampler[int]):
    """Shuffle the images the same way for a given epoch, from any position in it.

    Parameters
    ----------
    size : int
        Number of images.
    seed : int
        Training seed (see ``fashion_seg.progress.epoch_order``).

    Attributes
    ----------
    size : int
        Number of images.
    seed : int
        Shuffle seed.
    epoch : int
        Epoch whose order is produced.
    start : int
        Position in that order to start from (images already trained on are skipped).
    """

    size: int
    seed: int
    epoch: int
    start: int

    def __init__(self, size: int, seed: int):
        super().__init__()
        self.size, self.seed = size, seed
        self.epoch = self.start = 0

    def set_position(self, epoch: int, start: int) -> None:
        """Select the epoch and the position to iterate from.

        Parameters
        ----------
        epoch : int
            Epoch, from 0.
        start : int
            Number of images of the epoch to skip.
        """
        self.epoch, self.start = epoch, start

    def __iter__(self) -> Iterator[int]:
        """Iterate over the epoch's shuffled image indices, from ``start``.

        Returns
        -------
        Iterator[int]
            Dataset indices.
        """
        return iter(epoch_order(self.size, self.seed, self.epoch)[self.start :])

    def __len__(self) -> int:
        """Count the images left in the epoch.

        Returns
        -------
        int
            ``size - start``.
        """
        return max(0, self.size - self.start)


def ignore_interrupts(worker_id: int) -> None:  # pylint: disable=unused-argument  # torch's hook
    """Let data-loading workers ignore Ctrl+C: the main process stops the run cleanly.

    Parameters
    ----------
    worker_id : int
        Worker index, given by torch.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def make_loaders(
    datasets: tuple[Dataset, Dataset],
    collate: Callable[[list[Any]], Any],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Build the training loader (``EpochSampler`` order) and the validation loader.

    Parameters
    ----------
    datasets : tuple[Dataset, Dataset]
        Training and validation datasets.
    collate : Callable[[list[Any]], Any]
        Assembles a batch from dataset items.
    batch_size : int
        Items per batch.
    num_workers : int
        Data loading processes.
    seed : int
        Training seed.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training loader and validation loader.
    """
    train, val = datasets
    loaders = []
    for dataset, sampler in ((train, EpochSampler(len(train), seed)), (val, None)):
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                collate_fn=collate,
                worker_init_fn=ignore_interrupts,
            )
        )
    return loaders[0], loaders[1]
