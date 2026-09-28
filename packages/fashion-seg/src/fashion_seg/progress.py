"""Where a training is in its data: the same for every family, so every family can resume.

Pure functions: the shuffle order of an epoch is reproducible from a seed, so a training stopped
mid-epoch resumes on the images it had not seen yet.
"""

import numpy as np


def epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    """Shuffle the training images for one epoch, the same way every time.

    Parameters
    ----------
    size : int
        Number of training images.
    seed : int
        Training seed (``TrainConfig.seed``).
    epoch : int
        Epoch, from 0: each epoch has its own order.

    Returns
    -------
    list[int]
        A permutation of ``range(size)``.

    Examples
    --------
    >>> epoch_order(5, seed=0, epoch=1) == epoch_order(5, seed=0, epoch=1)
    True
    >>> sorted(epoch_order(5, seed=0, epoch=1))
    [0, 1, 2, 3, 4]
    """
    return np.random.default_rng([seed, epoch]).permutation(size).tolist()
