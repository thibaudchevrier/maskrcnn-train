"""Pick the compute device."""

import torch


def pick_device(name: str) -> torch.device:
    """Resolve a device name, ``"auto"`` meaning CUDA, then Apple GPU (MPS), then CPU.

    Parameters
    ----------
    name : str
        ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Returns
    -------
    torch.device
        The device.

    Examples
    --------
    >>> pick_device("cpu").type
    'cpu'
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
