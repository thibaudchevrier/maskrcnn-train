"""PyTorch toolkit shared by the families that train with our own loop.

- ``device``: CUDA, then Apple GPU (MPS), then CPU;
- ``data``: each epoch's reproducible order, resumable mid-epoch, and the data loaders;
- ``loop``: training state, checkpoints, and ``fit``: train, log, validate, stop and resume.

A family keeps its model logic (network, dataset, losses, export) and hands ``fit`` a ``Task``.
"""
