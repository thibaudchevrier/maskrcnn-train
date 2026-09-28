"""The workflow, generic over model families: prepare, train, package, evaluate.

Each module is one step. The steps know model families only through ``fashion_seg.ports``: the
composition root (``fashion_seg.__main__``) picks the family and passes it in.
"""
