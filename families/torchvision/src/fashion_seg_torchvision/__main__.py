"""Entrypoint of the torchvision family: the generic command line with this family injected.

``python -m fashion_seg_torchvision train | package <model> | evaluate <model>``, see
``fashion_seg.cli``.
"""

import fashion_seg_torchvision
from fashion_seg.cli import main

main(fashion_seg_torchvision)
