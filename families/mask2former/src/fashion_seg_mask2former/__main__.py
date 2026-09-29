"""Entrypoint of the mask2former family: the generic command line with this family injected.

``python -m fashion_seg_mask2former train | package <model> | evaluate <model>``, see
``fashion_seg.cli``.
"""

import fashion_seg_mask2former
from fashion_seg.cli import main

main(fashion_seg_mask2former)
