"""Entrypoint of the yolo family: the generic command line with this family injected.

``python -m fashion_seg_yolo train | package <model> | evaluate <model>``, see ``fashion_seg.cli``.
"""

import fashion_seg_yolo
from fashion_seg.cli import main

main(fashion_seg_yolo)
