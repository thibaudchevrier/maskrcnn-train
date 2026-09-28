"""pytest setup: collect each environment's code only in that environment.

The Matterport training code (TensorFlow 2.15) and its tests run in ``envs/matterport``
(``make test-matterport``); the root environment (``uv run pytest``) collects everything else.
"""

from fashion_seg.families import matterport
from fashion_seg.runtime import is_active

MATTERPORT_TRAINING = [
    "src/fashion_seg/families/matterport/dataset.py",
    "src/fashion_seg/families/matterport/training.py",
    "tests/matterport",
]

collect_ignore = [] if is_active(matterport.SPEC.train_runtime) else MATTERPORT_TRAINING
