"""``python -m fashion_seg prepare``: the step that needs no model family.

Per-image annotations and the frozen split, written to ``params.yaml:data.prepared_dir`` (the
``prepare`` DVC stage). Training, packaging and evaluation run in a family's environment, through
its entrypoint (``python -m fashion_seg_<family>``, see ``fashion_seg.cli``).
"""

import argparse
import logging
from pathlib import Path

from fashion_seg.config import load_params
from fashion_seg.service import preparation

logger = logging.getLogger("fashion_seg")


def main(argv: list[str] | None = None) -> None:
    """Write the prepared annotations and split from ``params.yaml``.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments, ``sys.argv[1:]`` when ``None``. By default ``None``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m fashion_seg", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="per-image annotations and the frozen split")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = load_params(args.params)
    logger.info("Prepared: %s", preparation.prepare(params.data, params.split))


if __name__ == "__main__":
    main()
