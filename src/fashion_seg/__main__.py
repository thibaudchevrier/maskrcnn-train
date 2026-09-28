"""Command line of the pipeline, and its composition root.

::

    python -m fashion_seg prepare
    python -m fashion_seg train <family> [--smoke]       # matterport, torchvision
    python -m fashion_seg package <model>                # a key of params.yaml:models
    python -m fashion_seg evaluate <model> [--max-images N] [--output FILE]

This is the only module that depends on all the others: it reads ``params.yaml``, gets the model
family from the registry and passes it to the workflow step (``fashion_seg.service``). ``train``
runs itself again in the family's environment when the current Python is not the right one
(Matterport: Python 3.11, ``envs/matterport``). The DVC stages call these commands.
"""

import argparse
import logging
import sys
from pathlib import Path

from fashion_seg import registry, runtime
from fashion_seg.config import Params, load_params
from fashion_seg.service import evaluation, packaging, preparation, training

logger = logging.getLogger("fashion_seg")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``prepare``, ``train``, ``package`` and ``evaluate`` commands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m fashion_seg", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="per-image annotations and the frozen split")
    train = commands.add_parser("train", help="train a model family")
    train.add_argument("family", choices=registry.names())
    train.add_argument("--smoke", action="store_true", help="tiny run on locally pulled images")
    package = commands.add_parser("package", help="package and register a served model")
    package.add_argument("model", help="a key of params.yaml:models, e.g. legacy")
    evaluate = commands.add_parser("evaluate", help="score a packaged model")
    evaluate.add_argument("model", help="a key of params.yaml:models, e.g. legacy")
    evaluate.add_argument("--max-images", type=int, help="override evaluate.max_images")
    evaluate.add_argument("--output", type=Path, help="metrics file")
    return parser


def run_train(args: argparse.Namespace, params: Params, argv: list[str]) -> None:
    """Train a family, in its own environment if needed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``train`` arguments.
    params : Params
        The pipeline parameters.
    argv : list[str]
        The raw arguments, to run the same command in another environment.

    Raises
    ------
    SystemExit
        With the exit code of the command, when it ran in another environment.
    """
    family = registry.get_family(args.family)
    if not runtime.is_active(family.SPEC.train_runtime):
        logger.info("Switching to %s", family.SPEC.train_runtime.project)
        raise SystemExit(runtime.run_in(family.SPEC.train_runtime, argv))
    config = family.Config.model_validate(params.train[args.family])
    result = training.train(family, config, params, smoke=args.smoke)
    logger.info("Trained %s: %s", args.family, result.metrics)


def main(argv: list[str] | None = None) -> None:
    """Run a pipeline command.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments, ``sys.argv[1:]`` when ``None``. By default ``None``.

    Raises
    ------
    SystemExit
        If the served model is not in ``params.yaml:models``.
    """
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = load_params(args.params)

    if args.command == "prepare":
        logger.info("Prepared: %s", preparation.prepare(params.data, params.split))
        return
    if args.command == "train":
        run_train(args, params, argv)
        return
    if args.model not in params.models:
        raise SystemExit(f"Unknown model {args.model!r}: expected one of {sorted(params.models)}")
    model = params.models[args.model]
    if args.command == "package":
        provenance = packaging.package(registry.get_family(model.family), args.model, model, params)
        logger.info(
            "Registered %s v%s; saved to %s",
            provenance["registered_name"],
            provenance["registered_version"],
            model.output_dir,
        )
    else:
        metrics = evaluation.evaluate(
            args.model, model, params, max_images=args.max_images, output=args.output
        )
        logger.info(
            "mask mAP %.4f | box mAP %.4f | %d images",
            metrics["mask_map"],
            metrics["box_map"],
            metrics["n_images"],
        )


if __name__ == "__main__":
    main()
