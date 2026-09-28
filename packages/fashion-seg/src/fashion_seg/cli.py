"""The generic command line, run with a model family injected by the family's entrypoint.

Each family has its own uv project (its Python and framework) and an entrypoint that passes the
family to ``main``::

    # families/torchvision/src/fashion_seg_torchvision/__main__.py
    import fashion_seg_torchvision
    from fashion_seg.cli import main

    main(fashion_seg_torchvision)

which gives, in the family's environment::

    python -m fashion_seg_<family> train [--smoke]
    python -m fashion_seg_<family> package <model>                  # a key of params.yaml:models
    python -m fashion_seg_<family> evaluate <model> [--max-images N] [--output FILE]

``main`` wires the family, the parameters and the workflow steps (``fashion_seg.service``) and
knows no family by name.
"""

import argparse
import logging
import sys
from pathlib import Path

from fashion_seg.config import PackagedModel, Params, load_params
from fashion_seg.ports import ModelFamily
from fashion_seg.service import evaluation, packaging, training

logger = logging.getLogger("fashion_seg")


def build_parser(family: ModelFamily) -> argparse.ArgumentParser:
    """Build the command-line parser of a family.

    Parameters
    ----------
    family : ModelFamily
        The model family (its name goes in the help).

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``train``, ``package`` and ``evaluate`` commands.
    """
    name = family.SPEC.name
    parser = argparse.ArgumentParser(
        prog=f"python -m fashion_seg_{name}", description=f"Train, package and evaluate {name}."
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help=f"train {name} (params.yaml:train.{name})")
    train.add_argument("--smoke", action="store_true", help="tiny run on locally pulled images")
    package = commands.add_parser("package", help="package and register a served model")
    package.add_argument("model", help="a key of params.yaml:models")
    evaluate = commands.add_parser("evaluate", help="score a packaged model")
    evaluate.add_argument("model", help="a key of params.yaml:models")
    evaluate.add_argument("--max-images", type=int, help="override evaluate.max_images")
    evaluate.add_argument("--output", type=Path, help="metrics file")
    return parser


def served_model(family: ModelFamily, params: Params, name: str) -> PackagedModel:
    """Find a served model of this family in ``params.yaml:models``.

    Parameters
    ----------
    family : ModelFamily
        The model family running the command.
    params : Params
        The pipeline parameters.
    name : str
        Key of ``params.yaml:models``.

    Returns
    -------
    PackagedModel
        The served model.

    Raises
    ------
    SystemExit
        If the model is unknown or belongs to another family (another environment).
    """
    own = sorted(k for k, m in params.models.items() if m.family == family.SPEC.name)
    if name not in own:
        raise SystemExit(f"{name!r} is not a {family.SPEC.name} model: expected one of {own}")
    return params.models[name]


def main(family: ModelFamily, argv: list[str] | None = None) -> None:
    """Run a command of the workflow with the given family.

    Parameters
    ----------
    family : ModelFamily
        The model family, injected by its entrypoint.
    argv : list[str] | None
        Command-line arguments, ``sys.argv[1:]`` when ``None``. By default ``None``.

    Raises
    ------
    TypeError
        If ``family`` does not implement ``ModelFamily``.
    """
    if not isinstance(family, ModelFamily):
        raise TypeError(f"{family!r} does not implement fashion_seg.ports.ModelFamily")
    args = build_parser(family).parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = load_params(args.params)
    name = family.SPEC.name

    if args.command == "train":
        config = family.Config.model_validate(params.train[name])
        result = training.train(family, config, params, smoke=args.smoke)
        logger.info("Trained %s: %s", name, result.metrics)
    elif args.command == "package":
        model = served_model(family, params, args.model)
        provenance = packaging.package(family, args.model, model, params)
        logger.info(
            "Registered %s v%s; saved to %s",
            provenance["registered_name"],
            provenance["registered_version"],
            model.output_dir,
        )
    else:
        model = served_model(family, params, args.model)
        metrics = evaluation.evaluate(
            args.model, model, params, max_images=args.max_images, output=args.output
        )
        logger.info(
            "mask mAP %.4f | box mAP %.4f | %d images",
            metrics["mask_map"],
            metrics["box_map"],
            metrics["n_images"],
        )
