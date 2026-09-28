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

``main`` is the composition root: it wires the injected family, the parameters, the
infrastructure (the MLflow adapters) and the workflow steps (``fashion_seg.service``), and writes
their results to files. It knows no family by name.
"""

import argparse
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from fashion_seg.adapters import mlflow_models, mlflow_tracking, signals
from fashion_seg.config import PackagedModel, Params, load_params
from fashion_seg.ports import Infrastructure, ModelFamily
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


def write_json(path: Path, content: dict[str, Any]) -> None:
    """Write a result file (DVC metrics), creating its directory.

    Parameters
    ----------
    path : Path
        File to write.
    content : dict[str, Any]
        JSON-serializable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def run_train(family: ModelFamily, args: argparse.Namespace, params: Params) -> None:
    """Train a family, stoppable with Ctrl+C or SIGTERM, then write ``metrics.json``.

    Parameters
    ----------
    family : ModelFamily
        The model family.
    args : argparse.Namespace
        Parsed ``train`` arguments.
    params : Params
        The pipeline parameters.

    Raises
    ------
    SystemExit
        If the training was stopped: a checkpoint is saved, running the command again resumes.
    """
    config = family.Config.model_validate(params.train[family.SPEC.name])
    with signals.stop_on_signal() as stop:  # only here: other commands stop on Ctrl+C as usual
        infra = Infrastructure(mlflow_tracking, mlflow_models, stop)
        result = training.train(family, config, params, infra, smoke=args.smoke)
    if result.stopped_at is not None:
        raise SystemExit(
            f"Stopped at step {result.stopped_at}, checkpoint saved: "
            "run the same command to resume."
        )
    write_json(training.output_dir_of(config, args.smoke) / "metrics.json", result.metrics)
    logger.info("Trained %s: %s", family.SPEC.name, result.metrics)


def run_command(family: ModelFamily, args: argparse.Namespace, params: Params) -> None:
    """Run one command, with the MLflow infrastructure.

    Parameters
    ----------
    family : ModelFamily
        The model family.
    args : argparse.Namespace
        Parsed command line.
    params : Params
        The pipeline parameters.
    """
    if args.command == "train":
        run_train(family, args, params)
        return
    infra = Infrastructure(mlflow_tracking, mlflow_models, threading.Event())
    if args.command == "package":
        model = served_model(family, params, args.model)
        provenance = packaging.package(family, args.model, model, params, infra)
        logger.info(
            "Registered %s v%s; saved to %s",
            provenance["registered_name"],
            provenance["registered_version"],
            model.output_dir,
        )
    else:
        model = served_model(family, params, args.model)
        metrics = evaluation.evaluate(args.model, model, params, infra, args.max_images)
        write_json(
            args.output or params.evaluate.output_dir / f"evaluate-{args.model}.json", metrics
        )
        logger.info(
            "mask mAP %.4f | box mAP %.4f | %d images",
            metrics["mask_map"],
            metrics["box_map"],
            metrics["n_images"],
        )


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
    run_command(family, args, load_params(args.params))
