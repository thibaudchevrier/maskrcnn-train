"""Run a command in the Python environment it needs.

Families may need another Python than the root environment (Matterport trains on Python 3.11
with TensorFlow 2.15, in ``envs/matterport``). ``python -m fashion_seg`` checks the current
interpreter and, if it is not the right one, runs itself again in that environment with uv.
"""

import subprocess
import sys

from fashion_seg.ports import Runtime


def is_active(runtime: Runtime) -> bool:
    """Tell whether the current interpreter is the runtime's Python version.

    Parameters
    ----------
    runtime : Runtime
        The required environment.

    Returns
    -------
    bool
        ``True`` if the ``major.minor`` versions match.
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}" == runtime.python


def command(runtime: Runtime, argv: list[str]) -> list[str]:
    """Build the command running ``python -m fashion_seg`` in the runtime's environment.

    Parameters
    ----------
    runtime : Runtime
        The required environment.
    argv : list[str]
        Arguments of ``python -m fashion_seg``.

    Returns
    -------
    list[str]
        The command.

    Raises
    ------
    RuntimeError
        If the runtime is the root environment (it has no project to switch to).

    Examples
    --------
    >>> command(Runtime(python="3.11", project="envs/x"), ["train", "x"])[:5]
    ['uv', 'run', '--project', 'envs/x', '--locked']
    """
    if runtime.project is None:
        raise RuntimeError(
            f"Needs Python {runtime.python} (the root environment): "
            "run `uv run python -m fashion_seg`"
        )
    return [
        "uv",
        "run",
        "--project",
        runtime.project,
        "--locked",
        "python",
        "-m",
        "fashion_seg",
        *argv,
    ]


def run_in(runtime: Runtime, argv: list[str]) -> int:
    """Run ``python -m fashion_seg`` in the runtime's environment and wait for it.

    Parameters
    ----------
    runtime : Runtime
        The required environment.
    argv : list[str]
        Arguments of ``python -m fashion_seg``.

    Returns
    -------
    int
        The exit code.
    """
    return subprocess.call(command(runtime, argv))
