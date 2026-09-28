"""Factory of model families: name -> the ``fashion_seg.families.<name>`` module.

Families are discovered, not listed: adding ``fashion_seg/families/<name>/`` is enough. Importing a
family is cheap (its framework loads only when it trains or predicts), so any environment can
look one up.
"""

import importlib
import pkgutil

from fashion_seg import families
from fashion_seg.ports import ModelFamily


def names() -> list[str]:
    """List the available model families.

    Returns
    -------
    list[str]
        Family names, sorted.

    Examples
    --------
    >>> names()
    ['matterport', 'torchvision']
    """
    return sorted(module.name for module in pkgutil.iter_modules(families.__path__) if module.ispkg)


def get_family(name: str) -> ModelFamily:
    """Load a model family by name.

    Parameters
    ----------
    name : str
        Family name, e.g. ``"torchvision"``.

    Returns
    -------
    ModelFamily
        The family module.

    Raises
    ------
    ValueError
        If no family has this name.
    TypeError
        If the module does not implement ``ModelFamily``.

    Examples
    --------
    >>> get_family("torchvision").SPEC.train_runtime.python
    '3.12'
    """
    if name not in names():
        raise ValueError(f"Unknown model family {name!r}: expected one of {names()}")
    family = importlib.import_module(f"{families.__name__}.{name}")
    if not isinstance(family, ModelFamily):
        raise TypeError(f"{family.__name__} does not implement fashion_seg.ports.ModelFamily")
    return family
