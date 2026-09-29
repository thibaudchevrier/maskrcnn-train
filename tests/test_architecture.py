"""Tests of the architecture: the library and the families respect the dependency rules.

Checked on the source (imports parsed, not executed), so the root environment checks the
families too without installing their frameworks.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LIBRARY = ROOT / "packages" / "fashion-seg" / "src" / "fashion_seg"
FAMILIES = sorted((ROOT / "families").glob("*/src/fashion_seg_*"))
FAMILY_PACKAGES = {family.name for family in FAMILIES}
# Code shared by several families (e.g. the PyTorch training loop): a family's helper.
SHARED = sorted((ROOT / "packages").glob("*/src/fashion_seg_*"))

# Which fashion_seg modules each module of the library may import: inner layers know nothing of
# outer ones, and nothing but the command line knows the workflow.
ALLOWED = {
    "config": set(),
    "data": set(),
    "scoring": set(),
    "progress": set(),
    "ports": {"config"},
    "serving": {"ports"},
    "adapters": {"ports", "serving"},
    "service": {"config", "data", "ports", "scoring"},
    "cli": {"adapters", "config", "ports", "service"},
    "__main__": {"config", "service"},
}
# Infrastructure libraries, only in the adapters and the serving wrapper (itself an MLflow
# model): the workflow reaches them through the ports, families through a MetricLogger.
INFRASTRUCTURE = {"mlflow"}
INFRASTRUCTURE_LAYERS = {"adapters", "serving"}
FRAMEWORKS = {"torch", "torchvision", "transformers", "ultralytics", "tensorflow", "keras", "mrcnn"}


def _imports(path: Path) -> set[str]:
    """List the modules a file imports (``from a import b`` gives ``a.b``)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _roots(names: set[str]) -> set[str]:
    """Keep the top-level package of each imported module."""
    return {name.split(".")[0] for name in names}


@pytest.mark.parametrize(
    "path", sorted(LIBRARY.rglob("*.py")), ids=lambda p: str(p.relative_to(LIBRARY))
)
def test_library_modules_respect_the_dependency_rules(path):
    """The library imports no family nor framework, and each layer only its allowed layers."""
    imports = _imports(path)
    assert not _roots(imports) & FRAMEWORKS
    assert not _roots(imports) & FAMILY_PACKAGES
    layer = path.relative_to(LIBRARY).parts[0].removesuffix(".py")
    if layer == "__init__":
        return
    used = {name.split(".")[1] for name in imports if name.startswith("fashion_seg.")} - {layer}
    assert used <= ALLOWED[layer], f"{layer} imports {used - ALLOWED[layer]}"
    if layer not in INFRASTRUCTURE_LAYERS:
        assert not _roots(imports) & INFRASTRUCTURE


@pytest.mark.parametrize(
    "path",
    sorted(p for family in FAMILIES for p in family.rglob("*.py")),
    ids=lambda p: str(p.relative_to(ROOT / "families")),
)
def test_families_only_use_the_ports(path):
    """A family uses the library's config, data, ports, progress (entrypoint: CLI); no MLflow."""
    imports = _imports(path)
    library = {name.split(".")[1] for name in imports if name.startswith("fashion_seg.")}
    allowed = {"config", "data", "ports", "progress"} | (
        {"cli"} if path.name == "__main__.py" else set()
    )
    assert library <= allowed, f"imports fashion_seg.{library - allowed}"
    own = path.parts[path.parts.index("src") + 1]
    assert not _roots(imports) & (FAMILY_PACKAGES - {own})
    assert "mlflow" not in _roots(imports)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda p: p.name)
def test_family_package_imports_its_framework_lazily(family):
    """A family's package module imports no framework: each environment may lack one of them."""
    assert not _roots(_imports(family / "__init__.py")) & FRAMEWORKS


@pytest.mark.parametrize(
    "path",
    sorted(p for package in SHARED for p in package.rglob("*.py")),
    ids=lambda p: str(p.relative_to(ROOT / "packages")),
)
def test_shared_packages_are_family_helpers(path):
    """Shared code uses what a family may use (config, data, ports, progress), and no family."""
    imports = _imports(path)
    library = {name.split(".")[1] for name in imports if name.startswith("fashion_seg.")}
    assert library <= {"config", "data", "ports", "progress"}
    assert not _roots(imports) & FAMILY_PACKAGES
    assert "mlflow" not in _roots(imports)
