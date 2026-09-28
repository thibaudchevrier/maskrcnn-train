# CLAUDE.md

Guidance for working in this repository. Read it before changing anything.

## What this repo is

`fashion-seg-train`: where every fashion segmentation model is **trained, compared and packaged**.
It owns the data (DVC on Google Drive), the frozen train/val split, one module per model family,
experiment tracking (MLflow) and the packaging of the served models into `models/<name>`, which
[fashion-serving](https://github.com/thibaudchevrier/fashion-serving) imports.

One Python package, `fashion_seg` (`src/fashion_seg/`), runs in every environment: the root one
(Python 3.12: serving of every family, torchvision training, DVC, lint) and `envs/matterport`
(Python 3.11 + TensorFlow 2.15, Matterport training only).

## Architecture: ports and adapters

The pattern is **ports and adapters** (hexagonal architecture): a workflow written once against
interfaces (the ports), one adapter per model family, and a registry used as a factory, the way
Detectron2 or MMDetection build models from a name. `python -m fashion_seg` is the composition
root: the only module that depends on all the others.

```
                         __main__ (composition root: CLI, params, wiring)
                          │           │              │
                   registry ──► families/<name>   service/ (the workflow)
                   (factory)    (adapters)          preparation, training,
                          │           │             packaging, evaluation
                          ▼           ▼                 │
                        ports (Protocols, shared types) ◄┘
                          │
                        config (pydantic: params.yaml)
   data/ (annotations, split)   scoring (COCO mAP)   tracking (MLflow)   serving/ (pyfunc)
```

| Module | Role | May import (from `fashion_seg`) |
|--------|------|------|
| `config.py` | pydantic models of `params.yaml`; `TrainConfig`, the base of every family's config | nothing |
| `data/` | Annotations (`annotations.py`) and the frozen split (`split.py`): pure dataset logic | nothing |
| `scoring.py` | COCO mask and box mAP of contract predictions (pycocotools) | nothing |
| `tracking.py` | MLflow tracking store and experiments | nothing |
| `ports.py` | The interfaces: `ModelFamily`, `Predictor`, `MetricLogger` (Protocols), `Detections`, `TrainInputs`, `TrainResult`, `FamilySpec` | `config` |
| `runtime.py` | Runs `python -m fashion_seg` in the Python environment a family needs (uv) | `ports` |
| `registry.py` | Factory: family name → module of `families/` (discovered, not listed) | `families`, `ports` |
| `families/<name>/` | **Everything about one model family** (adapter): `SPEC`, `Config`, `train`, `load_predictor`, `describe` in `__init__.py`; network, dataset, training loop, predictor in submodules | `config`, `ports`, itself |
| `serving/pyfunc.py` | MLflow pyfunc wrapper serving any family through the contract | `registry`, `ports` |
| `service/` | **The workflow, generic over families**: `preparation`, `training` (selection, MLflow run, metrics), `packaging` (MLflow model, registry), `evaluation` | `config`, `data`, `ports`, `scoring`, `serving`, `tracking` |
| `__main__.py` | CLI and composition root: reads `params.yaml`, gets the family from the registry, calls the service | everything |

`tests/test_architecture.py` enforces these rules (and that only `service`, `serving` and
`tracking` import MLflow): a new import across layers fails the tests.

### Design rules

- **Functions first.** A module is a set of functions. Use a class only when it is truly needed:
  a pydantic model (configuration, validated data), a frozen dataclass (a record passed between
  modules), a framework's base class (Keras callback, torch `Dataset`, MLflow `PythonModel`), or
  state that genuinely changes together (`TrainingState`). No class to group functions, no
  inheritance to share code.
- **Modules depend on interfaces, not on each other.** Modules talk through the Protocols of
  `ports.py`; a family is duck-typed (a module with the attributes of `ModelFamily`, checked by
  the registry), not a subclass. Inner modules (`config`, `data`, `scoring`, `tracking`, `ports`)
  never import outer ones (`families`, `service`, `__main__`).
- **Inject dependencies; build them in the composition root.** The service receives the family
  and the parameters; families receive a `MetricLogger` instead of calling MLflow. Wiring happens
  in `__main__.py`, using the registry (factory) to build the family from its name.
- **Configuration is validated.** Every `params.yaml` section has a pydantic model (frozen,
  unknown keys forbidden); code reads attributes, never `params["..."]` dicts.
- **Framework imports stay in their family and load lazily.** `families/<name>/__init__.py` imports
  no deep-learning framework (only inside `train` / `load_predictor`), so the registry, the
  service and the serving wrapper load in every environment. `ports.py` doesn't import polars at
  run time either: the serving image of a packaged model has no polars.
- **Shared behaviour goes in the service, model logic in the family.** Image selection, MLflow
  runs, `metrics.json`, packaging and evaluation are written once in `service/`; a family only
  trains, exports and predicts.

### Rules specific to this repo

- **DVC**
  - Run stages one at a time: `uv run dvc repro --single-item <stage>`. A plain `dvc repro` with a
    partial `data/` pull re-records `data.dvc` with only the local files; if it happens,
    `git checkout data.dvc`.
  - Changing a stage's code or parameters makes it stale: re-run it, `uv run dvc push`, and commit
    `dvc.lock` in the same PR. CI fails on a stale `dvc.lock` (when Drive credentials are set).
    A packaged model bundles all of `src/fashion_seg`: any code change re-packages it.
  - Never commit data, models, `mlflow.db` or `mlartifacts/`; never edit `dvc.lock` or `*.dvc` by
    hand.
- **Models are served through the contract**: every family is packaged by
  `service/packaging.py` (`FashionSegmentationModel` + the family's `load_predictor` returning
  `Detections`); a packaged model ships only its family's framework (`SPEC.serving`). Never log a
  raw framework flavor (`mlflow.pytorch`...) for serving: fashion-serving only understands the
  contract.
- **Every model trains and is evaluated on the same data**: `prepared/annotations.parquet` and
  `prepared/split.json`, selected by `service/training.py`. Models are compared with the
  `evaluate` stage only (COCO mAP of the packaged model on the val split), never with training
  losses.
- **MLflow**: experiments and the registered model name are in `params.yaml:tracking`. Every
  environment pins the same MLflow minor version (they share `mlflow.db`).
- **Environments**: base dependencies of `fashion-seg` (`[project]`) must stay framework-free and
  install on Python 3.11 and 3.12; frameworks go in the `orchestration` dependency group (root) or
  `envs/<name>/pyproject.toml`. Code that only runs in `envs/matterport` is listed in the
  Makefile (`MATTERPORT_CODE`), `conftest.py` and `.pre-commit-config.yaml`.
- **Shared code**: anything about the model's response goes in `fashion-seg-contract`; Matterport
  network code goes in `maskrcnn-matterport`. Don't copy code between repositories.
- **Compute**: full training needs the whole dataset (`uv run dvc pull data.dvc`, ~24 GB) and a
  GPU; locally, use `make pull-sample` and `make train-<family>-smoke`.
- Google Drive credentials live in the git-ignored `.dvc/config.local`.

### Adding a model family

1. Create `src/fashion_seg/families/<name>/`. Its `__init__.py` implements `ports.ModelFamily`:
   `SPEC` (name, training `Runtime`, serving requirements), `Config` (a `config.TrainConfig`
   subclass with `smoke_overrides`), and `train`, `load_predictor`, `describe`, importing the
   framework lazily. Put the model logic in submodules (`network`, `dataset`, `training`,
   `predictor`). The registry finds it by its directory name.
2. If it needs another Python or conflicting dependencies, add `envs/<name>/pyproject.toml` (like
   `envs/matterport`) and set `Runtime(python=..., project="envs/<name>")`; otherwise add its
   framework to the `orchestration` group.
3. `params.yaml`: a `train.<name>` section, and a `models.<served name>` entry for its export.
4. `dvc.yaml`: a `train_<name>` stage (`python -m fashion_seg train <name>`); `package@` and
   `evaluate@` stages come from `models`.
5. Tests: training smoke test, predictor test; `tests/test_architecture.py` checks the port.
   A `make train-<name>-smoke` target.

## Commands

```bash
make install                  # root env + envs/matterport
make hooks                    # once: install the pre-commit and commit-msg git hooks
make format                   # ruff format + ruff --fix
make lint                     # all pre-commit hooks on all files (exactly what CI runs)
make test                     # both environments' tests, including docstring examples
make test-matterport          # envs/matterport only
make check                    # lint + test: run before every commit
make prepare                  # dvc repro --single-item prepare
make pull-sample              # a few images for smoke runs
make train-matterport-smoke   # tiny training run on the pulled images
make train-torchvision-smoke  # same for torchvision (Apple GPU if available)
make mlflow-ui                # http://localhost:5002
make pull-val                 # validation images (VAL_IMAGES=200 for a subset)
make evaluate-quick MODEL=legacy   # score a packaged model on 200 val images (not DVC-tracked)
uv run python -m fashion_seg --help              # prepare | train | package | evaluate
uv run dvc repro --single-item evaluate@legacy   # score it on the whole split
uv run dvc repro --single-item package@legacy    # re-package a model (or package@torchvision)
```

## Standards

These standards are the same in the four repositories of the project (fashion-seg-contract,
maskrcnn-matterport-tf2, fashion-seg-train, fashion-serving). Keep them in sync.

### Environment

- **uv only** (never `pip install`): `uv add` / `uv add --dev` change dependencies and update
  `pyproject.toml` and `uv.lock` together; commit both.
- Packages from the other repositories are referenced by their **release wheel URL** in
  `[tool.uv.sources]`, like registry packages. Upgrade by changing the URL, then `uv lock`.
- Don't edit by hand: `uv.lock`, `CHANGELOG.md`, the `version` in `pyproject.toml` (commitizen owns
  the last two).
- Never commit secrets (`.dvc/*.local`, tokens) or large files (`check-added-large-files`
  blocks files over 1 MB: data and models go to DVC).

### Code quality: `make lint` = pre-commit hooks = CI

`.pre-commit-config.yaml` is the single definition of the checks. The git hooks (`make hooks`),
`make lint` and the CI lint job all run it, so a commit that passes locally passes in CI.

- **ruff format** (line length 100) and **ruff check**: pycodestyle, pyflakes, isort, pyupgrade,
  bugbear, comprehensions, simplify, and pydocstyle (numpy convention).
- **pydoclint**: every parameter, return value, yielded value, raised exception and class attribute
  is documented, with types matching the annotations.
- **pylint**: 10/10.
- Hygiene hooks: trailing whitespace, end of files, YAML/TOML syntax, merge conflicts, large files.
- A `# noqa: <code>` or `# pylint: disable=<name>` needs a reason on the same line. Never disable a
  check globally or raise a limit to make code pass: fix the code.

### Docstrings: numpy style, everywhere

Every module, class and function, public or private, has a
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstring:

```python
def decode(rle: str, height: int, width: int = 1) -> np.ndarray:
    """Decode an RLE string into a boolean mask.

    Parameters
    ----------
    rle : str
        Space-separated ``start length`` pairs, 1-indexed, column-major.
    height : int
        Mask height in pixels.
    width : int
        Mask width in pixels. By default 1.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(height, width)``.

    Raises
    ------
    ValueError
        If the RLE has an odd number of values.

    Examples
    --------
    >>> decode("1 2", height=2).tolist()
    [[True], [True]]
    """
```

- Summary line in the imperative mood, ending with a period, then a blank line before sections.
- Types are written exactly like the annotations (`str | Path`, `dict[str, Any]`): pydoclint
  compares them. No `, optional` suffix: state the default in the description ("By default 1.").
- `Raises` lists the exceptions the function raises itself; mention exceptions propagated from
  callees in the description.
- Classes document their constructor parameters (`Parameters`) and attributes (`Attributes`) in the
  class docstring, not in `__init__`. Instance attributes are declared in the class body
  (`timeout: float`) so they can be checked.
- `Examples` are doctests: pytest runs them (`--doctest-modules`), so they must stay correct.
- Tests: a module docstring, and a one-line docstring per test saying which behaviour it checks.

### Code style

- Type-annotate every function signature, including private helpers.
- Import submodules explicitly (`from skimage import io, transform`), not several `import
  skimage.x` lines: linters can't tell which of those is unused. No re-export blocks: import from
  the module that defines the name.
- `pathlib.Path` over `os.path`, f-strings, no mutable default arguments, `logging` rather than
  `print` in library code, error messages that say what to do.
- Keep functions small enough for pylint's limits; split them rather than raising the limits.
- No duplicated code across repositories: shared code goes in a released package
  (fashion-seg-contract for the model's response, maskrcnn-matterport for Matterport code).

### Tests

- pytest, fast and offline by default. Every behaviour change or bug fix comes with a test.
- Tests that need data, models or services skip cleanly when they are missing (and run in CI when
  the credentials are set).
- `make check` (lint + tests) before every commit.

### Commits, PRs and releases

- [Conventional Commits](https://www.conventionalcommits.org/), checked by the `commit-msg` hook and
  on every PR: `type(scope): summary`, imperative, lower case, no final period. The body explains
  *why*. One logical change per commit.
- Types and their effect on the version (commitizen, `major_version_zero = true`):

  | Type | Release |
  |------|---------|
  | `feat` | minor |
  | `fix`, `perf`, `refactor` | patch |
  | `!` after the type, or a `BREAKING CHANGE:` footer | minor while < 1.0 (then major) |
  | `docs`, `style`, `test`, `ci`, `build`, `chore` | none |

- Work on a branch, open a PR, merge only when CI is green, with a **merge commit** (not squash: the
  individual conventional commits build the changelog). Never push to `main` directly: only the
  release workflow does (bump commit + tag).
- On merge, `release.yml` bumps the version, updates `CHANGELOG.md`, tags `vX.Y.Z` and publishes a
  GitHub Release (with the wheel and sdist for libraries).
