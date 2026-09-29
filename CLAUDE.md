# CLAUDE.md

Guidance for working in this repository. Read it before changing anything.

## What this repo is

`fashion-seg-train`: where every fashion segmentation model is **trained, compared and packaged**.
It owns the data (DVC on Google Drive), the frozen train/val split, one uv project per model
family, experiment tracking (MLflow) and the packaging of the served models into `models/<name>`,
which [fashion-serving](https://github.com/thibaudchevrier/fashion-serving) imports.

| Path | Content | Environment (uv project) |
|------|---------|------|
| `packages/fashion-seg/` | `fashion_seg`: the shared library, framework-free (ports, config, data, workflow, scoring, serving wrapper, generic CLI), and its tests | installed in every environment |
| `families/torchvision/` | `fashion_seg_torchvision`: all the torchvision logic + its entrypoint | Python 3.12, PyTorch: trains, packages, evaluates |
| `families/matterport/` | `fashion_seg_matterport`: all the Matterport logic + its entrypoint | Python 3.11, TensorFlow 2.15: trains |
| `families/matterport/serve/` | Serving environment of the same package | Python 3.12, TensorFlow 2.18+: packages, evaluates |
| `pyproject.toml` (root) | The repository: version (commitizen), DVC, lint, architecture tests (`tests/`), `prepare` | Python 3.12, no framework |
| `dvc.yaml`, `params.yaml`, `dvc.lock` | Pipeline stages (each names its environment), parameters, hashes | |

## Architecture: ports and adapters, one entrypoint per family

The pattern is **ports and adapters** (hexagonal architecture): the workflow is written once
against interfaces (the ports, `fashion_seg.ports`), and everything it drives is an adapter
implementing them: each model family, and the infrastructure (MLflow tracking and model registry,
`fashion_seg.adapters`). Each family's **entrypoint is its composition root**: it injects the
family into the generic command line, which adds the infrastructure. No lookup by name, no
dynamic dispatch: the code that runs is the code the entrypoint imports, in the environment the
command names.

```
uv run --project families/torchvision python -m fashion_seg_torchvision train
                                                     │
 families/torchvision/src/fashion_seg_torchvision/__main__.py     (composition root)
     import fashion_seg_torchvision            ── the adapter: SPEC, Config, train,
     from fashion_seg.cli import main              load_predictor, describe
     main(fashion_seg_torchvision)             ── injection
                                                     │
 packages/fashion-seg: cli.main(family)     builds Infrastructure(tracker=mlflow_tracking,
                                                               repository=mlflow_models)
     ─► service/training.train(family, config, params, tracker)
        service/packaging.package(..., infra), service/evaluation.evaluate(..., infra)
        (the workflow, written once, knows only the ports; the CLI writes the result files)
```

| Module (`packages/fashion-seg/src/fashion_seg/`) | Role | May import (from `fashion_seg`) |
|--------|------|------|
| `config.py` | pydantic models of `params.yaml`; `TrainConfig`, the base of every family's config | nothing |
| `data/` | Pure dataset logic (`annotations.py`: per-image grouping and selection; `split.py`: the frozen split) and its files (`files.py`: `train.csv`, prepared annotations and split, images on disk) | itself |
| `scoring.py` | COCO mask and box mAP of contract predictions (pycocotools) | nothing |
| `ports.py` | The interfaces: `ModelFamily`, `Predictor`, `MetricLogger`, `StopSignal`, `Tracker`, `ModelRepository` (Protocols), `Detections`, `TrainInputs`, `TrainingSession`, `TrainResult`, `ModelPackage`, `Infrastructure`, `FamilySpec` | `config` |
| `progress.py` | Where a training is in its data: each epoch's reproducible shuffle order (pure) | nothing |
| `serving/` | `response.py`: the contract response built from `Detections` (pure); `pyfunc.py`: the MLflow pyfunc wrapper serving any family (the family's `load_predictor` is injected and pickled with the model) | `ports` |
| `adapters/` | **Infrastructure**, behind the ports: `mlflow_tracking` (a `Tracker`), `mlflow_models` (a `ModelRepository`: log, register, save, load, tag), `signals` (Ctrl+C/SIGTERM as a `StopSignal`). Modules of functions, duck-typed like the families | `ports`, `serving` |
| `service/` | **The workflow, generic over families and infrastructure**: `preparation`, `training`, `packaging`, `evaluation`. Imports no MLflow | `config`, `data`, `ports`, `scoring` |
| `cli.py` | `main(family, argv)`: builds the infrastructure, runs `train`, `package <model>` or `evaluate <model>`, writes their result files | `adapters`, `config`, `ports`, `service` |
| `__main__.py` | `python -m fashion_seg prepare`: the only step without a family | `config`, `service` |

A family package (`families/<name>/src/fashion_seg_<name>/`) holds **all of its model logic**:
`__init__.py` implements `ModelFamily` (`SPEC`, `Config`, `train`, `load_predictor`, `describe`),
submodules hold the network, dataset, training loop and predictor, and `__main__.py` is its
entrypoint. It may import `fashion_seg.config` and `fashion_seg.ports` only (its entrypoint,
`fashion_seg.cli`), never MLflow nor another family.

Test helpers (a synthetic dataset and its `params.yaml`) are in `packages/fashion-seg-testing`,
a dev-only dependency, so they never ship with a packaged model. The workflow steps are tested
with in-memory fakes of the ports (`packages/fashion-seg/tests/test_services.py`), the adapters
against a temporary MLflow store.

`tests/test_architecture.py` enforces these rules on the source of the library and of every
family: a new import across layers, or MLflow outside the adapters and the serving wrapper,
fails the tests.

**Every family can be stopped and resumed.** `TrainConfig` carries `resume`, `checkpoint_every`,
`log_every` and `seed`; `family.train` receives a `TrainingSession` (metrics logger and stop
signal). A family checkpoints every `checkpoint_every` steps, and when `session.stop` is set it
saves and returns `TrainResult(stopped_at=step)`; the service tags the run and the CLI exits
asking to resume. Resuming restores the checkpoint in `inputs.checkpoint_dir`, as exactly as the
framework allows (use `fashion_seg.progress.epoch_order` for a reproducible order).

### Design rules

- **Functions first.** A module is a set of functions. Use a class only when it is truly needed:
  a pydantic model (configuration, validated data), a frozen dataclass (a record passed between
  modules), a framework's base class (Keras callback, torch `Dataset`, MLflow `PythonModel`), or
  state that genuinely changes together (`TrainingState`). No class to group functions, no
  inheritance to share code.
- **Modules depend on interfaces, not on each other.** Modules talk through the Protocols of
  `ports.py`; a family is duck-typed (a module with the attributes of `ModelFamily`, checked by
  `cli.main`), not a subclass. The library never imports a family nor a framework.
- **Explicit wiring: inject dependencies in the entrypoint.** Each family's `__main__.py` passes
  the family to `cli.main`, which builds the infrastructure (`Infrastructure`: tracker and model
  repository) and passes both to the workflow; families receive a `MetricLogger` instead of
  calling MLflow; the serving wrapper receives `load_predictor`. No
  registry, no lookup by name, no switching environments at run time: the command (DVC stage,
  Makefile) names the environment and the entrypoint.
- **One uv project per family, with its Python and framework.** A family needing different
  environments to train and to serve has one per purpose (`families/matterport` and
  `families/matterport/serve`), running the same package.
- **Configuration is validated.** Every `params.yaml` section has a pydantic model (frozen,
  unknown keys forbidden); code reads attributes, never `params["..."]` dicts.
- **Framework imports stay in their family and load lazily** in the family's `__init__.py`
  (inside `train` / `load_predictor`): a family's two environments each lack one framework.
  `ports.py` doesn't import polars at run time either: a packaged model's serving image has none.
- **Business logic is separate from infrastructure.** The workflow (`service/`) and pure modules
  (`data/annotations.py`, `data/split.py`, `scoring.py`, `serving/response.py`) never import MLflow
  nor write result files; infrastructure lives in `adapters/` (and `data/files.py` for the
  dataset's files), reached through the ports. A new backend (another tracker or registry) is a
  new adapter module, wired in `cli.py`; the workflow doesn't change.
- **Shared behaviour goes in the service, model logic in the family.** Image selection, tracked
  runs, packaging and evaluation are written once in `service/`; a family only trains, exports
  and predicts.

### Rules specific to this repo

- **DVC**
  - Run stages one at a time: `uv run dvc repro --single-item <stage>`. A plain `dvc repro` with a
    partial `data/` pull re-records `data.dvc` with only the local files; if it happens,
    `git checkout data.dvc`.
  - Changing a stage's code or parameters makes it stale: re-run it, `uv run dvc push`, and commit
    `dvc.lock` in the same PR. CI fails on a stale `dvc.lock` (when Drive credentials are set).
    A packaged model bundles the library and its family's code, and depends on its environment's
    lock file: changing them re-packages it.
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
  environment pins the same MLflow minor version (they share `mlflow.db`). The store is versioned by
  DVC: after a training, packaging or evaluation, `make mlflow-snapshot` (when nothing is writing to
  it) and commit `mlflow.db.dvc` / `mlartifacts.dvc` with the PR.
- **Environments**: the library's dependencies (`packages/fashion-seg/pyproject.toml`) stay
  framework-free and install on Python 3.11 and 3.12; frameworks go in the family's project. Its
  version is fixed (not the repository's): family lock files record it, so bump it only with its
  interface, then `uv lock` every environment. Every environment has a `make test-*` target, a
  pylint hook and a CI test job.
- **Shared code**: anything about the model's request or response goes in `fashion-seg-contract`;
  Matterport network code goes in `maskrcnn-matterport`. Don't copy code between repositories.
- **Compute**: full training needs the whole dataset (`uv run dvc pull data.dvc`, ~24 GB) and a
  GPU; locally, use `make pull-sample` and `make train-<family>-smoke`.
- Google Drive credentials live in the git-ignored `.dvc/config.local`.

### Adding a model family

1. `families/<name>/pyproject.toml`: package `fashion-seg-<name>` (`src/fashion_seg_<name>/`),
   its Python, its framework, `fashion-seg` by path (editable), `jsonschema`, `pylint` and
   `pytest` in `dev` (and `fashion-seg-testing` by path); `uv lock --project families/<name>`. Add a `serve/` environment only if
   serving needs other versions than training.
2. `src/fashion_seg_<name>/__init__.py` implements `ports.ModelFamily` (including stop and resume,
   see above): `SPEC` (name, serving
   requirements), `Config` (a `config.TrainConfig` subclass with `smoke_overrides`), `train`,
   `load_predictor`, `describe`, importing the framework lazily. The model logic goes in
   submodules (`network`, `dataset`, `training`, `predictor`). `__main__.py`:
   `main(fashion_seg_<name>)`.
3. `params.yaml`: a `train.<name>` section, and a `models.<served name>` entry for its export.
4. `dvc.yaml`: `train_<name>`, `package_<served name>` and `evaluate_<served name>` stages, each
   `uv run --project families/<name> --locked python -m fashion_seg_<name> ...`.
5. Tests in `families/<name>/tests` (port, training smoke run with `fashion_seg_testing`,
   predictor, packaging round trip); `make test-<name>` and `train-<name>-smoke` targets, a
   pylint hook, a CI matrix entry, `make install`.

## Commands

```bash
make install                  # all environments (root + each family's)
make hooks                    # once: install the pre-commit and commit-msg git hooks
make format                   # ruff format + ruff --fix
make lint                     # all pre-commit hooks on all files (exactly what CI runs)
make test                     # every environment's tests, including docstring examples
make test-library             # the library (packages/fashion-seg/tests, doctests)
make test-architecture        # dependency rules across the library and the families (tests/)
                              # test-torchvision, test-matterport-train, test-matterport-serve
make check                    # lint + test: run before every commit
make prepare                  # dvc repro --single-item prepare
make pull-sample              # a few images for smoke runs
make train-matterport-smoke   # tiny training run on the pulled images
make train-torchvision-smoke  # same for torchvision (Apple GPU if available)
make train FAMILY=torchvision # full training of a family in the background (resumable)
make train-log / train-stop   # follow it / save and stop it (resume: make train again)
make mlflow-ui                # http://localhost:5002
make pull-val                 # validation images (VAL_IMAGES=200 for a subset)
make evaluate-quick MODEL=legacy   # score a packaged model on 200 val images (not DVC-tracked)
uv run --project families/torchvision python -m fashion_seg_torchvision --help
uv run dvc repro --single-item evaluate_legacy   # score it on the whole split
uv run dvc repro --single-item package_legacy    # re-package a model (or package_torchvision)
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
  (fashion-seg-contract for the model's request and response, maskrcnn-matterport for Matterport
  code).

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
