# CLAUDE.md

Guidance for working in this repository. Read it before changing anything.

## What this repo is

`fashion-seg-train`: where every fashion segmentation model is **trained, compared and packaged**.
It owns the data (DVC on Google Drive), the frozen train/val split, one trainer per model family,
experiment tracking (MLflow) and the packaging of the served model into `models/fashion-maskrcnn`,
which [fashion-serving](https://github.com/thibaudchevrier/fashion-serving) imports.

| Path | Content | Environment |
|------|---------|-------------|
| `packages/core/` | `fashion-seg-core`: prepared annotations, MLflow setup (training-side code shared by every trainer) | all |
| `src/fashion_seg/` | `prepare` stage, packaging, MLflow serving wrapper (`serving/`), predictor adapters (`predictors/`) | orchestration (Python 3.12, TF 2.21) |
| `trainers/matterport/` | Matterport trainer, own uv project | Python 3.11, `maskrcnn-matterport[train]` (TF 2.15) |
| `trainers/torchvision/` | torchvision Mask R-CNN v2 trainer (`fashion-seg-torchvision`), workspace member | orchestration (PyTorch; CPU wheels on Linux) |
| `dvc.yaml`, `params.yaml`, `dvc.lock` | Pipeline stages, their parameters, the hashes of their inputs/outputs | |

Released packages used here: `fashion-seg-contract` (response schema, RLE, labels) and
`maskrcnn-matterport` (Matterport network, training, TensorFlow-free inference), by wheel URL.

### Rules specific to this repo

- **DVC**
  - Run stages one at a time: `uv run dvc repro --single-item <stage>`. A plain `dvc repro` with a
    partial `data/` pull re-records `data.dvc` with only the local files; if it happens,
    `git checkout data.dvc`.
  - Changing a stage's code or parameters makes it stale: re-run it, `uv run dvc push`, and commit
    `dvc.lock` in the same PR. CI fails on a stale `dvc.lock` (when Drive credentials are set).
  - Never commit data, models, `mlflow.db` or `mlartifacts/`; never edit `dvc.lock` or `*.dvc` by
    hand.
- **Models are served through the contract**: package every model with
  `python -m fashion_seg.package` (`FashionSegmentationModel` + a predictor in
  `src/fashion_seg/predictors/` returning `Detections`); a packaged model ships only its own
  framework (see `FAMILIES` in `package.py`). Never log a raw framework flavor (`mlflow.pytorch`...) for
  serving: fashion-serving only understands the contract.
- **Every model trains and is evaluated on the same data**: `prepared/annotations.parquet` and
  `prepared/split.json`, read with `fashion_seg_core.annotations`. Models are compared with the
  `evaluate` stage only (COCO mAP of the packaged model on the val split), never with training
  losses.
- **MLflow**: training runs go to the `fashion-seg-training` experiment with a `model_family` tag.
  Every environment pins the same MLflow minor version (they share `mlflow.db`).
- **Shared code**: training-side helpers go in `packages/core`; anything about the model's response
  goes in `fashion-seg-contract`; Matterport code goes in `maskrcnn-matterport`. Don't copy code
  between these places.
- **Compute**: full training needs the whole dataset (`uv run dvc pull data.dvc`, ~24 GB) and a GPU;
  locally, use `make pull-sample` and `make train-matterport-smoke`.
- Google Drive credentials live in the git-ignored `.dvc/config.local`.

### Adding a model family

Follow `trainers/torchvision/` (see the README, "Add a model family"): trainer project (workspace
member when it shares the orchestration Python, own environment otherwise), `train_<name>` and
`package_<name>` stages, a predictor registered in `load_predictor`, its requirements in
`FAMILIES`, an `evaluate.models` entry, lint hooks and tests.

## Commands

```bash
make install                  # orchestration env + Matterport trainer env
make hooks                    # once: install the pre-commit and commit-msg git hooks
make format                   # ruff format + ruff --fix
make lint                     # all pre-commit hooks on all files (exactly what CI runs)
make test                     # both environments' tests, including docstring examples
make check                    # lint + test: run before every commit
make prepare                  # dvc repro --single-item prepare
make pull-sample              # a few images for smoke runs
make train-matterport-smoke   # tiny training run on the pulled images
make train-torchvision-smoke  # same for torchvision (Apple GPU if available)
make mlflow-ui                # http://localhost:5002
make pull-val                 # validation images (VAL_IMAGES=200 for a subset)
make evaluate-quick MODEL=legacy   # score a packaged model on 200 val images (not DVC-tracked)
uv run dvc repro --single-item evaluate@legacy   # score it on the whole split
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
