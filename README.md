# fashion-seg-train

Fashion instance segmentation: detect and segment the clothes in a photo
([iMaterialist-Fashion 2020](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7), 46 categories).

This repository is **where every model is trained**: it owns the data, the shared train/val split,
one module per model family, experiment tracking, and the packaging of the chosen model into a
self-contained MLflow model that [fashion-serving](https://github.com/thibaudchevrier/fashion-serving)
imports and serves.

## How it fits together

```
 Google Drive (DVC remote)                     fashion-seg-train (this repo)
 ┌──────────────────┐  dvc pull  ┌─────────┐   ┌──────────────────────────┐   ┌──────────────┐
 │ data/ (images,   │ ─────────► │ prepare │──►│ train_<model>            │──►│ package      │
 │  train.csv)      │            │ + split │   │ (one per model family,   │   │ MLflow pyfunc│
 │ weights/ (COCO)  │ ─────────────────────────►│  own environment)        │   │ + contract   │
 │ deployement/     │            └─────────┘   └────────────┬─────────────┘   └──────┬───────┘
 │ (2021 model)     │ ─────────────────────────────────────────────────────────────►  │
 │                  │                                        │ losses, params         │
 │                  │                                        ▼                        ▼
 │                  │                                 MLflow (local mlflow.db)  models/fashion-maskrcnn
 │                  │ ◄──────────────────────── dvc push ─────────────────────────────┘
 └──────────────────┘                                            │ dvc import (pinned commit)
                                                                 ▼
                                                   fashion-serving: mlflow models serve + webapp
```

| Tool | Role |
|------|------|
| **git** | Code, `params.yaml`, `dvc.yaml`, and small pointer files (`*.dvc`, `dvc.lock`) holding the hashes of the large files |
| **DVC** | Stores large files on Google Drive, runs the pipeline stages, transports the packaged model to fashion-serving |
| **MLflow** | Experiment tracking (params, per-epoch losses) and model registry, all local; the model package format |
| **uv** | One environment per purpose: the root one (DVC, lint, `prepare`) and one per model family with its own Python and framework |

### Repository layout

| Path | Content | Stored in |
|------|---------|-----------|
| `packages/fashion-seg/` | `fashion_seg`, the shared library: workflow, ports, data, scoring, serving wrapper, CLI (see [Code architecture](#code-architecture)) | git |
| `families/torchvision/` | torchvision family: code + uv project (Python 3.12, PyTorch) | git |
| `families/matterport/` | Matterport family: code + training uv project (Python 3.11, TensorFlow 2.15); `serve/`: its serving uv project (Python 3.12, TensorFlow 2.21) | git |
| `data/` | iMaterialist images + `train.csv` + `label_descriptions.json` (~23.7 GB, 48k files) | DVC (`data.dvc`) |
| `prepared/` | `annotations.parquet` (one row per image) + `split.json` (frozen train/val ids) | DVC (`prepare` stage) |
| `weights/` | COCO starting weights of each family, imported from their releases | DVC (`import-url`) |
| `outputs/<family>/` | Trained model (`model/`) and `metrics.json` | DVC (`train_<family>` stage) |
| `deployement/` | 2021 Matterport model (TF SavedModel + `config.json`) | DVC (`deployement.dvc`) |
| `models/fashion-maskrcnn/` | **Build output**: the packaged 2021 model, the one fashion-serving imports today | DVC (`package_legacy` stage) |
| `models/fashion-maskrcnn-torchvision/` | **Build output**: the packaged torchvision model | DVC (`package_torchvision` stage) |
| `metrics/` | Evaluation scores of each packaged model | git (`evaluate_<model>` stage) |
| `mlruns/` | Archive: a Nov 2024 MLflow 1.30 log of the 2021 weights (no params or metrics) | DVC (`mlruns.dvc`) |
| `mlflow.db`, `mlartifacts/` | Local MLflow tracking store | not versioned |

The deployed model is still the 2021 one, re-packaged: TensorFlow 2.21 loads its SavedModel, and
maskrcnn-matterport's `mrcnn.serving` runs it with Matterport's own pre/post-processing. Models
trained here are exported in the same format, so they are packaged and served the same way.

### Released packages

Code shared with other repositories comes from released packages, not copies:

| Package | Provides | Used by |
|---------|----------|---------|
| [fashion-seg-contract](https://github.com/thibaudchevrier/fashion-seg-contract) | Response JSON Schema, RLE `encode`/`decode`, label mapping | `fashion_seg` (every environment), fashion-serving |
| [maskrcnn-matterport](https://github.com/thibaudchevrier/maskrcnn-matterport-tf2) | Matterport network and training (`[train]`), TensorFlow-free inference helpers and export runner | Matterport family: training (`[train]`, `families/matterport`), serving (base, `families/matterport/serve`) |

They are referenced by the URL of their released wheel in `[tool.uv.sources]` (the library's and
each family's `pyproject.toml`), as from a package registry: GitHub Packages has no Python
registry, so each release attaches its wheel. The packaged model pins the same URLs in its
`requirements.txt`. To upgrade, change the URL, run `uv lock` in each environment
(`uv lock --project families/<name>`), then re-run the `package_<model>` stages: they depend on
their environment's `uv.lock`, so the models are re-packaged.

## Code architecture

The code follows **ports and adapters** (hexagonal architecture). The workflow is written once in
the shared library, against interfaces. Each model family is an adapter implementing them, in its
own uv project with its own Python and framework. Each family's entrypoint wires it explicitly:

```python
# families/torchvision/src/fashion_seg_torchvision/__main__.py
import fashion_seg_torchvision
from fashion_seg.cli import main

main(fashion_seg_torchvision)  # the generic CLI, with this family injected
```

```
uv run --project families/torchvision python -m fashion_seg_torchvision train [--smoke]
uv run --project families/torchvision python -m fashion_seg_torchvision package torchvision
uv run --project families/matterport/serve python -m fashion_seg_matterport evaluate legacy
```

| Module | Role |
|--------|------|
| `fashion_seg.ports` | The interfaces: `ModelFamily` (what a family provides), `Predictor`, `MetricLogger`, and the types they exchange |
| `fashion_seg.service` | The workflow steps, identical for every family: `preparation`, `training`, `packaging`, `evaluation` |
| `fashion_seg.cli` | `main(family)`: the generic command line (`train`, `package`, `evaluate`) |
| `fashion_seg.config`, `.data`, `.scoring`, `.tracking`, `.serving` | Parameters (pydantic), dataset logic, COCO mAP, MLflow setup, the MLflow serving wrapper: independent building blocks |
| `fashion_seg_<family>` | One model family: `SPEC` (serving requirements), `Config` (its `params.yaml` section), `train`, `load_predictor`, `describe`; network, dataset, training loop and predictor in submodules; `__main__` its entrypoint |

The command (DVC stage or Makefile target) names the environment and the entrypoint: nothing is
looked up by name or switched at run time. The design rules (functions first, dependencies
through Protocols, injection in the entrypoint) are in [`CLAUDE.md`](CLAUDE.md) and checked by
`tests/test_architecture.py`.

## Pipeline

```
data/imaterialist/train.csv ─► prepare ─► prepared/ ─┐
data/imaterialist/train/ ────────────────────────────┼─► train_matterport  ─► outputs/matterport/
weights/mask_rcnn_coco.h5 ───────────────────────────┤
weights/maskrcnn_resnet50_fpn_v2_coco.pth ───────────┴─► train_torchvision ─► outputs/torchvision/
deployement/ (2021 model) ────────────────────────────► package_legacy      ─► models/fashion-maskrcnn/
outputs/torchvision/model ────────────────────────────► package_torchvision ─► models/fashion-maskrcnn-torchvision/
models/<packaged> + prepared/ + val images ───────────► evaluate_<model>    ─► metrics/evaluate-<model>.json
```

Each stage runs in the environment it names: `python -m fashion_seg prepare` (root), or a family's entrypoint `uv run --project families/<...> python -m fashion_seg_<family> <command>`.

| Stage | Environment | What it does |
|-------|-------------|--------------|
| `prepare` | root | Groups `train.csv` (333k masks) into one row per image, and writes the frozen split: images sorted by id, `KFold(8, shuffle, seed 42)`, fold 0 = validation (39,920 train / 5,703 val), the procedure of the 2021 notebook |
| `train_matterport` | `families/matterport` | Starts from COCO weights, trains with the 2021 notebook's settings (`params.yaml:train.matterport`), logs every epoch's losses to the MLflow experiment `fashion-seg-training`, exports the inference model and `metrics.json` |
| `train_torchvision` | `families/torchvision` | Fine-tunes torchvision Mask R-CNN v2 from COCO (`params.yaml:train.torchvision`) on CUDA, the Apple GPU or CPU; logs losses, learning rate and per-epoch validation losses to `fashion-seg-training`; checkpoints every epoch (`resume: true` continues an interrupted run); exports `config.json` + `model.pt` |
| `package_legacy`, `package_torchvision` | `families/matterport/serve`, `families/torchvision` | One per entry of `params.yaml:models`: wraps a model export as an MLflow pyfunc model that serves the contract, with its own requirements (TensorFlow or PyTorch, never both); registers a new `fashion-maskrcnn` version tagged with its `model_family`; writes `models/<name>/` |
| `evaluate_legacy`, `evaluate_torchvision` | `families/matterport/serve`, `families/torchvision` | Loads a packaged model as it is served, predicts the validation split, computes COCO mask and box mAP (plus AP50, AP75, recall, per-class AP) with pycocotools; logs to the MLflow experiment `fashion-seg-evaluation`, tags the registered model version (`val_mask_map`...), writes `metrics/evaluate-<model>.json` |

Each stage re-runs only when its inputs (files, code, `params.yaml` section) changed since `dvc.lock`.

> **Always target a stage** (`dvc repro --single-item <stage>`) unless the whole `data/` folder is
> pulled. A plain `dvc repro` also re-checks `data.dvc` and, with a partial pull, would re-record the
> dataset as only the files present locally. If that happens: `git checkout data.dvc`.

## One-time setup

**1. Install the environments:**

```bash
make install     # the root env and each family's (families/*/.venv, families/matterport/serve/.venv)
```

**2. Give DVC access to Google Drive.** Google blocks DVC's shared OAuth app, so use your own:

1. In [Google Cloud Console](https://console.cloud.google.com/): create or pick a project and enable the
   **Google Drive API**.
2. **Google Auth Platform → Audience**: publishing status **Testing**, add your Google account as a
   **test user**.
3. **Google Auth Platform → Clients → Create client**, type **Desktop app**. Copy the client ID and secret.
4. Store them in the git-ignored `.dvc/config.local`:

   ```bash
   uv run dvc remote modify --local myremote gdrive_client_id '<client-id>'
   uv run dvc remote modify --local myremote gdrive_client_secret '<client-secret>'
   ```

5. Run any `dvc pull`: a browser opens. Google warns that the app is unverified (it's yours):
   **Advanced → Go to … → Continue**. The token is cached in `~/Library/Caches/pydrive2fs/`. While
   the app is in *Testing*, Google expires it after 7 days: the browser step then comes back.

**3. Pull what you need:**

| Goal | Command | Size |
|------|---------|------|
| Package / serve the 2021 model | `uv run dvc pull deployement.dvc data/imaterialist/label_descriptions.json` | ~265 MB |
| Smoke-train on a few images | `uv run dvc pull prepare weights/mask_rcnn_coco.h5.dvc data/imaterialist/label_descriptions.json && make pull-sample` | ~1.2 GB |
| Full training | `uv run dvc pull data.dvc prepare weights/mask_rcnn_coco.h5.dvc` | ~25 GB (slow on Drive) |

## Workflow

### Train a model

**torchvision (recommended)**, PyTorch on the Apple GPU (MPS) or CUDA:

```bash
make pull-sample && make train-torchvision-smoke        # end-to-end check, ~15 s
uv run dvc repro --single-item train_torchvision        # needs data.dvc (~24 GB) + weights/
uv run dvc repro --single-item package_torchvision
uv run dvc repro --single-item evaluate_torchvision      # compare with evaluate_legacy
```

Measured on an M4 Pro: 0.8 s per step (2 images, 1024 px) on the Apple GPU, 4.9 s on the CPU, so
about 5 hours per epoch (~20,000 steps). Checkpoints are written after each epoch: with
`resume: true`, re-running the stage continues an interrupted training.

**Matterport** (the 2021 model's architecture; TensorFlow 2.15, CPU-only on a Mac):

```bash
make train-matterport-smoke
uv run dvc repro --single-item train_matterport
```

Tune by editing `params.yaml:train.<family>` (epochs, learning rate, image size...), or without
editing it: `uv run dvc exp run --single-item train_torchvision -S train.torchvision.epochs=2`.
Unknown keys are rejected when the parameters are loaded.
Compare trainings in MLflow (`make mlflow-ui`); compare models with the `evaluate` stage.

### Package and publish

```bash
uv run dvc repro --single-item package_legacy   # or package_torchvision; skipped if unchanged
make check                                      # ruff, pylint, all tests
uv run dvc push                                 # upload new outputs to Google Drive
git add -A && git commit -m "..." && git push   # open a PR; CI checks lint, tests, dvc.lock freshness
```

After the merge, deploy it in fashion-serving with `dvc update` (see its README).

### Evaluate a packaged model

```bash
make pull-val                                        # the 5,703 validation images (~3 GB)
uv run dvc repro --single-item evaluate_legacy       # whole split (~2 h on CPU), tracked by DVC
uv run dvc metrics show                              # mask_map, box_map, ... per model

VAL_IMAGES=200 make pull-val && make evaluate-quick MODEL=legacy   # 200 images, ~5 min
```

The model is loaded from `models/<name>` exactly as fashion-serving loads it, so the score describes
the deployed model, whatever its framework. Each run is logged to MLflow (`fashion-seg-evaluation`)
with the per-class AP table as an artifact; a whole-split run also tags the registered model
version, so the registry shows its score. Quick runs write `metrics/evaluate-<model>-quick.json`
and don't tag the registry.

### Browse experiments

```bash
make mlflow-ui    # http://localhost:5002 (Ctrl+C to stop; MLFLOW_PORT=... to change the port)
```

- **Experiments → `fashion-seg-training`**: one run per training, tagged `model_family`, with all
  params, per-epoch losses (train and validation) and, for real runs, the exported model.
- **Experiments → `fashion-maskrcnn`**: one run per packaging.
- **Models → `fashion-maskrcnn`**: registered versions. The packaged one is recorded in
  `models/fashion-maskrcnn/provenance.json`.

No Docker or server is needed: the UI reads `mlflow.db` and `mlartifacts/` directly. A shared tracking
server becomes useful once runs come from several machines (e.g. training on a cloud GPU): start one
and set `MLFLOW_TRACKING_URI`. All environments pin the same MLflow version (3.16) because they
write to the same store.

There is no 2021 training history in MLflow: the Colab notebook logged to TensorBoard, in Google Drive
under `Final_project/model/train_results` (outside DVC).

### Add a model family

`families/torchvision/` is the reference implementation; the step-by-step checklist is in
[`CLAUDE.md`](CLAUDE.md#adding-a-model-family). In short: a uv project `families/<name>/` with its
Python and framework, a package `fashion_seg_<name>` implementing `ports.ModelFamily` and an
entrypoint calling `main(fashion_seg_<name>)`, a `train.<name>` section and a `models.<name>`
entry in `params.yaml`, and its `train_`, `package_` and `evaluate_` stages. The library doesn't
change.

## Model contract

`models/fashion-maskrcnn` is a standard MLflow model: anything that can run `mlflow models serve`
can serve it.

```bash
uv run mlflow models serve -m models/fashion-maskrcnn --env-manager local -p 5001
```

Request, one row per image:

```json
POST /invocations
{"dataframe_records": [{"image": "<base64 jpeg/png>"}], "params": {"min_score": 0.8}}
```

Response, one entry per image, specified by the JSON Schema in
[fashion-seg-contract](https://github.com/thibaudchevrier/fashion-seg-contract):

```json
{"predictions": [{"height": 400, "width": 300, "instances": [
  {"class_id": 24, "label": "dress", "score": 0.97, "box": [12, 40, 380, 260], "mask_rle": "5230 12 5630 14 ..."}
]}]}
```

- `box` is `[y1, x1, y2, x2]` in pixels; `mask_rle` uses the iMaterialist encoding (1-indexed
  `start length` pairs, column-major). Decode with `fashion_seg_contract.rle.decode(mask_rle, height, width)`.
- The 2021 model drops detections below 0.7 inside the network, so `min_score` can only raise that.
- **Keeping the contract is what makes models swappable.** Every family is packaged by
  `service/packaging.py` through `FashionSegmentationModel`, with the family's predictor returning
  `Detections`, never with a raw framework flavor such as `mlflow.pytorch`. The contract tests fail
  otherwise.
- The contract is versioned in its own package: adding optional fields is backward compatible;
  renaming or removing fields is a breaking release of fashion-seg-contract, which consumers adopt
  before the producer ships it.

## Development

```bash
make install   # every environment
make hooks     # once: pre-commit and commit-msg git hooks
make format    # ruff format + autofix
make check     # lint (all pre-commit hooks, exactly what CI runs) + tests of every environment
```

Code quality is defined once, in `.pre-commit-config.yaml`: ruff (format, lint, numpy docstrings)
and pydoclint (every parameter, return and exception documented) for all code, pylint (10/10) in
each environment, and hygiene checks (incl. a guard against committing large files). The git hooks,
`make lint` and CI all run it. Conventions for contributors (and for Claude Code) are in
[`CLAUDE.md`](CLAUDE.md).

### Commits, versions and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat: ...`,
`fix(prepare): ...`, `refactor(data): ...`, `docs: ...`, `ci: ...`. They are checked by the
`commit-msg` hook and on every PR by CI. `uv run cz commit` writes one interactively.

Releases are automatic. On every merge to `main`, [commitizen](https://commitizen-tools.github.io/commitizen/)
reads the commits since the last tag. A `feat` (minor), `fix`/`perf`/`refactor` (patch) or breaking change
(minor while < 1.0) bumps the version in `pyproject.toml` and `uv.lock`, updates `CHANGELOG.md`,
tags `vX.Y.Z` and publishes a GitHub Release with the changelog entry. Other types (`docs`, `ci`, `test`, `build`, `chore`...) never release.

The version tracks the code; model versions are tracked separately by the MLflow registry and
`dvc.lock`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on `main`:

| Job | What it checks |
|-----|----------------|
| **Lint** | All pre-commit hooks (`make lint`): ruff, pydoclint, pylint per environment, hygiene |
| **Tests** (one job per environment) | Library (data, scoring, serving wrapper, CLI, architecture rules), torchvision (training smoke run, predictor, packaging), Matterport training (smoke run, TF 2.15, Python 3.11) and serving; responses validated with `fashion_seg_contract.schema` |
| **ML checks** | Pulls the 2021 model and the published package from Drive, fails if `dvc.lock` is stale for `package_legacy`, runs the tests against the real model |

The **ML checks** need Drive access and are skipped until the `GDRIVE_CREDENTIALS_DATA` secret
exists. Use a service account (a personal OAuth token expires after 7 days while the app is in
Testing):

1. Cloud Console → **IAM & Admin → Service accounts → Create**, no roles needed. Open it →
   **Keys → Add key → JSON**.
2. In Google Drive, share the DVC folder (`1JQNGq5d1GiA8lC_GA-N5v3iQc1X9vHaK`) with the service
   account's email, as **Viewer** (CI only reads).
3. GitHub → repository **Settings → Secrets and variables → Actions → New repository secret**:
   name `GDRIVE_CREDENTIALS_DATA`, value = the whole JSON key file.

## Roadmap

1. **First torchvision training** (1 epoch, ~5 h on the Apple GPU) and its evaluation against the
   2021 baseline.
2. **Promotion**: mark the best evaluated model `@champion` in the MLflow registry and serve it;
   CI comments `dvc metrics diff` on the PR.
3. **Pre-resized images** (a `prepare` output at 1024 px) to speed up data loading.
4. **Augmentation** for the Matterport trainer (imgaug is unmaintained; needs a compatible substitute).
