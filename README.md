# fashion-seg-train

Fashion instance segmentation: detect and segment the clothes in a photo
([iMaterialist-Fashion 2020](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7), 46 categories).

This repository is **where every model is trained**: it owns the data, the shared train/val split,
one trainer per model family, experiment tracking, and the packaging of the chosen model into a
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
| **uv** | One environment per purpose: orchestration (Python 3.12) and one per trainer (e.g. Python 3.11 + TF 2.15) |

### Repository layout

| Path | Content | Stored in |
|------|---------|-----------|
| `packages/core/` | `fashion-seg-core`: prepared annotations and MLflow setup, shared by the orchestration env and every trainer | git |
| `src/fashion_seg/` | Orchestration (Python 3.12): `prepare` stage, packaging, the MLflow serving wrapper (`serving/`), one predictor adapter per model family (`predictors/`) | git |
| `trainers/matterport/` | Matterport Mask R-CNN trainer: its own uv project (Python 3.11, `maskrcnn-matterport[train]`, i.e. TensorFlow 2.15) | git |
| `data/` | iMaterialist images + `train.csv` + `label_descriptions.json` (~23.7 GB, 48k files) | DVC (`data.dvc`) |
| `prepared/` | `annotations.parquet` (one row per image) + `split.json` (frozen train/val ids) | DVC (`prepare` stage) |
| `weights/mask_rcnn_coco.h5` | COCO starting weights, imported from Matterport's release | DVC (`import-url`) |
| `outputs/<model>/` | Trained model (`model/`: `config.json` + SavedModel) and `metrics.json` | DVC (`train_<model>` stage) |
| `deployement/` | 2021 Matterport model (TF SavedModel + `config.json`) | DVC (`deployement.dvc`) |
| `models/fashion-maskrcnn/` | **Build output**: the packaged MLflow model consumed by fashion-serving | DVC (`package_legacy` stage) |
| `mlruns/` | Archive: a Nov 2024 MLflow 1.30 log of the 2021 weights (no params or metrics) | DVC (`mlruns.dvc`) |
| `mlflow.db`, `mlartifacts/` | Local MLflow tracking store | not versioned |

The deployed model is still the 2021 one, re-packaged: TensorFlow 2.21 loads its SavedModel, and
maskrcnn-matterport's `mrcnn.serving` runs it with Matterport's own pre/post-processing. Models
trained here are exported in the same format, so they are packaged and served the same way.

### Released packages

Code shared with other repositories comes from released packages, not copies:

| Package | Provides | Used by |
|---------|----------|---------|
| [fashion-seg-contract](https://github.com/thibaudchevrier/fashion-seg-contract) | Response JSON Schema, RLE `encode`/`decode`, label mapping | serving wrapper, Matterport trainer, fashion-serving |
| [maskrcnn-matterport](https://github.com/thibaudchevrier/maskrcnn-matterport-tf2) | Matterport network and training (`[train]`), TensorFlow-free inference helpers and export runner | Matterport trainer (`[train]`), serving wrapper (base) |

They are referenced by the URL of their released wheel in `[tool.uv.sources]` (`pyproject.toml` and
`trainers/matterport/pyproject.toml`), as from a package registry: GitHub Packages has no Python
registry, so each release attaches its wheel. The packaged model pins the same URLs in its
`requirements.txt`. To upgrade, change the URL in both files, run `uv lock` (and
`uv lock --project trainers/matterport`), then `uv run dvc repro --single-item package_legacy`: DVC
tracks `tool.uv.sources` as a parameter of that stage, so the model is re-packaged.

## Pipeline

```
data/imaterialist/train.csv ─► prepare ─► prepared/ ─┐
data/imaterialist/train/ ────────────────────────────┼─► train_matterport ─► outputs/matterport/
weights/mask_rcnn_coco.h5 ───────────────────────────┘
deployement/ + labels ──────────────────────────────────► package_legacy ──► models/fashion-maskrcnn/
```

| Stage | Environment | What it does |
|-------|-------------|--------------|
| `prepare` | orchestration | Groups `train.csv` (333k masks) into one row per image, and writes the frozen split: images sorted by id, `KFold(8, shuffle, seed 42)`, fold 0 = validation (39,920 train / 5,703 val), the procedure of the 2021 notebook |
| `train_matterport` | `trainers/matterport` | Starts from COCO weights, trains with the 2021 notebook's settings (`params.yaml:train_matterport`), logs every epoch's losses to the MLflow experiment `fashion-seg-training`, exports the inference model and `metrics.json` |
| `package_legacy` | orchestration | Wraps the 2021 model as an MLflow pyfunc model, registers a new `fashion-maskrcnn` version, writes `models/fashion-maskrcnn/` |

Each stage re-runs only when its inputs (files, code, `params.yaml` section) changed since `dvc.lock`.

> **Always target a stage** (`dvc repro --single-item <stage>`) unless the whole `data/` folder is
> pulled. A plain `dvc repro` also re-checks `data.dvc` and, with a partial pull, would re-record the
> dataset as only the files present locally. If that happens: `git checkout data.dvc`.

## One-time setup

**1. Install the environments:**

```bash
make install     # orchestration env (.venv) + Matterport trainer env (trainers/matterport/.venv)
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

```bash
# quick end-to-end check on the sample images (~1 min, logged as run "matterport-smoke")
make train-matterport-smoke

# real training: needs the full dataset. On CPU this takes days; use a GPU machine
uv run dvc repro --single-item train_matterport
```

Tune by editing `params.yaml:train_matterport` (epochs, learning rate, layers, image size...), or
without editing it: `uv run dvc exp run --single-item train_matterport -S train_matterport.learning_rate=0.001`.
Compare in MLflow (`make mlflow-ui`) and with `uv run dvc metrics show`.

### Package and publish

```bash
uv run dvc repro --single-item package_legacy   # re-packages only if its inputs changed
make check                                      # ruff, pylint, all tests
uv run dvc push                                 # upload new outputs to Google Drive
git add -A && git commit -m "..." && git push   # open a PR; CI checks lint, tests, dvc.lock freshness
```

After the merge, deploy it in fashion-serving with `dvc update` (see its README).

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

1. Create `trainers/<name>/`: a uv project depending on `fashion-seg-core` (path dependency),
   `fashion-seg-contract` (wheel URL) and its framework. Read `prepared/annotations.parquet` and `prepared/split.json` so every model uses the
   same data and split.
2. Log to the `fashion-seg-training` experiment with a `model_family` tag; write `outputs/<name>/`.
3. Add a `train_<name>` stage to `dvc.yaml` and a `train_<name>` section to `params.yaml`.
4. To serve it: add `src/fashion_seg/predictors/<name>.py`, a predictor returning `Detections`
   (see `predictors/base.py`), and package it through `FashionSegmentationModel`, so the response
   contract stays the same.

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
- **Keeping the contract is what makes models swappable.** Any new model must be packaged through
  `FashionSegmentationModel` with a predictor returning `Detections`, never with a raw framework
  flavor such as `mlflow.pytorch`. The contract tests fail otherwise.
- The contract is versioned in its own package: adding optional fields is backward compatible;
  renaming or removing fields is a breaking release of fashion-seg-contract, which consumers adopt
  before the producer ships it.

## Development

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg   # once
make format    # ruff format + autofix, all environments
make check     # ruff, pylint (each env), pytest (orchestration + Matterport smoke training)
```

The hooks format and lint staged Python files with ruff and check the commit message.

### Commits, versions and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat: ...`,
`fix(prepare): ...`, `refactor(data): ...`, `docs: ...`, `ci: ...`. They are checked by the
`commit-msg` hook and on every PR by CI. `uv run cz commit` writes one interactively.

Releases are automatic. On every merge to `main`, [commitizen](https://commitizen-tools.github.io/commitizen/)
reads the commits since the last tag. A `feat` (minor), `fix`/`perf` (patch) or breaking change
(minor while < 1.0) bumps the version in `pyproject.toml` and `uv.lock`, updates `CHANGELOG.md`,
tags `vX.Y.Z` and publishes a GitHub Release with the changelog entry. Other types never release.

The version tracks the code; model versions are tracked separately by the MLflow registry and
`dvc.lock`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on `main`:

| Job | What it checks |
|-----|----------------|
| **Lint** | `ruff format --check`, `ruff check`, `pylint` in each environment (same as `make lint`) |
| **Unit and contract tests** | Annotations and split, MLflow wrapper; responses validated with `fashion_seg_contract.schema` |
| **Matterport trainer** | Trains a tiny model on synthetic data and exports it (TF 2.15, Python 3.11) |
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

1. **`evaluate` stage**: one COCO mask-mAP evaluator on the validation split for every model,
   including the 2021 model as baseline. Metrics to MLflow and `dvc metrics`.
2. **Generic `package` stage**: package the best evaluated model (not only the 2021 one) and mark it
   `@champion` in the MLflow registry; CI comments `dvc metrics diff` on the PR.
3. **torchvision trainer** (`trainers/torchvision/`, PyTorch, runs on Apple Silicon GPUs).
4. **2021 model anchors**: its exported `config.json` uses anchor scales 32–512 while it was trained
   with 16–256; the evaluation stage will tell whether correcting it improves the baseline.
5. **Augmentation** for the Matterport trainer (imgaug is unmaintained; needs a compatible substitute).
