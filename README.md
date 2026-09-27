# maskrcnn-train

Fashion instance segmentation: detect and segment the clothes in a photo (Mask R-CNN,
[iMaterialist 2019](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6), 46 categories).

This repository owns **data, experiments and model packaging**. It publishes a
self-contained MLflow model that
[fashion-serving](https://github.com/thibaudchevrier/fashion-serving) imports and serves.

## How it fits together

```
                      maskrcnn-train (this repo)                               fashion-serving
 ┌──────────────┐   dvc pull   ┌──────────────────────────────┐
 │ Google Drive │ ───────────► │ data/          deployement/  │
 │ (DVC remote) │              │   (images,       (2021 TF     │
 │              │              │   annotations)   weights)     │
 │              │              └──────────────┬───────────────┘
 │              │                             │ dvc repro
 │              │                             ▼
 │              │              ┌──────────────────────────────┐   logs run,
 │              │              │ package stage                │── registers ──► MLflow (local)
 │              │              │ weights + code + labels      │   version
 │              │              │ → MLflow pyfunc model        │
 │              │   dvc push   └──────────────┬───────────────┘
 │              │ ◄──────────── models/fashion-maskrcnn        dvc import     ┌───────────────────────┐
 │              │ ──────────────────────────────────────────────────────────► │ mlflow models serve   │
 └──────────────┘                (git: dvc.lock pins the exact version)       │ + Flask upload webapp │
                                                                              └───────────────────────┘
```

| Tool | Role |
|------|------|
| **git** | Code, `params.yaml`, `dvc.yaml`, and small pointer files (`*.dvc`, `dvc.lock`) holding the hashes of the large files |
| **DVC** | Stores large files on Google Drive, runs the pipeline, and transports the packaged model to fashion-serving |
| **MLflow** | Experiment tracking and model registry (local), and the model package format (`MLmodel` + code + pinned requirements) |
| **uv** | Python environment and dependencies (`pyproject.toml`, `uv.lock`) |

### What is in the repository

| Path | Content | Stored in |
|------|---------|-----------|
| `src/fashion_seg/` | Source code: RLE codec, labels, Mask R-CNN pre/post-processing, MLflow wrapper, packaging script | git |
| `data/` | iMaterialist images + `train.csv` + `label_descriptions.json` (~23.7 GB, 48k files) | DVC (`data.dvc`) |
| `deployement/` | 2021 Matterport Mask R-CNN weights (TF SavedModel + `config.json`) | DVC (`deployement.dvc`) |
| `mlruns/` | 2021 MLflow runs, read-only history | DVC (`mlruns.dvc`) |
| `models/fashion-maskrcnn/` | **Build output**: the packaged MLflow model consumed by fashion-serving | DVC (`dvc.lock`) |
| `contracts/prediction.schema.json` | JSON Schema of the model's response, shared with fashion-serving | git |
| `mlflow.db`, `mlartifacts/` | Local MLflow tracking store | not versioned |

The model in `models/` is the 2021 model re-packaged, not retrained: TensorFlow 2.21 still loads the
SavedModel, and the Matterport pre/post-processing was ported to numpy
(`src/fashion_seg/legacy/matterport.py`). Retraining comes next (see [Roadmap](#roadmap)).

## One-time setup

**1. Install the environment** (Python 3.12, TensorFlow, MLflow, DVC):

```bash
uv sync
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
   **Advanced → Go to … → Continue**. The token is then cached in `~/Library/Caches/pydrive2fs/`.

**3. Pull what you need:**

```bash
uv run dvc pull deployement.dvc                              # 2021 weights (~265 MB)
uv run dvc pull data/imaterialist/label_descriptions.json    # labels only: enough to package the model
uv run dvc pull data.dvc                                     # full dataset (~23.7 GB, slow on Drive)
uv run dvc pull mlruns.dvc                                   # 2021 MLflow history
```

## Workflow

### 1. Change something

Code in `src/fashion_seg/`, settings in `params.yaml`, or data/weights (then `uv run dvc add <path>`).

### 2. Check quality

```bash
make format   # ruff format + autofix
make check    # ruff, pylint, pytest (the real-model test runs when deployement/ is pulled)
```

### 3. Rebuild the model

```bash
uv run dvc repro --single-item package_legacy
```

DVC compares the stage inputs (`deployement/`, the labels file, `src/fashion_seg/`,
`params.yaml:legacy_model`) with the hashes in `dvc.lock`:

- **nothing changed** → the stage is skipped;
- **something changed** → it runs `python -m fashion_seg.package_legacy`, which logs an MLflow run,
  registers a new `fashion-maskrcnn` version, rewrites `models/fashion-maskrcnn/` (with a
  `provenance.json` pointing back to the MLflow run), and updates `dvc.lock`.

> Use `--single-item` unless the whole `data/` folder is pulled. A plain `dvc repro` also re-checks
> `data.dvc` and, with a partial pull, would re-record the dataset as only the files present
> locally. If that happens: `git checkout data.dvc`.

### 4. Publish

```bash
uv run dvc push                       # upload new data/model files to Google Drive
git add -A && git commit -m "..."     # dvc.lock pins exactly which model version was built
git push                              # open a PR; CI checks lint, tests, contract, dvc.lock freshness
```

After the merge, deploy it in fashion-serving with `dvc update` (see its README).

### Browse experiments

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # runs and registered versions
uv run mlflow ui --backend-store-uri ./mlruns --port 5001    # 2021 history
```

Set `MLFLOW_TRACKING_URI` to log to a remote MLflow server instead of the local SQLite store.

## Model contract

`models/fashion-maskrcnn` is a standard MLflow model: anything that can run
`mlflow models serve` can serve it.

```bash
uv run mlflow models serve -m models/fashion-maskrcnn --env-manager local -p 5001
```

Request, one row per image:

```json
POST /invocations
{"dataframe_records": [{"image": "<base64 jpeg/png>"}], "params": {"min_score": 0.8}}
```

Response, one entry per image, specified by `contracts/prediction.schema.json`:

```json
{"predictions": [{"height": 400, "width": 300, "instances": [
  {"class_id": 24, "label": "dress", "score": 0.97, "box": [12, 40, 380, 260], "mask_rle": "5230 12 5630 14 ..."}
]}]}
```

- `box` is `[y1, x1, y2, x2]` in pixels; `mask_rle` uses the iMaterialist encoding (1-indexed
  `start length` pairs, column-major). Decode with `fashion_seg.rle.decode(mask_rle, height, width)`.
- The 2021 model drops detections below 0.7 inside the network, so `min_score` can only raise that.
- **Keeping the contract is what makes models swappable.** Any new model (e.g. torchvision) must be
  packaged through `FashionSegmentationModel` with a predictor returning `Detections`, never with a
  raw framework flavor such as `mlflow.pytorch`. The contract tests fail otherwise.
- The schema also lives in fashion-serving: change both copies together. Adding optional fields is
  backward compatible; renaming or removing fields is not.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on `main`:

| Job | What it checks |
|-----|----------------|
| **Lint** | `ruff format --check`, `ruff check`, `pylint` (same as `make lint`) |
| **Unit and contract tests** | RLE, pre/post-processing, MLflow wrapper; responses match `contracts/prediction.schema.json` |
| **ML checks** | Pulls the pipeline inputs and the published model from Drive, fails if `dvc.lock` is stale (code or params changed without re-packaging), runs the tests against the real model |

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

- **Retraining**: add `prepare` (train/val split) → `train` → `evaluate` stages, likely on torchvision
  Mask R-CNN (maintained, runs on Apple Silicon). Log params/metrics to MLflow, write metrics to
  `metrics.json` so `dvc metrics diff` and `dvc exp show` compare experiments. Then a generic
  `package` stage exports the best run through the same wrapper. fashion-serving needs no change.
- **CI for experiments**: once training exists, post `dvc params diff` / `dvc metrics diff` as a PR
  comment.
