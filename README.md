# maskrcnn-train

Fashion instance segmentation (Mask R-CNN, iMaterialist 2019, 46 categories).
This repo owns the **data, experiments and model packaging**:

```
Google Drive (DVC remote)            this repo                         serving repo
 data/ (images + annotations) ──►  dvc pipeline ──► MLflow runs + registry
 deployement/ (2021 SavedModel)          │
                                         └──► models/fashion-maskrcnn  ──dvc import──►  mlflow models serve
                                              (MLflow pyfunc, DVC-tracked)
```

## Setup

```bash
uv sync                 # Python 3.12 env with TensorFlow, MLflow, DVC (+ gdrive)
```

### Google Drive access for DVC

Google blocks DVC's shared OAuth app, so use your own OAuth client (one-time):

1. In Google Cloud Console: create a project, enable the **Google Drive API**, configure the
   OAuth consent screen (External, add yourself as test user), then create an
   **OAuth client ID** of type *Desktop app*.
2. Store it locally (written to the git-ignored `.dvc/config.local`):

   ```bash
   uv run dvc remote modify --local myremote gdrive_client_id '<client-id>'
   uv run dvc remote modify --local myremote gdrive_client_secret '<client-secret>'
   ```

3. The first `dvc pull`/`push` opens a browser to authorize; the token is cached afterwards.

### Pull the data and models

```bash
uv run dvc pull deployement.dvc   # legacy 2021 model, ~265 MB
uv run dvc pull data.dvc          # iMaterialist images + annotations, ~23.7 GB / 48k files
uv run dvc pull data/imaterialist/label_descriptions.json   # just the labels (enough to package the model)
uv run dvc pull mlruns.dvc        # 2021 MLflow runs (read-only history)
```

`data/` is large; the first pull from Drive takes a while.

## Pipeline

Stages are defined in `dvc.yaml` and configured in `params.yaml`. DVC re-runs only the
stages whose inputs changed.

| Stage | What it does |
|-------|--------------|
| `package_legacy` | Wraps the 2021 Matterport SavedModel as an MLflow pyfunc model, logs a run, registers a new version of `fashion-maskrcnn`, and exports it to `models/fashion-maskrcnn` |

> **Partial pulls:** if you only pulled part of `data/` (e.g.
> `dvc pull data/imaterialist/label_descriptions.json`), run stages with
> `dvc repro --single-item <stage>`. A plain `dvc repro` would re-commit `data.dvc` with only
> the files present locally. Restore it with `git checkout data.dvc` if that happens.

```bash
uv run dvc repro        # run the pipeline (requires data/ fully pulled)
uv run dvc push         # upload new outputs (models/...) to Drive
git add dvc.lock && git commit -m "..."   # dvc.lock pins exactly which data/model versions were produced
```

### Experiments (MLflow)

Runs are logged to a local SQLite store (`mlflow.db`, artifacts in `mlartifacts/`). Point
`MLFLOW_TRACKING_URI` at a server to log elsewhere.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # current experiments
uv run mlflow ui --backend-store-uri ./mlruns --port 5001    # 2021 history
```

Only the chosen model is exported to `models/` and versioned with DVC; the MLflow store
stays local.

## Model contract

`models/fashion-maskrcnn` is a self-contained MLflow model (weights, code, pinned
requirements). Serve it with:

```bash
uv run mlflow models serve -m models/fashion-maskrcnn --env-manager local -p 5000
```

Request:

```json
POST /invocations
{"dataframe_records": [{"image": "<base64 jpeg/png>"}], "params": {"min_score": 0.8}}
```

Response, one entry per image:

```json
{"predictions": [{"height": 400, "width": 300, "instances": [
  {"class_id": 24, "label": "dress", "score": 0.97, "box": [y1, x1, y2, x2], "mask_rle": "12 3 40 5 ..."}
]}]}
```

- `mask_rle` uses the iMaterialist annotation encoding (1-indexed `start length` pairs,
  column-major); decode with `fashion_seg.rle.decode(mask_rle, height, width)`.
- The legacy model filters detections below 0.7 inside the graph, so `min_score` can only
  raise that threshold.

## Development

```bash
uv run pytest           # the real-model test runs when deployement/ is pulled
uv run ruff check src tests && uv run ruff format src tests
```
