# `lint` runs the pre-commit hooks on every file: the same checks as the git hooks and CI.
MATTERPORT = uv run --project envs/matterport
# Code that only runs in envs/matterport (Python 3.11, TensorFlow 2.15), and its tests.
MATTERPORT_CODE = src/fashion_seg/families/matterport/dataset.py src/fashion_seg/families/matterport/training.py tests/matterport
# 5000 is taken by AirPlay on macOS, 5001 by fashion-serving's inference service.
MLFLOW_PORT ?= 5002
SAMPLE_IMAGES ?= 12

.PHONY: install hooks format lint test test-matterport check mlflow-ui prepare pull-sample pull-val train-matterport-smoke train-torchvision-smoke evaluate-quick

install:
	uv sync --locked
	uv sync --locked --project envs/matterport

hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run pre-commit run --all-files --show-diff-on-failure

test: test-matterport
	uv run pytest

test-matterport:
	TF_CPP_MIN_LOG_LEVEL=3 $(MATTERPORT) pytest -p no:warnings $(MATTERPORT_CODE)

check: lint test

# Local experiment tracking UI (runs + model registry). Ctrl+C to stop.
mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

# Per-image annotations + frozen train/val split (needs data/imaterialist/train.csv).
prepare:
	uv run dvc repro --single-item prepare

# A few train/val images for smoke runs, without pulling the whole 23.7 GB dataset.
pull-sample:
	uv run python -c "from fashion_seg.data.annotations import load_split as s; d = s('prepared/split.json'); n = $(SAMPLE_IMAGES); print('\n'.join(f'data/imaterialist/train/{i}.jpg' for i in d['train'][:n] + d['val'][:max(2, n // 4)]))" | xargs uv run dvc pull

# Validation images (~3 GB for the 5,703 of the split; VAL_IMAGES=200 for a subset).
VAL_IMAGES ?=
pull-val:
	uv run python -c "from fashion_seg.data.annotations import load_split as s; ids = s('prepared/split.json')['val']; n = '$(VAL_IMAGES)'; ids = ids[:int(n)] if n else ids; print('\n'.join(f'data/imaterialist/train/{i}.jpg' for i in ids))" | xargs -n 500 uv run dvc pull

# Score a packaged model (MODEL=legacy|torchvision, a key of params.yaml:models) on the first 200 pulled val images
# (not tracked by DVC, no registry tags).
MODEL ?= legacy
evaluate-quick:
	TF_CPP_MIN_LOG_LEVEL=3 MLFLOW_DISABLE_AGENT_HINT=1 uv run python -m fashion_seg evaluate $(MODEL) --max-images 200 --output metrics/evaluate-$(MODEL)-quick.json

# Tiny torchvision run on the pulled images (Apple GPU if available): checks data, training,
# checkpointing, MLflow logging and export.
train-torchvision-smoke:
	MLFLOW_DISABLE_AGENT_HINT=1 uv run python -m fashion_seg train torchvision --smoke

# Tiny Matterport run on the pulled images: checks data, training, MLflow logging and export.
# Runs in envs/matterport (the command switches environment on its own).
train-matterport-smoke:
	TF_CPP_MIN_LOG_LEVEL=3 uv run python -m fashion_seg train matterport --smoke
