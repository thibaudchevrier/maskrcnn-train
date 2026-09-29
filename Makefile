# One uv environment per purpose; each family runs through its own entrypoint:
#   uv run --project <env> python -m fashion_seg_<family> train | package <model> | evaluate <model>
# `lint` runs the pre-commit hooks on every file: the same checks as the git hooks and CI.
TORCHVISION = uv run --project families/torchvision
MATTERPORT_TRAIN = TF_CPP_MIN_LOG_LEVEL=3 uv run --project families/matterport
MATTERPORT_SERVE = TF_CPP_MIN_LOG_LEVEL=3 uv run --project families/matterport/serve
# 5000 is taken by AirPlay on macOS, 5001 by fashion-serving's inference service.
MLFLOW_PORT ?= 5002
SAMPLE_IMAGES ?= 12
export MLFLOW_DISABLE_AGENT_HINT = 1

.PHONY: install hooks format lint test test-architecture test-library test-torchvision test-matterport-train \
	test-matterport-serve check mlflow-ui prepare pull-sample pull-val \
	train-matterport-smoke train-torchvision-smoke evaluate-quick \
	train train-log train-stop mlflow-snapshot

install:
	uv sync --locked
	uv sync --locked --project families/torchvision
	uv sync --locked --project families/matterport
	uv sync --locked --project families/matterport/serve

hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run pre-commit run --all-files --show-diff-on-failure

# Each environment runs its own tests (docstring examples included).
test: test-architecture test-library test-torchvision test-matterport-train test-matterport-serve

# Dependency rules across the library and every family (reads the source: root environment).
test-architecture:
	uv run pytest

test-library:
	cd packages/fashion-seg && uv run pytest -p no:warnings

test-torchvision:
	cd families/torchvision && uv run pytest -p no:warnings

test-matterport-train:
	cd families/matterport && TF_CPP_MIN_LOG_LEVEL=3 uv run pytest -p no:warnings

test-matterport-serve:
	cd families/matterport && TF_CPP_MIN_LOG_LEVEL=3 uv run --project serve pytest -p no:warnings tests/serve

check: lint test

# Version the MLflow store (runs, curves, registry) with DVC and upload it; commit the two .dvc
# pointers afterwards. Run it when no training, packaging or evaluation is writing to it.
mlflow-snapshot:
	uv run dvc add mlflow.db mlartifacts
	uv run dvc push mlflow.db.dvc mlartifacts.dvc

# Local experiment tracking UI (runs + model registry). Ctrl+C to stop.
mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

# Per-image annotations + frozen train/val split (needs data/imaterialist/train.csv).
prepare:
	uv run dvc repro --single-item prepare

# A few train/val images for smoke runs, without pulling the whole 23.7 GB dataset.
pull-sample:
	uv run python -c "from fashion_seg.data.files import load_split as s; d = s('prepared'); n = $(SAMPLE_IMAGES); print('\n'.join(f'data/imaterialist/train/{i}.jpg' for i in d['train'][:n] + d['val'][:max(2, n // 4)]))" | xargs uv run dvc pull

# Validation images (~3 GB for the 5,703 of the split; VAL_IMAGES=200 for a subset).
VAL_IMAGES ?=
pull-val:
	uv run python -c "from fashion_seg.data.files import load_split as s; ids = s('prepared')['val']; n = '$(VAL_IMAGES)'; ids = ids[:int(n)] if n else ids; print('\n'.join(f'data/imaterialist/train/{i}.jpg' for i in ids))" | xargs -n 500 uv run dvc pull

# Tiny runs on the pulled images: check data, training, checkpointing, MLflow logging and export.
train-torchvision-smoke:
	$(TORCHVISION) python -m fashion_seg_torchvision train --smoke

train-matterport-smoke:
	$(MATTERPORT_TRAIN) python -m fashion_seg_matterport train --smoke

# Full training of a family (DVC stage train_<family>), in the background: it survives closing
# the terminal and keeps the Mac awake (caffeinate; closing the lid still sleeps it). Resumable:
# `make train-stop` finishes the current step, saves and stops; `make train` again resumes.
FAMILY ?= torchvision
TRAIN_LOG = outputs/train-$(FAMILY).log
train:
	@mkdir -p outputs
	nohup caffeinate -i uv run dvc repro --single-item train_$(FAMILY) >> $(TRAIN_LOG) 2>&1 &
	@echo "Training $(FAMILY) in the background: 'make train-log FAMILY=$(FAMILY)' to follow it, 'make train-stop' to stop it."

train-log:
	tail -f $(TRAIN_LOG)

# SIGINT to the Python process only (not its uv and DVC parents): it saves, then exits.
train-stop:
	pkill -INT -f "^[^ ]*python[0-9.]* -m fashion_seg_(torchvision|matterport) train" \
		|| echo "No training running."

# Score a packaged model on the first 200 pulled val images, in its family's environment
# (not tracked by DVC, no registry tags): make evaluate-quick MODEL=legacy|torchvision
MODEL ?= legacy
EVALUATE_legacy = $(MATTERPORT_SERVE) python -m fashion_seg_matterport
EVALUATE_torchvision = $(TORCHVISION) python -m fashion_seg_torchvision
evaluate-quick:
	$(EVALUATE_$(MODEL)) evaluate $(MODEL) --max-images 200 --output metrics/evaluate-$(MODEL)-quick.json
