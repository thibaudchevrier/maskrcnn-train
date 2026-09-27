# Same commands locally and in CI (.github/workflows/ci.yml).
PYLINT_TESTS_DISABLE = missing-module-docstring,missing-class-docstring,missing-function-docstring,unbalanced-tuple-unpacking,import-outside-toplevel
# 5000 is taken by AirPlay on macOS, 5001 by fashion-serving's inference service.
MLFLOW_PORT ?= 5002

.PHONY: install format lint test check mlflow-ui

install:
	uv sync --locked

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run pylint src
	uv run pylint tests --disable=$(PYLINT_TESTS_DISABLE)

test:
	uv run pytest

check: lint test

# Local experiment tracking UI (runs + model registry). Ctrl+C to stop.
mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)
