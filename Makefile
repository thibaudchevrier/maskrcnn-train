# Same commands locally and in CI (.github/workflows/ci.yml).
PYLINT_TESTS_DISABLE = missing-module-docstring,missing-class-docstring,missing-function-docstring,unbalanced-tuple-unpacking,import-outside-toplevel

.PHONY: install format lint test check

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
