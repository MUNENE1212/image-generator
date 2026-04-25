.PHONY: help install install-eval dev test test-fast lint format typecheck check clean

# Silences "Failed to hardlink" when uv cache (~/.cache/uv) and project .venv are on different mounts.
export UV_LINK_MODE := copy

help:
	@echo "Targets:"
	@echo "  install       Install core deps + dev tools (fast)"
	@echo "  install-eval  Install heavy ML deps for the eval harness"
	@echo "  dev           Run the Streamlit app"
	@echo "  test          Run all tests"
	@echo "  test-fast     Skip tests marked 'slow' or 'integration'"
	@echo "  lint          Ruff lint"
	@echo "  format        Ruff format"
	@echo "  typecheck     Mypy"
	@echo "  check         lint + typecheck + test-fast"
	@echo "  clean         Remove caches + build artifacts"

install:
	uv sync --extra dev

install-eval:
	uv sync --extra dev --extra eval

dev:
	uv run streamlit run app/Home.py

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow and not integration"

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

check: lint typecheck test-fast

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
