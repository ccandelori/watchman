.PHONY: quality test lint format-check typecheck boundaries

quality: lint format-check typecheck boundaries test

test:
	uv run --extra dev pytest

lint:
	uv run --extra dev ruff check src/aegis tests/aegis scripts

format-check:
	uv run --extra dev ruff format --check src/aegis tests/aegis scripts

typecheck:
	uv run --extra dev mypy src/aegis scripts

boundaries:
	uv run python scripts/check_import_boundaries.py
