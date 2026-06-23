.PHONY: quality test lint format-check typecheck boundaries artifacts

quality: lint format-check typecheck boundaries artifacts test

test:
	uv run --extra dev pytest

lint:
	uv run --extra dev ruff check src/aegis src/detect tests/aegis tests/dp_honey scripts

format-check:
	uv run --extra dev ruff format --check src/aegis src/detect tests/aegis tests/dp_honey scripts

typecheck:
	uv run --extra dev mypy src/aegis src/detect scripts

boundaries:
	uv run python scripts/check_import_boundaries.py

artifacts:
	uv run python scripts/check_artifact_boundaries.py
