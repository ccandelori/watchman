from __future__ import annotations

import importlib.util
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType


def _load_boundary_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_artifact_boundaries.py"
    spec = importlib.util.spec_from_file_location("check_artifact_boundaries", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BOUNDARY_SCRIPT = _load_boundary_script()
artifact_boundary_violations = _BOUNDARY_SCRIPT.artifact_boundary_violations


def test_generated_trace_and_worktree_paths_are_rejected() -> None:
    violations = artifact_boundary_violations(
        (
            PurePosixPath(".worktrees/trace-collection-harness/README.md"),
            PurePosixPath("data/trace_collection/records.generated.jsonl"),
            PurePosixPath("data/redteam_results/aegis-local.jsonl"),
            PurePosixPath("introspection/data/trace_collection/structured_prompts.generated.jsonl"),
            PurePosixPath("results/redteam/aegis-local.jsonl"),
        )
    )

    assert {str(violation.path) for violation in violations} == {
        ".worktrees/trace-collection-harness/README.md",
        "data/trace_collection/records.generated.jsonl",
        "data/redteam_results/aegis-local.jsonl",
        "introspection/data/trace_collection/structured_prompts.generated.jsonl",
        "results/redteam/aegis-local.jsonl",
    }


def test_python_cache_artifacts_are_rejected() -> None:
    violations = artifact_boundary_violations(
        (
            PurePosixPath("src/aegis/__pycache__/contracts.cpython-312.pyc"),
            PurePosixPath("tests/aegis/test_contracts.py"),
        )
    )

    assert [str(violation.path) for violation in violations] == ["src/aegis/__pycache__/contracts.cpython-312.pyc"]


def test_runtime_raw_model_artifacts_are_rejected_but_research_artifacts_are_allowed() -> None:
    violations = artifact_boundary_violations(
        (
            PurePosixPath("src/aegis/detectors/model.pt"),
            PurePosixPath("tests/aegis/fixtures/probe.pkl"),
            PurePosixPath("introspection/data/models/research_probe.pkl"),
            PurePosixPath("introspection/data/activations/qwen.pt"),
        )
    )

    assert {str(violation.path) for violation in violations} == {
        "src/aegis/detectors/model.pt",
        "tests/aegis/fixtures/probe.pkl",
    }
