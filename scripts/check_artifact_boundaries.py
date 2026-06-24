from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

GENERATED_PREFIXES: tuple[PurePosixPath, ...] = (
    PurePosixPath(".worktrees"),
    PurePosixPath("data/trace_collection"),
    PurePosixPath("data/redteam_results"),
    PurePosixPath("introspection/data/trace_collection"),
    PurePosixPath("results/redteam"),
)

RUNTIME_PREFIXES: tuple[PurePosixPath, ...] = (
    PurePosixPath("src/aegis"),
    PurePosixPath("tests/aegis"),
    PurePosixPath("scripts"),
)

RAW_MODEL_SUFFIXES: tuple[str, ...] = (
    ".bin",
    ".gguf",
    ".joblib",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".safetensors",
)


@dataclass(frozen=True)
class ArtifactBoundaryViolation:
    path: PurePosixPath
    reason: str


def artifact_boundary_violations(paths: tuple[PurePosixPath, ...]) -> tuple[ArtifactBoundaryViolation, ...]:
    violations: list[ArtifactBoundaryViolation] = []
    for path in paths:
        violation = artifact_boundary_violation(path)
        if violation is not None:
            violations.append(violation)
    return tuple(violations)


def artifact_boundary_violation(path: PurePosixPath) -> ArtifactBoundaryViolation | None:
    if has_forbidden_prefix(path, GENERATED_PREFIXES):
        return ArtifactBoundaryViolation(
            path=path,
            reason="generated local worktrees and trace-collection outputs must not be committed",
        )
    if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
        return ArtifactBoundaryViolation(
            path=path,
            reason="Python cache artifacts must not be committed",
        )
    if has_forbidden_prefix(path, RUNTIME_PREFIXES) and path.suffix in RAW_MODEL_SUFFIXES:
        return ArtifactBoundaryViolation(
            path=path,
            reason="runtime paths must consume promoted JSON fixtures, not raw model or activation artifacts",
        )
    return None


def has_forbidden_prefix(path: PurePosixPath, prefixes: tuple[PurePosixPath, ...]) -> bool:
    return any(path == prefix or prefix in path.parents for prefix in prefixes)


def tracked_file_paths(repository_root: Path) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        ("git", "ls-files"),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed with exit code {result.returncode}: {result.stderr.strip()}")
    return tuple(PurePosixPath(line) for line in result.stdout.splitlines() if line != "")


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    violations = artifact_boundary_violations(tracked_file_paths(repository_root))
    if len(violations) == 0:
        return 0
    for violation in violations:
        sys.stderr.write(f"{violation.path}: {violation.reason}.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
