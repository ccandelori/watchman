from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.artifacts import (  # noqa: E402
    ActivationArtifact,
    ActivationArtifactMetadata,
    load_activation_artifact_allowing_sealed_holdout,
)


class MergeActivationArtifactsError(ValueError):
    """Raised when activation artifacts cannot be merged safely."""


@dataclass(frozen=True)
class MergeActivationArtifactsConfig:
    input_paths: tuple[Path, ...]
    output_path: Path
    overwrite: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge compatible torch activation artifacts.")
    parser.add_argument("--input", required=True, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> MergeActivationArtifactsConfig:
    namespace = _build_parser().parse_args(argv)
    input_paths = tuple(Path(str(value)) for value in namespace.input)
    if len(input_paths) < 2:
        raise MergeActivationArtifactsError("at least two --input paths are required.")
    return MergeActivationArtifactsConfig(
        input_paths=input_paths,
        output_path=Path(str(namespace.output)),
        overwrite=bool(namespace.overwrite),
    )


def merge_activation_artifacts(config: MergeActivationArtifactsConfig) -> int:
    if config.output_path.exists() and not config.overwrite:
        raise MergeActivationArtifactsError(f"output path already exists: {config.output_path}.")
    artifacts = tuple(load_activation_artifact_allowing_sealed_holdout(path) for path in config.input_paths)
    first = artifacts[0]
    for artifact_index, artifact in enumerate(artifacts[1:], start=2):
        _validate_compatible_artifact(reference=first, candidate=artifact, artifact_index=artifact_index)
    merged = _merged_artifact(artifacts)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, config.output_path)
    return len(merged["example_ids"])


def _validate_compatible_artifact(
    reference: ActivationArtifact,
    candidate: ActivationArtifact,
    artifact_index: int,
) -> None:
    _validate_metadata(reference["metadata"], candidate["metadata"], artifact_index)
    reference_feature_keys = set(reference["features"])
    candidate_feature_keys = set(candidate["features"])
    if candidate_feature_keys != reference_feature_keys:
        raise MergeActivationArtifactsError(
            f"artifact {artifact_index} feature keys do not match the first artifact."
        )
    for feature_key, reference_tensor in reference["features"].items():
        candidate_tensor = candidate["features"][feature_key]
        if tuple(candidate_tensor.shape[1:]) != tuple(reference_tensor.shape[1:]):
            raise MergeActivationArtifactsError(
                f"artifact {artifact_index} feature '{feature_key}' width does not match the first artifact."
            )


def _validate_metadata(
    reference: ActivationArtifactMetadata,
    candidate: ActivationArtifactMetadata,
    artifact_index: int,
) -> None:
    for field_name, reference_value in reference.items():
        candidate_value = candidate.get(field_name)
        if candidate_value != reference_value:
            raise MergeActivationArtifactsError(
                f"artifact {artifact_index} metadata.{field_name} does not match the first artifact."
            )
    extra_fields = set(candidate) - set(reference)
    if len(extra_fields) > 0:
        joined_fields = ", ".join(sorted(extra_fields))
        raise MergeActivationArtifactsError(f"artifact {artifact_index} has extra metadata fields: {joined_fields}.")


def _merged_artifact(artifacts: tuple[ActivationArtifact, ...]) -> ActivationArtifact:
    first = artifacts[0]
    features = {
        feature_key: torch.cat(tuple(artifact["features"][feature_key] for artifact in artifacts), dim=0)
        for feature_key in first["features"]
    }
    return {
        "metadata": first["metadata"],
        "example_ids": _merge_string_rows(artifact["example_ids"] for artifact in artifacts),
        "labels": _merge_string_rows(artifact["labels"] for artifact in artifacts),
        "families": _merge_string_rows(artifact["families"] for artifact in artifacts),
        "texts": _merge_string_rows(artifact["texts"] for artifact in artifacts),
        "tags": _merge_tag_rows(artifact["tags"] for artifact in artifacts),
        "features": features,
    }


def _merge_string_rows(rows: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    merged: list[str] = []
    for row in rows:
        merged.extend(row)
    return tuple(merged)


def _merge_tag_rows(rows: Sequence[tuple[tuple[str, ...], ...]]) -> tuple[tuple[str, ...], ...]:
    merged: list[tuple[str, ...]] = []
    for row in rows:
        merged.extend(row)
    return tuple(merged)


def main(argv: Sequence[str]) -> int:
    try:
        row_count = merge_activation_artifacts(_parse_args(argv))
    except MergeActivationArtifactsError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    print(f"Wrote merged activation artifact rows: {row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
