from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.artifacts import ActivationArtifact, load_activation_artifact


@dataclass(frozen=True)
class InspectionConfig:
    artifact_path: Path
    example_count: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a saved Aegis activation feature artifact.")
    parser.add_argument(
        "artifact",
        nargs="?",
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_features.pt"),
    )
    parser.add_argument("--examples", required=False, type=int, default=3)
    return parser


def _parse_args(argv: Sequence[str]) -> InspectionConfig:
    namespace = _build_parser().parse_args(argv)
    example_count = int(namespace.examples)
    if example_count < 0:
        raise ValueError("--examples must be greater than or equal to 0.")
    return InspectionConfig(
        artifact_path=Path(namespace.artifact),
        example_count=example_count,
    )


def _print_metadata(artifact: ActivationArtifact) -> None:
    metadata = artifact["metadata"]
    print("metadata")
    print(f"  model_id: {metadata['model_id']}")
    print(f"  revision: {metadata['revision']}")
    print(f"  selected_device: {metadata['selected_device']}")
    print(f"  layer_indices: {metadata['layer_indices']}")
    print(f"  pooling_methods: {metadata['pooling_methods']}")


def _print_rows(artifact: ActivationArtifact) -> None:
    labels = artifact["labels"]
    counts = Counter(labels)
    print("rows")
    print(f"  examples: {len(artifact['example_ids'])}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


def _print_features(artifact: ActivationArtifact) -> None:
    print("features")
    for key, tensor in artifact["features"].items():
        print(f"  {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def _print_examples(artifact: ActivationArtifact, example_count: int) -> None:
    if example_count == 0:
        return

    limit = min(example_count, len(artifact["example_ids"]))
    print("examples")
    for index in range(limit):
        print(f"  {artifact['example_ids'][index]} [{artifact['labels'][index]}]")
        print(f"    tags: {artifact['tags'][index]}")
        print(f"    text: {artifact['texts'][index]}")


def run_inspection(config: InspectionConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    print(f"artifact: {config.artifact_path}")
    _print_metadata(artifact)
    _print_rows(artifact)
    _print_features(artifact)
    _print_examples(artifact, config.example_count)


def main(argv: Sequence[str]) -> None:
    run_inspection(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
