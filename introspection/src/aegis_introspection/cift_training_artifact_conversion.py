from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from aegis_introspection.cift_model_training import (
    cift_training_artifact_to_pickle_record,
    load_cift_training_artifact_with_unseal_policy,
)
from aegis_introspection.lineage import sha256_file
from aegis_introspection.sealed_holdout_policy import SealedHoldoutPolicyError, assert_unsealed_path


class CiftTrainingArtifactConversionError(ValueError):
    """Raised when a CIFT training artifact cannot be converted to the dependency-clean format."""


@dataclass(frozen=True)
class CiftTrainingArtifactConversionConfig:
    source_path: Path
    output_path: Path
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftTrainingArtifactConversionReport:
    source_path: Path
    output_path: Path
    source_artifact_sha256: str
    example_count: int
    feature_keys: tuple[str, ...]


def convert_cift_training_artifact(
    config: CiftTrainingArtifactConversionConfig,
) -> CiftTrainingArtifactConversionReport:
    try:
        assert_unsealed_path(
            path=config.output_path,
            allow_sealed_holdout=config.allow_sealed_holdout,
            context="CIFT training artifact conversion",
        )
        artifact = load_cift_training_artifact_with_unseal_policy(
            path=config.source_path,
            allow_sealed_holdout=config.allow_sealed_holdout,
            context="CIFT training artifact conversion",
        )
    except (SealedHoldoutPolicyError, ValueError) as exc:
        raise CiftTrainingArtifactConversionError(str(exc)) from exc

    record = cift_training_artifact_to_pickle_record(artifact)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("wb") as file:
        pickle.dump(record, file)
    return CiftTrainingArtifactConversionReport(
        source_path=config.source_path,
        output_path=config.output_path,
        source_artifact_sha256=sha256_file(config.source_path),
        example_count=len(artifact.example_ids),
        feature_keys=tuple(sorted(artifact.features.keys())),
    )
