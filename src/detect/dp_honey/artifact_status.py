"""Typed inspection and validation status for DP-HONEY model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .errors import DPHoneyError, UnknownFormatError
from .formats import get_format
from .model_io import ArtifactSource, load_model, read_artifact_dict


class SnapshotStatus(StrEnum):
    """Lenient format snapshot status for an inspected model artifact."""

    OK = "OK"
    DRIFT = "DRIFT"
    UNKNOWN_FORMAT = "UNKNOWN_FORMAT"


@dataclass(frozen=True)
class ArtifactInspection:
    """Lenient, JSON-friendly metadata extracted from a model artifact."""

    schema_version: object
    format_slug: object
    registry_version: object
    epsilon: object
    clip: object
    corpus_size: object
    train_seed: object
    alphabet_size: int
    snapshot_status: SnapshotStatus
    safety: object

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format": self.format_slug,
            "registry_version": self.registry_version,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "corpus_size": self.corpus_size,
            "train_seed": self.train_seed,
            "alphabet_size": self.alphabet_size,
            "snapshot_status": self.snapshot_status.value,
            "safety": self.safety,
        }


@dataclass(frozen=True)
class ArtifactValidation:
    """Strict artifact validation result that does not raise at the adapter seam."""

    valid: bool
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "error": self.error}


def inspect_artifact(source: ArtifactSource) -> ArtifactInspection:
    """Return lenient artifact metadata, including drift status."""
    data = read_artifact_dict(source)
    fmt = _mapping(data.get("format"))
    privacy = _mapping(data.get("privacy"))
    alphabet = _mapping(data.get("alphabet"))
    symbols = alphabet.get("symbols")
    slug = fmt.get("slug", "?")
    return ArtifactInspection(
        schema_version=data.get("schema_version"),
        format_slug=slug,
        registry_version=fmt.get("registry_version"),
        epsilon=privacy.get("epsilon"),
        clip=privacy.get("clip"),
        corpus_size=privacy.get("corpus_size"),
        train_seed=privacy.get("train_seed"),
        alphabet_size=len(symbols) if isinstance(symbols, list) else 0,
        snapshot_status=snapshot_status(slug, fmt.get("spec_hash")),
        safety=data.get("safety", {}),
    )


def validate_artifact(source: ArtifactSource) -> ArtifactValidation:
    """Strictly validate an artifact and return a typed result."""
    try:
        load_model(source)
    except DPHoneyError as exc:
        return ArtifactValidation(valid=False, error=str(exc))
    return ArtifactValidation(valid=True, error=None)


def snapshot_status(slug: object, stored_hash: object) -> SnapshotStatus:
    """Resolve a stored format hash against the live registry."""
    if not isinstance(slug, str):
        return SnapshotStatus.UNKNOWN_FORMAT
    try:
        live = get_format(slug)
    except UnknownFormatError:
        return SnapshotStatus.UNKNOWN_FORMAT
    return SnapshotStatus.OK if stored_hash == live.spec_hash() else SnapshotStatus.DRIFT


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return value
    return {}
