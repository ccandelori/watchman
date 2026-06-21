from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, cast


class LineageError(ValueError):
    """Raised when experiment lineage data is malformed or inconsistent."""


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    path: Path
    sha256: str
    purpose: str
    label_counts: dict[str, int]
    family_count: int


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    path: Path
    sha256: str
    dataset_id: str
    model_id: str
    feature_count: int


@dataclass(frozen=True)
class ReportRecord:
    id: str
    path: Path
    sha256: str
    dataset_id: str
    artifact_id: str
    evaluation_strategy: str


@dataclass(frozen=True)
class LineageManifest:
    schema_version: int
    created_on: str
    datasets: tuple[DatasetRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]
    reports: tuple[ReportRecord, ...]


def _read_hash_chunk(file: BinaryIO) -> bytes:
    return file.read(1024 * 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: _read_hash_chunk(file), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise LineageError(f"Expected {description} to be a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, description: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise LineageError(f"Expected {description} field '{field_name}' to be a string.")
    if value == "":
        raise LineageError(f"Expected {description} field '{field_name}' to be non-empty.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, description: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise LineageError(f"Expected {description} field '{field_name}' to be an integer.")
    if value < 0:
        raise LineageError(f"Expected {description} field '{field_name}' to be non-negative.")
    return value


def _required_list(record: Mapping[str, object], field_name: str, description: str) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise LineageError(f"Expected {description} field '{field_name}' to be a list.")
    return value


def _required_label_counts(record: Mapping[str, object], description: str) -> dict[str, int]:
    value = record.get("label_counts")
    mapping = _as_mapping(value, f"{description} field 'label_counts'")
    label_counts: dict[str, int] = {}
    for label, count in mapping.items():
        if not isinstance(label, str):
            raise LineageError(f"Expected {description} label_counts keys to be strings.")
        if not isinstance(count, int) or count < 0:
            raise LineageError(
                f"Expected {description} label_counts value for '{label}' to be a non-negative integer."
            )
        label_counts[label] = count
    if len(label_counts) == 0:
        raise LineageError(f"Expected {description} label_counts to contain at least one label.")
    return label_counts


def _validate_sha256_text(value: str, description: str) -> None:
    if len(value) != 64:
        raise LineageError(f"Expected {description} sha256 to contain 64 hexadecimal characters.")
    valid_chars = set("0123456789abcdef")
    if any(character not in valid_chars for character in value):
        raise LineageError(f"Expected {description} sha256 to be lowercase hexadecimal.")


def _dataset_record(value: object, index: int) -> DatasetRecord:
    description = f"dataset record {index}"
    record = _as_mapping(value, description)
    sha256 = _required_string(record, "sha256", description)
    _validate_sha256_text(sha256, description)
    return DatasetRecord(
        id=_required_string(record, "id", description),
        path=Path(_required_string(record, "path", description)),
        sha256=sha256,
        purpose=_required_string(record, "purpose", description),
        label_counts=_required_label_counts(record, description),
        family_count=_required_int(record, "family_count", description),
    )


def _artifact_record(value: object, index: int) -> ArtifactRecord:
    description = f"artifact record {index}"
    record = _as_mapping(value, description)
    sha256 = _required_string(record, "sha256", description)
    _validate_sha256_text(sha256, description)
    return ArtifactRecord(
        id=_required_string(record, "id", description),
        path=Path(_required_string(record, "path", description)),
        sha256=sha256,
        dataset_id=_required_string(record, "dataset_id", description),
        model_id=_required_string(record, "model_id", description),
        feature_count=_required_int(record, "feature_count", description),
    )


def _report_record(value: object, index: int) -> ReportRecord:
    description = f"report record {index}"
    record = _as_mapping(value, description)
    sha256 = _required_string(record, "sha256", description)
    _validate_sha256_text(sha256, description)
    return ReportRecord(
        id=_required_string(record, "id", description),
        path=Path(_required_string(record, "path", description)),
        sha256=sha256,
        dataset_id=_required_string(record, "dataset_id", description),
        artifact_id=_required_string(record, "artifact_id", description),
        evaluation_strategy=_required_string(record, "evaluation_strategy", description),
    )


def _unique_ids(ids: tuple[str, ...], description: str) -> None:
    seen: set[str] = set()
    for record_id in ids:
        if record_id in seen:
            raise LineageError(f"Duplicate {description} id '{record_id}'.")
        seen.add(record_id)


def load_lineage_manifest(path: Path) -> LineageManifest:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LineageError(f"Invalid lineage JSON in {path}: {exc.msg}.") from exc

    record = _as_mapping(decoded, "lineage manifest")
    schema_version = _required_int(record, "schema_version", "lineage manifest")
    if schema_version != 1:
        raise LineageError(f"Unsupported lineage schema_version {schema_version}.")

    datasets = tuple(
        _dataset_record(item, index)
        for index, item in enumerate(_required_list(record, "datasets", "lineage manifest"))
    )
    artifacts = tuple(
        _artifact_record(item, index)
        for index, item in enumerate(_required_list(record, "artifacts", "lineage manifest"))
    )
    reports = tuple(
        _report_record(item, index)
        for index, item in enumerate(_required_list(record, "reports", "lineage manifest"))
    )
    _unique_ids(tuple(item.id for item in datasets), "dataset")
    _unique_ids(tuple(item.id for item in artifacts), "artifact")
    _unique_ids(tuple(item.id for item in reports), "report")

    return LineageManifest(
        schema_version=schema_version,
        created_on=_required_string(record, "created_on", "lineage manifest"),
        datasets=datasets,
        artifacts=artifacts,
        reports=reports,
    )


def _validate_file_hash(root_path: Path, relative_path: Path, expected_sha256: str, record_id: str) -> None:
    full_path = root_path / relative_path
    if not full_path.exists():
        raise LineageError(f"Lineage record '{record_id}' points to missing file {relative_path}.")
    actual_sha256 = sha256_file(full_path)
    if actual_sha256 != expected_sha256:
        raise LineageError(
            f"Lineage record '{record_id}' hash mismatch for {relative_path}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def validate_lineage_manifest(manifest: LineageManifest, root_path: Path) -> None:
    dataset_ids = {dataset.id for dataset in manifest.datasets}
    artifact_ids = {artifact.id for artifact in manifest.artifacts}

    for dataset in manifest.datasets:
        _validate_file_hash(root_path, dataset.path, dataset.sha256, dataset.id)

    for artifact in manifest.artifacts:
        if artifact.dataset_id not in dataset_ids:
            raise LineageError(f"Artifact '{artifact.id}' references unknown dataset '{artifact.dataset_id}'.")
        _validate_file_hash(root_path, artifact.path, artifact.sha256, artifact.id)

    for report in manifest.reports:
        if report.dataset_id not in dataset_ids:
            raise LineageError(f"Report '{report.id}' references unknown dataset '{report.dataset_id}'.")
        if report.artifact_id not in artifact_ids:
            raise LineageError(f"Report '{report.id}' references unknown artifact '{report.artifact_id}'.")
        _validate_file_hash(root_path, report.path, report.sha256, report.id)
