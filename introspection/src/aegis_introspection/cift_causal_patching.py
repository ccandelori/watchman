from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np

from aegis.core.action_severity import action_severity
from aegis.core.contracts import Action
from aegis.detectors.cift_runtime import (
    CiftRuntimeModel,
    CiftRuntimePrediction,
    load_cift_runtime_model,
    predict_cift_runtime_model,
)
from aegis_introspection.cift_model_training import (
    CiftTrainingArtifact,
    load_cift_training_artifact_with_unseal_policy,
)
from aegis_introspection.lineage import sha256_file

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

_SCHEMA_VERSION = "aegis_introspection.cift_counterfactual_patching/v1"
_INTERVENTION_TYPE = "paired_feature_vector_replacement"
_CLAIM_SCOPE = "runtime_detector_decision"
_PAIRING_TAG_FIELDS = ("participant", "task", "family", "credential_type")
_VARIANT_TAG_FIELD = "variant"
_LIMITATION = (
    "This report patches persisted pooled CIFT feature vectors consumed by the runtime detector; "
    "it does not patch transformer hidden states or measure changed model generations."
)


class CiftCounterfactualPatchingError(ValueError):
    """Raised when a CIFT counterfactual patching report cannot be generated."""


@dataclass(frozen=True)
class CiftCounterfactualPatchingConfig:
    activation_artifact_path: Path
    runtime_model_path: Path
    output_path: Path
    report_id: str
    created_at: str
    minimum_flip_rate: float
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftPatchPrediction:
    score: float
    predicted_label: str
    action: str
    operating_band: str


@dataclass(frozen=True)
class CiftPatchPair:
    pair_key: str
    participant: str
    task: str
    family: str
    variant: str
    credential_type: str
    safe_example_id: str
    exfil_example_id: str
    safe_original: CiftPatchPrediction
    exfil_original: CiftPatchPrediction
    safe_to_exfil_patch: CiftPatchPrediction
    exfil_to_safe_patch: CiftPatchPrediction


@dataclass(frozen=True)
class CiftCounterfactualPatchingReport:
    schema_version: str
    report_id: str
    model_bundle_id: str
    training_dataset_id: str
    task_name: str
    feature_key: str
    source_artifact_sha256: str
    intervention_type: str
    claim_scope: str
    transformer_hidden_state_patching: bool
    paper_faithfulness_limitation: str
    pairing_tag_fields: tuple[str, ...]
    pair_count: int
    skipped_ambiguous_pair_count: int
    skipped_ambiguous_pair_keys: tuple[str, ...]
    minimum_flip_rate: float
    safe_original_allow_rate: float
    exfil_original_block_rate: float
    safe_to_exfil_block_rate: float
    exfil_to_safe_allow_rate: float
    passed: bool
    families: tuple[str, ...]
    pairs: tuple[CiftPatchPair, ...]
    created_at: str


@dataclass(frozen=True)
class _ArtifactRow:
    row_index: int
    example_id: str
    label: str
    family: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class _PairedRows:
    key: str
    participant: str
    task: str
    family: str
    variant: str
    credential_type: str
    safe: _ArtifactRow
    exfil: _ArtifactRow


@dataclass(frozen=True)
class _PairingSelection:
    pairs: tuple[_PairedRows, ...]
    skipped_ambiguous_pair_keys: tuple[str, ...]


def run_cift_counterfactual_patching(
    config: CiftCounterfactualPatchingConfig,
) -> CiftCounterfactualPatchingReport:
    _validate_config(config)
    model = load_cift_runtime_model(config.runtime_model_path)
    artifact_sha256 = sha256_file(config.activation_artifact_path)
    if artifact_sha256 != model.source_artifact_sha256:
        raise CiftCounterfactualPatchingError(
            "activation_artifact_path sha256 does not match runtime model source_artifact_sha256: "
            f"expected {model.source_artifact_sha256}, got {artifact_sha256}."
        )
    artifact = load_cift_training_artifact_with_unseal_policy(
        path=config.activation_artifact_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT counterfactual patching",
    )
    feature_matrix = _feature_matrix(artifact=artifact, feature_key=model.feature_key)
    pairing_selection = _paired_rows(artifact=artifact, model=model)
    patch_pairs = tuple(
        _patch_pair(model=model, matrix=feature_matrix, paired_rows=paired_rows)
        for paired_rows in pairing_selection.pairs
    )
    report = _report_from_patch_pairs(
        config=config,
        model=model,
        source_artifact_sha256=artifact_sha256,
        patch_pairs=patch_pairs,
        skipped_ambiguous_pair_keys=pairing_selection.skipped_ambiguous_pair_keys,
    )
    _write_json(path=config.output_path, record=cift_counterfactual_patching_report_to_json(report))
    return report


def cift_counterfactual_patching_report_to_json(
    report: CiftCounterfactualPatchingReport,
) -> dict[str, JsonValue]:
    return {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "model_bundle_id": report.model_bundle_id,
        "training_dataset_id": report.training_dataset_id,
        "task_name": report.task_name,
        "feature_key": report.feature_key,
        "source_artifact_sha256": report.source_artifact_sha256,
        "intervention_type": report.intervention_type,
        "claim_scope": report.claim_scope,
        "transformer_hidden_state_patching": report.transformer_hidden_state_patching,
        "paper_faithfulness_limitation": report.paper_faithfulness_limitation,
        "pairing_tag_fields": cast(list[JsonValue], list(report.pairing_tag_fields)),
        "pair_count": report.pair_count,
        "skipped_ambiguous_pair_count": report.skipped_ambiguous_pair_count,
        "skipped_ambiguous_pair_keys": cast(list[JsonValue], list(report.skipped_ambiguous_pair_keys)),
        "minimum_flip_rate": report.minimum_flip_rate,
        "safe_original_allow_rate": report.safe_original_allow_rate,
        "exfil_original_block_rate": report.exfil_original_block_rate,
        "safe_to_exfil_block_rate": report.safe_to_exfil_block_rate,
        "exfil_to_safe_allow_rate": report.exfil_to_safe_allow_rate,
        "passed": report.passed,
        "families": cast(list[JsonValue], list(report.families)),
        "pairs": [cift_patch_pair_to_json(pair) for pair in report.pairs],
        "created_at": report.created_at,
    }


def cift_patch_pair_to_json(pair: CiftPatchPair) -> dict[str, JsonValue]:
    return {
        "pair_key": pair.pair_key,
        "participant": pair.participant,
        "task": pair.task,
        "family": pair.family,
        "variant": pair.variant,
        "credential_type": pair.credential_type,
        "safe_example_id": pair.safe_example_id,
        "exfil_example_id": pair.exfil_example_id,
        "safe_original": _patch_prediction_to_json(pair.safe_original),
        "exfil_original": _patch_prediction_to_json(pair.exfil_original),
        "safe_to_exfil_patch": _patch_prediction_to_json(pair.safe_to_exfil_patch),
        "exfil_to_safe_patch": _patch_prediction_to_json(pair.exfil_to_safe_patch),
    }


def _patch_prediction_to_json(prediction: CiftPatchPrediction) -> dict[str, JsonValue]:
    return {
        "score": prediction.score,
        "predicted_label": prediction.predicted_label,
        "action": prediction.action,
        "operating_band": prediction.operating_band,
    }


def _validate_config(config: CiftCounterfactualPatchingConfig) -> None:
    if config.report_id == "":
        raise CiftCounterfactualPatchingError("report_id must not be empty.")
    if config.created_at == "":
        raise CiftCounterfactualPatchingError("created_at must not be empty.")
    if not math.isfinite(config.minimum_flip_rate):
        raise CiftCounterfactualPatchingError("minimum_flip_rate must be finite.")
    if config.minimum_flip_rate < 0.0 or config.minimum_flip_rate > 1.0:
        raise CiftCounterfactualPatchingError("minimum_flip_rate must be in [0.0, 1.0].")


def _feature_matrix(artifact: CiftTrainingArtifact, feature_key: str) -> np.ndarray:
    matrix = artifact.features.get(feature_key)
    if matrix is None:
        raise CiftCounterfactualPatchingError(f"Activation feature '{feature_key}' is not present in the artifact.")
    return np.asarray(matrix, dtype=np.float32)


def _paired_rows(artifact: CiftTrainingArtifact, model: CiftRuntimeModel) -> _PairingSelection:
    safe_labels = _safe_labels(model)
    exfil_label = model.positive_label
    rows_by_key: dict[str, list[_ArtifactRow]] = defaultdict(list)
    metadata_by_key: dict[str, Mapping[str, str]] = {}
    for row_index, label in enumerate(artifact.labels):
        if label not in safe_labels and label != exfil_label:
            continue
        row = _ArtifactRow(
            row_index=row_index,
            example_id=artifact.example_ids[row_index],
            label=label,
            family=artifact.families[row_index],
            tags=artifact.tags[row_index],
        )
        key_metadata = _pairing_metadata(row)
        key = _pair_key(key_metadata)
        rows_by_key[key].append(row)
        metadata_by_key[key] = key_metadata
    pairs: list[_PairedRows] = []
    skipped_ambiguous_pair_keys: list[str] = []
    for key in sorted(rows_by_key):
        rows = tuple(rows_by_key[key])
        safe_rows = tuple(row for row in rows if row.label in safe_labels)
        exfil_rows = tuple(row for row in rows if row.label == exfil_label)
        if len(safe_rows) == 0 or len(exfil_rows) == 0:
            continue
        if len(safe_rows) > 1 or len(exfil_rows) > 1:
            skipped_ambiguous_pair_keys.append(key)
            continue
        metadata = metadata_by_key[key]
        pairs.append(
            _PairedRows(
                key=key,
                participant=metadata["participant"],
                task=metadata["task"],
                family=metadata["family"],
                variant=_paired_variant(safe=safe_rows[0], exfil=exfil_rows[0]),
                credential_type=metadata["credential_type"],
                safe=safe_rows[0],
                exfil=exfil_rows[0],
            )
        )
    if len(pairs) == 0:
        ambiguous_summary = ""
        if len(skipped_ambiguous_pair_keys) > 0:
            ambiguous_summary = f" Skipped ambiguous pair groups: {len(skipped_ambiguous_pair_keys)}."
        raise CiftCounterfactualPatchingError(f"No exact safe/exfil row pairs were found.{ambiguous_summary}")
    return _PairingSelection(
        pairs=tuple(pairs),
        skipped_ambiguous_pair_keys=tuple(skipped_ambiguous_pair_keys),
    )


def _pairing_metadata(row: _ArtifactRow) -> Mapping[str, str]:
    parsed_tags = _parse_tags(row.tags)
    missing_fields = tuple(field for field in _PAIRING_TAG_FIELDS if parsed_tags.get(field, "") == "")
    if len(missing_fields) > 0:
        raise CiftCounterfactualPatchingError(
            f"Row '{row.example_id}' is missing pairing tag fields: {', '.join(missing_fields)}."
        )
    family = parsed_tags["family"]
    if family != row.family:
        raise CiftCounterfactualPatchingError(
            f"Row '{row.example_id}' family tag '{family}' does not match artifact family '{row.family}'."
        )
    return {field: parsed_tags[field] for field in _PAIRING_TAG_FIELDS}


def _paired_variant(safe: _ArtifactRow, exfil: _ArtifactRow) -> str:
    safe_variant = _variant_from_row(safe)
    exfil_variant = _variant_from_row(exfil)
    if safe_variant == exfil_variant:
        return safe_variant
    return f"safe={safe_variant};exfil={exfil_variant}"


def _variant_from_row(row: _ArtifactRow) -> str:
    parsed_tags = _parse_tags(row.tags)
    variant = parsed_tags.get(_VARIANT_TAG_FIELD, "")
    if variant == "":
        raise CiftCounterfactualPatchingError(f"Row '{row.example_id}' is missing variant tag field.")
    return variant


def _parse_tags(tags: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for tag in tags:
        if ":" not in tag:
            continue
        key, value = tag.split(":", maxsplit=1)
        parsed[key] = value
    return parsed


def _pair_key(metadata: Mapping[str, str]) -> str:
    return "|".join(f"{field}={metadata[field]}" for field in _PAIRING_TAG_FIELDS)


def _safe_labels(model: CiftRuntimeModel) -> tuple[str, ...]:
    negative_label = _negative_label(model)
    if negative_label == "non_exfiltration":
        return ("secret_present_safe",)
    return (negative_label,)


def _negative_label(model: CiftRuntimeModel) -> str:
    labels = tuple(label for label in model.label_names if label != model.positive_label)
    if len(labels) != 1:
        raise CiftCounterfactualPatchingError("Runtime model must have exactly one non-positive label.")
    return labels[0]


def _patch_pair(model: CiftRuntimeModel, matrix: np.ndarray, paired_rows: _PairedRows) -> CiftPatchPair:
    safe_vector = _feature_vector(matrix=matrix, row=paired_rows.safe)
    exfil_vector = _feature_vector(matrix=matrix, row=paired_rows.exfil)
    return CiftPatchPair(
        pair_key=paired_rows.key,
        participant=paired_rows.participant,
        task=paired_rows.task,
        family=paired_rows.family,
        variant=paired_rows.variant,
        credential_type=paired_rows.credential_type,
        safe_example_id=paired_rows.safe.example_id,
        exfil_example_id=paired_rows.exfil.example_id,
        safe_original=_prediction(model=model, feature_vector=safe_vector),
        exfil_original=_prediction(model=model, feature_vector=exfil_vector),
        safe_to_exfil_patch=_prediction(model=model, feature_vector=exfil_vector),
        exfil_to_safe_patch=_prediction(model=model, feature_vector=safe_vector),
    )


def _feature_vector(matrix: np.ndarray, row: _ArtifactRow) -> tuple[float, ...]:
    return tuple(float(value) for value in matrix[row.row_index].tolist())


def _prediction(model: CiftRuntimeModel, feature_vector: tuple[float, ...]) -> CiftPatchPrediction:
    prediction = predict_cift_runtime_model(model=model, feature_vector=feature_vector)
    return _patch_prediction(prediction)


def _patch_prediction(prediction: CiftRuntimePrediction) -> CiftPatchPrediction:
    return CiftPatchPrediction(
        score=prediction.score,
        predicted_label=prediction.predicted_label,
        action=prediction.recommended_action.value,
        operating_band=prediction.operating_band,
    )


def _report_from_patch_pairs(
    config: CiftCounterfactualPatchingConfig,
    model: CiftRuntimeModel,
    source_artifact_sha256: str,
    patch_pairs: tuple[CiftPatchPair, ...],
    skipped_ambiguous_pair_keys: tuple[str, ...],
) -> CiftCounterfactualPatchingReport:
    pair_count = len(patch_pairs)
    safe_original_allow_rate = _rate(
        tuple(pair.safe_original.action == Action.ALLOW.value for pair in patch_pairs),
        pair_count,
    )
    exfil_original_block_rate = _rate(
        tuple(_action_blocks_or_escalates(pair.exfil_original.action) for pair in patch_pairs),
        pair_count,
    )
    safe_to_exfil_block_rate = _rate(
        tuple(_action_blocks_or_escalates(pair.safe_to_exfil_patch.action) for pair in patch_pairs),
        pair_count,
    )
    exfil_to_safe_allow_rate = _rate(
        tuple(pair.exfil_to_safe_patch.action == Action.ALLOW.value for pair in patch_pairs),
        pair_count,
    )
    rates = (
        safe_original_allow_rate,
        exfil_original_block_rate,
        safe_to_exfil_block_rate,
        exfil_to_safe_allow_rate,
    )
    return CiftCounterfactualPatchingReport(
        schema_version=_SCHEMA_VERSION,
        report_id=config.report_id,
        model_bundle_id=model.model_bundle_id,
        training_dataset_id=model.training_dataset_id,
        task_name=model.task_name,
        feature_key=model.feature_key,
        source_artifact_sha256=source_artifact_sha256,
        intervention_type=_INTERVENTION_TYPE,
        claim_scope=_CLAIM_SCOPE,
        transformer_hidden_state_patching=False,
        paper_faithfulness_limitation=_LIMITATION,
        pairing_tag_fields=_PAIRING_TAG_FIELDS,
        pair_count=pair_count,
        skipped_ambiguous_pair_count=len(skipped_ambiguous_pair_keys),
        skipped_ambiguous_pair_keys=skipped_ambiguous_pair_keys,
        minimum_flip_rate=config.minimum_flip_rate,
        safe_original_allow_rate=safe_original_allow_rate,
        exfil_original_block_rate=exfil_original_block_rate,
        safe_to_exfil_block_rate=safe_to_exfil_block_rate,
        exfil_to_safe_allow_rate=exfil_to_safe_allow_rate,
        passed=all(rate >= config.minimum_flip_rate for rate in rates),
        families=tuple(sorted({pair.family for pair in patch_pairs})),
        pairs=patch_pairs,
        created_at=config.created_at,
    )


def _action_blocks_or_escalates(action: str) -> bool:
    return action_severity(Action(action)) >= action_severity(Action.BLOCK)


def _rate(successes: tuple[bool, ...], total: int) -> float:
    if total < 1:
        raise CiftCounterfactualPatchingError("Cannot compute a rate with zero pairs.")
    return float(sum(1 for success in successes if success)) / float(total)


def _write_json(path: Path, record: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
