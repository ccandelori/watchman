from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aegis.core.contracts import JsonValue
from aegis_introspection.cift_model_bundle import CiftModelBundle, load_cift_model_bundle, predict_cift_model_bundle
from aegis_introspection.cift_model_training import (
    BinaryTaskDefinition,
    build_cift_binary_task_rows,
    cift_binary_task_definition,
    cift_feature_matrix_for_rows,
    load_cift_training_artifact_with_unseal_policy,
)
from aegis_introspection.detector_result_bridge import (
    CiftModelPredictionContext,
    RecommendedAction,
    TrainedCiftDetectorBridgeConfig,
    trained_cift_prediction_to_detector_result,
)
from aegis_introspection.sealed_holdout_policy import assert_unsealed_jsonl_tags


class TrainedDetectorExportError(ValueError):
    """Raised when trained detector results cannot be exported."""


@dataclass(frozen=True)
class TrainedDetectorExportConfig:
    runtime_turns_path: Path
    artifact_path: Path
    model_bundle_path: Path
    output_path: Path
    detector_name: str
    model_bundle_id: str
    capability_required: str
    positive_action: RecommendedAction
    negative_action: RecommendedAction
    confidence: float
    allow_sealed_holdout: bool


def export_trained_cift_detector_results(config: TrainedDetectorExportConfig) -> int:
    assert_unsealed_jsonl_tags(
        path=config.runtime_turns_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="trained CIFT DetectorResult export",
    )
    turns_by_example_id = load_runtime_turns_by_example_id(config.runtime_turns_path)
    artifact = load_cift_training_artifact_with_unseal_policy(
        path=config.artifact_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="trained CIFT DetectorResult export",
    )
    bundle = load_cift_model_bundle(config.model_bundle_path)
    definition = cift_binary_task_definition(bundle.metadata.task_name)
    task_rows = build_cift_binary_task_rows(artifact=artifact, definition=definition)
    matrix = cift_feature_matrix_for_rows(
        artifact=artifact,
        feature_key=bundle.metadata.activation_feature_key,
        selected_indices=task_rows.artifact_indices,
    )
    predictions = predict_cift_model_bundle(bundle=bundle, feature_matrix=matrix)
    bridge_config = _bridge_config(config=config, bundle=bundle)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        for row_index, prediction in enumerate(predictions):
            example_id = task_rows.example_ids[row_index]
            turn = turns_by_example_id.get(example_id)
            if turn is None:
                raise TrainedDetectorExportError(f"Missing runtime turn for trained prediction example '{example_id}'.")
            detector_result = trained_cift_prediction_to_detector_result(
                prediction=prediction,
                context=CiftModelPredictionContext(
                    example_id=example_id,
                    family=task_rows.families[row_index],
                    source_label=task_rows.source_labels[row_index],
                    true_label=task_rows.target_labels[row_index],
                ),
                config=bridge_config,
            )
            row: dict[str, JsonValue] = {
                "trace_id": _required_string(turn.get("trace_id"), "trace_id", example_id),
                "session_id": _required_string(turn.get("session_id"), "session_id", example_id),
                "turn_index": _required_int(turn.get("turn_index"), "turn_index", example_id),
                "example_id": example_id,
                "detector_result": detector_result,
            }
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    return len(predictions)


def load_runtime_turns_by_example_id(path: Path) -> dict[str, Mapping[str, object]]:
    turns_by_example_id: dict[str, Mapping[str, object]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            record = _as_mapping(value=decoded, line_number=line_number)
            metadata = _as_mapping(value=record.get("metadata"), line_number=line_number)
            example_id = metadata.get("example_id")
            if not isinstance(example_id, str) or example_id == "":
                raise TrainedDetectorExportError(f"Line {line_number}: metadata.example_id must be a non-empty string.")
            if example_id in turns_by_example_id:
                raise TrainedDetectorExportError(f"Line {line_number}: duplicate example_id '{example_id}'.")
            turns_by_example_id[example_id] = record

    if len(turns_by_example_id) == 0:
        raise TrainedDetectorExportError(f"No runtime turns found in {path}.")
    return turns_by_example_id


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    try:
        return cift_binary_task_definition(task_name)
    except ValueError as exc:
        raise TrainedDetectorExportError(str(exc)) from exc


def _bridge_config(
    config: TrainedDetectorExportConfig,
    bundle: CiftModelBundle,
) -> TrainedCiftDetectorBridgeConfig:
    return TrainedCiftDetectorBridgeConfig(
        detector_name=config.detector_name,
        feature_key=bundle.metadata.activation_feature_key,
        task_name=bundle.metadata.task_name,
        model_bundle_id=config.model_bundle_id,
        capability_required=config.capability_required,
        positive_action=config.positive_action,
        negative_action=config.negative_action,
        confidence=config.confidence,
    )


def _as_mapping(value: object, line_number: int) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TrainedDetectorExportError(f"Line {line_number}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(value: object, field_name: str, example_id: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TrainedDetectorExportError(
            f"Runtime turn for example '{example_id}' field '{field_name}' must be a non-empty string."
        )
    return value


def _required_int(value: object, field_name: str, example_id: str) -> int:
    if not isinstance(value, int):
        raise TrainedDetectorExportError(
            f"Runtime turn for example '{example_id}' field '{field_name}' must be an integer."
        )
    return value
