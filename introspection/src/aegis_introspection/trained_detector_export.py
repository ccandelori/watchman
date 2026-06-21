from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aegis_introspection.artifacts import load_activation_artifact
from aegis_introspection.binary_tasks import (
    BinaryTaskDefinition,
    activation_feature_tensor,
    build_binary_task_dataset,
    default_binary_task_definitions,
)
from aegis_introspection.calibrated_detector_export import load_runtime_turns_by_example_id
from aegis_introspection.cift_model_bundle import CiftModelBundle, load_cift_model_bundle, predict_cift_model_bundle
from aegis_introspection.detector_result_bridge import (
    CiftModelPredictionContext,
    RecommendedAction,
    TrainedCiftDetectorBridgeConfig,
    trained_cift_prediction_to_detector_result,
)
from aegis_introspection.probe import JsonValue, tensor_to_float_matrix


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


def export_trained_cift_detector_results(config: TrainedDetectorExportConfig) -> int:
    turns_by_example_id = load_runtime_turns_by_example_id(config.runtime_turns_path)
    artifact = load_activation_artifact(config.artifact_path)
    bundle = load_cift_model_bundle(config.model_bundle_path)
    definition = _task_definition(bundle.metadata.task_name)
    dataset = build_binary_task_dataset(artifact=artifact, definition=definition)
    feature_tensor = activation_feature_tensor(artifact=artifact, feature_key=bundle.metadata.activation_feature_key)
    selected_indices = tuple(artifact["example_ids"].index(example_id) for example_id in dataset.example_ids)
    matrix = tensor_to_float_matrix(feature_tensor)[list(selected_indices)]
    predictions = predict_cift_model_bundle(bundle=bundle, feature_matrix=matrix)
    bridge_config = _bridge_config(config=config, bundle=bundle)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        for row_index, prediction in enumerate(predictions):
            example_id = dataset.example_ids[row_index]
            turn = turns_by_example_id.get(example_id)
            if turn is None:
                raise TrainedDetectorExportError(f"Missing runtime turn for trained prediction example '{example_id}'.")
            detector_result = trained_cift_prediction_to_detector_result(
                prediction=prediction,
                context=CiftModelPredictionContext(
                    example_id=example_id,
                    family=dataset.families[row_index],
                    source_label=dataset.source_labels[row_index],
                    true_label=dataset.target_labels[row_index],
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


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise TrainedDetectorExportError(f"Unknown binary task '{task_name}'.")
    return matches[0]


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
