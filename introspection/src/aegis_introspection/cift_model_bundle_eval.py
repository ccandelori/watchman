from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aegis_introspection.cift_model_bundle import CiftModelBundle, load_cift_model_bundle, predict_cift_model_bundle
from aegis_introspection.cift_model_training import (
    CiftModelTrainingError,
    CiftTrainingArtifact,
    build_cift_binary_task_rows,
    cift_binary_task_definition,
    cift_feature_matrix_for_rows,
    load_cift_training_artifact_with_unseal_policy,
)
from aegis_introspection.lineage import sha256_file
from aegis_introspection.probe import JsonValue
from aegis_introspection.sealed_holdout_policy import tag_rows_are_sealed_holdout


class CiftModelBundleEvalError(ValueError):
    """Raised when a CIFT model bundle evaluation cannot be materialized."""


@dataclass(frozen=True)
class CiftModelBundleEvalConfig:
    activation_artifact_path: Path
    model_bundle_path: Path
    output_path: Path
    report_id: str
    evaluation_split_id: str
    metric_name: str
    created_at: str
    task_name: str
    allow_sealed_holdout: bool


def materialize_cift_model_bundle_eval(config: CiftModelBundleEvalConfig) -> dict[str, JsonValue]:
    _validate_config(config)
    artifact = load_cift_training_artifact_with_unseal_policy(
        path=config.activation_artifact_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT model bundle evaluation",
    )
    bundle = load_cift_model_bundle(config.model_bundle_path)
    _validate_binding(artifact=artifact, bundle=bundle, task_name=config.task_name)
    rows = build_cift_binary_task_rows(
        artifact=artifact,
        definition=cift_binary_task_definition(config.task_name),
    )
    matrix = cift_feature_matrix_for_rows(
        artifact=artifact,
        feature_key=bundle.metadata.activation_feature_key,
        selected_indices=rows.artifact_indices,
    )
    if not np.isfinite(matrix).all():
        raise CiftModelBundleEvalError(
            f"Activation feature '{bundle.metadata.activation_feature_key}' contains non-finite values."
        )
    predictions = predict_cift_model_bundle(bundle=bundle, feature_matrix=matrix)
    predicted_labels = tuple(prediction.predicted_label for prediction in predictions)
    metrics = _classification_metrics(
        true_labels=rows.target_labels,
        predicted_labels=predicted_labels,
        label_names=bundle.metadata.label_names,
        positive_label=bundle.metadata.positive_label,
        metric_name=config.metric_name,
    )
    record: dict[str, JsonValue] = {
        "schema_version": "aegis_introspection.cift_model_bundle_eval/v1",
        "report_id": config.report_id,
        "created_at": config.created_at,
        "evaluation_split_id": config.evaluation_split_id,
        "metric_name": config.metric_name,
        "metric_value": metrics["metric_value"],
        "model_bundle_path": str(config.model_bundle_path),
        "model_bundle_sha256": sha256_file(config.model_bundle_path),
        "activation_artifact_path": str(config.activation_artifact_path),
        "activation_artifact_sha256": sha256_file(config.activation_artifact_path),
        "sealed_holdout": tag_rows_are_sealed_holdout(artifact.tags),
        "source_model_id": bundle.metadata.source_model_id,
        "source_revision": bundle.metadata.source_revision,
        "source_selected_device": bundle.metadata.source_selected_device,
        "source_hidden_size": bundle.metadata.source_hidden_size,
        "source_layer_count": bundle.metadata.source_layer_count,
        "tokenizer_fingerprint_sha256": bundle.metadata.tokenizer_fingerprint_sha256,
        "special_tokens_map_sha256": bundle.metadata.special_tokens_map_sha256,
        "chat_template_sha256": bundle.metadata.chat_template_sha256,
        "training_dataset_id": bundle.metadata.training_dataset_id,
        "task_name": bundle.metadata.task_name,
        "activation_feature_key": bundle.metadata.activation_feature_key,
        "feature_count": bundle.metadata.feature_count,
        "label_names": list(bundle.metadata.label_names),
        "positive_label": bundle.metadata.positive_label,
        "decision_threshold": bundle.metadata.decision_threshold,
        "score_semantics": bundle.metadata.score_semantics,
        "candidate_status": bundle.metadata.candidate_status,
    }
    record.update(metrics)
    _write_json(path=config.output_path, record=record)
    return record


def _validate_config(config: CiftModelBundleEvalConfig) -> None:
    if config.report_id == "":
        raise CiftModelBundleEvalError("report_id must not be empty.")
    if config.evaluation_split_id == "":
        raise CiftModelBundleEvalError("evaluation_split_id must not be empty.")
    if config.metric_name == "":
        raise CiftModelBundleEvalError("metric_name must not be empty.")
    if config.created_at == "":
        raise CiftModelBundleEvalError("created_at must not be empty.")
    if config.task_name == "":
        raise CiftModelBundleEvalError("task_name must not be empty.")


def _validate_binding(artifact: CiftTrainingArtifact, bundle: CiftModelBundle, task_name: str) -> None:
    metadata = artifact.metadata
    expected_pairs = (
        ("source_model_id", bundle.metadata.source_model_id, metadata.model_id),
        ("source_revision", bundle.metadata.source_revision, metadata.revision),
        ("source_selected_device", bundle.metadata.source_selected_device, metadata.selected_device),
        ("source_hidden_size", bundle.metadata.source_hidden_size, metadata.hidden_size),
        ("source_layer_count", bundle.metadata.source_layer_count, metadata.layer_count),
        (
            "tokenizer_fingerprint_sha256",
            bundle.metadata.tokenizer_fingerprint_sha256,
            metadata.tokenizer_fingerprint_sha256,
        ),
        ("special_tokens_map_sha256", bundle.metadata.special_tokens_map_sha256, metadata.special_tokens_map_sha256),
        ("chat_template_sha256", bundle.metadata.chat_template_sha256, metadata.chat_template_sha256),
    )
    mismatches = tuple(
        f"{field_name}: bundle={bundle_value!r} artifact={artifact_value!r}"
        for field_name, bundle_value, artifact_value in expected_pairs
        if bundle_value != artifact_value
    )
    if len(mismatches) > 0:
        raise CiftModelBundleEvalError("Bundle/artifact binding mismatch: " + "; ".join(mismatches))
    if bundle.metadata.task_name != task_name:
        raise CiftModelBundleEvalError(
            f"Bundle task '{bundle.metadata.task_name}' does not match requested task '{task_name}'."
        )
    if bundle.metadata.activation_feature_key not in artifact.features:
        raise CiftModelBundleEvalError(
            f"Bundle activation feature '{bundle.metadata.activation_feature_key}' is absent from artifact."
        )
    feature_matrix = artifact.features[bundle.metadata.activation_feature_key]
    if int(feature_matrix.shape[1]) != bundle.metadata.feature_count:
        raise CiftModelBundleEvalError(
            f"Bundle expects {bundle.metadata.feature_count} features, but artifact feature "
            f"'{bundle.metadata.activation_feature_key}' has width {int(feature_matrix.shape[1])}."
        )


def _classification_metrics(
    true_labels: tuple[str, ...],
    predicted_labels: tuple[str, ...],
    label_names: tuple[str, ...],
    positive_label: str,
    metric_name: str,
) -> dict[str, JsonValue]:
    if len(true_labels) != len(predicted_labels):
        raise CiftModelBundleEvalError("true_labels and predicted_labels must have the same length.")
    if len(label_names) != 2:
        raise CiftModelBundleEvalError("label_names must contain exactly two labels.")
    if positive_label not in label_names:
        raise CiftModelBundleEvalError("positive_label must be present in label_names.")
    label_set = set(label_names)
    unexpected_true_labels = tuple(sorted(set(true_labels) - label_set))
    unexpected_predicted_labels = tuple(sorted(set(predicted_labels) - label_set))
    if len(unexpected_true_labels) > 0:
        raise CiftModelBundleEvalError(f"Unexpected true labels: {unexpected_true_labels}.")
    if len(unexpected_predicted_labels) > 0:
        raise CiftModelBundleEvalError(f"Unexpected predicted labels: {unexpected_predicted_labels}.")

    negative_label = next(label for label in label_names if label != positive_label)
    true_positive = _count_matching(true_labels, predicted_labels, positive_label, positive_label)
    true_negative = _count_matching(true_labels, predicted_labels, negative_label, negative_label)
    false_positive = _count_matching(true_labels, predicted_labels, negative_label, positive_label)
    false_negative = _count_matching(true_labels, predicted_labels, positive_label, negative_label)
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    example_count = len(true_labels)
    accuracy = _safe_ratio(true_positive + true_negative, example_count)
    macro_f1 = _macro_f1(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        label_names=label_names,
    )
    metric_value = _metric_value(metric_name=metric_name, macro_f1=macro_f1)
    confusion_matrix = [
        [
            _count_matching(true_labels, predicted_labels, true_label, predicted_label)
            for predicted_label in label_names
        ]
        for true_label in label_names
    ]
    return {
        "example_count": example_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "true_positive_count": true_positive,
        "true_negative_count": true_negative,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "false_positive_rate": _safe_ratio(false_positive, negative_count),
        "false_negative_rate": _safe_ratio(false_negative, positive_count),
        "true_positive_rate": _safe_ratio(true_positive, positive_count),
        "true_negative_rate": _safe_ratio(true_negative, negative_count),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "metric_value": metric_value,
        "confusion_matrix": confusion_matrix,
    }


def _metric_value(metric_name: str, macro_f1: float) -> float:
    if metric_name != "macro_f1":
        raise CiftModelBundleEvalError(f"Unsupported metric_name '{metric_name}'.")
    return macro_f1


def _macro_f1(true_labels: tuple[str, ...], predicted_labels: tuple[str, ...], label_names: tuple[str, ...]) -> float:
    return float(np.mean(np.asarray(tuple(_label_f1(true_labels, predicted_labels, label) for label in label_names))))


def _label_f1(true_labels: tuple[str, ...], predicted_labels: tuple[str, ...], label: str) -> float:
    true_positive = _count_matching(true_labels, predicted_labels, label, label)
    false_positive = sum(
        1
        for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True)
        if true_label != label and predicted_label == label
    )
    false_negative = sum(
        1
        for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True)
        if true_label == label and predicted_label != label
    )
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _count_matching(
    true_labels: tuple[str, ...],
    predicted_labels: tuple[str, ...],
    true_label: str,
    predicted_label: str,
) -> int:
    return sum(
        1
        for actual, predicted in zip(true_labels, predicted_labels, strict=True)
        if actual == true_label and predicted == predicted_label
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _write_json(path: Path, record: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
