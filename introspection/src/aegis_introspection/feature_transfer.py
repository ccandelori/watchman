from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryTaskConfig,
    BinaryTaskDefinition,
    activation_feature_tensor,
    build_activation_classifier,
    build_binary_task_dataset,
    default_binary_task_definitions,
)
from aegis_introspection.probe import FloatMatrix, IntVector, JsonValue, encode_labels, tensor_to_float_matrix


class FeatureTransferError(ValueError):
    """Raised when a cross-profile feature transfer evaluation cannot be run."""


@dataclass(frozen=True)
class FeatureTransferDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class FeatureTransferConfig:
    task_name: str
    activation_feature_key: str
    positive_label: str
    decision_threshold: float
    random_seed: int
    max_iter: int
    regularization_c: float


@dataclass(frozen=True)
class ProfileTaskMatrix:
    dataset_id: str
    example_ids: tuple[str, ...]
    target_labels: tuple[str, ...]
    matrix: FloatMatrix


@dataclass(frozen=True)
class FeatureTransferMetric:
    dataset_id: str
    example_count: int
    accuracy: float
    macro_f1: float
    positive_precision: float
    positive_recall: float
    positive_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FeatureTransferReport:
    evaluation_strategy: str
    task_name: str
    task_description: str
    activation_feature_key: str
    positive_label: str
    decision_threshold: float
    train_dataset_ids: tuple[str, ...]
    test_dataset_ids: tuple[str, ...]
    train_example_count: int
    feature_count: int
    label_names: tuple[str, ...]
    random_seed: int
    regularization_c: float
    max_iter: int
    train_in_sample: FeatureTransferMetric
    tests: tuple[FeatureTransferMetric, ...]


def evaluate_feature_transfer(
    train_datasets: tuple[FeatureTransferDataset, ...],
    test_datasets: tuple[FeatureTransferDataset, ...],
    config: FeatureTransferConfig,
) -> FeatureTransferReport:
    _validate_config(config)
    if len(train_datasets) == 0:
        raise FeatureTransferError("At least one training dataset is required.")
    if len(test_datasets) == 0:
        raise FeatureTransferError("At least one test dataset is required.")

    definition = _task_definition(config.task_name)
    train_profiles = tuple(
        _profile_task_matrix(dataset, definition, config.activation_feature_key) for dataset in train_datasets
    )
    test_profiles = tuple(
        _profile_task_matrix(dataset, definition, config.activation_feature_key) for dataset in test_datasets
    )
    _validate_feature_counts(train_profiles + test_profiles)

    train_matrix = _stack_matrices(train_profiles)
    train_labels = _stack_labels(train_profiles)
    label_encoding = encode_labels(train_labels)
    if config.positive_label not in label_encoding.label_to_index:
        raise FeatureTransferError(f"positive_label '{config.positive_label}' is not present in training labels.")

    classifier = build_activation_classifier(_binary_task_config(config))
    classifier.fit(train_matrix, label_encoding.encoded_labels)

    positive_label_index = label_encoding.label_to_index[config.positive_label]
    train_predictions = _threshold_predictions(
        classifier=classifier,
        matrix=train_matrix,
        positive_label_index=positive_label_index,
        label_count=len(label_encoding.label_names),
        threshold=config.decision_threshold,
    )
    train_metric = _metric(
        dataset_id="__train__",
        true_labels=label_encoding.encoded_labels,
        predictions=train_predictions,
        label_names=label_encoding.label_names,
        positive_label_index=positive_label_index,
    )
    test_metrics = tuple(
        _evaluate_profile(
            classifier=classifier,
            profile=profile,
            label_names=label_encoding.label_names,
            label_to_index=label_encoding.label_to_index,
            positive_label_index=positive_label_index,
            threshold=config.decision_threshold,
        )
        for profile in test_profiles
    )

    return FeatureTransferReport(
        evaluation_strategy="train_profiles_to_test_profiles",
        task_name=definition.name,
        task_description=definition.description,
        activation_feature_key=config.activation_feature_key,
        positive_label=config.positive_label,
        decision_threshold=config.decision_threshold,
        train_dataset_ids=tuple(dataset.dataset_id for dataset in train_datasets),
        test_dataset_ids=tuple(dataset.dataset_id for dataset in test_datasets),
        train_example_count=int(train_matrix.shape[0]),
        feature_count=int(train_matrix.shape[1]),
        label_names=label_encoding.label_names,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        train_in_sample=train_metric,
        tests=test_metrics,
    )


def _validate_config(config: FeatureTransferConfig) -> None:
    if config.task_name == "":
        raise FeatureTransferError("task_name must not be empty.")
    if config.activation_feature_key == "":
        raise FeatureTransferError("activation_feature_key must not be empty.")
    if config.positive_label == "":
        raise FeatureTransferError("positive_label must not be empty.")
    if not 0.0 <= config.decision_threshold <= 1.0:
        raise FeatureTransferError("decision_threshold must be between 0.0 and 1.0.")
    if config.max_iter < 1:
        raise FeatureTransferError("max_iter must be at least 1.")
    if config.regularization_c <= 0.0:
        raise FeatureTransferError("regularization_c must be greater than 0.0.")


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise FeatureTransferError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _binary_task_config(config: FeatureTransferConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.activation_feature_key,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


def _profile_task_matrix(
    dataset: FeatureTransferDataset,
    definition: BinaryTaskDefinition,
    activation_feature_key: str,
) -> ProfileTaskMatrix:
    task_dataset = build_binary_task_dataset(artifact=dataset.artifact, definition=definition)
    feature_tensor = activation_feature_tensor(dataset.artifact, activation_feature_key)
    artifact_index_by_example_id = {
        example_id: index for index, example_id in enumerate(dataset.artifact["example_ids"])
    }
    selected_indices = tuple(artifact_index_by_example_id[example_id] for example_id in task_dataset.example_ids)
    matrix = tensor_to_float_matrix(feature_tensor)[list(selected_indices)]
    return ProfileTaskMatrix(
        dataset_id=dataset.dataset_id,
        example_ids=task_dataset.example_ids,
        target_labels=task_dataset.target_labels,
        matrix=matrix,
    )


def _validate_feature_counts(profiles: tuple[ProfileTaskMatrix, ...]) -> None:
    if len(profiles) == 0:
        raise FeatureTransferError("At least one profile is required.")
    expected_feature_count = int(profiles[0].matrix.shape[1])
    for profile in profiles:
        feature_count = int(profile.matrix.shape[1])
        if feature_count != expected_feature_count:
            raise FeatureTransferError(
                f"Profile '{profile.dataset_id}' has {feature_count} features; expected {expected_feature_count}."
            )


def _stack_matrices(profiles: tuple[ProfileTaskMatrix, ...]) -> FloatMatrix:
    matrices = tuple(profile.matrix for profile in profiles)
    return np.vstack(matrices).astype(np.float32, copy=False)


def _stack_labels(profiles: tuple[ProfileTaskMatrix, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for profile in profiles:
        labels.extend(profile.target_labels)
    return tuple(labels)


def _encoded_labels(labels: tuple[str, ...], label_to_index: dict[str, int]) -> IntVector:
    encoded: list[int] = []
    for label in labels:
        label_index = label_to_index.get(label)
        if label_index is None:
            raise FeatureTransferError(f"Test label '{label}' was not present in the training labels.")
        encoded.append(label_index)
    return np.asarray(encoded, dtype=np.int64)


def _threshold_predictions(
    classifier: Pipeline,
    matrix: FloatMatrix,
    positive_label_index: int,
    label_count: int,
    threshold: float,
) -> IntVector:
    if label_count != 2:
        raise FeatureTransferError(f"Expected binary labels, received {label_count} labels.")

    probabilities = classifier.predict_proba(matrix)
    class_indices = _classifier_class_indices(classifier)
    if positive_label_index not in class_indices:
        raise FeatureTransferError(f"Positive label index {positive_label_index} is missing from classifier classes.")
    positive_column_index = class_indices.index(positive_label_index)
    negative_label_index = _negative_label_index(label_count=label_count, positive_label_index=positive_label_index)
    positive_scores = probabilities[:, positive_column_index]
    return np.asarray(
        [positive_label_index if score >= threshold else negative_label_index for score in positive_scores],
        dtype=np.int64,
    )


def _classifier_class_indices(classifier: Pipeline) -> tuple[int, ...]:
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise FeatureTransferError("Classifier does not expose fitted classes_.")
    class_array = np.asarray(classes, dtype=np.int64)
    return tuple(int(value) for value in class_array.tolist())


def _negative_label_index(label_count: int, positive_label_index: int) -> int:
    label_indices = tuple(index for index in range(label_count) if index != positive_label_index)
    if len(label_indices) != 1:
        raise FeatureTransferError("Binary transfer evaluation requires exactly one negative label.")
    return label_indices[0]


def _evaluate_profile(
    classifier: Pipeline,
    profile: ProfileTaskMatrix,
    label_names: tuple[str, ...],
    label_to_index: dict[str, int],
    positive_label_index: int,
    threshold: float,
) -> FeatureTransferMetric:
    true_labels = _encoded_labels(profile.target_labels, label_to_index)
    predictions = _threshold_predictions(
        classifier=classifier,
        matrix=profile.matrix,
        positive_label_index=positive_label_index,
        label_count=len(label_names),
        threshold=threshold,
    )
    return _metric(
        dataset_id=profile.dataset_id,
        true_labels=true_labels,
        predictions=predictions,
        label_names=label_names,
        positive_label_index=positive_label_index,
    )


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _metric(
    dataset_id: str,
    true_labels: IntVector,
    predictions: IntVector,
    label_names: tuple[str, ...],
    positive_label_index: int,
) -> FeatureTransferMetric:
    label_indices = np.arange(len(label_names), dtype=np.int64)
    confusion = confusion_matrix(true_labels, predictions, labels=label_indices).astype(np.int64, copy=False)
    return FeatureTransferMetric(
        dataset_id=dataset_id,
        example_count=int(true_labels.shape[0]),
        accuracy=float(accuracy_score(true_labels, predictions)),
        macro_f1=float(
            f1_score(
                true_labels,
                predictions,
                average="macro",
                labels=label_indices,
                zero_division=0,
            )
        ),
        positive_precision=float(
            precision_score(
                true_labels,
                predictions,
                labels=label_indices,
                pos_label=positive_label_index,
                zero_division=0,
            )
        ),
        positive_recall=float(
            recall_score(
                true_labels,
                predictions,
                labels=label_indices,
                pos_label=positive_label_index,
                zero_division=0,
            )
        ),
        positive_f1=float(
            f1_score(
                true_labels,
                predictions,
                labels=label_indices,
                pos_label=positive_label_index,
                zero_division=0,
            )
        ),
        confusion_matrix=_matrix_to_tuple(confusion),
    )


def _metric_to_json(metric: FeatureTransferMetric) -> dict[str, JsonValue]:
    return {
        "dataset_id": metric.dataset_id,
        "example_count": metric.example_count,
        "accuracy": metric.accuracy,
        "macro_f1": metric.macro_f1,
        "positive_precision": metric.positive_precision,
        "positive_recall": metric.positive_recall,
        "positive_f1": metric.positive_f1,
        "confusion_matrix": [list(row) for row in metric.confusion_matrix],
    }


def feature_transfer_report_to_json(report: FeatureTransferReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "activation_feature_key": report.activation_feature_key,
        "positive_label": report.positive_label,
        "decision_threshold": report.decision_threshold,
        "train_dataset_ids": list(report.train_dataset_ids),
        "test_dataset_ids": list(report.test_dataset_ids),
        "train_example_count": report.train_example_count,
        "feature_count": report.feature_count,
        "label_names": list(report.label_names),
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "train_in_sample": _metric_to_json(report.train_in_sample),
        "tests": [_metric_to_json(metric) for metric in report.tests],
    }


def write_feature_transfer_json(path: Path, report: FeatureTransferReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(feature_transfer_report_to_json(report), file, indent=2)
        file.write("\n")


def render_feature_transfer_markdown(report: FeatureTransferReport) -> str:
    lines = [
        "# Feature Transfer Evaluation",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Feature: `{report.activation_feature_key}`",
        f"- Positive label: `{report.positive_label}`",
        f"- Decision threshold: `{report.decision_threshold:.4f}`",
        f"- Train datasets: `{', '.join(report.train_dataset_ids)}`",
        f"- Test datasets: `{', '.join(report.test_dataset_ids)}`",
        f"- Train examples: `{report.train_example_count}`",
        f"- Feature count: `{report.feature_count}`",
        "",
        "## Metrics",
        "",
        "| Dataset | Examples | Accuracy | Macro F1 | Positive Precision | Positive Recall | Positive F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _metric_table_row(report.train_in_sample),
    ]
    for metric in report.tests:
        lines.append(_metric_table_row(metric))

    lines.extend(
        [
            "",
            "## Confusion Matrices",
            "",
            f"Label order: `{', '.join(report.label_names)}`",
            "",
        ]
    )
    for metric in (report.train_in_sample, *report.tests):
        lines.append(f"### {metric.dataset_id}")
        lines.append("")
        lines.append("```text")
        for row in metric.confusion_matrix:
            lines.append(str(list(row)))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _metric_table_row(metric: FeatureTransferMetric) -> str:
    return (
        f"| `{metric.dataset_id}` | "
        f"{metric.example_count} | "
        f"{metric.accuracy:.4f} | "
        f"{metric.macro_f1:.4f} | "
        f"{metric.positive_precision:.4f} | "
        f"{metric.positive_recall:.4f} | "
        f"{metric.positive_f1:.4f} |"
    )


def write_feature_transfer_markdown(path: Path, report: FeatureTransferReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_feature_transfer_markdown(report), encoding="utf-8")
