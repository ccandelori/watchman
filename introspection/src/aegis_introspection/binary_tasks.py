from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.probe import JsonValue, encode_labels, tensor_to_float_matrix


BinaryMethodName: TypeAlias = Literal["activation_probe", "word_tfidf", "char_tfidf"]
EvaluationStrategy: TypeAlias = Literal["stratified_kfold", "stratified_group_kfold"]
IntVector: TypeAlias = NDArray[np.int64]


class BinaryTaskError(ValueError):
    """Raised when a binary task cannot be built or evaluated."""


@dataclass(frozen=True)
class BinaryTaskConfig:
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    activation_feature_key: str
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]


@dataclass(frozen=True)
class BinaryTaskDefinition:
    name: str
    description: str
    source_labels: tuple[str, ...]
    target_labels: tuple[str, ...]


@dataclass(frozen=True)
class BinaryTaskDataset:
    name: str
    description: str
    example_ids: tuple[str, ...]
    families: tuple[str, ...]
    texts: tuple[str, ...]
    source_labels: tuple[str, ...]
    target_labels: tuple[str, ...]


@dataclass(frozen=True)
class BinaryFoldMetrics:
    fold_index: int
    accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BinaryMethodReport:
    method_name: BinaryMethodName
    feature_name: str
    label_names: tuple[str, ...]
    example_count: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[BinaryFoldMetrics, ...]


@dataclass(frozen=True)
class BinaryTaskReport:
    task_name: str
    description: str
    label_names: tuple[str, ...]
    methods: tuple[BinaryMethodReport, ...]


@dataclass(frozen=True)
class BinaryTasksReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    activation_feature_key: str
    tasks: tuple[BinaryTaskReport, ...]


@dataclass(frozen=True)
class CrossValidationSplit:
    fold_index: int
    train_indices: IntVector
    test_indices: IntVector


def _parse_concat_feature_key(feature_key: str) -> tuple[str, ...] | None:
    prefix = "concat("
    suffix = ")"
    if not feature_key.startswith(prefix):
        return None
    if not feature_key.endswith(suffix):
        raise BinaryTaskError(f"Activation feature expression '{feature_key}' is missing a closing parenthesis.")

    inner_value = feature_key[len(prefix) : -len(suffix)]
    source_feature_keys = tuple(item.strip() for item in inner_value.split(",") if item.strip() != "")
    if len(source_feature_keys) < 2:
        raise BinaryTaskError(
            f"Activation feature expression '{feature_key}' must concatenate at least two source features."
        )
    return source_feature_keys


def _source_feature_tensor(
    artifact: ActivationArtifact,
    expression_key: str,
    source_feature_key: str,
) -> torch.Tensor:
    feature_tensor = artifact["features"].get(source_feature_key)
    if feature_tensor is None:
        raise BinaryTaskError(
            f"Activation feature expression '{expression_key}' references missing source feature "
            f"'{source_feature_key}'."
        )
    return feature_tensor


def _concat_feature_tensor(
    artifact: ActivationArtifact,
    expression_key: str,
    source_feature_keys: tuple[str, ...],
) -> torch.Tensor:
    tensors = tuple(
        _source_feature_tensor(
            artifact=artifact,
            expression_key=expression_key,
            source_feature_key=source_feature_key,
        )
        for source_feature_key in source_feature_keys
    )
    row_count = tensors[0].shape[0]
    for source_feature_key, tensor in zip(source_feature_keys, tensors, strict=True):
        if tensor.shape[0] != row_count:
            raise BinaryTaskError(
                f"Activation feature expression '{expression_key}' references source feature "
                f"'{source_feature_key}' with {tensor.shape[0]} rows, but expected {row_count} rows."
            )
    return torch.cat(tensors, dim=1)


def activation_feature_tensor(
    artifact: ActivationArtifact,
    feature_key: str,
) -> torch.Tensor:
    feature_tensor = artifact["features"].get(feature_key)
    if feature_tensor is not None:
        return feature_tensor

    source_feature_keys = _parse_concat_feature_key(feature_key)
    if source_feature_keys is not None:
        return _concat_feature_tensor(
            artifact=artifact,
            expression_key=feature_key,
            source_feature_keys=source_feature_keys,
        )

    raise BinaryTaskError(f"Activation feature '{feature_key}' is not present in the artifact.")


def default_binary_task_definitions() -> tuple[BinaryTaskDefinition, ...]:
    return (
        BinaryTaskDefinition(
            name="benign_vs_secret_related",
            description="Classify benign prompts against any prompt involving secret-like material.",
            source_labels=("benign", "secret_present_safe", "exfiltration_intent"),
            target_labels=("benign", "secret_related", "secret_related"),
        ),
        BinaryTaskDefinition(
            name="safe_secret_vs_exfiltration",
            description="Classify safe secret handling against exfiltration-oriented secret handling.",
            source_labels=("secret_present_safe", "exfiltration_intent"),
            target_labels=("secret_present_safe", "exfiltration_intent"),
        ),
    )


def build_binary_task_dataset(
    artifact: ActivationArtifact,
    definition: BinaryTaskDefinition,
) -> BinaryTaskDataset:
    if len(definition.source_labels) != len(definition.target_labels):
        raise BinaryTaskError(f"Task '{definition.name}' has mismatched source and target label counts.")

    label_pairs = tuple(zip(definition.source_labels, definition.target_labels, strict=True))
    example_ids: list[str] = []
    families: list[str] = []
    texts: list[str] = []
    source_labels: list[str] = []
    target_labels: list[str] = []

    for example_id, family, text, source_label in zip(
        artifact["example_ids"],
        artifact["families"],
        artifact["texts"],
        artifact["labels"],
        strict=True,
    ):
        matched_targets = tuple(target for source, target in label_pairs if source == source_label)
        if len(matched_targets) == 0:
            continue
        if len(matched_targets) > 1:
            raise BinaryTaskError(f"Task '{definition.name}' maps source label '{source_label}' more than once.")
        example_ids.append(example_id)
        families.append(family)
        texts.append(text)
        source_labels.append(source_label)
        target_labels.append(matched_targets[0])

    if len(set(target_labels)) != 2:
        raise BinaryTaskError(f"Task '{definition.name}' must produce exactly two target labels.")
    if len(target_labels) == 0:
        raise BinaryTaskError(f"Task '{definition.name}' produced no examples.")

    return BinaryTaskDataset(
        name=definition.name,
        description=definition.description,
        example_ids=tuple(example_ids),
        families=tuple(families),
        texts=tuple(texts),
        source_labels=tuple(source_labels),
        target_labels=tuple(target_labels),
    )


def _validate_cross_validation_inputs(
    row_count: int,
    encoded_labels: IntVector,
    config: BinaryTaskConfig,
) -> None:
    if config.fold_count < 2:
        raise BinaryTaskError("fold_count must be at least 2.")
    if config.max_iter < 1:
        raise BinaryTaskError("max_iter must be at least 1.")
    if config.regularization_c <= 0:
        raise BinaryTaskError("regularization_c must be greater than 0.")
    if row_count != encoded_labels.shape[0]:
        raise BinaryTaskError(f"Row count {row_count} does not match label count {encoded_labels.shape[0]}.")

    label_counts = np.bincount(encoded_labels)
    if len(label_counts) != 2:
        raise BinaryTaskError("Binary task must contain exactly two encoded labels.")
    smallest_class_count = int(label_counts.min())
    if smallest_class_count < config.fold_count:
        raise BinaryTaskError(
            f"fold_count={config.fold_count} exceeds the smallest class size {smallest_class_count}."
        )


def _validate_grouped_cross_validation_inputs(
    row_count: int,
    encoded_labels: IntVector,
    groups: tuple[str, ...],
    config: BinaryTaskConfig,
) -> None:
    _validate_cross_validation_inputs(row_count, encoded_labels, config)
    if len(groups) != row_count:
        raise BinaryTaskError(f"Group count {len(groups)} does not match row count {row_count}.")
    for index, group in enumerate(groups):
        if group == "":
            raise BinaryTaskError(f"Group at row {index} must not be empty.")

    for label_index in range(2):
        label_groups = {
            group
            for group, encoded_label in zip(groups, encoded_labels.tolist(), strict=True)
            if int(encoded_label) == label_index
        }
        if len(label_groups) < config.fold_count:
            raise BinaryTaskError(
                f"fold_count={config.fold_count} exceeds the group count {len(label_groups)} "
                f"for encoded label {label_index}."
            )


def stratified_splits(
    row_count: int,
    encoded_labels: IntVector,
    config: BinaryTaskConfig,
) -> tuple[CrossValidationSplit, ...]:
    _validate_cross_validation_inputs(row_count, encoded_labels, config)
    row_indices = np.arange(row_count, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=config.fold_count,
        shuffle=True,
        random_state=config.random_seed,
    )
    return tuple(
        CrossValidationSplit(
            fold_index=fold_index,
            train_indices=np.asarray(train_indices, dtype=np.int64),
            test_indices=np.asarray(test_indices, dtype=np.int64),
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(row_indices, encoded_labels),
            start=1,
        )
    )


def stratified_group_splits(
    encoded_labels: IntVector,
    groups: tuple[str, ...],
    config: BinaryTaskConfig,
) -> tuple[CrossValidationSplit, ...]:
    _validate_grouped_cross_validation_inputs(encoded_labels.shape[0], encoded_labels, groups, config)
    row_indices = np.arange(encoded_labels.shape[0], dtype=np.int64)
    group_array = np.asarray(groups, dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=config.fold_count,
        shuffle=True,
        random_state=config.random_seed,
    )
    return tuple(
        CrossValidationSplit(
            fold_index=fold_index,
            train_indices=np.asarray(train_indices, dtype=np.int64),
            test_indices=np.asarray(test_indices, dtype=np.int64),
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(row_indices, encoded_labels, groups=group_array),
            start=1,
        )
    )


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: tuple[float, ...]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def build_activation_classifier(config: BinaryTaskConfig) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.regularization_c,
            class_weight="balanced",
            max_iter=config.max_iter,
            random_state=config.random_seed,
        ),
    )


def build_text_classifier(method_name: BinaryMethodName, config: BinaryTaskConfig) -> Pipeline:
    if method_name == "word_tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            min_df=1,
            ngram_range=config.word_ngram_range,
        )
    elif method_name == "char_tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            min_df=1,
            ngram_range=config.char_ngram_range,
        )
    else:
        raise BinaryTaskError(f"Unsupported text classifier method '{method_name}'.")

    return make_pipeline(
        vectorizer,
        LogisticRegression(
            C=config.regularization_c,
            class_weight="balanced",
            max_iter=config.max_iter,
            random_state=config.random_seed,
        ),
    )


def _evaluate_predictions(
    method_name: BinaryMethodName,
    feature_name: str,
    label_names: tuple[str, ...],
    true_labels: IntVector,
    fold_predictions: tuple[tuple[int, IntVector, IntVector], ...],
) -> BinaryMethodReport:
    label_indices = np.arange(len(label_names), dtype=np.int64)
    confusion_total = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    folds: list[BinaryFoldMetrics] = []

    for fold_index, y_true, predictions in fold_predictions:
        fold_confusion = confusion_matrix(
            y_true,
            predictions,
            labels=label_indices,
        ).astype(np.int64, copy=False)
        confusion_total += fold_confusion
        folds.append(
            BinaryFoldMetrics(
                fold_index=fold_index,
                accuracy=float(accuracy_score(y_true, predictions)),
                macro_f1=float(
                    f1_score(
                        y_true,
                        predictions,
                        average="macro",
                        labels=label_indices,
                        zero_division=0,
                    )
                ),
                confusion_matrix=_matrix_to_tuple(fold_confusion),
            )
        )

    accuracies = tuple(fold.accuracy for fold in folds)
    macro_f1_scores = tuple(fold.macro_f1 for fold in folds)
    return BinaryMethodReport(
        method_name=method_name,
        feature_name=feature_name,
        label_names=label_names,
        example_count=int(true_labels.shape[0]),
        accuracy_mean=_mean(accuracies),
        accuracy_std=_std(accuracies),
        macro_f1_mean=_mean(macro_f1_scores),
        macro_f1_std=_std(macro_f1_scores),
        confusion_matrix=_matrix_to_tuple(confusion_total),
        folds=tuple(folds),
    )


def evaluate_activation_method(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    config: BinaryTaskConfig,
) -> BinaryMethodReport:
    feature_tensor = activation_feature_tensor(artifact, config.activation_feature_key)
    selected_indices = tuple(artifact["example_ids"].index(example_id) for example_id in dataset.example_ids)
    matrix = tensor_to_float_matrix(feature_tensor)[list(selected_indices)]
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    splits = stratified_splits(matrix.shape[0], encoded_labels, config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []

    for split in splits:
        classifier = build_activation_classifier(config)
        classifier.fit(matrix[split.train_indices], encoded_labels[split.train_indices])
        predictions = classifier.predict(matrix[split.test_indices]).astype(np.int64, copy=False)
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))

    return _evaluate_predictions(
        method_name="activation_probe",
        feature_name=config.activation_feature_key,
        label_names=label_encoding.label_names,
        true_labels=encoded_labels,
        fold_predictions=tuple(fold_predictions),
    )


def evaluate_grouped_activation_method(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    config: BinaryTaskConfig,
) -> BinaryMethodReport:
    feature_tensor = activation_feature_tensor(artifact, config.activation_feature_key)
    selected_indices = tuple(artifact["example_ids"].index(example_id) for example_id in dataset.example_ids)
    matrix = tensor_to_float_matrix(feature_tensor)[list(selected_indices)]
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    splits = stratified_group_splits(encoded_labels, dataset.families, config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []

    for split in splits:
        classifier = build_activation_classifier(config)
        classifier.fit(matrix[split.train_indices], encoded_labels[split.train_indices])
        predictions = classifier.predict(matrix[split.test_indices]).astype(np.int64, copy=False)
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))

    return _evaluate_predictions(
        method_name="activation_probe",
        feature_name=config.activation_feature_key,
        label_names=label_encoding.label_names,
        true_labels=encoded_labels,
        fold_predictions=tuple(fold_predictions),
    )


def evaluate_text_method(
    dataset: BinaryTaskDataset,
    method_name: BinaryMethodName,
    config: BinaryTaskConfig,
) -> BinaryMethodReport:
    if method_name == "activation_probe":
        raise BinaryTaskError("Use evaluate_activation_method for activation probes.")

    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    text_array = np.asarray(dataset.texts, dtype=object)
    splits = stratified_splits(len(dataset.texts), encoded_labels, config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []

    for split in splits:
        classifier = build_text_classifier(method_name, config)
        classifier.fit(text_array[split.train_indices].tolist(), encoded_labels[split.train_indices])
        predictions = classifier.predict(text_array[split.test_indices].tolist()).astype(np.int64, copy=False)
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))

    if method_name == "word_tfidf":
        feature_name = f"word_tfidf_{config.word_ngram_range[0]}_{config.word_ngram_range[1]}"
    else:
        feature_name = f"char_wb_tfidf_{config.char_ngram_range[0]}_{config.char_ngram_range[1]}"
    return _evaluate_predictions(
        method_name=method_name,
        feature_name=feature_name,
        label_names=label_encoding.label_names,
        true_labels=encoded_labels,
        fold_predictions=tuple(fold_predictions),
    )


def evaluate_grouped_text_method(
    dataset: BinaryTaskDataset,
    method_name: BinaryMethodName,
    config: BinaryTaskConfig,
) -> BinaryMethodReport:
    if method_name == "activation_probe":
        raise BinaryTaskError("Use evaluate_grouped_activation_method for activation probes.")

    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    text_array = np.asarray(dataset.texts, dtype=object)
    splits = stratified_group_splits(encoded_labels, dataset.families, config)
    fold_predictions: list[tuple[int, IntVector, IntVector]] = []

    for split in splits:
        classifier = build_text_classifier(method_name, config)
        classifier.fit(text_array[split.train_indices].tolist(), encoded_labels[split.train_indices])
        predictions = classifier.predict(text_array[split.test_indices].tolist()).astype(np.int64, copy=False)
        fold_predictions.append((split.fold_index, encoded_labels[split.test_indices], predictions))

    if method_name == "word_tfidf":
        feature_name = f"word_tfidf_{config.word_ngram_range[0]}_{config.word_ngram_range[1]}"
    else:
        feature_name = f"char_wb_tfidf_{config.char_ngram_range[0]}_{config.char_ngram_range[1]}"
    return _evaluate_predictions(
        method_name=method_name,
        feature_name=feature_name,
        label_names=label_encoding.label_names,
        true_labels=encoded_labels,
        fold_predictions=tuple(fold_predictions),
    )


def evaluate_binary_task(
    artifact: ActivationArtifact,
    definition: BinaryTaskDefinition,
    config: BinaryTaskConfig,
) -> BinaryTaskReport:
    dataset = build_binary_task_dataset(artifact, definition)
    methods = (
        evaluate_activation_method(artifact, dataset, config),
        evaluate_text_method(dataset, "word_tfidf", config),
        evaluate_text_method(dataset, "char_tfidf", config),
    )
    return BinaryTaskReport(
        task_name=dataset.name,
        description=dataset.description,
        label_names=methods[0].label_names,
        methods=methods,
    )


def evaluate_grouped_binary_task(
    artifact: ActivationArtifact,
    definition: BinaryTaskDefinition,
    config: BinaryTaskConfig,
) -> BinaryTaskReport:
    dataset = build_binary_task_dataset(artifact, definition)
    methods = (
        evaluate_grouped_activation_method(artifact, dataset, config),
        evaluate_grouped_text_method(dataset, "word_tfidf", config),
        evaluate_grouped_text_method(dataset, "char_tfidf", config),
    )
    return BinaryTaskReport(
        task_name=dataset.name,
        description=dataset.description,
        label_names=methods[0].label_names,
        methods=methods,
    )


def evaluate_binary_tasks(
    artifact: ActivationArtifact,
    config: BinaryTaskConfig,
) -> BinaryTasksReport:
    task_reports = tuple(
        evaluate_binary_task(
            artifact=artifact,
            definition=definition,
            config=config,
        )
        for definition in default_binary_task_definitions()
    )
    metadata = artifact["metadata"]
    return BinaryTasksReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_kfold",
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        activation_feature_key=config.activation_feature_key,
        tasks=task_reports,
    )


def evaluate_grouped_binary_tasks(
    artifact: ActivationArtifact,
    config: BinaryTaskConfig,
) -> BinaryTasksReport:
    task_reports = tuple(
        evaluate_grouped_binary_task(
            artifact=artifact,
            definition=definition,
            config=config,
        )
        for definition in default_binary_task_definitions()
    )
    metadata = artifact["metadata"]
    return BinaryTasksReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        activation_feature_key=config.activation_feature_key,
        tasks=task_reports,
    )


def _fold_to_json(fold: BinaryFoldMetrics) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "accuracy": fold.accuracy,
        "macro_f1": fold.macro_f1,
        "confusion_matrix": [list(row) for row in fold.confusion_matrix],
    }


def _method_to_json(method: BinaryMethodReport) -> dict[str, JsonValue]:
    return {
        "method_name": method.method_name,
        "feature_name": method.feature_name,
        "label_names": list(method.label_names),
        "example_count": method.example_count,
        "accuracy_mean": method.accuracy_mean,
        "accuracy_std": method.accuracy_std,
        "macro_f1_mean": method.macro_f1_mean,
        "macro_f1_std": method.macro_f1_std,
        "confusion_matrix": [list(row) for row in method.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in method.folds],
    }


def _task_to_json(task: BinaryTaskReport) -> dict[str, JsonValue]:
    return {
        "task_name": task.task_name,
        "description": task.description,
        "label_names": list(task.label_names),
        "methods": [_method_to_json(method) for method in task.methods],
    }


def binary_tasks_report_to_json(report: BinaryTasksReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "activation_feature_key": report.activation_feature_key,
        "tasks": [_task_to_json(task) for task in report.tasks],
    }


def write_binary_tasks_report_json(path: Path, report: BinaryTasksReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(binary_tasks_report_to_json(report), file, indent=2)
        file.write("\n")


def render_binary_tasks_markdown(report: BinaryTasksReport) -> str:
    lines = [
        "# Binary Task Evaluation Summary",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Activation feature: `{report.activation_feature_key}`",
        f"- Fold count: `{report.fold_count}`",
        "",
    ]

    for task in report.tasks:
        lines.extend(
            [
                f"## {task.task_name}",
                "",
                task.description,
                "",
                f"Labels: `{', '.join(task.label_names)}`",
                "",
                "| Method | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for method in sorted(task.methods, key=lambda item: (item.macro_f1_mean, item.accuracy_mean), reverse=True):
            lines.append(
                "| "
                f"`{method.method_name}` | `{method.feature_name}` | "
                f"{method.macro_f1_mean:.4f} | {method.accuracy_mean:.4f} | "
                f"{method.macro_f1_std:.4f} | {method.accuracy_std:.4f} |"
            )
        lines.extend(["", "Confusion matrices:", ""])
        for method in task.methods:
            lines.append(f"### {task.task_name} / {method.method_name}")
            lines.append("")
            lines.append("```text")
            for row in method.confusion_matrix:
                lines.append(str(list(row)))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def write_binary_tasks_markdown(path: Path, report: BinaryTasksReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_binary_tasks_markdown(report), encoding="utf-8")
