from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryTaskConfig,
    BinaryTaskDataset,
    BinaryTaskDefinition,
    BinaryTaskError,
    activation_feature_tensor,
    build_activation_classifier,
    build_binary_task_dataset,
    default_binary_task_definitions,
)
from aegis_introspection.probe import FloatMatrix, IntVector, JsonValue, encode_labels, tensor_to_float_matrix

ErrorKind: TypeAlias = Literal["false_positive", "false_negative"]


class FeatureTransferErrorAnalysisError(ValueError):
    """Raised when transfer error analysis cannot be computed."""


@dataclass(frozen=True)
class TransferErrorDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class StructuredPromptRecord:
    example_id: str
    rendered_prompt: str
    label: str
    family: str
    secret_char_span: tuple[int, int] | None
    secret_token_span: tuple[int, int] | None
    query_token_span: tuple[int, int] | None
    payload_token_span: tuple[int, int] | None
    readout_token_indices: tuple[int, ...]
    tool_call_name: str | None
    tool_argument_path: str | None


@dataclass(frozen=True)
class TransferErrorAnalysisConfig:
    task_name: str
    activation_feature_key: str
    positive_label: str
    decision_threshold: float
    random_seed: int
    max_iter: int
    regularization_c: float
    max_error_examples: int


@dataclass(frozen=True)
class TransferPrediction:
    example_id: str
    family: str
    true_label: str
    predicted_label: str
    positive_score: float
    is_correct: bool
    error_kind: ErrorKind | None
    readout_token_count: int | None
    secret_token_count: int | None
    query_token_count: int | None
    payload_token_count: int | None
    tool_call_name: str | None
    tool_argument_path: str | None
    redacted_excerpt: str


@dataclass(frozen=True)
class TransferScoreSummary:
    group_name: str
    example_count: int
    minimum: float
    p25: float
    median: float
    p75: float
    maximum: float
    mean: float


@dataclass(frozen=True)
class TransferFamilySummary:
    family: str
    example_count: int
    correct_count: int
    false_positive_count: int
    false_negative_count: int
    accuracy: float
    safe_mean_positive_score: float | None
    exfil_mean_positive_score: float | None


@dataclass(frozen=True)
class TransferSpanSummary:
    group_name: str
    example_count: int
    mean_readout_token_count: float | None
    mean_secret_token_count: float | None
    mean_query_token_count: float | None
    mean_payload_token_count: float | None


@dataclass(frozen=True)
class TransferErrorAnalysisReport:
    evaluation_strategy: str
    task_name: str
    task_description: str
    activation_feature_key: str
    positive_label: str
    decision_threshold: float
    train_dataset_ids: tuple[str, ...]
    test_dataset_id: str
    label_names: tuple[str, ...]
    train_example_count: int
    test_example_count: int
    feature_count: int
    accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    correct_count: int
    error_count: int
    false_positive_count: int
    false_negative_count: int
    score_summaries: tuple[TransferScoreSummary, ...]
    family_summaries: tuple[TransferFamilySummary, ...]
    span_summaries: tuple[TransferSpanSummary, ...]
    error_examples: tuple[TransferPrediction, ...]


def load_structured_prompt_records(path: Path) -> tuple[StructuredPromptRecord, ...]:
    records: list[StructuredPromptRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if stripped_line == "":
                continue
            decoded = json.loads(stripped_line)
            if not isinstance(decoded, dict):
                raise FeatureTransferErrorAnalysisError(f"Structured prompt line {line_number} is not a JSON object.")
            records.append(_structured_prompt_record(cast(Mapping[str, object], decoded), line_number))
    return tuple(records)


def analyze_feature_transfer_errors(
    train_datasets: tuple[TransferErrorDataset, ...],
    test_dataset: TransferErrorDataset,
    structured_prompt_records: tuple[StructuredPromptRecord, ...],
    config: TransferErrorAnalysisConfig,
) -> TransferErrorAnalysisReport:
    _validate_config(config)
    if len(train_datasets) == 0:
        raise FeatureTransferErrorAnalysisError("At least one training dataset is required.")

    definition = _task_definition(config.task_name)
    train_task_matrices = tuple(
        _task_matrix(dataset.artifact, definition, config.activation_feature_key) for dataset in train_datasets
    )
    test_task_matrix = _task_matrix(test_dataset.artifact, definition, config.activation_feature_key)
    _validate_feature_counts((*train_task_matrices, test_task_matrix))

    train_matrix = _stack_matrices(tuple(task.matrix for task in train_task_matrices))
    train_labels = _stack_labels(tuple(task.dataset.target_labels for task in train_task_matrices))
    label_encoding = encode_labels(train_labels)
    if config.positive_label not in label_encoding.label_to_index:
        raise FeatureTransferErrorAnalysisError(
            f"positive_label '{config.positive_label}' is not present in training labels."
        )

    classifier = build_activation_classifier(_binary_task_config(config))
    classifier.fit(train_matrix, label_encoding.encoded_labels)
    positive_label_index = label_encoding.label_to_index[config.positive_label]
    test_encoded_labels = _encoded_labels(test_task_matrix.dataset.target_labels, label_encoding.label_to_index)
    positive_scores = _positive_scores(
        classifier=classifier,
        matrix=test_task_matrix.matrix,
        positive_label_index=positive_label_index,
    )
    predicted_labels = _threshold_predictions(
        positive_scores=positive_scores,
        positive_label_index=positive_label_index,
        label_count=len(label_encoding.label_names),
        threshold=config.decision_threshold,
    )
    prompt_by_example_id = _records_by_example_id(structured_prompt_records)
    predictions = _predictions(
        dataset=test_task_matrix.dataset,
        label_names=label_encoding.label_names,
        true_labels=test_encoded_labels,
        predicted_labels=predicted_labels,
        positive_scores=positive_scores,
        positive_label=config.positive_label,
        prompt_by_example_id=prompt_by_example_id,
    )
    error_examples = _top_error_examples(predictions, config.max_error_examples, config.decision_threshold)
    label_indices = np.arange(len(label_encoding.label_names), dtype=np.int64)
    confusion = confusion_matrix(test_encoded_labels, predicted_labels, labels=label_indices).astype(
        np.int64,
        copy=False,
    )

    return TransferErrorAnalysisReport(
        evaluation_strategy="train_profiles_to_test_profile_error_analysis",
        task_name=definition.name,
        task_description=definition.description,
        activation_feature_key=config.activation_feature_key,
        positive_label=config.positive_label,
        decision_threshold=config.decision_threshold,
        train_dataset_ids=tuple(dataset.dataset_id for dataset in train_datasets),
        test_dataset_id=test_dataset.dataset_id,
        label_names=label_encoding.label_names,
        train_example_count=int(train_matrix.shape[0]),
        test_example_count=int(test_task_matrix.matrix.shape[0]),
        feature_count=int(train_matrix.shape[1]),
        accuracy=float(accuracy_score(test_encoded_labels, predicted_labels)),
        macro_f1=float(
            f1_score(test_encoded_labels, predicted_labels, average="macro", labels=label_indices, zero_division=0)
        ),
        confusion_matrix=_matrix_to_tuple(confusion),
        correct_count=sum(1 for prediction in predictions if prediction.is_correct),
        error_count=sum(1 for prediction in predictions if not prediction.is_correct),
        false_positive_count=sum(1 for prediction in predictions if prediction.error_kind == "false_positive"),
        false_negative_count=sum(1 for prediction in predictions if prediction.error_kind == "false_negative"),
        score_summaries=_score_summaries(predictions),
        family_summaries=_family_summaries(predictions, config.positive_label),
        span_summaries=_span_summaries(predictions),
        error_examples=error_examples,
    )


@dataclass(frozen=True)
class _TaskMatrix:
    dataset: BinaryTaskDataset
    matrix: FloatMatrix


def _validate_config(config: TransferErrorAnalysisConfig) -> None:
    if config.task_name == "":
        raise FeatureTransferErrorAnalysisError("task_name must not be empty.")
    if config.activation_feature_key == "":
        raise FeatureTransferErrorAnalysisError("activation_feature_key must not be empty.")
    if config.positive_label == "":
        raise FeatureTransferErrorAnalysisError("positive_label must not be empty.")
    if not 0.0 <= config.decision_threshold <= 1.0:
        raise FeatureTransferErrorAnalysisError("decision_threshold must be between 0.0 and 1.0.")
    if config.max_iter < 1:
        raise FeatureTransferErrorAnalysisError("max_iter must be at least 1.")
    if config.regularization_c <= 0.0:
        raise FeatureTransferErrorAnalysisError("regularization_c must be greater than 0.0.")
    if config.max_error_examples < 1:
        raise FeatureTransferErrorAnalysisError("max_error_examples must be at least 1.")


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise FeatureTransferErrorAnalysisError(
            f"Expected exactly one binary task named '{task_name}', found {len(matches)}."
        )
    return matches[0]


def _binary_task_config(config: TransferErrorAnalysisConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=2,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.activation_feature_key,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


def _task_matrix(
    artifact: ActivationArtifact,
    definition: BinaryTaskDefinition,
    activation_feature_key: str,
) -> _TaskMatrix:
    dataset = build_binary_task_dataset(artifact=artifact, definition=definition)
    feature_tensor = activation_feature_tensor(artifact, activation_feature_key)
    artifact_index_by_example_id = {example_id: index for index, example_id in enumerate(artifact["example_ids"])}
    selected_indices: list[int] = []
    for example_id in dataset.example_ids:
        artifact_index = artifact_index_by_example_id.get(example_id)
        if artifact_index is None:
            raise BinaryTaskError(f"Artifact does not contain binary task example '{example_id}'.")
        selected_indices.append(artifact_index)
    return _TaskMatrix(dataset=dataset, matrix=tensor_to_float_matrix(feature_tensor)[selected_indices])


def _validate_feature_counts(task_matrices: tuple[_TaskMatrix, ...]) -> None:
    expected_feature_count = int(task_matrices[0].matrix.shape[1])
    for task_matrix in task_matrices:
        feature_count = int(task_matrix.matrix.shape[1])
        if feature_count != expected_feature_count:
            raise FeatureTransferErrorAnalysisError(
                f"Feature count mismatch: expected {expected_feature_count}, received {feature_count}."
            )


def _stack_matrices(matrices: tuple[FloatMatrix, ...]) -> FloatMatrix:
    return np.vstack(matrices).astype(np.float32, copy=False)


def _stack_labels(label_groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for label_group in label_groups:
        labels.extend(label_group)
    return tuple(labels)


def _encoded_labels(labels: tuple[str, ...], label_to_index: dict[str, int]) -> IntVector:
    encoded_labels: list[int] = []
    for label in labels:
        label_index = label_to_index.get(label)
        if label_index is None:
            raise FeatureTransferErrorAnalysisError(f"Test label '{label}' was not present in the training labels.")
        encoded_labels.append(label_index)
    return np.asarray(encoded_labels, dtype=np.int64)


def _positive_scores(
    classifier: object,
    matrix: FloatMatrix,
    positive_label_index: int,
) -> NDArray[np.float64]:
    probabilities = classifier.predict_proba(matrix)
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise FeatureTransferErrorAnalysisError("Classifier does not expose fitted classes_.")
    class_indices = tuple(int(value) for value in np.asarray(classes, dtype=np.int64).tolist())
    if positive_label_index not in class_indices:
        raise FeatureTransferErrorAnalysisError(
            f"Positive label index {positive_label_index} is missing from classifier classes."
        )
    return np.asarray(probabilities[:, class_indices.index(positive_label_index)], dtype=np.float64)


def _threshold_predictions(
    positive_scores: NDArray[np.float64],
    positive_label_index: int,
    label_count: int,
    threshold: float,
) -> IntVector:
    negative_label_indices = tuple(index for index in range(label_count) if index != positive_label_index)
    if len(negative_label_indices) != 1:
        raise FeatureTransferErrorAnalysisError("Transfer error analysis requires exactly one negative label.")
    negative_label_index = negative_label_indices[0]
    return np.asarray(
        [positive_label_index if score >= threshold else negative_label_index for score in positive_scores.tolist()],
        dtype=np.int64,
    )


def _records_by_example_id(records: tuple[StructuredPromptRecord, ...]) -> dict[str, StructuredPromptRecord]:
    record_by_example_id: dict[str, StructuredPromptRecord] = {}
    for record in records:
        if record.example_id in record_by_example_id:
            raise FeatureTransferErrorAnalysisError(f"Duplicate structured prompt example_id '{record.example_id}'.")
        record_by_example_id[record.example_id] = record
    return record_by_example_id


def _predictions(
    dataset: BinaryTaskDataset,
    label_names: tuple[str, ...],
    true_labels: IntVector,
    predicted_labels: IntVector,
    positive_scores: NDArray[np.float64],
    positive_label: str,
    prompt_by_example_id: dict[str, StructuredPromptRecord],
) -> tuple[TransferPrediction, ...]:
    rows: list[TransferPrediction] = []
    for row_index, example_id in enumerate(dataset.example_ids):
        prompt_record = prompt_by_example_id.get(example_id)
        if prompt_record is None:
            raise FeatureTransferErrorAnalysisError(f"Missing structured prompt record for example '{example_id}'.")
        true_label = label_names[int(true_labels[row_index])]
        predicted_label = label_names[int(predicted_labels[row_index])]
        rows.append(
            TransferPrediction(
                example_id=example_id,
                family=dataset.families[row_index],
                true_label=true_label,
                predicted_label=predicted_label,
                positive_score=float(positive_scores[row_index]),
                is_correct=predicted_label == true_label,
                error_kind=_error_kind(
                    true_label=true_label,
                    predicted_label=predicted_label,
                    positive_label=positive_label,
                ),
                readout_token_count=len(prompt_record.readout_token_indices),
                secret_token_count=_span_length(prompt_record.secret_token_span),
                query_token_count=_span_length(prompt_record.query_token_span),
                payload_token_count=_span_length(prompt_record.payload_token_span),
                tool_call_name=prompt_record.tool_call_name,
                tool_argument_path=prompt_record.tool_argument_path,
                redacted_excerpt=_redacted_excerpt(prompt_record),
            )
        )
    return tuple(rows)


def _error_kind(true_label: str, predicted_label: str, positive_label: str) -> ErrorKind | None:
    if true_label == predicted_label:
        return None
    if true_label != positive_label and predicted_label == positive_label:
        return "false_positive"
    return "false_negative"


def _span_length(span: tuple[int, int] | None) -> int | None:
    if span is None:
        return None
    return span[1] - span[0]


def _redacted_excerpt(record: StructuredPromptRecord) -> str:
    text = record.rendered_prompt
    if record.secret_char_span is not None:
        start, end = record.secret_char_span
        text = f"{text[:start]}[SECRET]{text[end:]}"
    query_marker = "[message:user:1]"
    query_start = text.find(query_marker)
    excerpt = text[query_start:] if query_start >= 0 else text
    return " ".join(excerpt[:700].split())


def _top_error_examples(
    predictions: tuple[TransferPrediction, ...],
    max_error_examples: int,
    decision_threshold: float,
) -> tuple[TransferPrediction, ...]:
    errors = tuple(prediction for prediction in predictions if not prediction.is_correct)
    return tuple(
        sorted(
            errors,
            key=lambda prediction: (
                -abs(prediction.positive_score - decision_threshold),
                prediction.family,
                prediction.example_id,
            ),
        )[:max_error_examples]
    )


def _score_summaries(predictions: tuple[TransferPrediction, ...]) -> tuple[TransferScoreSummary, ...]:
    groups: dict[str, list[float]] = {}
    for prediction in predictions:
        groups.setdefault(f"true:{prediction.true_label}", []).append(prediction.positive_score)
        if prediction.error_kind is not None:
            groups.setdefault(f"error:{prediction.error_kind}", []).append(prediction.positive_score)
    return tuple(_score_summary(group_name, tuple(scores)) for group_name, scores in sorted(groups.items()))


def _score_summary(group_name: str, scores: tuple[float, ...]) -> TransferScoreSummary:
    if len(scores) == 0:
        raise FeatureTransferErrorAnalysisError(f"Cannot summarize empty score group '{group_name}'.")
    values = np.asarray(scores, dtype=np.float64)
    return TransferScoreSummary(
        group_name=group_name,
        example_count=int(values.shape[0]),
        minimum=float(np.min(values)),
        p25=float(np.quantile(values, 0.25)),
        median=float(np.quantile(values, 0.5)),
        p75=float(np.quantile(values, 0.75)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
    )


def _family_summaries(
    predictions: tuple[TransferPrediction, ...],
    positive_label: str,
) -> tuple[TransferFamilySummary, ...]:
    family_names = tuple(sorted({prediction.family for prediction in predictions}))
    summaries: list[TransferFamilySummary] = []
    for family in family_names:
        family_predictions = tuple(prediction for prediction in predictions if prediction.family == family)
        safe_scores = tuple(
            prediction.positive_score for prediction in family_predictions if prediction.true_label != positive_label
        )
        exfil_scores = tuple(
            prediction.positive_score for prediction in family_predictions if prediction.true_label == positive_label
        )
        correct_count = sum(1 for prediction in family_predictions if prediction.is_correct)
        summaries.append(
            TransferFamilySummary(
                family=family,
                example_count=len(family_predictions),
                correct_count=correct_count,
                false_positive_count=sum(
                    1 for prediction in family_predictions if prediction.error_kind == "false_positive"
                ),
                false_negative_count=sum(
                    1 for prediction in family_predictions if prediction.error_kind == "false_negative"
                ),
                accuracy=float(correct_count / len(family_predictions)),
                safe_mean_positive_score=_optional_mean(safe_scores),
                exfil_mean_positive_score=_optional_mean(exfil_scores),
            )
        )
    return tuple(sorted(summaries, key=lambda summary: (summary.accuracy, summary.family)))


def _span_summaries(predictions: tuple[TransferPrediction, ...]) -> tuple[TransferSpanSummary, ...]:
    groups: dict[str, tuple[TransferPrediction, ...]] = {
        "all": predictions,
        "correct": tuple(prediction for prediction in predictions if prediction.is_correct),
        "errors": tuple(prediction for prediction in predictions if not prediction.is_correct),
        "false_positive": tuple(prediction for prediction in predictions if prediction.error_kind == "false_positive"),
        "false_negative": tuple(prediction for prediction in predictions if prediction.error_kind == "false_negative"),
    }
    return tuple(_span_summary(group_name, group_predictions) for group_name, group_predictions in groups.items())


def _span_summary(
    group_name: str,
    predictions: tuple[TransferPrediction, ...],
) -> TransferSpanSummary:
    return TransferSpanSummary(
        group_name=group_name,
        example_count=len(predictions),
        mean_readout_token_count=_optional_mean(
            _present_values(tuple(item.readout_token_count for item in predictions))
        ),
        mean_secret_token_count=_optional_mean(_present_values(tuple(item.secret_token_count for item in predictions))),
        mean_query_token_count=_optional_mean(_present_values(tuple(item.query_token_count for item in predictions))),
        mean_payload_token_count=_optional_mean(
            _present_values(tuple(item.payload_token_count for item in predictions))
        ),
    )


def _present_values(values: tuple[int | None, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values if value is not None)


def _optional_mean(values: tuple[float, ...]) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _structured_prompt_record(record: Mapping[str, object], line_number: int) -> StructuredPromptRecord:
    return StructuredPromptRecord(
        example_id=_required_string(record, "example_id", line_number),
        rendered_prompt=_required_string(record, "rendered_prompt", line_number),
        label=_required_string(record, "label", line_number),
        family=_required_string(record, "family", line_number),
        secret_char_span=_optional_int_pair(record.get("secret_char_span"), "secret_char_span", line_number),
        secret_token_span=_optional_int_pair(record.get("secret_token_span"), "secret_token_span", line_number),
        query_token_span=_optional_int_pair(record.get("query_token_span"), "query_token_span", line_number),
        payload_token_span=_optional_int_pair(record.get("payload_token_span"), "payload_token_span", line_number),
        readout_token_indices=_int_tuple(record.get("readout_token_indices"), "readout_token_indices", line_number),
        tool_call_name=_optional_string(record.get("tool_call_name"), "tool_call_name", line_number),
        tool_argument_path=_optional_string(record.get("tool_argument_path"), "tool_argument_path", line_number),
    )


def _required_string(record: Mapping[str, object], field_name: str, line_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise FeatureTransferErrorAnalysisError(f"Line {line_number}: field '{field_name}' must be a string.")
    return value


def _optional_string(value: object, field_name: str, line_number: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FeatureTransferErrorAnalysisError(
            f"Line {line_number}: field '{field_name}' must be a string when present."
        )
    return value


def _optional_int_pair(value: object, field_name: str, line_number: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise FeatureTransferErrorAnalysisError(f"Line {line_number}: field '{field_name}' must be a two-item list.")
    first, second = value
    if not isinstance(first, int) or not isinstance(second, int):
        raise FeatureTransferErrorAnalysisError(f"Line {line_number}: field '{field_name}' must contain integers.")
    return (first, second)


def _int_tuple(value: object, field_name: str, line_number: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise FeatureTransferErrorAnalysisError(f"Line {line_number}: field '{field_name}' must be a list.")
    values: list[int] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, int):
            raise FeatureTransferErrorAnalysisError(
                f"Line {line_number}: field '{field_name}' item {item_index} must be an integer."
            )
        values.append(item)
    return tuple(values)


def _score_summary_to_json(summary: TransferScoreSummary) -> dict[str, JsonValue]:
    return {
        "group_name": summary.group_name,
        "example_count": summary.example_count,
        "minimum": summary.minimum,
        "p25": summary.p25,
        "median": summary.median,
        "p75": summary.p75,
        "maximum": summary.maximum,
        "mean": summary.mean,
    }


def _family_summary_to_json(summary: TransferFamilySummary) -> dict[str, JsonValue]:
    return {
        "family": summary.family,
        "example_count": summary.example_count,
        "correct_count": summary.correct_count,
        "false_positive_count": summary.false_positive_count,
        "false_negative_count": summary.false_negative_count,
        "accuracy": summary.accuracy,
        "safe_mean_positive_score": summary.safe_mean_positive_score,
        "exfil_mean_positive_score": summary.exfil_mean_positive_score,
    }


def _span_summary_to_json(summary: TransferSpanSummary) -> dict[str, JsonValue]:
    return {
        "group_name": summary.group_name,
        "example_count": summary.example_count,
        "mean_readout_token_count": summary.mean_readout_token_count,
        "mean_secret_token_count": summary.mean_secret_token_count,
        "mean_query_token_count": summary.mean_query_token_count,
        "mean_payload_token_count": summary.mean_payload_token_count,
    }


def _prediction_to_json(prediction: TransferPrediction) -> dict[str, JsonValue]:
    return {
        "example_id": prediction.example_id,
        "family": prediction.family,
        "true_label": prediction.true_label,
        "predicted_label": prediction.predicted_label,
        "positive_score": prediction.positive_score,
        "is_correct": prediction.is_correct,
        "error_kind": prediction.error_kind,
        "readout_token_count": prediction.readout_token_count,
        "secret_token_count": prediction.secret_token_count,
        "query_token_count": prediction.query_token_count,
        "payload_token_count": prediction.payload_token_count,
        "tool_call_name": prediction.tool_call_name,
        "tool_argument_path": prediction.tool_argument_path,
        "redacted_excerpt": prediction.redacted_excerpt,
    }


def feature_transfer_error_analysis_to_json(report: TransferErrorAnalysisReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "activation_feature_key": report.activation_feature_key,
        "positive_label": report.positive_label,
        "decision_threshold": report.decision_threshold,
        "train_dataset_ids": list(report.train_dataset_ids),
        "test_dataset_id": report.test_dataset_id,
        "label_names": list(report.label_names),
        "train_example_count": report.train_example_count,
        "test_example_count": report.test_example_count,
        "feature_count": report.feature_count,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "confusion_matrix": [list(row) for row in report.confusion_matrix],
        "correct_count": report.correct_count,
        "error_count": report.error_count,
        "false_positive_count": report.false_positive_count,
        "false_negative_count": report.false_negative_count,
        "score_summaries": [_score_summary_to_json(summary) for summary in report.score_summaries],
        "family_summaries": [_family_summary_to_json(summary) for summary in report.family_summaries],
        "span_summaries": [_span_summary_to_json(summary) for summary in report.span_summaries],
        "error_examples": [_prediction_to_json(prediction) for prediction in report.error_examples],
    }


def write_feature_transfer_error_analysis_json(path: Path, report: TransferErrorAnalysisReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(feature_transfer_error_analysis_to_json(report), file, indent=2)
        file.write("\n")


def render_feature_transfer_error_analysis_markdown(report: TransferErrorAnalysisReport) -> str:
    lines = [
        "# Feature Transfer Error Analysis",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Feature: `{report.activation_feature_key}`",
        f"- Positive label: `{report.positive_label}`",
        f"- Decision threshold: `{report.decision_threshold:.4f}`",
        f"- Train datasets: `{', '.join(report.train_dataset_ids)}`",
        f"- Test dataset: `{report.test_dataset_id}`",
        f"- Label order: `{', '.join(report.label_names)}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {report.accuracy:.4f} |",
        f"| Macro F1 | {report.macro_f1:.4f} |",
        f"| Correct | {report.correct_count} |",
        f"| Errors | {report.error_count} |",
        f"| False positives | {report.false_positive_count} |",
        f"| False negatives | {report.false_negative_count} |",
        "",
        "Confusion matrix:",
        "",
        "```text",
    ]
    for row in report.confusion_matrix:
        lines.append(str(list(row)))
    lines.extend(
        [
            "```",
            "",
            "## Score Summaries",
            "",
            "| Group | Count | Min | P25 | Median | P75 | Max | Mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in report.score_summaries:
        lines.append(
            f"| `{summary.group_name}` | {summary.example_count} | {summary.minimum:.4f} | "
            f"{summary.p25:.4f} | {summary.median:.4f} | {summary.p75:.4f} | "
            f"{summary.maximum:.4f} | {summary.mean:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Worst Families",
            "",
            "| Family | Count | Accuracy | FP | FN | Safe Mean Score | Exfil Mean Score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in report.family_summaries:
        lines.append(
            f"| `{summary.family}` | {summary.example_count} | {summary.accuracy:.4f} | "
            f"{summary.false_positive_count} | {summary.false_negative_count} | "
            f"{_format_optional(summary.safe_mean_positive_score)} | "
            f"{_format_optional(summary.exfil_mean_positive_score)} |"
        )

    lines.extend(
        [
            "",
            "## Span Geometry",
            "",
            "| Group | Count | Mean Readout Tokens | Mean Secret Tokens | Mean Query Tokens | Mean Payload Tokens |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in report.span_summaries:
        lines.append(
            f"| `{summary.group_name}` | {summary.example_count} | "
            f"{_format_optional(summary.mean_readout_token_count)} | "
            f"{_format_optional(summary.mean_secret_token_count)} | "
            f"{_format_optional(summary.mean_query_token_count)} | "
            f"{_format_optional(summary.mean_payload_token_count)} |"
        )

    lines.extend(
        [
            "",
            "## Most Confident Errors",
            "",
            "| Example | Family | Kind | True | Predicted | Score | Readout Tokens | Excerpt |",
            "|---|---|---|---|---|---:|---:|---|",
        ]
    )
    for prediction in report.error_examples:
        lines.append(
            f"| `{prediction.example_id}` | `{prediction.family}` | `{prediction.error_kind}` | "
            f"`{prediction.true_label}` | `{prediction.predicted_label}` | {prediction.positive_score:.4f} | "
            f"{prediction.readout_token_count if prediction.readout_token_count is not None else ''} | "
            f"{_escape_markdown_table(prediction.redacted_excerpt)} |"
        )

    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|")


def write_feature_transfer_error_analysis_markdown(path: Path, report: TransferErrorAnalysisReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_feature_transfer_error_analysis_markdown(report), encoding="utf-8")
