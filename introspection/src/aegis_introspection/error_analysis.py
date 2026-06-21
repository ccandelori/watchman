from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryMethodName,
    BinaryTaskConfig,
    BinaryTaskDataset,
    BinaryTaskError,
    EvaluationStrategy,
    activation_feature_tensor,
    build_activation_classifier,
    build_text_classifier,
    build_binary_task_dataset,
    default_binary_task_definitions,
    stratified_group_splits,
)
from aegis_introspection.probe import IntVector, JsonValue, encode_labels, tensor_to_float_matrix


@dataclass(frozen=True)
class BinaryExamplePrediction:
    fold_index: int
    example_id: str
    family: str
    source_label: str
    true_label: str
    predicted_label: str
    is_correct: bool


@dataclass(frozen=True)
class BinaryFamilyErrorSummary:
    family: str
    true_label: str
    example_count: int
    correct_count: int
    error_count: int
    accuracy: float
    predicted_label_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class BinaryMethodErrorAnalysis:
    method_name: BinaryMethodName
    feature_name: str
    label_names: tuple[str, ...]
    prediction_count: int
    correct_count: int
    error_count: int
    accuracy: float
    family_summaries: tuple[BinaryFamilyErrorSummary, ...]
    predictions: tuple[BinaryExamplePrediction, ...]


@dataclass(frozen=True)
class BinaryTaskErrorAnalysis:
    task_name: str
    description: str
    label_names: tuple[str, ...]
    methods: tuple[BinaryMethodErrorAnalysis, ...]


@dataclass(frozen=True)
class BinaryErrorAnalysisReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    activation_feature_key: str
    tasks: tuple[BinaryTaskErrorAnalysis, ...]


def summarize_family_predictions(
    predictions: tuple[BinaryExamplePrediction, ...],
) -> tuple[BinaryFamilyErrorSummary, ...]:
    grouped_predictions: dict[tuple[str, str], list[BinaryExamplePrediction]] = {}
    for prediction in predictions:
        key = (prediction.family, prediction.true_label)
        grouped_predictions.setdefault(key, []).append(prediction)

    summaries: list[BinaryFamilyErrorSummary] = []
    for (family, true_label), family_predictions in grouped_predictions.items():
        predicted_label_counts = tuple(
            (label, sum(1 for prediction in family_predictions if prediction.predicted_label == label))
            for label in sorted({prediction.predicted_label for prediction in family_predictions})
        )
        correct_count = sum(1 for prediction in family_predictions if prediction.is_correct)
        example_count = len(family_predictions)
        summaries.append(
            BinaryFamilyErrorSummary(
                family=family,
                true_label=true_label,
                example_count=example_count,
                correct_count=correct_count,
                error_count=example_count - correct_count,
                accuracy=float(correct_count / example_count),
                predicted_label_counts=predicted_label_counts,
            )
        )
    return tuple(summaries)


def _selected_artifact_indices(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
) -> tuple[int, ...]:
    example_index_by_id = {example_id: index for index, example_id in enumerate(artifact["example_ids"])}
    selected_indices: list[int] = []
    for example_id in dataset.example_ids:
        index = example_index_by_id.get(example_id)
        if index is None:
            raise BinaryTaskError(f"Artifact does not contain binary task example '{example_id}'.")
        selected_indices.append(index)
    return tuple(selected_indices)


def _predictions_from_encoded_labels(
    dataset: BinaryTaskDataset,
    label_names: tuple[str, ...],
    fold_index: int,
    test_indices: IntVector,
    predictions: IntVector,
) -> tuple[BinaryExamplePrediction, ...]:
    rows: list[BinaryExamplePrediction] = []
    for row_index, predicted_index in zip(test_indices.tolist(), predictions.tolist(), strict=True):
        predicted_label = label_names[int(predicted_index)]
        true_label = dataset.target_labels[row_index]
        rows.append(
            BinaryExamplePrediction(
                fold_index=fold_index,
                example_id=dataset.example_ids[row_index],
                family=dataset.families[row_index],
                source_label=dataset.source_labels[row_index],
                true_label=true_label,
                predicted_label=predicted_label,
                is_correct=predicted_label == true_label,
            )
        )
    return tuple(rows)


def _build_method_error_analysis(
    method_name: BinaryMethodName,
    feature_name: str,
    label_names: tuple[str, ...],
    predictions: tuple[BinaryExamplePrediction, ...],
) -> BinaryMethodErrorAnalysis:
    if len(predictions) == 0:
        raise BinaryTaskError(f"Method '{method_name}' produced no predictions.")
    correct_count = sum(1 for prediction in predictions if prediction.is_correct)
    prediction_count = len(predictions)
    return BinaryMethodErrorAnalysis(
        method_name=method_name,
        feature_name=feature_name,
        label_names=label_names,
        prediction_count=prediction_count,
        correct_count=correct_count,
        error_count=prediction_count - correct_count,
        accuracy=float(correct_count / prediction_count),
        family_summaries=summarize_family_predictions(predictions),
        predictions=predictions,
    )


def _text_feature_name(method_name: BinaryMethodName, config: BinaryTaskConfig) -> str:
    if method_name == "word_tfidf":
        return f"word_tfidf_{config.word_ngram_range[0]}_{config.word_ngram_range[1]}"
    if method_name == "char_tfidf":
        return f"char_wb_tfidf_{config.char_ngram_range[0]}_{config.char_ngram_range[1]}"
    raise BinaryTaskError(f"Unsupported text classifier method '{method_name}'.")


def collect_grouped_activation_predictions(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    config: BinaryTaskConfig,
) -> BinaryMethodErrorAnalysis:
    feature_tensor = activation_feature_tensor(artifact, config.activation_feature_key)
    matrix = tensor_to_float_matrix(feature_tensor)[list(_selected_artifact_indices(artifact, dataset))]
    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    splits = stratified_group_splits(encoded_labels, dataset.families, config)
    predictions: list[BinaryExamplePrediction] = []

    for split in splits:
        classifier = build_activation_classifier(config)
        classifier.fit(matrix[split.train_indices], encoded_labels[split.train_indices])
        predicted_labels = classifier.predict(matrix[split.test_indices]).astype(np.int64, copy=False)
        predictions.extend(
            _predictions_from_encoded_labels(
                dataset=dataset,
                label_names=label_encoding.label_names,
                fold_index=split.fold_index,
                test_indices=split.test_indices,
                predictions=predicted_labels,
            )
        )

    return _build_method_error_analysis(
        method_name="activation_probe",
        feature_name=config.activation_feature_key,
        label_names=label_encoding.label_names,
        predictions=tuple(predictions),
    )


def collect_grouped_text_predictions(
    dataset: BinaryTaskDataset,
    method_name: BinaryMethodName,
    config: BinaryTaskConfig,
) -> BinaryMethodErrorAnalysis:
    if method_name == "activation_probe":
        raise BinaryTaskError("Use collect_grouped_activation_predictions for activation probes.")

    label_encoding = encode_labels(dataset.target_labels)
    encoded_labels = label_encoding.encoded_labels
    text_array = np.asarray(dataset.texts, dtype=object)
    splits = stratified_group_splits(encoded_labels, dataset.families, config)
    predictions: list[BinaryExamplePrediction] = []

    for split in splits:
        classifier = build_text_classifier(method_name, config)
        classifier.fit(text_array[split.train_indices].tolist(), encoded_labels[split.train_indices])
        predicted_labels = classifier.predict(text_array[split.test_indices].tolist()).astype(np.int64, copy=False)
        predictions.extend(
            _predictions_from_encoded_labels(
                dataset=dataset,
                label_names=label_encoding.label_names,
                fold_index=split.fold_index,
                test_indices=split.test_indices,
                predictions=predicted_labels,
            )
        )

    return _build_method_error_analysis(
        method_name=method_name,
        feature_name=_text_feature_name(method_name, config),
        label_names=label_encoding.label_names,
        predictions=tuple(predictions),
    )


def evaluate_grouped_binary_error_analysis(
    artifact: ActivationArtifact,
    config: BinaryTaskConfig,
) -> BinaryErrorAnalysisReport:
    task_reports: list[BinaryTaskErrorAnalysis] = []
    for definition in default_binary_task_definitions():
        dataset = build_binary_task_dataset(artifact, definition)
        methods = (
            collect_grouped_activation_predictions(artifact, dataset, config),
            collect_grouped_text_predictions(dataset, "word_tfidf", config),
            collect_grouped_text_predictions(dataset, "char_tfidf", config),
        )
        task_reports.append(
            BinaryTaskErrorAnalysis(
                task_name=dataset.name,
                description=dataset.description,
                label_names=methods[0].label_names,
                methods=methods,
            )
        )

    metadata = artifact["metadata"]
    return BinaryErrorAnalysisReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        activation_feature_key=config.activation_feature_key,
        tasks=tuple(task_reports),
    )


def _prediction_to_json(prediction: BinaryExamplePrediction) -> dict[str, JsonValue]:
    return {
        "fold_index": prediction.fold_index,
        "example_id": prediction.example_id,
        "family": prediction.family,
        "source_label": prediction.source_label,
        "true_label": prediction.true_label,
        "predicted_label": prediction.predicted_label,
        "is_correct": prediction.is_correct,
    }


def _family_summary_to_json(summary: BinaryFamilyErrorSummary) -> dict[str, JsonValue]:
    return {
        "family": summary.family,
        "true_label": summary.true_label,
        "example_count": summary.example_count,
        "correct_count": summary.correct_count,
        "error_count": summary.error_count,
        "accuracy": summary.accuracy,
        "predicted_label_counts": [
            {"label": label, "count": count}
            for label, count in summary.predicted_label_counts
        ],
    }


def _method_to_json(method: BinaryMethodErrorAnalysis) -> dict[str, JsonValue]:
    return {
        "method_name": method.method_name,
        "feature_name": method.feature_name,
        "label_names": list(method.label_names),
        "prediction_count": method.prediction_count,
        "correct_count": method.correct_count,
        "error_count": method.error_count,
        "accuracy": method.accuracy,
        "family_summaries": [_family_summary_to_json(summary) for summary in method.family_summaries],
        "predictions": [_prediction_to_json(prediction) for prediction in method.predictions],
    }


def _task_to_json(task: BinaryTaskErrorAnalysis) -> dict[str, JsonValue]:
    return {
        "task_name": task.task_name,
        "description": task.description,
        "label_names": list(task.label_names),
        "methods": [_method_to_json(method) for method in task.methods],
    }


def binary_error_analysis_report_to_json(report: BinaryErrorAnalysisReport) -> dict[str, JsonValue]:
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


def write_binary_error_analysis_json(path: Path, report: BinaryErrorAnalysisReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(binary_error_analysis_report_to_json(report), file, indent=2)
        file.write("\n")


def _predicted_label_counts_text(summary: BinaryFamilyErrorSummary) -> str:
    return ", ".join(f"{label}={count}" for label, count in summary.predicted_label_counts)


def render_binary_error_analysis_markdown(report: BinaryErrorAnalysisReport) -> str:
    lines = [
        "# Binary Error Analysis",
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
                "| Method | Accuracy | Errors | Predictions |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in sorted(task.methods, key=lambda item: (item.error_count, -item.accuracy, item.method_name)):
            lines.append(
                f"| `{method.method_name}` | {method.accuracy:.4f} | "
                f"{method.error_count} | {method.prediction_count} |"
            )
        lines.append("")

        for method in task.methods:
            lines.extend(
                [
                    f"### {task.task_name} / {method.method_name}",
                    "",
                ]
            )
            sorted_summaries = sorted(
                (summary for summary in method.family_summaries if summary.error_count > 0),
                key=lambda summary: (-summary.error_count, summary.accuracy, summary.family, summary.true_label),
            )
            if len(sorted_summaries) == 0:
                lines.extend(["No family-level errors.", ""])
                continue

            lines.extend(
                [
                    "| Method | Family | True Label | Examples | Errors | Accuracy | Predicted Labels |",
                    "|---|---|---|---:|---:|---:|---|",
                ]
            )
            for summary in sorted_summaries:
                lines.append(
                    f"| `{method.method_name}` | `{summary.family}` | `{summary.true_label}` | "
                    f"{summary.example_count} | {summary.error_count} | {summary.accuracy:.4f} | "
                    f"`{_predicted_label_counts_text(summary)}` |"
                )
            lines.append("")

    return "\n".join(lines)


def write_binary_error_analysis_markdown(path: Path, report: BinaryErrorAnalysisReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_binary_error_analysis_markdown(report), encoding="utf-8")
