from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryTaskConfig,
    BinaryTaskDataset,
    BinaryTaskDefinition,
    activation_feature_tensor,
    build_activation_classifier,
    build_binary_task_dataset,
    default_binary_task_definitions,
    stratified_group_splits,
)
from aegis_introspection.probe import JsonValue, encode_labels, tensor_to_float_matrix


class CiftCalibrationError(ValueError):
    """Raised when CIFT detector calibration cannot be computed."""


@dataclass(frozen=True)
class CiftCalibrationConfig:
    task_name: str
    positive_label: str
    activation_feature_key: str
    fold_count: int
    inner_fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    decision_threshold: float


@dataclass(frozen=True)
class CalibratedCiftPrediction:
    fold_index: int
    example_id: str
    family: str
    source_label: str
    true_label: str
    predicted_label: str
    is_correct: bool
    positive_label: str
    positive_probability: float


@dataclass(frozen=True)
class CalibrationBinSummary:
    bin_index: int
    lower_bound: float
    upper_bound: float
    example_count: int
    mean_probability: float
    empirical_positive_rate: float
    absolute_gap: float


@dataclass(frozen=True)
class CiftCalibrationReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: str
    score_semantics: str
    task_name: str
    positive_label: str
    activation_feature_key: str
    fold_count: int
    inner_fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    decision_threshold: float
    accuracy: float
    macro_f1: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    bin_summaries: tuple[CalibrationBinSummary, ...]
    predictions: tuple[CalibratedCiftPrediction, ...]


def collect_grouped_calibrated_cift_predictions(
    artifact: ActivationArtifact,
    config: CiftCalibrationConfig,
) -> CiftCalibrationReport:
    _validate_config(config)
    definition = _task_definition(config.task_name)
    dataset = build_binary_task_dataset(artifact, definition)
    feature_tensor = activation_feature_tensor(artifact, config.activation_feature_key)
    selected_indices = tuple(artifact["example_ids"].index(example_id) for example_id in dataset.example_ids)
    matrix = tensor_to_float_matrix(feature_tensor)[list(selected_indices)]
    label_encoding = encode_labels(dataset.target_labels)
    positive_index = _positive_index(label_encoding.label_to_index, config.positive_label)
    encoded_labels = label_encoding.encoded_labels
    outer_task_config = _binary_task_config(config, config.fold_count)
    outer_splits = stratified_group_splits(encoded_labels, dataset.families, outer_task_config)
    predictions: list[CalibratedCiftPrediction] = []

    for split in outer_splits:
        train_matrix = matrix[split.train_indices]
        train_labels = encoded_labels[split.train_indices]
        train_groups = tuple(dataset.families[index] for index in split.train_indices.tolist())
        calibrator = _fit_inner_platt_calibrator(
            matrix=train_matrix,
            labels=train_labels,
            groups=train_groups,
            positive_index=positive_index,
            config=config,
        )
        classifier = build_activation_classifier(_binary_task_config(config, config.fold_count))
        classifier.fit(train_matrix, train_labels)
        test_matrix = matrix[split.test_indices]
        positive_probabilities = _positive_probabilities(classifier, test_matrix, positive_index)
        calibrated_probabilities = calibrator.predict_proba(positive_probabilities.reshape(-1, 1))[:, 1]
        predictions.extend(
            _calibrated_predictions(
                dataset=dataset,
                label_names=label_encoding.label_names,
                positive_index=positive_index,
                fold_index=split.fold_index,
                test_indices=split.test_indices,
                positive_probabilities=calibrated_probabilities,
                config=config,
            )
        )

    return _calibration_report(
        artifact=artifact,
        config=config,
        label_names=label_encoding.label_names,
        predictions=tuple(predictions),
    )


def cift_calibration_report_to_json(report: CiftCalibrationReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "score_semantics": report.score_semantics,
        "task_name": report.task_name,
        "positive_label": report.positive_label,
        "activation_feature_key": report.activation_feature_key,
        "fold_count": report.fold_count,
        "inner_fold_count": report.inner_fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "decision_threshold": report.decision_threshold,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "brier_score": report.brier_score,
        "log_loss": report.log_loss,
        "expected_calibration_error": report.expected_calibration_error,
        "confusion_matrix": [list(row) for row in report.confusion_matrix],
        "bin_summaries": [_bin_summary_to_json(summary) for summary in report.bin_summaries],
        "predictions": [_prediction_to_json(prediction) for prediction in report.predictions],
    }


def write_cift_calibration_json(path: Path, report: CiftCalibrationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_calibration_report_to_json(report), file, indent=2)
        file.write("\n")


def load_cift_calibration_report_json(path: Path) -> CiftCalibrationReport:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftCalibrationError(f"Invalid calibration JSON in {path}: {exc.msg}.") from exc
    record = _as_mapping(decoded, "calibration report")
    return CiftCalibrationReport(
        source_model_id=_required_string(record, "source_model_id", "calibration report"),
        source_revision=_required_string(record, "source_revision", "calibration report"),
        source_selected_device=_required_string(record, "source_selected_device", "calibration report"),
        evaluation_strategy=_required_string(record, "evaluation_strategy", "calibration report"),
        score_semantics=_required_string(record, "score_semantics", "calibration report"),
        task_name=_required_string(record, "task_name", "calibration report"),
        positive_label=_required_string(record, "positive_label", "calibration report"),
        activation_feature_key=_required_string(record, "activation_feature_key", "calibration report"),
        fold_count=_required_int(record, "fold_count", "calibration report"),
        inner_fold_count=_required_int(record, "inner_fold_count", "calibration report"),
        random_seed=_required_int(record, "random_seed", "calibration report"),
        regularization_c=_required_float(record, "regularization_c", "calibration report"),
        max_iter=_required_int(record, "max_iter", "calibration report"),
        decision_threshold=_required_float(record, "decision_threshold", "calibration report"),
        accuracy=_required_float(record, "accuracy", "calibration report"),
        macro_f1=_required_float(record, "macro_f1", "calibration report"),
        brier_score=_required_float(record, "brier_score", "calibration report"),
        log_loss=_required_float(record, "log_loss", "calibration report"),
        expected_calibration_error=_required_float(record, "expected_calibration_error", "calibration report"),
        confusion_matrix=_confusion_matrix_from_json(record.get("confusion_matrix")),
        bin_summaries=tuple(
            _bin_summary_from_json(item, item_index)
            for item_index, item in enumerate(_required_list(record, "bin_summaries", "calibration report"))
        ),
        predictions=tuple(
            _prediction_from_json(item, item_index)
            for item_index, item in enumerate(_required_list(record, "predictions", "calibration report"))
        ),
    )


def render_cift_calibration_markdown(report: CiftCalibrationReport) -> str:
    lines = [
        "# CIFT Detector Score Calibration",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Score semantics: `{report.score_semantics}`",
        f"- Task: `{report.task_name}`",
        f"- Positive label: `{report.positive_label}`",
        f"- Activation feature: `{report.activation_feature_key}`",
        f"- Outer folds: `{report.fold_count}`",
        f"- Inner calibration folds: `{report.inner_fold_count}`",
        f"- Decision threshold: `{report.decision_threshold:.4f}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {report.accuracy:.4f} |",
        f"| Macro F1 | {report.macro_f1:.4f} |",
        f"| Brier score | {report.brier_score:.4f} |",
        f"| Log loss | {report.log_loss:.4f} |",
        f"| Expected calibration error | {report.expected_calibration_error:.4f} |",
        "",
        "## Calibration Bins",
        "",
        "| Bin | Range | Examples | Mean Probability | Empirical Positive Rate | Absolute Gap |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for summary in report.bin_summaries:
        lines.append(
            f"| {summary.bin_index} | "
            f"[{summary.lower_bound:.2f}, {summary.upper_bound:.2f}] | "
            f"{summary.example_count} | {summary.mean_probability:.4f} | "
            f"{summary.empirical_positive_rate:.4f} | {summary.absolute_gap:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_cift_calibration_markdown(path: Path, report: CiftCalibrationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_calibration_markdown(report), encoding="utf-8")


def _fit_inner_platt_calibrator(
    matrix: NDArray[np.float32],
    labels: NDArray[np.int64],
    groups: tuple[str, ...],
    positive_index: int,
    config: CiftCalibrationConfig,
) -> LogisticRegression:
    inner_task_config = _binary_task_config(config, config.inner_fold_count)
    inner_splits = stratified_group_splits(labels, groups, inner_task_config)
    oof_probabilities = np.zeros(labels.shape[0], dtype=np.float64)
    positive_targets = (labels == positive_index).astype(np.int64, copy=False)

    for split in inner_splits:
        classifier = build_activation_classifier(_binary_task_config(config, config.fold_count))
        classifier.fit(matrix[split.train_indices], labels[split.train_indices])
        oof_probabilities[split.test_indices] = _positive_probabilities(
            classifier,
            matrix[split.test_indices],
            positive_index,
        )

    calibrator = LogisticRegression(
        C=1.0,
        max_iter=config.max_iter,
        random_state=config.random_seed,
    )
    calibrator.fit(oof_probabilities.reshape(-1, 1), positive_targets)
    return calibrator


def _positive_probabilities(classifier: Pipeline, matrix: NDArray[np.float32], positive_index: int) -> NDArray[np.float64]:
    probabilities = classifier.predict_proba(matrix)
    classes = classifier.classes_
    matching_indices = np.where(classes == positive_index)[0]
    if matching_indices.shape[0] != 1:
        raise CiftCalibrationError(f"Classifier does not expose positive class index {positive_index}.")
    return probabilities[:, int(matching_indices[0])].astype(np.float64, copy=False)


def _calibrated_predictions(
    dataset: BinaryTaskDataset,
    label_names: tuple[str, ...],
    positive_index: int,
    fold_index: int,
    test_indices: NDArray[np.int64],
    positive_probabilities: NDArray[np.float64],
    config: CiftCalibrationConfig,
) -> tuple[CalibratedCiftPrediction, ...]:
    rows: list[CalibratedCiftPrediction] = []
    negative_index = _negative_index(label_names, positive_index)
    for row_index, positive_probability in zip(test_indices.tolist(), positive_probabilities.tolist(), strict=True):
        predicted_index = positive_index if positive_probability >= config.decision_threshold else negative_index
        predicted_label = label_names[predicted_index]
        true_label = dataset.target_labels[row_index]
        rows.append(
            CalibratedCiftPrediction(
                fold_index=fold_index,
                example_id=dataset.example_ids[row_index],
                family=dataset.families[row_index],
                source_label=dataset.source_labels[row_index],
                true_label=true_label,
                predicted_label=predicted_label,
                is_correct=predicted_label == true_label,
                positive_label=config.positive_label,
                positive_probability=float(positive_probability),
            )
        )
    return tuple(rows)


def _calibration_report(
    artifact: ActivationArtifact,
    config: CiftCalibrationConfig,
    label_names: tuple[str, ...],
    predictions: tuple[CalibratedCiftPrediction, ...],
) -> CiftCalibrationReport:
    true_binary = np.asarray(
        [1 if prediction.true_label == config.positive_label else 0 for prediction in predictions],
        dtype=np.int64,
    )
    probability = np.asarray([prediction.positive_probability for prediction in predictions], dtype=np.float64)
    predicted_labels = tuple(prediction.predicted_label for prediction in predictions)
    true_labels = tuple(prediction.true_label for prediction in predictions)
    metadata = artifact["metadata"]
    return CiftCalibrationReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold_with_inner_platt_calibration",
        score_semantics="inner_cv_platt_calibrated_probability",
        task_name=config.task_name,
        positive_label=config.positive_label,
        activation_feature_key=config.activation_feature_key,
        fold_count=config.fold_count,
        inner_fold_count=config.inner_fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        decision_threshold=config.decision_threshold,
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        macro_f1=float(f1_score(true_labels, predicted_labels, average="macro", labels=label_names, zero_division=0)),
        brier_score=float(brier_score_loss(true_binary, probability)),
        log_loss=float(log_loss(true_binary, np.clip(probability, 1e-9, 1.0 - 1e-9), labels=(0, 1))),
        expected_calibration_error=_expected_calibration_error(true_binary, probability, bin_count=10),
        confusion_matrix=_matrix_to_tuple(confusion_matrix(true_labels, predicted_labels, labels=label_names)),
        bin_summaries=_calibration_bins(true_binary, probability, bin_count=10),
        predictions=predictions,
    )


def _expected_calibration_error(
    true_binary: NDArray[np.int64],
    probability: NDArray[np.float64],
    bin_count: int,
) -> float:
    summaries = _calibration_bins(true_binary, probability, bin_count)
    total_count = int(true_binary.shape[0])
    return float(sum((summary.example_count / total_count) * summary.absolute_gap for summary in summaries))


def _calibration_bins(
    true_binary: NDArray[np.int64],
    probability: NDArray[np.float64],
    bin_count: int,
) -> tuple[CalibrationBinSummary, ...]:
    summaries: list[CalibrationBinSummary] = []
    for bin_index in range(bin_count):
        lower_bound = bin_index / bin_count
        upper_bound = (bin_index + 1) / bin_count
        if bin_index == bin_count - 1:
            mask = (probability >= lower_bound) & (probability <= upper_bound)
        else:
            mask = (probability >= lower_bound) & (probability < upper_bound)
        example_count = int(mask.sum())
        if example_count == 0:
            mean_probability = 0.0
            empirical_positive_rate = 0.0
        else:
            mean_probability = float(probability[mask].mean())
            empirical_positive_rate = float(true_binary[mask].mean())
        summaries.append(
            CalibrationBinSummary(
                bin_index=bin_index + 1,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                example_count=example_count,
                mean_probability=mean_probability,
                empirical_positive_rate=empirical_positive_rate,
                absolute_gap=abs(mean_probability - empirical_positive_rate),
            )
        )
    return tuple(summaries)


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise CiftCalibrationError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _binary_task_config(config: CiftCalibrationConfig, fold_count: int) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key=config.activation_feature_key,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
    )


def _positive_index(label_to_index: dict[str, int], positive_label: str) -> int:
    index = label_to_index.get(positive_label)
    if index is None:
        raise CiftCalibrationError(f"Positive label '{positive_label}' is not present in task labels.")
    return index


def _negative_index(label_names: tuple[str, ...], positive_index: int) -> int:
    indices = tuple(index for index in range(len(label_names)) if index != positive_index)
    if len(indices) != 1:
        raise CiftCalibrationError("Calibrated CIFT detector requires exactly one negative label.")
    return indices[0]


def _validate_config(config: CiftCalibrationConfig) -> None:
    if config.task_name == "":
        raise CiftCalibrationError("task_name must not be empty.")
    if config.positive_label == "":
        raise CiftCalibrationError("positive_label must not be empty.")
    if config.activation_feature_key == "":
        raise CiftCalibrationError("activation_feature_key must not be empty.")
    if config.fold_count < 2:
        raise CiftCalibrationError("fold_count must be at least 2.")
    if config.inner_fold_count < 2:
        raise CiftCalibrationError("inner_fold_count must be at least 2.")
    if config.max_iter < 1:
        raise CiftCalibrationError("max_iter must be at least 1.")
    if config.regularization_c <= 0.0:
        raise CiftCalibrationError("regularization_c must be greater than 0.")
    if config.decision_threshold < 0.0 or config.decision_threshold > 1.0:
        raise CiftCalibrationError("decision_threshold must be in [0.0, 1.0].")


def _prediction_to_json(prediction: CalibratedCiftPrediction) -> dict[str, JsonValue]:
    return {
        "fold_index": prediction.fold_index,
        "example_id": prediction.example_id,
        "family": prediction.family,
        "source_label": prediction.source_label,
        "true_label": prediction.true_label,
        "predicted_label": prediction.predicted_label,
        "is_correct": prediction.is_correct,
        "positive_label": prediction.positive_label,
        "positive_probability": prediction.positive_probability,
    }


def _bin_summary_to_json(summary: CalibrationBinSummary) -> dict[str, JsonValue]:
    return {
        "bin_index": summary.bin_index,
        "lower_bound": summary.lower_bound,
        "upper_bound": summary.upper_bound,
        "example_count": summary.example_count,
        "mean_probability": summary.mean_probability,
        "empirical_positive_rate": summary.empirical_positive_rate,
        "absolute_gap": summary.absolute_gap,
    }


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CiftCalibrationError(f"{description}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, description: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise CiftCalibrationError(f"{description}: field '{field_name}' must be a string.")
    if value == "":
        raise CiftCalibrationError(f"{description}: field '{field_name}' must not be empty.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, description: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise CiftCalibrationError(f"{description}: field '{field_name}' must be an integer.")
    return value


def _required_float(record: Mapping[str, object], field_name: str, description: str) -> float:
    value = record.get(field_name)
    if not isinstance(value, int) and not isinstance(value, float):
        raise CiftCalibrationError(f"{description}: field '{field_name}' must be numeric.")
    return float(value)


def _required_bool(record: Mapping[str, object], field_name: str, description: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise CiftCalibrationError(f"{description}: field '{field_name}' must be boolean.")
    return value


def _required_list(record: Mapping[str, object], field_name: str, description: str) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise CiftCalibrationError(f"{description}: field '{field_name}' must be a list.")
    return value


def _confusion_matrix_from_json(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise CiftCalibrationError("calibration report: confusion_matrix must be a list.")
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise CiftCalibrationError(f"calibration report: confusion_matrix row {row_index} must be a list.")
        values: list[int] = []
        for column_index, item in enumerate(row):
            if not isinstance(item, int):
                raise CiftCalibrationError(
                    f"calibration report: confusion_matrix item {row_index},{column_index} must be an integer."
                )
            values.append(item)
        rows.append(tuple(values))
    return tuple(rows)


def _bin_summary_from_json(value: object, index: int) -> CalibrationBinSummary:
    record = _as_mapping(value, f"bin summary {index}")
    return CalibrationBinSummary(
        bin_index=_required_int(record, "bin_index", f"bin summary {index}"),
        lower_bound=_required_float(record, "lower_bound", f"bin summary {index}"),
        upper_bound=_required_float(record, "upper_bound", f"bin summary {index}"),
        example_count=_required_int(record, "example_count", f"bin summary {index}"),
        mean_probability=_required_float(record, "mean_probability", f"bin summary {index}"),
        empirical_positive_rate=_required_float(record, "empirical_positive_rate", f"bin summary {index}"),
        absolute_gap=_required_float(record, "absolute_gap", f"bin summary {index}"),
    )


def _prediction_from_json(value: object, index: int) -> CalibratedCiftPrediction:
    record = _as_mapping(value, f"calibrated prediction {index}")
    return CalibratedCiftPrediction(
        fold_index=_required_int(record, "fold_index", f"calibrated prediction {index}"),
        example_id=_required_string(record, "example_id", f"calibrated prediction {index}"),
        family=_required_string(record, "family", f"calibrated prediction {index}"),
        source_label=_required_string(record, "source_label", f"calibrated prediction {index}"),
        true_label=_required_string(record, "true_label", f"calibrated prediction {index}"),
        predicted_label=_required_string(record, "predicted_label", f"calibrated prediction {index}"),
        is_correct=_required_bool(record, "is_correct", f"calibrated prediction {index}"),
        positive_label=_required_string(record, "positive_label", f"calibrated prediction {index}"),
        positive_probability=_required_float(record, "positive_probability", f"calibrated prediction {index}"),
    )
