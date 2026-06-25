from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from aegis_introspection.cift_model_training import (
    CiftLinearLogisticClassifier,
    CiftModelTrainingError,
    CiftTrainingArtifact,
    build_cift_binary_task_rows,
    cift_binary_task_definition,
    cift_feature_matrix_for_rows,
    encode_labels,
)
from aegis_introspection.cift_paper_mlp import CiftPaperMlpClassifier, CiftPaperMlpConfig
from aegis_introspection.cift_probe_competition import (
    CiftProbeCompetitionConfig,
    CiftProbeCompetitionReport,
    CiftProbeRun,
    cift_probe_competition_report_to_json,
    compare_cift_probe_candidates,
)

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
FloatMatrix: TypeAlias = NDArray[np.float32]
IntVector: TypeAlias = NDArray[np.int64]
CiftFeatureRepresentation: TypeAlias = Literal["raw_activation", "diagonal_mahalanobis_cci"]


class CiftProbeHeadToHeadError(ValueError):
    """Raised when CIFT probe head-to-head evidence cannot be generated."""


@dataclass(frozen=True)
class CiftProbeHeadToHeadConfig:
    report_id: str
    artifact: CiftTrainingArtifact
    training_dataset_id: str
    task_name: str
    positive_label: str
    feature_representation: CiftFeatureRepresentation
    activation_feature_key: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    fold_count: int
    random_seeds: tuple[int, ...]
    decision_threshold: float
    linear_max_epochs: int
    linear_regularization_c: float
    paper_mlp_max_epochs: int
    paper_mlp_learning_rate: float
    paper_mlp_l1_softplus_weight: float
    paper_mlp_batch_size: int
    paper_hyperparameter_search_trials: int
    candidate_hyperparameter_search_trials: int
    evaluation_split_id: str
    evaluation_split_manifest_id: str
    metric_name: str
    created_at: str


@dataclass(frozen=True)
class CiftProbeHeadToHeadSeedResult:
    random_seed: int
    fold_count: int
    paper_probe_architecture: str
    candidate_probe_architecture: str
    paper_macro_f1: float
    candidate_macro_f1: float
    paper_false_positive_rate: float
    paper_true_positive_rate: float
    candidate_false_positive_rate: float
    candidate_true_positive_rate: float


@dataclass(frozen=True)
class CiftProbeHeadToHeadReport:
    competition_report: CiftProbeCompetitionReport
    feature_representation: CiftFeatureRepresentation
    activation_feature_key: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    seed_results: tuple[CiftProbeHeadToHeadSeedResult, ...]


@dataclass(frozen=True)
class _GroupedSplit:
    fold_index: int
    train_indices: IntVector
    test_indices: IntVector


@dataclass(frozen=True)
class _FoldPredictions:
    fold_index: int
    test_example_ids: tuple[str, ...]
    true_labels: IntVector
    paper_predictions: IntVector
    candidate_predictions: IntVector


@dataclass(frozen=True)
class _SeedEvaluation:
    seed_result: CiftProbeHeadToHeadSeedResult
    fold_predictions: tuple[_FoldPredictions, ...]


def evaluate_cift_probe_head_to_head(config: CiftProbeHeadToHeadConfig) -> CiftProbeHeadToHeadReport:
    _validate_config(config)
    task_rows = build_cift_binary_task_rows(
        artifact=config.artifact,
        definition=cift_binary_task_definition(config.task_name),
    )
    feature_matrices = _feature_matrices_for_config(config=config, selected_indices=task_rows.artifact_indices)
    label_encoding = encode_labels(task_rows.target_labels)
    positive_index = _positive_index(label_encoding.label_to_index, config.positive_label)
    seed_evaluations = tuple(
        _evaluate_seed(
            config=config,
            feature_matrices=feature_matrices,
            labels=label_encoding.encoded_labels,
            groups=task_rows.families,
            example_ids=task_rows.example_ids,
            source_labels=task_rows.source_labels,
            positive_index=positive_index,
            random_seed=random_seed,
        )
        for random_seed in config.random_seeds
    )
    split_manifest_sha256 = _split_manifest_sha256(seed_evaluations)
    training_dataset_sha256 = _training_dataset_sha256(
        config=config,
        feature_matrices=feature_matrices,
        example_ids=task_rows.example_ids,
        labels=task_rows.target_labels,
        families=task_rows.families,
    )
    paper_probe = _probe_run(
        config=config,
        source_report_id=f"{config.report_id}:paper_mlp",
        probe_architecture="mlp_128_64_1",
        training_loss="bce_with_l1_softplus_weight_sparsity",
        training_dataset_sha256=training_dataset_sha256,
        evaluation_split_sha256=split_manifest_sha256,
        metric_value=_mean(tuple(item.seed_result.paper_macro_f1 for item in seed_evaluations)),
        metric_values=tuple(item.seed_result.paper_macro_f1 for item in seed_evaluations),
        false_positive_rate=_mean(tuple(item.seed_result.paper_false_positive_rate for item in seed_evaluations)),
        true_positive_rate=_mean(tuple(item.seed_result.paper_true_positive_rate for item in seed_evaluations)),
        hyperparameter_search_trials=config.paper_hyperparameter_search_trials,
    )
    candidate_probe = _probe_run(
        config=config,
        source_report_id=f"{config.report_id}:linear_logistic_regression",
        probe_architecture="linear_logistic_regression",
        training_loss="regularized_logistic_loss",
        training_dataset_sha256=training_dataset_sha256,
        evaluation_split_sha256=split_manifest_sha256,
        metric_value=_mean(tuple(item.seed_result.candidate_macro_f1 for item in seed_evaluations)),
        metric_values=tuple(item.seed_result.candidate_macro_f1 for item in seed_evaluations),
        false_positive_rate=_mean(tuple(item.seed_result.candidate_false_positive_rate for item in seed_evaluations)),
        true_positive_rate=_mean(tuple(item.seed_result.candidate_true_positive_rate for item in seed_evaluations)),
        hyperparameter_search_trials=config.candidate_hyperparameter_search_trials,
    )
    competition_report = compare_cift_probe_candidates(
        CiftProbeCompetitionConfig(
            report_id=config.report_id,
            paper_probe=paper_probe,
            candidate_probe=candidate_probe,
            higher_is_better=True,
            created_at=config.created_at,
        )
    )
    return CiftProbeHeadToHeadReport(
        competition_report=competition_report,
        feature_representation=config.feature_representation,
        activation_feature_key=config.activation_feature_key,
        source_feature_keys=config.source_feature_keys,
        calibration_source_labels=config.calibration_source_labels,
        ridge=config.ridge,
        seed_results=tuple(item.seed_result for item in seed_evaluations),
    )


def cift_probe_head_to_head_report_to_json(report: CiftProbeHeadToHeadReport) -> dict[str, JsonValue]:
    record = cift_probe_competition_report_to_json(report.competition_report)
    record["feature_representation"] = report.feature_representation
    record["activation_feature_key"] = report.activation_feature_key
    record["source_feature_keys"] = list(report.source_feature_keys)
    record["calibration_source_labels"] = list(report.calibration_source_labels)
    record["ridge"] = report.ridge
    record["seed_results"] = [_seed_result_to_json(seed_result) for seed_result in report.seed_results]
    return record


def write_cift_probe_head_to_head_json(path: Path, report: CiftProbeHeadToHeadReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cift_probe_head_to_head_report_to_json(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evaluate_seed(
    config: CiftProbeHeadToHeadConfig,
    feature_matrices: dict[str, FloatMatrix],
    labels: IntVector,
    groups: tuple[str, ...],
    example_ids: tuple[str, ...],
    source_labels: tuple[str, ...],
    positive_index: int,
    random_seed: int,
) -> _SeedEvaluation:
    fold_predictions: list[_FoldPredictions] = []
    for split in _stratified_group_splits(
        labels=labels,
        groups=groups,
        fold_count=config.fold_count,
        random_seed=random_seed,
    ):
        train_matrix, test_matrix = _fold_feature_matrices(
            config=config,
            feature_matrices=feature_matrices,
            source_labels=source_labels,
            split=split,
        )
        paper = CiftPaperMlpClassifier(
            CiftPaperMlpConfig(
                input_dim=int(train_matrix.shape[1]),
                hidden_layer_sizes=(128, 64),
                learning_rate=config.paper_mlp_learning_rate,
                max_epochs=config.paper_mlp_max_epochs,
                batch_size=config.paper_mlp_batch_size,
                l1_softplus_weight=config.paper_mlp_l1_softplus_weight,
                random_seed=random_seed,
            )
        ).fit(train_matrix, labels[split.train_indices])
        candidate = CiftLinearLogisticClassifier(
            input_dim=int(train_matrix.shape[1]),
            max_epochs=config.linear_max_epochs,
            regularization_c=config.linear_regularization_c,
            random_seed=random_seed,
        ).fit(train_matrix, labels[split.train_indices])
        fold_predictions.append(
            _FoldPredictions(
                fold_index=split.fold_index,
                test_example_ids=tuple(example_ids[index] for index in split.test_indices.tolist()),
                true_labels=labels[split.test_indices],
                paper_predictions=_predictions_from_probabilities(
                    probabilities=paper.predict_proba(test_matrix),
                    positive_index=positive_index,
                    threshold=config.decision_threshold,
                ),
                candidate_predictions=_predictions_from_probabilities(
                    probabilities=candidate.predict_proba(test_matrix),
                    positive_index=positive_index,
                    threshold=config.decision_threshold,
                ),
            )
        )
    return _seed_evaluation(
        random_seed=random_seed,
        fold_predictions=tuple(fold_predictions),
        positive_index=positive_index,
    )


def _feature_matrices_for_config(
    config: CiftProbeHeadToHeadConfig,
    selected_indices: tuple[int, ...],
) -> dict[str, FloatMatrix]:
    matrices: dict[str, FloatMatrix] = {}
    for feature_key in _feature_keys_for_config(config):
        try:
            matrix = cift_feature_matrix_for_rows(
                artifact=config.artifact,
                feature_key=feature_key,
                selected_indices=selected_indices,
            )
        except CiftModelTrainingError as exc:
            raise CiftProbeHeadToHeadError(str(exc)) from exc
        if matrix.ndim != 2:
            raise CiftProbeHeadToHeadError(f"Feature '{feature_key}' must be a 2D matrix.")
        matrices[feature_key] = np.asarray(matrix, dtype=np.float32)
    return matrices


def _feature_keys_for_config(config: CiftProbeHeadToHeadConfig) -> tuple[str, ...]:
    if config.feature_representation == "raw_activation":
        return (config.activation_feature_key,)
    if config.feature_representation == "diagonal_mahalanobis_cci":
        return config.source_feature_keys
    raise CiftProbeHeadToHeadError(f"Unsupported feature_representation '{config.feature_representation}'.")


def _fold_feature_matrices(
    config: CiftProbeHeadToHeadConfig,
    feature_matrices: dict[str, FloatMatrix],
    source_labels: tuple[str, ...],
    split: _GroupedSplit,
) -> tuple[FloatMatrix, FloatMatrix]:
    if config.feature_representation == "raw_activation":
        matrix = feature_matrices[config.activation_feature_key]
        return (
            matrix[split.train_indices].astype(np.float32, copy=False),
            matrix[split.test_indices].astype(np.float32, copy=False),
        )
    if config.feature_representation == "diagonal_mahalanobis_cci":
        return _diagonal_mahalanobis_cci_fold_matrices(
            config=config,
            feature_matrices=feature_matrices,
            source_labels=source_labels,
            split=split,
        )
    raise CiftProbeHeadToHeadError(f"Unsupported feature_representation '{config.feature_representation}'.")


def _diagonal_mahalanobis_cci_fold_matrices(
    config: CiftProbeHeadToHeadConfig,
    feature_matrices: dict[str, FloatMatrix],
    source_labels: tuple[str, ...],
    split: _GroupedSplit,
) -> tuple[FloatMatrix, FloatMatrix]:
    calibration_indices = _calibration_indices(
        source_labels=source_labels,
        train_indices=split.train_indices,
        calibration_source_labels=config.calibration_source_labels,
    )
    if len(calibration_indices) == 0:
        raise CiftProbeHeadToHeadError("CIFT diagonal calibration requires at least one training calibration row.")

    train_scores: list[NDArray[np.float64]] = []
    test_scores: list[NDArray[np.float64]] = []
    calibration_index_vector = np.asarray(calibration_indices, dtype=np.int64)
    for source_feature_key in config.source_feature_keys:
        matrix = feature_matrices[source_feature_key].astype(np.float64)
        calibration_rows = matrix[calibration_index_vector]
        mean = calibration_rows.mean(axis=0)
        variance = calibration_rows.var(axis=0)
        train_scores.append(
            _diagonal_mahalanobis_scores(
                matrix=matrix,
                row_indices=split.train_indices,
                mean=mean,
                variance=variance,
                ridge=config.ridge,
            )
        )
        test_scores.append(
            _diagonal_mahalanobis_scores(
                matrix=matrix,
                row_indices=split.test_indices,
                mean=mean,
                variance=variance,
                ridge=config.ridge,
            )
        )
    return (
        np.stack(train_scores, axis=1).astype(np.float32, copy=False),
        np.stack(test_scores, axis=1).astype(np.float32, copy=False),
    )


def _calibration_indices(
    source_labels: tuple[str, ...],
    train_indices: IntVector,
    calibration_source_labels: tuple[str, ...],
) -> tuple[int, ...]:
    calibration_label_set = set(calibration_source_labels)
    return tuple(int(index) for index in train_indices.tolist() if source_labels[int(index)] in calibration_label_set)


def _diagonal_mahalanobis_scores(
    matrix: NDArray[np.float64],
    row_indices: IntVector,
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    ridge: float,
) -> NDArray[np.float64]:
    rows = matrix[row_indices]
    denominator = variance + ridge
    squared_distance = ((rows - mean) ** 2) / denominator
    return np.sqrt(squared_distance.sum(axis=1))


def _seed_evaluation(
    random_seed: int,
    fold_predictions: tuple[_FoldPredictions, ...],
    positive_index: int,
) -> _SeedEvaluation:
    true_labels = np.concatenate(tuple(item.true_labels for item in fold_predictions)).astype(np.int64, copy=False)
    paper_predictions = np.concatenate(tuple(item.paper_predictions for item in fold_predictions)).astype(
        np.int64,
        copy=False,
    )
    candidate_predictions = np.concatenate(tuple(item.candidate_predictions for item in fold_predictions)).astype(
        np.int64,
        copy=False,
    )
    paper_rates = _classification_rates(
        true_labels=true_labels,
        predictions=paper_predictions,
        positive_index=positive_index,
    )
    candidate_rates = _classification_rates(
        true_labels=true_labels,
        predictions=candidate_predictions,
        positive_index=positive_index,
    )
    return _SeedEvaluation(
        seed_result=CiftProbeHeadToHeadSeedResult(
            random_seed=random_seed,
            fold_count=len(fold_predictions),
            paper_probe_architecture="mlp_128_64_1",
            candidate_probe_architecture="linear_logistic_regression",
            paper_macro_f1=_macro_f1(true_labels=true_labels, predictions=paper_predictions),
            candidate_macro_f1=_macro_f1(true_labels=true_labels, predictions=candidate_predictions),
            paper_false_positive_rate=paper_rates[0],
            paper_true_positive_rate=paper_rates[1],
            candidate_false_positive_rate=candidate_rates[0],
            candidate_true_positive_rate=candidate_rates[1],
        ),
        fold_predictions=fold_predictions,
    )


def _predictions_from_probabilities(
    probabilities: NDArray[np.float64],
    positive_index: int,
    threshold: float,
) -> IntVector:
    positive_probabilities = probabilities[:, positive_index]
    negative_index = 1 - positive_index
    return np.asarray(
        [positive_index if probability >= threshold else negative_index for probability in positive_probabilities],
        dtype=np.int64,
    )


def _classification_rates(true_labels: IntVector, predictions: IntVector, positive_index: int) -> tuple[float, float]:
    negative_index = 1 - positive_index
    true_negative = float(np.sum((true_labels == negative_index) & (predictions == negative_index)))
    false_positive = float(np.sum((true_labels == negative_index) & (predictions == positive_index)))
    false_negative = float(np.sum((true_labels == positive_index) & (predictions == negative_index)))
    true_positive = float(np.sum((true_labels == positive_index) & (predictions == positive_index)))
    false_positive_rate = _safe_ratio(numerator=false_positive, denominator=false_positive + true_negative)
    true_positive_rate = _safe_ratio(numerator=true_positive, denominator=true_positive + false_negative)
    return (false_positive_rate, true_positive_rate)


def _macro_f1(true_labels: IntVector, predictions: IntVector) -> float:
    label_scores = tuple(_label_f1(true_labels=true_labels, predictions=predictions, label=label) for label in (0, 1))
    return _mean(label_scores)


def _label_f1(true_labels: IntVector, predictions: IntVector, label: int) -> float:
    true_positive = float(np.sum((true_labels == label) & (predictions == label)))
    false_positive = float(np.sum((true_labels != label) & (predictions == label)))
    false_negative = float(np.sum((true_labels == label) & (predictions != label)))
    precision = _safe_ratio(numerator=true_positive, denominator=true_positive + false_positive)
    recall = _safe_ratio(numerator=true_positive, denominator=true_positive + false_negative)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _stratified_group_splits(
    labels: IntVector,
    groups: tuple[str, ...],
    fold_count: int,
    random_seed: int,
) -> tuple[_GroupedSplit, ...]:
    _validate_grouped_split_inputs(labels=labels, groups=groups, fold_count=fold_count)
    group_indices = _group_indices(groups)
    group_counts = {
        group: _group_label_counts(labels=labels, indices=indices) for group, indices in group_indices.items()
    }
    rng = np.random.default_rng(random_seed)
    shuffled_groups = list(group_indices)
    rng.shuffle(shuffled_groups)
    ordered_groups = sorted(
        shuffled_groups,
        key=lambda group: (-max(group_counts[group]), -sum(group_counts[group]), group),
    )
    fold_groups: list[list[str]] = [[] for _index in range(fold_count)]
    fold_counts = np.zeros((fold_count, 2), dtype=np.int64)
    fold_totals = np.zeros(fold_count, dtype=np.int64)
    for group in ordered_groups:
        group_count = np.asarray(group_counts[group], dtype=np.int64)
        fold_index = _best_fold_index(fold_counts=fold_counts, fold_totals=fold_totals, group_count=group_count)
        fold_groups[fold_index].append(group)
        fold_counts[fold_index] = fold_counts[fold_index] + group_count
        fold_totals[fold_index] = fold_totals[fold_index] + int(group_count.sum())
    row_indices = np.arange(labels.shape[0], dtype=np.int64)
    splits: list[_GroupedSplit] = []
    for fold_index, groups_for_fold in enumerate(fold_groups, start=1):
        test_indices = np.asarray(
            [index for group in groups_for_fold for index in group_indices[group]],
            dtype=np.int64,
        )
        test_index_set = set(test_indices.tolist())
        train_indices = np.asarray(
            [index for index in row_indices.tolist() if index not in test_index_set],
            dtype=np.int64,
        )
        splits.append(_GroupedSplit(fold_index=fold_index, train_indices=train_indices, test_indices=test_indices))
    return tuple(splits)


def _group_indices(groups: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    indices_by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        indices_by_group.setdefault(group, []).append(index)
    return {group: tuple(indices) for group, indices in indices_by_group.items()}


def _group_label_counts(labels: IntVector, indices: tuple[int, ...]) -> tuple[int, int]:
    selected = labels[list(indices)]
    return (int(np.sum(selected == 0)), int(np.sum(selected == 1)))


def _best_fold_index(fold_counts: IntVector, fold_totals: IntVector, group_count: IntVector) -> int:
    best_index = 0
    best_score: tuple[int, int, int] | None = None
    for fold_index in range(fold_counts.shape[0]):
        candidate_counts = fold_counts.copy()
        candidate_totals = fold_totals.copy()
        candidate_counts[fold_index] = candidate_counts[fold_index] + group_count
        candidate_totals[fold_index] = candidate_totals[fold_index] + int(group_count.sum())
        label_imbalance = int(
            (candidate_counts[:, 0].max() - candidate_counts[:, 0].min())
            + (candidate_counts[:, 1].max() - candidate_counts[:, 1].min())
        )
        total_imbalance = int(candidate_totals.max() - candidate_totals.min())
        score = (label_imbalance, total_imbalance, int(candidate_totals[fold_index]))
        if best_score is None or score < best_score:
            best_score = score
            best_index = fold_index
    return best_index


def _validate_grouped_split_inputs(labels: IntVector, groups: tuple[str, ...], fold_count: int) -> None:
    if labels.ndim != 1:
        raise CiftProbeHeadToHeadError("labels must be a 1D vector.")
    if labels.shape[0] != len(groups):
        raise CiftProbeHeadToHeadError("groups length must match labels length.")
    if set(int(value) for value in labels.tolist()) != {0, 1}:
        raise CiftProbeHeadToHeadError("labels must contain both binary classes 0 and 1.")
    for label in (0, 1):
        label_groups = {
            group for group, encoded_label in zip(groups, labels.tolist(), strict=True) if int(encoded_label) == label
        }
        if len(label_groups) < fold_count:
            raise CiftProbeHeadToHeadError(f"fold_count={fold_count} exceeds group count for label {label}.")


def _probe_run(
    config: CiftProbeHeadToHeadConfig,
    source_report_id: str,
    probe_architecture: str,
    training_loss: str,
    training_dataset_sha256: str,
    evaluation_split_sha256: str,
    metric_value: float,
    metric_values: tuple[float, ...],
    false_positive_rate: float,
    true_positive_rate: float,
    hyperparameter_search_trials: int,
) -> CiftProbeRun:
    return CiftProbeRun(
        source_report_id=source_report_id,
        probe_architecture=probe_architecture,
        training_loss=training_loss,
        training_dataset_id=config.training_dataset_id,
        training_dataset_sha256=training_dataset_sha256,
        task_name=config.task_name,
        evaluation_split_id=config.evaluation_split_id,
        evaluation_split_manifest_id=config.evaluation_split_manifest_id,
        evaluation_split_sha256=evaluation_split_sha256,
        metric_name=config.metric_name,
        metric_value=metric_value,
        metric_confidence_interval_low=min(metric_values),
        metric_confidence_interval_high=max(metric_values),
        random_seeds=config.random_seeds,
        hyperparameter_search_trials=hyperparameter_search_trials,
        operating_threshold=config.decision_threshold,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
    )


def _split_manifest_sha256(seed_evaluations: tuple[_SeedEvaluation, ...]) -> str:
    manifest: list[dict[str, JsonValue]] = []
    for seed_evaluation in seed_evaluations:
        random_seed = seed_evaluation.seed_result.random_seed
        for fold in seed_evaluation.fold_predictions:
            manifest.append(
                {
                    "random_seed": random_seed,
                    "fold_index": fold.fold_index,
                    "test_example_ids": list(fold.test_example_ids),
                }
            )
    return _sha256_json({"splits": manifest})


def _training_dataset_sha256(
    config: CiftProbeHeadToHeadConfig,
    feature_matrices: dict[str, FloatMatrix],
    example_ids: tuple[str, ...],
    labels: tuple[str, ...],
    families: tuple[str, ...],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(config.training_dataset_id.encode("utf-8"))
    hasher.update(config.task_name.encode("utf-8"))
    hasher.update(config.feature_representation.encode("utf-8"))
    hasher.update(config.activation_feature_key.encode("utf-8"))
    hasher.update(json.dumps(list(config.source_feature_keys), sort_keys=True).encode("utf-8"))
    hasher.update(json.dumps(list(config.calibration_source_labels), sort_keys=True).encode("utf-8"))
    hasher.update(str(config.ridge).encode("utf-8"))
    hasher.update(json.dumps(list(example_ids), sort_keys=True).encode("utf-8"))
    hasher.update(json.dumps(list(labels), sort_keys=True).encode("utf-8"))
    hasher.update(json.dumps(list(families), sort_keys=True).encode("utf-8"))
    for feature_key in _feature_keys_for_config(config):
        hasher.update(feature_key.encode("utf-8"))
        hasher.update(np.asarray(feature_matrices[feature_key], dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _seed_result_to_json(seed_result: CiftProbeHeadToHeadSeedResult) -> dict[str, JsonValue]:
    return {
        "random_seed": seed_result.random_seed,
        "fold_count": seed_result.fold_count,
        "paper_probe_architecture": seed_result.paper_probe_architecture,
        "candidate_probe_architecture": seed_result.candidate_probe_architecture,
        "paper_macro_f1": seed_result.paper_macro_f1,
        "candidate_macro_f1": seed_result.candidate_macro_f1,
        "paper_false_positive_rate": seed_result.paper_false_positive_rate,
        "paper_true_positive_rate": seed_result.paper_true_positive_rate,
        "candidate_false_positive_rate": seed_result.candidate_false_positive_rate,
        "candidate_true_positive_rate": seed_result.candidate_true_positive_rate,
    }


def _sha256_json(record: dict[str, JsonValue]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _positive_index(label_to_index: dict[str, int], positive_label: str) -> int:
    index = label_to_index.get(positive_label)
    if index is None:
        raise CiftProbeHeadToHeadError(f"positive_label '{positive_label}' is not present in task labels.")
    return index


def _validate_config(config: CiftProbeHeadToHeadConfig) -> None:
    for field_name, value in (
        ("report_id", config.report_id),
        ("training_dataset_id", config.training_dataset_id),
        ("task_name", config.task_name),
        ("positive_label", config.positive_label),
        ("evaluation_split_id", config.evaluation_split_id),
        ("evaluation_split_manifest_id", config.evaluation_split_manifest_id),
        ("metric_name", config.metric_name),
        ("created_at", config.created_at),
    ):
        if value == "":
            raise CiftProbeHeadToHeadError(f"{field_name} must not be empty.")
    if config.feature_representation not in ("raw_activation", "diagonal_mahalanobis_cci"):
        raise CiftProbeHeadToHeadError(f"Unsupported feature_representation '{config.feature_representation}'.")
    if config.feature_representation == "raw_activation" and config.activation_feature_key == "":
        raise CiftProbeHeadToHeadError("activation_feature_key must not be empty for raw_activation.")
    if len(set(config.source_feature_keys)) != len(config.source_feature_keys):
        raise CiftProbeHeadToHeadError("source_feature_keys must not contain duplicates.")
    if any(feature_key == "" for feature_key in config.source_feature_keys):
        raise CiftProbeHeadToHeadError("source_feature_keys must not contain empty values.")
    if len(set(config.calibration_source_labels)) != len(config.calibration_source_labels):
        raise CiftProbeHeadToHeadError("calibration_source_labels must not contain duplicates.")
    if any(source_label == "" for source_label in config.calibration_source_labels):
        raise CiftProbeHeadToHeadError("calibration_source_labels must not contain empty values.")
    if config.feature_representation == "diagonal_mahalanobis_cci":
        if len(config.source_feature_keys) == 0:
            raise CiftProbeHeadToHeadError("source_feature_keys must not be empty for diagonal_mahalanobis_cci.")
        if len(config.calibration_source_labels) == 0:
            raise CiftProbeHeadToHeadError("calibration_source_labels must not be empty for diagonal_mahalanobis_cci.")
    if len(config.random_seeds) < 3:
        raise CiftProbeHeadToHeadError("random_seeds must include at least three repeated-evaluation seeds.")
    if len(set(config.random_seeds)) != len(config.random_seeds):
        raise CiftProbeHeadToHeadError("random_seeds must not contain duplicate seeds.")
    for field_name, value in (
        ("decision_threshold", config.decision_threshold),
        ("linear_regularization_c", config.linear_regularization_c),
        ("paper_mlp_learning_rate", config.paper_mlp_learning_rate),
        ("paper_mlp_l1_softplus_weight", config.paper_mlp_l1_softplus_weight),
        ("ridge", config.ridge),
    ):
        if not math.isfinite(value):
            raise CiftProbeHeadToHeadError(f"{field_name} must be finite.")
    if config.decision_threshold < 0.0 or config.decision_threshold > 1.0:
        raise CiftProbeHeadToHeadError("decision_threshold must be in [0.0, 1.0].")
    if config.fold_count < 2:
        raise CiftProbeHeadToHeadError("fold_count must be at least 2.")
    for field_name, value in (
        ("linear_max_epochs", config.linear_max_epochs),
        ("paper_mlp_max_epochs", config.paper_mlp_max_epochs),
        ("paper_mlp_batch_size", config.paper_mlp_batch_size),
        ("paper_hyperparameter_search_trials", config.paper_hyperparameter_search_trials),
        ("candidate_hyperparameter_search_trials", config.candidate_hyperparameter_search_trials),
    ):
        if value < 1:
            raise CiftProbeHeadToHeadError(f"{field_name} must be at least 1.")
    if config.linear_regularization_c <= 0.0:
        raise CiftProbeHeadToHeadError("linear_regularization_c must be greater than 0.")
    if config.paper_mlp_learning_rate <= 0.0:
        raise CiftProbeHeadToHeadError("paper_mlp_learning_rate must be greater than 0.")
    if config.paper_mlp_l1_softplus_weight < 0.0:
        raise CiftProbeHeadToHeadError("paper_mlp_l1_softplus_weight must be nonnegative.")
    if config.ridge <= 0.0:
        raise CiftProbeHeadToHeadError("ridge must be greater than 0.")
