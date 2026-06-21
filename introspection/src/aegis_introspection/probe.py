from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis_introspection.artifacts import ActivationArtifact


FloatMatrix: TypeAlias = NDArray[np.float32]
IntVector: TypeAlias = NDArray[np.int64]
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class ProbeTrainingError(ValueError):
    """Raised when a probe cannot be trained with the provided artifact/configuration."""


@dataclass(frozen=True)
class ProbeTrainingConfig:
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float


@dataclass(frozen=True)
class LabelEncoding:
    label_names: tuple[str, ...]
    label_to_index: dict[str, int]
    encoded_labels: IntVector


@dataclass(frozen=True)
class FoldProbeMetrics:
    fold_index: int
    accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FeatureProbeReport:
    feature_key: str
    example_count: int
    feature_count: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[FoldProbeMetrics, ...]


@dataclass(frozen=True)
class ProbeTrainingReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    label_names: tuple[str, ...]
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    best_feature_key: str
    features: tuple[FeatureProbeReport, ...]


def encode_labels(labels: tuple[str, ...]) -> LabelEncoding:
    if len(labels) == 0:
        raise ProbeTrainingError("Cannot encode an empty label set.")

    label_names = tuple(sorted(set(labels)))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    encoded = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)
    return LabelEncoding(
        label_names=label_names,
        label_to_index=label_to_index,
        encoded_labels=encoded,
    )


def tensor_to_float_matrix(tensor: torch.Tensor) -> FloatMatrix:
    if tensor.ndim != 2:
        raise ProbeTrainingError(f"Expected a 2D feature tensor, received shape {tuple(tensor.shape)}.")
    matrix = tensor.detach().cpu().to(dtype=torch.float32).numpy()
    return matrix.astype(np.float32, copy=False)


def _validate_cross_validation_inputs(
    matrix: FloatMatrix,
    encoded_labels: IntVector,
    config: ProbeTrainingConfig,
) -> None:
    if config.fold_count < 2:
        raise ProbeTrainingError("fold_count must be at least 2.")
    if config.max_iter < 1:
        raise ProbeTrainingError("max_iter must be at least 1.")
    if config.regularization_c <= 0:
        raise ProbeTrainingError("regularization_c must be greater than 0.")
    if matrix.shape[0] != encoded_labels.shape[0]:
        raise ProbeTrainingError(
            f"Feature matrix has {matrix.shape[0]} rows, but labels have {encoded_labels.shape[0]} rows."
        )

    label_counts = np.bincount(encoded_labels)
    smallest_class_count = int(label_counts.min())
    if smallest_class_count < config.fold_count:
        raise ProbeTrainingError(
            f"fold_count={config.fold_count} exceeds the smallest class size {smallest_class_count}."
        )


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: tuple[float, ...]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def train_feature_probe(
    feature_key: str,
    feature_tensor: torch.Tensor,
    label_encoding: LabelEncoding,
    config: ProbeTrainingConfig,
) -> FeatureProbeReport:
    matrix = tensor_to_float_matrix(feature_tensor)
    encoded_labels = label_encoding.encoded_labels
    _validate_cross_validation_inputs(matrix, encoded_labels, config)

    splitter = StratifiedKFold(
        n_splits=config.fold_count,
        shuffle=True,
        random_state=config.random_seed,
    )
    label_indices = np.arange(len(label_encoding.label_names), dtype=np.int64)

    folds: list[FoldProbeMetrics] = []
    confusion_total = np.zeros((len(label_encoding.label_names), len(label_encoding.label_names)), dtype=np.int64)

    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(matrix, encoded_labels), start=1):
        classifier: Pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config.regularization_c,
                class_weight="balanced",
                max_iter=config.max_iter,
                random_state=config.random_seed,
            ),
        )
        classifier.fit(matrix[train_indices], encoded_labels[train_indices])
        predictions = classifier.predict(matrix[test_indices])

        fold_confusion = confusion_matrix(
            encoded_labels[test_indices],
            predictions,
            labels=label_indices,
        ).astype(np.int64, copy=False)
        confusion_total += fold_confusion

        folds.append(
            FoldProbeMetrics(
                fold_index=fold_index,
                accuracy=float(accuracy_score(encoded_labels[test_indices], predictions)),
                macro_f1=float(
                    f1_score(
                        encoded_labels[test_indices],
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

    return FeatureProbeReport(
        feature_key=feature_key,
        example_count=int(matrix.shape[0]),
        feature_count=int(matrix.shape[1]),
        accuracy_mean=_mean(accuracies),
        accuracy_std=_std(accuracies),
        macro_f1_mean=_mean(macro_f1_scores),
        macro_f1_std=_std(macro_f1_scores),
        confusion_matrix=_matrix_to_tuple(confusion_total),
        folds=tuple(folds),
    )


def train_probe_report(artifact: ActivationArtifact, config: ProbeTrainingConfig) -> ProbeTrainingReport:
    label_encoding = encode_labels(artifact["labels"])
    feature_reports = tuple(
        train_feature_probe(
            feature_key=feature_key,
            feature_tensor=feature_tensor,
            label_encoding=label_encoding,
            config=config,
        )
        for feature_key, feature_tensor in artifact["features"].items()
    )

    if len(feature_reports) == 0:
        raise ProbeTrainingError("Cannot train a probe report with no feature matrices.")

    best_feature = max(feature_reports, key=lambda report: (report.macro_f1_mean, report.accuracy_mean))
    metadata = artifact["metadata"]
    return ProbeTrainingReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        label_names=label_encoding.label_names,
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        best_feature_key=best_feature.feature_key,
        features=feature_reports,
    )


def _fold_metrics_to_json(fold: FoldProbeMetrics) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "accuracy": fold.accuracy,
        "macro_f1": fold.macro_f1,
        "confusion_matrix": [list(row) for row in fold.confusion_matrix],
    }


def _feature_report_to_json(feature: FeatureProbeReport) -> dict[str, JsonValue]:
    return {
        "feature_key": feature.feature_key,
        "example_count": feature.example_count,
        "feature_count": feature.feature_count,
        "accuracy_mean": feature.accuracy_mean,
        "accuracy_std": feature.accuracy_std,
        "macro_f1_mean": feature.macro_f1_mean,
        "macro_f1_std": feature.macro_f1_std,
        "confusion_matrix": [list(row) for row in feature.confusion_matrix],
        "folds": [_fold_metrics_to_json(fold) for fold in feature.folds],
    }


def probe_report_to_json(report: ProbeTrainingReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "label_names": list(report.label_names),
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "best_feature_key": report.best_feature_key,
        "features": [_feature_report_to_json(feature) for feature in report.features],
    }


def write_probe_report_json(path: Path, report: ProbeTrainingReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(probe_report_to_json(report), file, indent=2)
        file.write("\n")
