from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.probe import JsonValue, encode_labels


IntVector: TypeAlias = NDArray[np.int64]


class TextBaselineTrainingError(ValueError):
    """Raised when the text baseline cannot be trained with the provided artifact/configuration."""


@dataclass(frozen=True)
class TextBaselineTrainingConfig:
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    lowercase: bool
    min_df: int
    ngram_range: tuple[int, int]


@dataclass(frozen=True)
class FoldTextBaselineMetrics:
    fold_index: int
    accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TextBaselineTrainingReport:
    baseline_name: str
    source_model_id: str
    source_revision: str
    label_names: tuple[str, ...]
    example_count: int
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    lowercase: bool
    min_df: int
    ngram_range: tuple[int, int]
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[FoldTextBaselineMetrics, ...]


def _validate_text_baseline_inputs(
    texts: tuple[str, ...],
    encoded_labels: IntVector,
    config: TextBaselineTrainingConfig,
) -> None:
    if len(texts) == 0:
        raise TextBaselineTrainingError("Cannot train a text baseline with no texts.")
    if config.fold_count < 2:
        raise TextBaselineTrainingError("fold_count must be at least 2.")
    if config.max_iter < 1:
        raise TextBaselineTrainingError("max_iter must be at least 1.")
    if config.regularization_c <= 0:
        raise TextBaselineTrainingError("regularization_c must be greater than 0.")
    if config.min_df < 1:
        raise TextBaselineTrainingError("min_df must be at least 1.")
    if config.ngram_range[0] < 1:
        raise TextBaselineTrainingError("ngram_range lower bound must be at least 1.")
    if config.ngram_range[1] < config.ngram_range[0]:
        raise TextBaselineTrainingError("ngram_range upper bound must be greater than or equal to the lower bound.")
    if len(texts) != encoded_labels.shape[0]:
        raise TextBaselineTrainingError(f"Text count {len(texts)} does not match label count {encoded_labels.shape[0]}.")

    label_counts = np.bincount(encoded_labels)
    smallest_class_count = int(label_counts.min())
    if smallest_class_count < config.fold_count:
        raise TextBaselineTrainingError(
            f"fold_count={config.fold_count} exceeds the smallest class size {smallest_class_count}."
        )


def _matrix_to_tuple(matrix: NDArray[np.int64]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: tuple[float, ...]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def train_text_baseline_report(
    artifact: ActivationArtifact,
    config: TextBaselineTrainingConfig,
) -> TextBaselineTrainingReport:
    label_encoding = encode_labels(artifact["labels"])
    texts = artifact["texts"]
    encoded_labels = label_encoding.encoded_labels
    _validate_text_baseline_inputs(texts, encoded_labels, config)

    splitter = StratifiedKFold(
        n_splits=config.fold_count,
        shuffle=True,
        random_state=config.random_seed,
    )
    label_indices = np.arange(len(label_encoding.label_names), dtype=np.int64)
    confusion_total = np.zeros((len(label_encoding.label_names), len(label_encoding.label_names)), dtype=np.int64)
    text_array = np.asarray(texts, dtype=object)
    folds: list[FoldTextBaselineMetrics] = []

    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(text_array, encoded_labels), start=1):
        classifier: Pipeline = make_pipeline(
            TfidfVectorizer(
                lowercase=config.lowercase,
                min_df=config.min_df,
                ngram_range=config.ngram_range,
            ),
            LogisticRegression(
                C=config.regularization_c,
                class_weight="balanced",
                max_iter=config.max_iter,
                random_state=config.random_seed,
            ),
        )
        classifier.fit(text_array[train_indices].tolist(), encoded_labels[train_indices])
        predictions = classifier.predict(text_array[test_indices].tolist())

        fold_confusion = confusion_matrix(
            encoded_labels[test_indices],
            predictions,
            labels=label_indices,
        ).astype(np.int64, copy=False)
        confusion_total += fold_confusion
        folds.append(
            FoldTextBaselineMetrics(
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
    metadata = artifact["metadata"]

    return TextBaselineTrainingReport(
        baseline_name="tfidf_logistic_regression",
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        label_names=label_encoding.label_names,
        example_count=len(texts),
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        lowercase=config.lowercase,
        min_df=config.min_df,
        ngram_range=config.ngram_range,
        accuracy_mean=_mean(accuracies),
        accuracy_std=_std(accuracies),
        macro_f1_mean=_mean(macro_f1_scores),
        macro_f1_std=_std(macro_f1_scores),
        confusion_matrix=_matrix_to_tuple(confusion_total),
        folds=tuple(folds),
    )


def _fold_metrics_to_json(fold: FoldTextBaselineMetrics) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "accuracy": fold.accuracy,
        "macro_f1": fold.macro_f1,
        "confusion_matrix": [list(row) for row in fold.confusion_matrix],
    }


def text_baseline_report_to_json(report: TextBaselineTrainingReport) -> dict[str, JsonValue]:
    return {
        "baseline_name": report.baseline_name,
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "label_names": list(report.label_names),
        "example_count": report.example_count,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "lowercase": report.lowercase,
        "min_df": report.min_df,
        "ngram_range": list(report.ngram_range),
        "accuracy_mean": report.accuracy_mean,
        "accuracy_std": report.accuracy_std,
        "macro_f1_mean": report.macro_f1_mean,
        "macro_f1_std": report.macro_f1_std,
        "confusion_matrix": [list(row) for row in report.confusion_matrix],
        "folds": [_fold_metrics_to_json(fold) for fold in report.folds],
    }


def write_text_baseline_report_json(path: Path, report: TextBaselineTrainingReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(text_baseline_report_to_json(report), file, indent=2)
        file.write("\n")
