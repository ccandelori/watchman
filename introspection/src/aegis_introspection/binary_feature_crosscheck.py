from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryFoldMetrics,
    BinaryMethodReport,
    BinaryTaskConfig,
    BinaryTaskDefinition,
    BinaryTaskError,
    EvaluationStrategy,
    build_binary_task_dataset,
    default_binary_task_definitions,
    evaluate_grouped_activation_method,
)
from aegis_introspection.probe import JsonValue


@dataclass(frozen=True)
class FeatureCrosscheckDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class FeatureCrosscheckMetric:
    feature_key: str
    label_names: tuple[str, ...]
    example_count: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[BinaryFoldMetrics, ...]


@dataclass(frozen=True)
class DatasetFeatureCrosscheck:
    dataset_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    reference: FeatureCrosscheckMetric
    candidate: FeatureCrosscheckMetric
    macro_f1_delta: float
    accuracy_delta: float
    winning_feature_key: str


@dataclass(frozen=True)
class FeatureCrosscheckReport:
    evaluation_strategy: EvaluationStrategy
    task_name: str
    task_description: str
    reference_feature_key: str
    candidate_feature_key: str
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    dataset_count: int
    candidate_win_count: int
    reference_win_count: int
    tie_count: int
    datasets: tuple[DatasetFeatureCrosscheck, ...]


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _metric_from_method(method: BinaryMethodReport) -> FeatureCrosscheckMetric:
    return FeatureCrosscheckMetric(
        feature_key=method.feature_name,
        label_names=method.label_names,
        example_count=method.example_count,
        accuracy_mean=method.accuracy_mean,
        accuracy_std=method.accuracy_std,
        macro_f1_mean=method.macro_f1_mean,
        macro_f1_std=method.macro_f1_std,
        confusion_matrix=method.confusion_matrix,
        folds=method.folds,
    )


def _evaluate_feature(
    artifact: ActivationArtifact,
    task_name: str,
    feature_key: str,
    config: BinaryTaskConfig,
) -> BinaryMethodReport:
    definition = _task_definition(task_name)
    dataset = build_binary_task_dataset(artifact, definition)
    feature_config = replace(config, activation_feature_key=feature_key)
    return evaluate_grouped_activation_method(
        artifact=artifact,
        dataset=dataset,
        config=feature_config,
    )


def _winning_feature_key(
    reference: FeatureCrosscheckMetric,
    candidate: FeatureCrosscheckMetric,
) -> str:
    reference_score = (reference.macro_f1_mean, reference.accuracy_mean)
    candidate_score = (candidate.macro_f1_mean, candidate.accuracy_mean)
    if candidate_score > reference_score:
        return candidate.feature_key
    if reference_score > candidate_score:
        return reference.feature_key
    return "tie"


def _compare_dataset(
    dataset: FeatureCrosscheckDataset,
    task_name: str,
    reference_feature_key: str,
    candidate_feature_key: str,
    config: BinaryTaskConfig,
) -> DatasetFeatureCrosscheck:
    reference = _metric_from_method(
        _evaluate_feature(
            artifact=dataset.artifact,
            task_name=task_name,
            feature_key=reference_feature_key,
            config=config,
        )
    )
    candidate = _metric_from_method(
        _evaluate_feature(
            artifact=dataset.artifact,
            task_name=task_name,
            feature_key=candidate_feature_key,
            config=config,
        )
    )
    metadata = dataset.artifact["metadata"]
    return DatasetFeatureCrosscheck(
        dataset_id=dataset.dataset_id,
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        reference=reference,
        candidate=candidate,
        macro_f1_delta=candidate.macro_f1_mean - reference.macro_f1_mean,
        accuracy_delta=candidate.accuracy_mean - reference.accuracy_mean,
        winning_feature_key=_winning_feature_key(reference, candidate),
    )


def compare_grouped_binary_features(
    datasets: tuple[FeatureCrosscheckDataset, ...],
    task_name: str,
    reference_feature_key: str,
    candidate_feature_key: str,
    config: BinaryTaskConfig,
) -> FeatureCrosscheckReport:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one dataset is required for feature crosscheck.")

    definition = _task_definition(task_name)
    dataset_reports = tuple(
        _compare_dataset(
            dataset=dataset,
            task_name=task_name,
            reference_feature_key=reference_feature_key,
            candidate_feature_key=candidate_feature_key,
            config=config,
        )
        for dataset in datasets
    )
    candidate_win_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == candidate_feature_key)
    reference_win_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == reference_feature_key)
    tie_count = sum(1 for dataset in dataset_reports if dataset.winning_feature_key == "tie")

    return FeatureCrosscheckReport(
        evaluation_strategy="stratified_group_kfold",
        task_name=definition.name,
        task_description=definition.description,
        reference_feature_key=reference_feature_key,
        candidate_feature_key=candidate_feature_key,
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        dataset_count=len(dataset_reports),
        candidate_win_count=candidate_win_count,
        reference_win_count=reference_win_count,
        tie_count=tie_count,
        datasets=dataset_reports,
    )


def _fold_to_json(fold: BinaryFoldMetrics) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "accuracy": fold.accuracy,
        "macro_f1": fold.macro_f1,
        "confusion_matrix": [list(row) for row in fold.confusion_matrix],
    }


def _metric_to_json(metric: FeatureCrosscheckMetric) -> dict[str, JsonValue]:
    return {
        "feature_key": metric.feature_key,
        "label_names": list(metric.label_names),
        "example_count": metric.example_count,
        "accuracy_mean": metric.accuracy_mean,
        "accuracy_std": metric.accuracy_std,
        "macro_f1_mean": metric.macro_f1_mean,
        "macro_f1_std": metric.macro_f1_std,
        "confusion_matrix": [list(row) for row in metric.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in metric.folds],
    }


def _dataset_to_json(dataset: DatasetFeatureCrosscheck) -> dict[str, JsonValue]:
    return {
        "dataset_id": dataset.dataset_id,
        "source_model_id": dataset.source_model_id,
        "source_revision": dataset.source_revision,
        "source_selected_device": dataset.source_selected_device,
        "reference": _metric_to_json(dataset.reference),
        "candidate": _metric_to_json(dataset.candidate),
        "macro_f1_delta": dataset.macro_f1_delta,
        "accuracy_delta": dataset.accuracy_delta,
        "winning_feature_key": dataset.winning_feature_key,
    }


def feature_crosscheck_report_to_json(report: FeatureCrosscheckReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "reference_feature_key": report.reference_feature_key,
        "candidate_feature_key": report.candidate_feature_key,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "dataset_count": report.dataset_count,
        "candidate_win_count": report.candidate_win_count,
        "reference_win_count": report.reference_win_count,
        "tie_count": report.tie_count,
        "datasets": [_dataset_to_json(dataset) for dataset in report.datasets],
    }


def write_feature_crosscheck_json(path: Path, report: FeatureCrosscheckReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(feature_crosscheck_report_to_json(report), file, indent=2)
        file.write("\n")


def render_feature_crosscheck_markdown(report: FeatureCrosscheckReport) -> str:
    lines = [
        "# Binary Feature Crosscheck",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Reference feature: `{report.reference_feature_key}`",
        f"- Candidate feature: `{report.candidate_feature_key}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Candidate wins: `{report.candidate_win_count}`",
        f"- Reference wins: `{report.reference_win_count}`",
        f"- Ties: `{report.tie_count}`",
        "",
        "## Dataset Comparison",
        "",
        "| Dataset | Reference Macro F1 | Candidate Macro F1 | Delta Macro F1 | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in report.datasets:
        lines.append(
            f"| `{dataset.dataset_id}` | "
            f"{dataset.reference.macro_f1_mean:.4f} | "
            f"{dataset.candidate.macro_f1_mean:.4f} | "
            f"{dataset.macro_f1_delta:+.4f} | "
            f"`{dataset.winning_feature_key}` |"
        )

    lines.extend(
        [
            "",
            "| Dataset | Reference Accuracy | Candidate Accuracy | Delta Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in report.datasets:
        lines.append(
            f"| `{dataset.dataset_id}` | "
            f"{dataset.reference.accuracy_mean:.4f} | "
            f"{dataset.candidate.accuracy_mean:.4f} | "
            f"{dataset.accuracy_delta:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrices",
            "",
        ]
    )
    for dataset in report.datasets:
        lines.append(f"### {dataset.dataset_id} / {dataset.reference.feature_key}")
        lines.append("")
        lines.append("```text")
        for row in dataset.reference.confusion_matrix:
            lines.append(str(list(row)))
        lines.append("```")
        lines.append("")
        lines.append(f"### {dataset.dataset_id} / {dataset.candidate.feature_key}")
        lines.append("")
        lines.append("```text")
        for row in dataset.candidate.confusion_matrix:
            lines.append(str(list(row)))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def write_feature_crosscheck_markdown(path: Path, report: FeatureCrosscheckReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_feature_crosscheck_markdown(report), encoding="utf-8")
