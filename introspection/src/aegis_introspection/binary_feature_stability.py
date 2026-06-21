from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from aegis_introspection.binary_feature_crosscheck import (
    FeatureCrosscheckDataset,
    FeatureCrosscheckMetric,
)
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
class DatasetFeatureStability:
    dataset_id: str
    source_model_id: str
    source_revision: str
    source_selected_device: str
    metrics: tuple[FeatureCrosscheckMetric, ...]
    winning_feature_keys: tuple[str, ...]


@dataclass(frozen=True)
class FeatureStabilitySummary:
    rank: int
    feature_key: str
    win_count: int
    mean_macro_f1: float
    mean_accuracy: float
    min_macro_f1: float
    max_macro_f1: float
    macro_f1_range: float


@dataclass(frozen=True)
class FeatureStabilityReport:
    evaluation_strategy: EvaluationStrategy
    task_name: str
    task_description: str
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    dataset_count: int
    feature_count: int
    feature_keys: tuple[str, ...]
    feature_summaries: tuple[FeatureStabilitySummary, ...]
    datasets: tuple[DatasetFeatureStability, ...]


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
    dataset: FeatureCrosscheckDataset,
    definition: BinaryTaskDefinition,
    feature_key: str,
    config: BinaryTaskConfig,
) -> FeatureCrosscheckMetric:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    feature_config = replace(config, activation_feature_key=feature_key)
    return _metric_from_method(
        evaluate_grouped_activation_method(
            artifact=dataset.artifact,
            dataset=task_dataset,
            config=feature_config,
        )
    )


def _winning_feature_keys(metrics: tuple[FeatureCrosscheckMetric, ...]) -> tuple[str, ...]:
    if len(metrics) == 0:
        raise BinaryTaskError("At least one feature metric is required.")
    best_score = max((metric.macro_f1_mean, metric.accuracy_mean) for metric in metrics)
    return tuple(
        metric.feature_key
        for metric in metrics
        if (metric.macro_f1_mean, metric.accuracy_mean) == best_score
    )


def _compare_dataset(
    dataset: FeatureCrosscheckDataset,
    definition: BinaryTaskDefinition,
    feature_keys: tuple[str, ...],
    config: BinaryTaskConfig,
) -> DatasetFeatureStability:
    metrics = tuple(
        _evaluate_feature(
            dataset=dataset,
            definition=definition,
            feature_key=feature_key,
            config=config,
        )
        for feature_key in feature_keys
    )
    metadata = dataset.artifact["metadata"]
    return DatasetFeatureStability(
        dataset_id=dataset.dataset_id,
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        metrics=metrics,
        winning_feature_keys=_winning_feature_keys(metrics),
    )


def _metric_by_feature(
    dataset: DatasetFeatureStability,
    feature_key: str,
) -> FeatureCrosscheckMetric:
    matches = tuple(metric for metric in dataset.metrics if metric.feature_key == feature_key)
    if len(matches) != 1:
        raise BinaryTaskError(
            f"Expected exactly one metric for feature '{feature_key}' in dataset '{dataset.dataset_id}'."
        )
    return matches[0]


def _feature_summary(
    feature_key: str,
    datasets: tuple[DatasetFeatureStability, ...],
    rank: int,
) -> FeatureStabilitySummary:
    metrics = tuple(_metric_by_feature(dataset, feature_key) for dataset in datasets)
    macro_f1_values = tuple(metric.macro_f1_mean for metric in metrics)
    accuracy_values = tuple(metric.accuracy_mean for metric in metrics)
    min_macro_f1 = min(macro_f1_values)
    max_macro_f1 = max(macro_f1_values)
    return FeatureStabilitySummary(
        rank=rank,
        feature_key=feature_key,
        win_count=sum(1 for dataset in datasets if feature_key in dataset.winning_feature_keys),
        mean_macro_f1=sum(macro_f1_values) / len(macro_f1_values),
        mean_accuracy=sum(accuracy_values) / len(accuracy_values),
        min_macro_f1=min_macro_f1,
        max_macro_f1=max_macro_f1,
        macro_f1_range=max_macro_f1 - min_macro_f1,
    )


def _rank_feature_summaries(
    feature_keys: tuple[str, ...],
    datasets: tuple[DatasetFeatureStability, ...],
) -> tuple[FeatureStabilitySummary, ...]:
    unranked = tuple(_feature_summary(feature_key, datasets, 0) for feature_key in feature_keys)
    sorted_summaries = sorted(
        unranked,
        key=lambda summary: (
            -summary.mean_macro_f1,
            -summary.min_macro_f1,
            -summary.win_count,
            summary.macro_f1_range,
            summary.feature_key,
        ),
    )
    return tuple(
        FeatureStabilitySummary(
            rank=index,
            feature_key=summary.feature_key,
            win_count=summary.win_count,
            mean_macro_f1=summary.mean_macro_f1,
            mean_accuracy=summary.mean_accuracy,
            min_macro_f1=summary.min_macro_f1,
            max_macro_f1=summary.max_macro_f1,
            macro_f1_range=summary.macro_f1_range,
        )
        for index, summary in enumerate(sorted_summaries, start=1)
    )


def compare_grouped_binary_feature_stability(
    datasets: tuple[FeatureCrosscheckDataset, ...],
    task_name: str,
    feature_keys: tuple[str, ...],
    config: BinaryTaskConfig,
) -> FeatureStabilityReport:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one dataset is required for feature stability comparison.")
    if len(feature_keys) == 0:
        raise BinaryTaskError("At least one feature is required for feature stability comparison.")
    if len(set(feature_keys)) != len(feature_keys):
        raise BinaryTaskError("Feature stability comparison requires unique feature keys.")

    definition = _task_definition(task_name)
    dataset_reports = tuple(
        _compare_dataset(
            dataset=dataset,
            definition=definition,
            feature_keys=feature_keys,
            config=config,
        )
        for dataset in datasets
    )
    return FeatureStabilityReport(
        evaluation_strategy="stratified_group_kfold",
        task_name=definition.name,
        task_description=definition.description,
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        dataset_count=len(dataset_reports),
        feature_count=len(feature_keys),
        feature_keys=feature_keys,
        feature_summaries=_rank_feature_summaries(feature_keys, dataset_reports),
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


def _dataset_to_json(dataset: DatasetFeatureStability) -> dict[str, JsonValue]:
    return {
        "dataset_id": dataset.dataset_id,
        "source_model_id": dataset.source_model_id,
        "source_revision": dataset.source_revision,
        "source_selected_device": dataset.source_selected_device,
        "winning_feature_keys": list(dataset.winning_feature_keys),
        "metrics": [_metric_to_json(metric) for metric in dataset.metrics],
    }


def _summary_to_json(summary: FeatureStabilitySummary) -> dict[str, JsonValue]:
    return {
        "rank": summary.rank,
        "feature_key": summary.feature_key,
        "win_count": summary.win_count,
        "mean_macro_f1": summary.mean_macro_f1,
        "mean_accuracy": summary.mean_accuracy,
        "min_macro_f1": summary.min_macro_f1,
        "max_macro_f1": summary.max_macro_f1,
        "macro_f1_range": summary.macro_f1_range,
    }


def feature_stability_report_to_json(report: FeatureStabilityReport) -> dict[str, JsonValue]:
    return {
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "dataset_count": report.dataset_count,
        "feature_count": report.feature_count,
        "feature_keys": list(report.feature_keys),
        "feature_summaries": [_summary_to_json(summary) for summary in report.feature_summaries],
        "datasets": [_dataset_to_json(dataset) for dataset in report.datasets],
    }


def write_feature_stability_json(path: Path, report: FeatureStabilityReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(feature_stability_report_to_json(report), file, indent=2)
        file.write("\n")


def _winner_text(dataset: DatasetFeatureStability) -> str:
    return ", ".join(f"`{feature_key}`" for feature_key in dataset.winning_feature_keys)


def _metric_cell(
    dataset: DatasetFeatureStability,
    feature_key: str,
    metric_name: str,
) -> str:
    metric = _metric_by_feature(dataset, feature_key)
    if metric_name == "macro_f1":
        return f"{metric.macro_f1_mean:.4f}"
    if metric_name == "accuracy":
        return f"{metric.accuracy_mean:.4f}"
    raise BinaryTaskError(f"Unsupported feature stability metric '{metric_name}'.")


def _render_dataset_metric_table(
    title: str,
    datasets: tuple[DatasetFeatureStability, ...],
    feature_keys: tuple[str, ...],
    metric_name: str,
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Dataset | Winner | " + " | ".join(f"`{feature_key}`" for feature_key in feature_keys) + " |",
        "|---|---|" + "---:|" * len(feature_keys),
    ]
    for dataset in datasets:
        metric_cells = " | ".join(_metric_cell(dataset, feature_key, metric_name) for feature_key in feature_keys)
        lines.append(f"| `{dataset.dataset_id}` | {_winner_text(dataset)} | {metric_cells} |")
    lines.append("")
    return lines


def render_feature_stability_markdown(report: FeatureStabilityReport) -> str:
    lines = [
        "# Binary Feature Stability",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Feature count: `{report.feature_count}`",
        f"- Fold count: `{report.fold_count}`",
        "",
        "## Feature Summary",
        "",
        "| Rank | Feature | Wins | Mean Macro F1 | Mean Accuracy | Min Macro F1 | Max Macro F1 | Macro F1 Range |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.feature_summaries:
        lines.append(
            f"| {summary.rank} | `{summary.feature_key}` | {summary.win_count} | "
            f"{summary.mean_macro_f1:.4f} | {summary.mean_accuracy:.4f} | "
            f"{summary.min_macro_f1:.4f} | {summary.max_macro_f1:.4f} | "
            f"{summary.macro_f1_range:.4f} |"
        )
    lines.append("")
    lines.extend(_render_dataset_metric_table("Macro F1 by Dataset", report.datasets, report.feature_keys, "macro_f1"))
    lines.extend(_render_dataset_metric_table("Accuracy by Dataset", report.datasets, report.feature_keys, "accuracy"))
    return "\n".join(lines)


def write_feature_stability_markdown(path: Path, report: FeatureStabilityReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_feature_stability_markdown(report), encoding="utf-8")
