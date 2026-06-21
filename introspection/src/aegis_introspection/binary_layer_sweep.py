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
class BinaryLayerSweepFeatureReport:
    rank: int
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
class BinaryLayerSweepReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    task_name: str
    task_description: str
    label_names: tuple[str, ...]
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    reference_feature_key: str
    best_feature_key: str
    feature_count: int
    features: tuple[BinaryLayerSweepFeatureReport, ...]


def _definition_by_name(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _feature_report_from_method(
    rank: int,
    method_report: BinaryMethodReport,
) -> BinaryLayerSweepFeatureReport:
    return BinaryLayerSweepFeatureReport(
        rank=rank,
        feature_name=method_report.feature_name,
        label_names=method_report.label_names,
        example_count=method_report.example_count,
        accuracy_mean=method_report.accuracy_mean,
        accuracy_std=method_report.accuracy_std,
        macro_f1_mean=method_report.macro_f1_mean,
        macro_f1_std=method_report.macro_f1_std,
        confusion_matrix=method_report.confusion_matrix,
        folds=method_report.folds,
    )


def evaluate_grouped_binary_layer_sweep(
    artifact: ActivationArtifact,
    task_name: str,
    config: BinaryTaskConfig,
) -> BinaryLayerSweepReport:
    if config.activation_feature_key not in artifact["features"]:
        raise BinaryTaskError(
            f"Reference activation feature '{config.activation_feature_key}' is not present in the artifact."
        )

    definition = _definition_by_name(task_name)
    dataset = build_binary_task_dataset(artifact, definition)
    raw_feature_reports = []

    for feature_name in artifact["features"]:
        feature_config = replace(config, activation_feature_key=feature_name)
        raw_feature_reports.append(
            evaluate_grouped_activation_method(
                artifact=artifact,
                dataset=dataset,
                config=feature_config,
            )
        )

    sorted_method_reports = tuple(
        sorted(
            raw_feature_reports,
            key=lambda report: (-report.macro_f1_mean, -report.accuracy_mean, report.feature_name),
        )
    )
    feature_reports = tuple(
        _feature_report_from_method(
            rank=rank,
            method_report=method_report,
        )
        for rank, method_report in enumerate(sorted_method_reports, start=1)
    )
    best_feature_key = feature_reports[0].feature_name
    metadata = artifact["metadata"]

    return BinaryLayerSweepReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        task_name=dataset.name,
        task_description=dataset.description,
        label_names=feature_reports[0].label_names,
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        reference_feature_key=config.activation_feature_key,
        best_feature_key=best_feature_key,
        feature_count=len(feature_reports),
        features=feature_reports,
    )


def _fold_to_json(fold: BinaryFoldMetrics) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "accuracy": fold.accuracy,
        "macro_f1": fold.macro_f1,
        "confusion_matrix": [list(row) for row in fold.confusion_matrix],
    }


def _feature_to_json(feature: BinaryLayerSweepFeatureReport) -> dict[str, JsonValue]:
    return {
        "rank": feature.rank,
        "feature_name": feature.feature_name,
        "label_names": list(feature.label_names),
        "example_count": feature.example_count,
        "accuracy_mean": feature.accuracy_mean,
        "accuracy_std": feature.accuracy_std,
        "macro_f1_mean": feature.macro_f1_mean,
        "macro_f1_std": feature.macro_f1_std,
        "confusion_matrix": [list(row) for row in feature.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in feature.folds],
    }


def binary_layer_sweep_report_to_json(report: BinaryLayerSweepReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "task_name": report.task_name,
        "task_description": report.task_description,
        "label_names": list(report.label_names),
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "reference_feature_key": report.reference_feature_key,
        "best_feature_key": report.best_feature_key,
        "feature_count": report.feature_count,
        "features": [_feature_to_json(feature) for feature in report.features],
    }


def write_binary_layer_sweep_json(path: Path, report: BinaryLayerSweepReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(binary_layer_sweep_report_to_json(report), file, indent=2)
        file.write("\n")


def render_binary_layer_sweep_markdown(report: BinaryLayerSweepReport) -> str:
    lines = [
        "# Binary Layer Sweep",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Reference feature: `{report.reference_feature_key}`",
        f"- Best feature: `{report.best_feature_key}`",
        f"- Feature count: `{report.feature_count}`",
        "",
        "## Feature Ranking",
        "",
        "| Rank | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for feature in report.features:
        reference_marker = " (reference)" if feature.feature_name == report.reference_feature_key else ""
        lines.append(
            f"| {feature.rank} | `{feature.feature_name}`{reference_marker} | "
            f"{feature.macro_f1_mean:.4f} | {feature.accuracy_mean:.4f} | "
            f"{feature.macro_f1_std:.4f} | {feature.accuracy_std:.4f} |"
        )

    top_features = report.features[:10]
    lines.extend(
        [
            "",
            "## Top Confusion Matrices",
            "",
        ]
    )
    for feature in top_features:
        lines.append(f"### {feature.rank}. {feature.feature_name}")
        lines.append("")
        lines.append("```text")
        for row in feature.confusion_matrix:
            lines.append(str(list(row)))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def write_binary_layer_sweep_markdown(path: Path, report: BinaryLayerSweepReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_binary_layer_sweep_markdown(report), encoding="utf-8")
