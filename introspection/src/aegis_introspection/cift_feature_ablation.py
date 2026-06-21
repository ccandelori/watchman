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
class CiftFeatureAblationVariant:
    variant_id: str
    feature_key: str


@dataclass(frozen=True)
class CiftFeatureAblationVariantReport:
    rank: int
    variant_id: str
    feature_key: str
    is_baseline: bool
    label_names: tuple[str, ...]
    example_count: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    folds: tuple[BinaryFoldMetrics, ...]


@dataclass(frozen=True)
class CiftFeatureAblationReport:
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
    baseline_variant_id: str
    baseline_feature_key: str
    best_variant_id: str
    best_feature_key: str
    variant_count: int
    variants: tuple[CiftFeatureAblationVariantReport, ...]


def evaluate_grouped_cift_feature_ablation(
    artifact: ActivationArtifact,
    task_name: str,
    variants: tuple[CiftFeatureAblationVariant, ...],
    baseline_variant_id: str,
    config: BinaryTaskConfig,
) -> CiftFeatureAblationReport:
    _validate_variants(variants=variants, baseline_variant_id=baseline_variant_id)
    definition = _definition_by_name(task_name)
    dataset = build_binary_task_dataset(artifact, definition)
    raw_reports = tuple(
        _evaluate_variant(
            artifact=artifact,
            variant=variant,
            dataset_definition=definition,
            config=config,
        )
        for variant in variants
    )
    variant_by_id = {variant.variant_id: variant for variant in variants}
    sorted_reports = tuple(
        sorted(
            raw_reports,
            key=lambda item: (-item[1].macro_f1_mean, -item[1].accuracy_mean, item[0].variant_id),
        )
    )
    variant_reports = tuple(
        _variant_report_from_method(
            rank=rank,
            variant=variant,
            method_report=method_report,
            baseline_variant_id=baseline_variant_id,
        )
        for rank, (variant, method_report) in enumerate(sorted_reports, start=1)
    )
    best_variant = variant_reports[0]
    baseline_variant = variant_by_id[baseline_variant_id]
    metadata = artifact["metadata"]

    return CiftFeatureAblationReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        task_name=dataset.name,
        task_description=dataset.description,
        label_names=variant_reports[0].label_names,
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        baseline_variant_id=baseline_variant_id,
        baseline_feature_key=baseline_variant.feature_key,
        best_variant_id=best_variant.variant_id,
        best_feature_key=best_variant.feature_key,
        variant_count=len(variant_reports),
        variants=variant_reports,
    )


def cift_feature_ablation_report_to_json(report: CiftFeatureAblationReport) -> dict[str, JsonValue]:
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
        "baseline_variant_id": report.baseline_variant_id,
        "baseline_feature_key": report.baseline_feature_key,
        "best_variant_id": report.best_variant_id,
        "best_feature_key": report.best_feature_key,
        "variant_count": report.variant_count,
        "variants": [_variant_to_json(variant) for variant in report.variants],
    }


def write_cift_feature_ablation_json(path: Path, report: CiftFeatureAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_feature_ablation_report_to_json(report), file, indent=2)
        file.write("\n")


def render_cift_feature_ablation_markdown(report: CiftFeatureAblationReport) -> str:
    lines = [
        "# CIFT Feature Ablation",
        "",
        "## Source",
        "",
        f"- Model: `{report.source_model_id}`",
        f"- Revision: `{report.source_revision}`",
        f"- Extraction device: `{report.source_selected_device}`",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Baseline variant: `{report.baseline_variant_id}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Best variant: `{report.best_variant_id}`",
        f"- Best feature: `{report.best_feature_key}`",
        f"- Variant count: `{report.variant_count}`",
        "",
        "## Variant Ranking",
        "",
        "| Rank | Variant | Feature | Macro F1 | Accuracy | Macro F1 Std | Accuracy Std |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for variant in report.variants:
        baseline_marker = " (baseline)" if variant.is_baseline else ""
        lines.append(
            f"| {variant.rank} | `{variant.variant_id}`{baseline_marker} | `{variant.feature_key}` | "
            f"{variant.macro_f1_mean:.4f} | {variant.accuracy_mean:.4f} | "
            f"{variant.macro_f1_std:.4f} | {variant.accuracy_std:.4f} |"
        )
    lines.extend(["", "## Top Confusion Matrices", ""])
    for variant in report.variants[:10]:
        lines.append(f"### {variant.rank}. {variant.variant_id}")
        lines.append("")
        lines.append("```text")
        for row in variant.confusion_matrix:
            lines.append(str(list(row)))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def write_cift_feature_ablation_markdown(path: Path, report: CiftFeatureAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_feature_ablation_markdown(report), encoding="utf-8")


def _evaluate_variant(
    artifact: ActivationArtifact,
    variant: CiftFeatureAblationVariant,
    dataset_definition: BinaryTaskDefinition,
    config: BinaryTaskConfig,
) -> tuple[CiftFeatureAblationVariant, BinaryMethodReport]:
    dataset = build_binary_task_dataset(artifact, dataset_definition)
    variant_config = replace(config, activation_feature_key=variant.feature_key)
    return (
        variant,
        evaluate_grouped_activation_method(
            artifact=artifact,
            dataset=dataset,
            config=variant_config,
        ),
    )


def _variant_report_from_method(
    rank: int,
    variant: CiftFeatureAblationVariant,
    method_report: BinaryMethodReport,
    baseline_variant_id: str,
) -> CiftFeatureAblationVariantReport:
    return CiftFeatureAblationVariantReport(
        rank=rank,
        variant_id=variant.variant_id,
        feature_key=method_report.feature_name,
        is_baseline=variant.variant_id == baseline_variant_id,
        label_names=method_report.label_names,
        example_count=method_report.example_count,
        accuracy_mean=method_report.accuracy_mean,
        accuracy_std=method_report.accuracy_std,
        macro_f1_mean=method_report.macro_f1_mean,
        macro_f1_std=method_report.macro_f1_std,
        confusion_matrix=method_report.confusion_matrix,
        folds=method_report.folds,
    )


def _definition_by_name(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _validate_variants(
    variants: tuple[CiftFeatureAblationVariant, ...],
    baseline_variant_id: str,
) -> None:
    if len(variants) == 0:
        raise BinaryTaskError("At least one CIFT feature ablation variant is required.")
    if baseline_variant_id == "":
        raise BinaryTaskError("baseline_variant_id must not be empty.")
    variant_ids = tuple(variant.variant_id for variant in variants)
    if len(set(variant_ids)) != len(variant_ids):
        raise BinaryTaskError("CIFT feature ablation variant ids must be unique.")
    feature_keys = tuple(variant.feature_key for variant in variants)
    if len(set(feature_keys)) != len(feature_keys):
        raise BinaryTaskError("CIFT feature ablation feature keys must be unique.")
    for variant in variants:
        if variant.variant_id == "":
            raise BinaryTaskError("CIFT feature ablation variant id must not be empty.")
        if variant.feature_key == "":
            raise BinaryTaskError(f"CIFT feature ablation variant '{variant.variant_id}' feature key must not be empty.")
    if baseline_variant_id not in variant_ids:
        raise BinaryTaskError(f"Baseline variant '{baseline_variant_id}' is not present in variants.")


def _fold_to_json(fold: BinaryFoldMetrics) -> dict[str, JsonValue]:
    return {
        "fold_index": fold.fold_index,
        "accuracy": fold.accuracy,
        "macro_f1": fold.macro_f1,
        "confusion_matrix": [list(row) for row in fold.confusion_matrix],
    }


def _variant_to_json(variant: CiftFeatureAblationVariantReport) -> dict[str, JsonValue]:
    return {
        "rank": variant.rank,
        "variant_id": variant.variant_id,
        "feature_key": variant.feature_key,
        "is_baseline": variant.is_baseline,
        "label_names": list(variant.label_names),
        "example_count": variant.example_count,
        "accuracy_mean": variant.accuracy_mean,
        "accuracy_std": variant.accuracy_std,
        "macro_f1_mean": variant.macro_f1_mean,
        "macro_f1_std": variant.macro_f1_std,
        "confusion_matrix": [list(row) for row in variant.confusion_matrix],
        "folds": [_fold_to_json(fold) for fold in variant.folds],
    }
