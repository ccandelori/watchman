from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryMethodName, BinaryTaskConfig, EvaluationStrategy
from aegis_introspection.cift_meta_head import CiftMetaDecisionRule, CiftMetaHeadVariant
from aegis_introspection.cift_meta_residuals import (
    CiftMetaResidualDataset,
    CiftMetaResidualSuiteReport,
    compare_cift_meta_residual_suite,
)
from aegis_introspection.probe import JsonValue


@dataclass(frozen=True)
class CiftMetaAblationDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaAblationVariant:
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    risk_label: str
    inner_fold_count: int
    decision_rule: CiftMetaDecisionRule


@dataclass(frozen=True)
class CiftMetaAblationDatasetVariantReport:
    dataset_id: str
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    decision_rule: CiftMetaDecisionRule
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int
    reference_accuracy: float
    candidate_accuracy: float


@dataclass(frozen=True)
class CiftMetaAblationVariantSummary:
    variant_id: str
    feature_name: str
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    decision_rule: CiftMetaDecisionRule
    dataset_count: int
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int
    mean_candidate_accuracy: float
    min_candidate_accuracy: float


@dataclass(frozen=True)
class CiftMetaAblationReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    task_name: str
    method_name: BinaryMethodName
    baseline_feature_key: str
    dataset_count: int
    variant_count: int
    best_variant_summary: CiftMetaAblationVariantSummary
    variant_summaries: tuple[CiftMetaAblationVariantSummary, ...]
    dataset_variants: tuple[CiftMetaAblationDatasetVariantReport, ...]


def _head_variant(variant: CiftMetaAblationVariant) -> CiftMetaHeadVariant:
    return CiftMetaHeadVariant(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        ridge=variant.ridge,
        risk_label=variant.risk_label,
        inner_fold_count=variant.inner_fold_count,
        decision_rule=variant.decision_rule,
    )


def _residual_datasets(
    datasets: tuple[CiftMetaAblationDataset, ...],
) -> tuple[CiftMetaResidualDataset, ...]:
    return tuple(
        CiftMetaResidualDataset(dataset_id=dataset.dataset_id, artifact=dataset.artifact)
        for dataset in datasets
    )


def _dataset_variant_report(
    variant: CiftMetaAblationVariant,
    residual_report: CiftMetaResidualSuiteReport,
) -> tuple[CiftMetaAblationDatasetVariantReport, ...]:
    reports: list[CiftMetaAblationDatasetVariantReport] = []
    for item in residual_report.comparisons:
        comparison = item.comparison
        reports.append(
            CiftMetaAblationDatasetVariantReport(
                dataset_id=item.dataset_id,
                variant_id=variant.variant_id,
                feature_name=variant.feature_name,
                source_feature_keys=variant.source_feature_keys,
                calibration_source_labels=variant.calibration_source_labels,
                decision_rule=variant.decision_rule,
                reference_error_count=comparison.reference_error_count,
                candidate_error_count=comparison.candidate_error_count,
                fixed_error_count=comparison.fixed_error_count,
                persistent_error_count=comparison.persistent_error_count,
                introduced_error_count=comparison.introduced_error_count,
                net_error_delta=comparison.introduced_error_count - comparison.fixed_error_count,
                reference_accuracy=comparison.reference_accuracy,
                candidate_accuracy=comparison.candidate_accuracy,
            )
        )
    return tuple(reports)


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _variant_summary(
    variant: CiftMetaAblationVariant,
    dataset_reports: tuple[CiftMetaAblationDatasetVariantReport, ...],
) -> CiftMetaAblationVariantSummary:
    candidate_accuracies = tuple(report.candidate_accuracy for report in dataset_reports)
    fixed_error_count = sum(report.fixed_error_count for report in dataset_reports)
    introduced_error_count = sum(report.introduced_error_count for report in dataset_reports)
    return CiftMetaAblationVariantSummary(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        decision_rule=variant.decision_rule,
        dataset_count=len({report.dataset_id for report in dataset_reports}),
        reference_error_count=sum(report.reference_error_count for report in dataset_reports),
        candidate_error_count=sum(report.candidate_error_count for report in dataset_reports),
        fixed_error_count=fixed_error_count,
        persistent_error_count=sum(report.persistent_error_count for report in dataset_reports),
        introduced_error_count=introduced_error_count,
        net_error_delta=introduced_error_count - fixed_error_count,
        mean_candidate_accuracy=_mean(candidate_accuracies),
        min_candidate_accuracy=min(candidate_accuracies),
    )


def _best_summary(
    summaries: tuple[CiftMetaAblationVariantSummary, ...],
) -> CiftMetaAblationVariantSummary:
    return min(
        summaries,
        key=lambda summary: (
            summary.net_error_delta,
            summary.introduced_error_count,
            summary.candidate_error_count,
            -summary.fixed_error_count,
            -summary.mean_candidate_accuracy,
        ),
    )


def compare_cift_meta_ablation(
    datasets: tuple[CiftMetaAblationDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variants: tuple[CiftMetaAblationVariant, ...],
    binary_config: BinaryTaskConfig,
) -> CiftMetaAblationReport:
    residual_datasets = _residual_datasets(datasets)
    all_dataset_reports: list[CiftMetaAblationDatasetVariantReport] = []
    summaries: list[CiftMetaAblationVariantSummary] = []
    first_residual_report: CiftMetaResidualSuiteReport | None = None

    for variant in variants:
        residual_report = compare_cift_meta_residual_suite(
            datasets=residual_datasets,
            task_name=task_name,
            baseline_feature_key=baseline_feature_key,
            variant=_head_variant(variant),
            binary_config=binary_config,
        )
        if first_residual_report is None:
            first_residual_report = residual_report
        dataset_reports = _dataset_variant_report(variant, residual_report)
        all_dataset_reports.extend(dataset_reports)
        summaries.append(_variant_summary(variant, dataset_reports))

    if first_residual_report is None:
        raise ValueError("At least one CIFT meta ablation variant is required.")

    summary_tuple = tuple(summaries)
    return CiftMetaAblationReport(
        source_model_id=first_residual_report.source_model_id,
        source_revision=first_residual_report.source_revision,
        source_selected_device=first_residual_report.source_selected_device,
        evaluation_strategy=first_residual_report.evaluation_strategy,
        fold_count=first_residual_report.fold_count,
        random_seed=first_residual_report.random_seed,
        regularization_c=first_residual_report.regularization_c,
        max_iter=first_residual_report.max_iter,
        task_name=first_residual_report.task_name,
        method_name=first_residual_report.method_name,
        baseline_feature_key=baseline_feature_key,
        dataset_count=len({dataset.dataset_id for dataset in datasets}),
        variant_count=len(variants),
        best_variant_summary=_best_summary(summary_tuple),
        variant_summaries=summary_tuple,
        dataset_variants=tuple(all_dataset_reports),
    )


def _summary_to_json(summary: CiftMetaAblationVariantSummary) -> dict[str, JsonValue]:
    return {
        "variant_id": summary.variant_id,
        "feature_name": summary.feature_name,
        "source_feature_keys": list(summary.source_feature_keys),
        "calibration_source_labels": list(summary.calibration_source_labels),
        "decision_rule": summary.decision_rule,
        "dataset_count": summary.dataset_count,
        "reference_error_count": summary.reference_error_count,
        "candidate_error_count": summary.candidate_error_count,
        "fixed_error_count": summary.fixed_error_count,
        "persistent_error_count": summary.persistent_error_count,
        "introduced_error_count": summary.introduced_error_count,
        "net_error_delta": summary.net_error_delta,
        "mean_candidate_accuracy": summary.mean_candidate_accuracy,
        "min_candidate_accuracy": summary.min_candidate_accuracy,
    }


def _dataset_variant_to_json(report: CiftMetaAblationDatasetVariantReport) -> dict[str, JsonValue]:
    return {
        "dataset_id": report.dataset_id,
        "variant_id": report.variant_id,
        "feature_name": report.feature_name,
        "source_feature_keys": list(report.source_feature_keys),
        "calibration_source_labels": list(report.calibration_source_labels),
        "decision_rule": report.decision_rule,
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "fixed_error_count": report.fixed_error_count,
        "persistent_error_count": report.persistent_error_count,
        "introduced_error_count": report.introduced_error_count,
        "net_error_delta": report.net_error_delta,
        "reference_accuracy": report.reference_accuracy,
        "candidate_accuracy": report.candidate_accuracy,
    }


def cift_meta_ablation_report_to_json(report: CiftMetaAblationReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "baseline_feature_key": report.baseline_feature_key,
        "dataset_count": report.dataset_count,
        "variant_count": report.variant_count,
        "best_variant_summary": _summary_to_json(report.best_variant_summary),
        "variant_summaries": [_summary_to_json(summary) for summary in report.variant_summaries],
        "dataset_variants": [_dataset_variant_to_json(item) for item in report.dataset_variants],
    }


def write_cift_meta_ablation_json(path: Path, report: CiftMetaAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_ablation_report_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def render_cift_meta_ablation_markdown(report: CiftMetaAblationReport) -> str:
    lines = [
        "# CIFT Meta-Head Ablation",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Variant count: `{report.variant_count}`",
        f"- Best variant: `{report.best_variant_summary.variant_id}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Calibration Labels | Source Count | Decision Rule | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.variant_summaries:
        lines.append(
            f"| `{summary.variant_id}` | `{_joined(summary.calibration_source_labels)}` | "
            f"{len(summary.source_feature_keys)} | `{summary.decision_rule}` | "
            f"{summary.candidate_error_count} | {summary.fixed_error_count} | "
            f"{summary.persistent_error_count} | {summary.introduced_error_count} | "
            f"{summary.net_error_delta} | {summary.mean_candidate_accuracy:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Variant Results",
            "",
            "| Dataset | Variant | Candidate Errors | Fixed | Persistent | Introduced | Candidate Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report.dataset_variants:
        lines.append(
            f"| `{item.dataset_id}` | `{item.variant_id}` | "
            f"{item.candidate_error_count} | {item.fixed_error_count} | "
            f"{item.persistent_error_count} | {item.introduced_error_count} | "
            f"{item.candidate_accuracy:.4f} |"
        )

    return "\n".join(lines)


def write_cift_meta_ablation_markdown(path: Path, report: CiftMetaAblationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_ablation_markdown(report), encoding="utf-8")
