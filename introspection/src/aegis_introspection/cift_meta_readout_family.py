from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import BinaryMethodName, BinaryTaskConfig, BinaryTaskError, EvaluationStrategy
from aegis_introspection.cift_meta_regularization_sweep import (
    CiftMetaRegularizationDatasetVariantReport,
    CiftMetaRegularizationDataset,
    CiftMetaRegularizationVariantSummary,
    CiftMetaRegularizationVariant,
    compare_cift_meta_regularization_sweep,
)
from aegis_introspection.probe import JsonValue


CiftMetaReadoutFamily: TypeAlias = Literal[
    "full_dual_readout",
    "final_token_only",
    "mean_pool_only",
]


@dataclass(frozen=True)
class CiftMetaReadoutFamilyDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaReadoutFamilyVariant:
    variant_id: str
    feature_name: str
    source_family: CiftMetaReadoutFamily
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    ridge: float
    risk_label: str
    inner_fold_count: int
    meta_regularization_c: float


@dataclass(frozen=True)
class CiftMetaReadoutFamilyDatasetVariantReport:
    dataset_id: str
    variant_id: str
    feature_name: str
    source_family: CiftMetaReadoutFamily
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    meta_regularization_c: float
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int
    reference_accuracy: float
    candidate_accuracy: float


@dataclass(frozen=True)
class CiftMetaReadoutFamilyVariantSummary:
    variant_id: str
    feature_name: str
    source_family: CiftMetaReadoutFamily
    source_feature_keys: tuple[str, ...]
    calibration_source_labels: tuple[str, ...]
    meta_regularization_c: float
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
class CiftMetaReadoutFamilyReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    inner_fold_count: int
    source_regularization_c: float
    meta_regularization_c: float
    random_seed: int
    max_iter: int
    task_name: str
    method_name: BinaryMethodName
    baseline_feature_key: str
    dataset_count: int
    variant_count: int
    best_variant_summary: CiftMetaReadoutFamilyVariantSummary
    variant_summaries: tuple[CiftMetaReadoutFamilyVariantSummary, ...]
    dataset_variants: tuple[CiftMetaReadoutFamilyDatasetVariantReport, ...]


def _validate_variant(variant: CiftMetaReadoutFamilyVariant) -> None:
    if variant.variant_id == "":
        raise BinaryTaskError("CIFT readout-family variant id must not be empty.")
    if variant.feature_name == "":
        raise BinaryTaskError(f"CIFT readout-family variant '{variant.variant_id}' feature name must not be empty.")
    if variant.source_family not in ("full_dual_readout", "final_token_only", "mean_pool_only"):
        raise BinaryTaskError(
            f"CIFT readout-family variant '{variant.variant_id}' has unsupported source family "
            f"'{variant.source_family}'."
        )
    if len(variant.source_feature_keys) == 0:
        raise BinaryTaskError(f"CIFT readout-family variant '{variant.variant_id}' requires source features.")
    if len(set(variant.source_feature_keys)) != len(variant.source_feature_keys):
        raise BinaryTaskError(f"CIFT readout-family variant '{variant.variant_id}' source features must be unique.")
    if len(variant.calibration_source_labels) == 0:
        raise BinaryTaskError(f"CIFT readout-family variant '{variant.variant_id}' requires calibration labels.")
    if variant.ridge <= 0:
        raise BinaryTaskError(f"CIFT readout-family variant '{variant.variant_id}' ridge must be greater than 0.")
    if variant.risk_label == "":
        raise BinaryTaskError(f"CIFT readout-family variant '{variant.variant_id}' risk label must not be empty.")
    if variant.inner_fold_count < 2:
        raise BinaryTaskError(f"CIFT readout-family variant '{variant.variant_id}' inner_fold_count must be at least 2.")
    if variant.meta_regularization_c <= 0:
        raise BinaryTaskError(
            f"CIFT readout-family variant '{variant.variant_id}' meta_regularization_c must be greater than 0."
        )


def _validate_inputs(
    datasets: tuple[CiftMetaReadoutFamilyDataset, ...],
    baseline_feature_key: str,
    variants: tuple[CiftMetaReadoutFamilyVariant, ...],
) -> None:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one CIFT readout-family dataset is required.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT readout-family baseline feature key must not be empty.")
    for index, dataset in enumerate(datasets):
        if dataset.dataset_id == "":
            raise BinaryTaskError(f"CIFT readout-family dataset {index} has an empty dataset id.")
    if len(variants) == 0:
        raise BinaryTaskError("At least one CIFT readout-family variant is required.")
    for variant in variants:
        _validate_variant(variant)
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT readout-family variant ids must be unique.")
    if len({variant.feature_name for variant in variants}) != len(variants):
        raise BinaryTaskError("CIFT readout-family feature names must be unique.")
    if len({variant.meta_regularization_c for variant in variants}) != 1:
        raise BinaryTaskError("CIFT readout-family variants must share one meta_regularization_c.")


def _regularization_datasets(
    datasets: tuple[CiftMetaReadoutFamilyDataset, ...],
) -> tuple[CiftMetaRegularizationDataset, ...]:
    return tuple(
        CiftMetaRegularizationDataset(dataset_id=dataset.dataset_id, artifact=dataset.artifact)
        for dataset in datasets
    )


def _regularization_variant(variant: CiftMetaReadoutFamilyVariant) -> CiftMetaRegularizationVariant:
    return CiftMetaRegularizationVariant(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        ridge=variant.ridge,
        risk_label=variant.risk_label,
        inner_fold_count=variant.inner_fold_count,
        meta_regularization_c=variant.meta_regularization_c,
    )


def _summary(
    variant: CiftMetaReadoutFamilyVariant,
    summary: CiftMetaRegularizationVariantSummary,
) -> CiftMetaReadoutFamilyVariantSummary:
    return CiftMetaReadoutFamilyVariantSummary(
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_family=variant.source_family,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        meta_regularization_c=variant.meta_regularization_c,
        dataset_count=summary.dataset_count,
        reference_error_count=summary.reference_error_count,
        candidate_error_count=summary.candidate_error_count,
        fixed_error_count=summary.fixed_error_count,
        persistent_error_count=summary.persistent_error_count,
        introduced_error_count=summary.introduced_error_count,
        net_error_delta=summary.net_error_delta,
        mean_candidate_accuracy=summary.mean_candidate_accuracy,
        min_candidate_accuracy=summary.min_candidate_accuracy,
    )


def _dataset_variant(
    variant: CiftMetaReadoutFamilyVariant,
    dataset_variant: CiftMetaRegularizationDatasetVariantReport,
) -> CiftMetaReadoutFamilyDatasetVariantReport:
    return CiftMetaReadoutFamilyDatasetVariantReport(
        dataset_id=dataset_variant.dataset_id,
        variant_id=variant.variant_id,
        feature_name=variant.feature_name,
        source_family=variant.source_family,
        source_feature_keys=variant.source_feature_keys,
        calibration_source_labels=variant.calibration_source_labels,
        meta_regularization_c=variant.meta_regularization_c,
        reference_error_count=dataset_variant.reference_error_count,
        candidate_error_count=dataset_variant.candidate_error_count,
        fixed_error_count=dataset_variant.fixed_error_count,
        persistent_error_count=dataset_variant.persistent_error_count,
        introduced_error_count=dataset_variant.introduced_error_count,
        net_error_delta=dataset_variant.net_error_delta,
        reference_accuracy=dataset_variant.reference_accuracy,
        candidate_accuracy=dataset_variant.candidate_accuracy,
    )


def _best_summary(
    summaries: tuple[CiftMetaReadoutFamilyVariantSummary, ...],
) -> CiftMetaReadoutFamilyVariantSummary:
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


def compare_cift_meta_readout_family(
    datasets: tuple[CiftMetaReadoutFamilyDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variants: tuple[CiftMetaReadoutFamilyVariant, ...],
    binary_config: BinaryTaskConfig,
) -> CiftMetaReadoutFamilyReport:
    _validate_inputs(datasets=datasets, baseline_feature_key=baseline_feature_key, variants=variants)
    variant_by_id = {variant.variant_id: variant for variant in variants}
    regularization_report = compare_cift_meta_regularization_sweep(
        datasets=_regularization_datasets(datasets),
        task_name=task_name,
        baseline_feature_key=baseline_feature_key,
        variants=tuple(_regularization_variant(variant) for variant in variants),
        binary_config=binary_config,
    )
    summaries = tuple(
        _summary(variant=variant_by_id[summary.variant_id], summary=summary)
        for summary in regularization_report.variant_summaries
    )
    dataset_variants = tuple(
        _dataset_variant(variant=variant_by_id[dataset_variant.variant_id], dataset_variant=dataset_variant)
        for dataset_variant in regularization_report.dataset_variants
    )
    return CiftMetaReadoutFamilyReport(
        source_model_id=regularization_report.source_model_id,
        source_revision=regularization_report.source_revision,
        source_selected_device=regularization_report.source_selected_device,
        evaluation_strategy=regularization_report.evaluation_strategy,
        fold_count=regularization_report.fold_count,
        inner_fold_count=regularization_report.inner_fold_count,
        source_regularization_c=regularization_report.source_regularization_c,
        meta_regularization_c=variants[0].meta_regularization_c,
        random_seed=regularization_report.random_seed,
        max_iter=regularization_report.max_iter,
        task_name=regularization_report.task_name,
        method_name=regularization_report.method_name,
        baseline_feature_key=regularization_report.baseline_feature_key,
        dataset_count=regularization_report.dataset_count,
        variant_count=regularization_report.variant_count,
        best_variant_summary=_best_summary(summaries),
        variant_summaries=summaries,
        dataset_variants=dataset_variants,
    )


def _summary_to_json(summary: CiftMetaReadoutFamilyVariantSummary) -> dict[str, JsonValue]:
    return {
        "variant_id": summary.variant_id,
        "feature_name": summary.feature_name,
        "source_family": summary.source_family,
        "source_feature_keys": list(summary.source_feature_keys),
        "calibration_source_labels": list(summary.calibration_source_labels),
        "meta_regularization_c": summary.meta_regularization_c,
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


def _dataset_variant_to_json(report: CiftMetaReadoutFamilyDatasetVariantReport) -> dict[str, JsonValue]:
    return {
        "dataset_id": report.dataset_id,
        "variant_id": report.variant_id,
        "feature_name": report.feature_name,
        "source_family": report.source_family,
        "source_feature_keys": list(report.source_feature_keys),
        "calibration_source_labels": list(report.calibration_source_labels),
        "meta_regularization_c": report.meta_regularization_c,
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "fixed_error_count": report.fixed_error_count,
        "persistent_error_count": report.persistent_error_count,
        "introduced_error_count": report.introduced_error_count,
        "net_error_delta": report.net_error_delta,
        "reference_accuracy": report.reference_accuracy,
        "candidate_accuracy": report.candidate_accuracy,
    }


def cift_meta_readout_family_to_json(report: CiftMetaReadoutFamilyReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "inner_fold_count": report.inner_fold_count,
        "source_regularization_c": report.source_regularization_c,
        "meta_regularization_c": report.meta_regularization_c,
        "random_seed": report.random_seed,
        "max_iter": report.max_iter,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "baseline_feature_key": report.baseline_feature_key,
        "dataset_count": report.dataset_count,
        "variant_count": report.variant_count,
        "best_variant_summary": _summary_to_json(report.best_variant_summary),
        "variant_summaries": [_summary_to_json(summary) for summary in report.variant_summaries],
        "dataset_variants": [_dataset_variant_to_json(dataset_variant) for dataset_variant in report.dataset_variants],
    }


def write_cift_meta_readout_family_json(path: Path, report: CiftMetaReadoutFamilyReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_readout_family_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def render_cift_meta_readout_family_markdown(report: CiftMetaReadoutFamilyReport) -> str:
    lines = [
        "# CIFT Meta-Head Readout Family Comparison",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        f"- Baseline feature: `{report.baseline_feature_key}`",
        f"- Source-head C: `{report.source_regularization_c}`",
        f"- Meta-head C: `{report.meta_regularization_c}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Variant count: `{report.variant_count}`",
        f"- Best variant: `{report.best_variant_summary.variant_id}`",
        "",
        "## Variant Summary",
        "",
        (
            "| Variant | Source Family | Meta C | Source Count | Calibration Labels | Candidate Errors | "
            "Fixed | Persistent | Introduced | Net Error Delta | Mean Accuracy |"
        ),
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.variant_summaries:
        lines.append(
            f"| `{summary.variant_id}` | "
            f"`{summary.source_family}` | "
            f"{summary.meta_regularization_c:.4g} | "
            f"{len(summary.source_feature_keys)} | "
            f"`{_joined(summary.calibration_source_labels)}` | "
            f"{summary.candidate_error_count} | "
            f"{summary.fixed_error_count} | "
            f"{summary.persistent_error_count} | "
            f"{summary.introduced_error_count} | "
            f"{summary.net_error_delta} | "
            f"{summary.mean_candidate_accuracy:.4f} |"
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
    for dataset_variant in report.dataset_variants:
        lines.append(
            f"| `{dataset_variant.dataset_id}` | "
            f"`{dataset_variant.variant_id}` | "
            f"{dataset_variant.candidate_error_count} | "
            f"{dataset_variant.fixed_error_count} | "
            f"{dataset_variant.persistent_error_count} | "
            f"{dataset_variant.introduced_error_count} | "
            f"{dataset_variant.candidate_accuracy:.4f} |"
        )
    return "\n".join(lines)


def write_cift_meta_readout_family_markdown(path: Path, report: CiftMetaReadoutFamilyReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_readout_family_markdown(report), encoding="utf-8")
