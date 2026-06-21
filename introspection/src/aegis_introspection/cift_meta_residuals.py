from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from aegis_introspection.artifacts import ActivationArtifact
from aegis_introspection.binary_tasks import (
    BinaryMethodName,
    BinaryTaskConfig,
    BinaryTaskDataset,
    BinaryTaskDefinition,
    BinaryTaskError,
    EvaluationStrategy,
    build_binary_task_dataset,
    default_binary_task_definitions,
)
from aegis_introspection.cift_meta_head import CiftMetaHeadVariant, collect_grouped_cift_meta_head_predictions
from aegis_introspection.error_analysis import (
    BinaryErrorAnalysisReport,
    BinaryMethodErrorAnalysis,
    BinaryTaskErrorAnalysis,
    collect_grouped_activation_predictions,
)
from aegis_introspection.probe import JsonValue
from aegis_introspection.residual_error_comparison import (
    DatasetResidualErrorComparison,
    ResidualErrorComparisonReport,
    compare_binary_error_residuals,
    residual_error_comparison_report_to_json,
)


@dataclass(frozen=True)
class CiftMetaResidualDataset:
    dataset_id: str
    artifact: ActivationArtifact


@dataclass(frozen=True)
class CiftMetaResidualSuiteReport:
    source_model_id: str
    source_revision: str
    source_selected_device: str
    evaluation_strategy: EvaluationStrategy
    fold_count: int
    inner_fold_count: int
    random_seed: int
    regularization_c: float
    max_iter: int
    task_name: str
    method_name: BinaryMethodName
    reference_feature_key: str
    candidate_feature_key: str
    candidate_variant_id: str
    candidate_source_feature_keys: tuple[str, ...]
    candidate_calibration_source_labels: tuple[str, ...]
    dataset_count: int
    comparison_count: int
    reference_error_count: int
    candidate_error_count: int
    fixed_error_count: int
    persistent_error_count: int
    introduced_error_count: int
    net_error_delta: int
    comparisons: tuple[DatasetResidualErrorComparison, ...]


def _task_definition(task_name: str) -> BinaryTaskDefinition:
    matches = tuple(definition for definition in default_binary_task_definitions() if definition.name == task_name)
    if len(matches) != 1:
        raise BinaryTaskError(f"Expected exactly one binary task named '{task_name}', found {len(matches)}.")
    return matches[0]


def _error_analysis_report(
    artifact: ActivationArtifact,
    dataset: BinaryTaskDataset,
    method: BinaryMethodErrorAnalysis,
    config: BinaryTaskConfig,
) -> BinaryErrorAnalysisReport:
    metadata = artifact["metadata"]
    return BinaryErrorAnalysisReport(
        source_model_id=metadata["model_id"],
        source_revision=metadata["revision"],
        source_selected_device=metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        regularization_c=config.regularization_c,
        max_iter=config.max_iter,
        activation_feature_key=method.feature_name,
        tasks=(
            BinaryTaskErrorAnalysis(
                task_name=dataset.name,
                description=dataset.description,
                label_names=method.label_names,
                methods=(method,),
            ),
        ),
    )


def _compare_dataset(
    dataset: CiftMetaResidualDataset,
    definition: BinaryTaskDefinition,
    baseline_feature_key: str,
    variant: CiftMetaHeadVariant,
    binary_config: BinaryTaskConfig,
) -> DatasetResidualErrorComparison:
    task_dataset = build_binary_task_dataset(dataset.artifact, definition)
    baseline_config = replace(binary_config, activation_feature_key=baseline_feature_key)
    baseline_method = collect_grouped_activation_predictions(
        artifact=dataset.artifact,
        dataset=task_dataset,
        config=baseline_config,
    )
    candidate_method = collect_grouped_cift_meta_head_predictions(
        artifact=dataset.artifact,
        dataset=task_dataset,
        binary_config=binary_config,
        variant=variant,
    )
    baseline_report = _error_analysis_report(
        artifact=dataset.artifact,
        dataset=task_dataset,
        method=baseline_method,
        config=baseline_config,
    )
    candidate_report = _error_analysis_report(
        artifact=dataset.artifact,
        dataset=task_dataset,
        method=candidate_method,
        config=binary_config,
    )
    return DatasetResidualErrorComparison(
        dataset_id=dataset.dataset_id,
        comparison=compare_binary_error_residuals(
            reference_report=baseline_report,
            candidate_report=candidate_report,
            task_name=definition.name,
            method_name="activation_probe",
        ),
    )


def _validate_inputs(
    datasets: tuple[CiftMetaResidualDataset, ...],
    baseline_feature_key: str,
) -> None:
    if len(datasets) == 0:
        raise BinaryTaskError("At least one CIFT meta residual dataset is required.")
    if baseline_feature_key == "":
        raise BinaryTaskError("CIFT meta residual baseline feature key must not be empty.")
    for index, dataset in enumerate(datasets):
        if dataset.dataset_id == "":
            raise BinaryTaskError(f"CIFT meta residual dataset {index} has an empty dataset id.")


def compare_cift_meta_residual_suite(
    datasets: tuple[CiftMetaResidualDataset, ...],
    task_name: str,
    baseline_feature_key: str,
    variant: CiftMetaHeadVariant,
    binary_config: BinaryTaskConfig,
) -> CiftMetaResidualSuiteReport:
    _validate_inputs(datasets, baseline_feature_key)
    definition = _task_definition(task_name)
    comparisons = tuple(
        _compare_dataset(
            dataset=dataset,
            definition=definition,
            baseline_feature_key=baseline_feature_key,
            variant=variant,
            binary_config=binary_config,
        )
        for dataset in datasets
    )
    first_metadata = datasets[0].artifact["metadata"]
    reference_error_count = sum(item.comparison.reference_error_count for item in comparisons)
    candidate_error_count = sum(item.comparison.candidate_error_count for item in comparisons)
    fixed_error_count = sum(item.comparison.fixed_error_count for item in comparisons)
    persistent_error_count = sum(item.comparison.persistent_error_count for item in comparisons)
    introduced_error_count = sum(item.comparison.introduced_error_count for item in comparisons)

    return CiftMetaResidualSuiteReport(
        source_model_id=first_metadata["model_id"],
        source_revision=first_metadata["revision"],
        source_selected_device=first_metadata["selected_device"],
        evaluation_strategy="stratified_group_kfold",
        fold_count=binary_config.fold_count,
        inner_fold_count=variant.inner_fold_count,
        random_seed=binary_config.random_seed,
        regularization_c=binary_config.regularization_c,
        max_iter=binary_config.max_iter,
        task_name=definition.name,
        method_name="activation_probe",
        reference_feature_key=baseline_feature_key,
        candidate_feature_key=variant.feature_name,
        candidate_variant_id=variant.variant_id,
        candidate_source_feature_keys=variant.source_feature_keys,
        candidate_calibration_source_labels=variant.calibration_source_labels,
        dataset_count=len({item.dataset_id for item in comparisons}),
        comparison_count=len(comparisons),
        reference_error_count=reference_error_count,
        candidate_error_count=candidate_error_count,
        fixed_error_count=fixed_error_count,
        persistent_error_count=persistent_error_count,
        introduced_error_count=introduced_error_count,
        net_error_delta=introduced_error_count - fixed_error_count,
        comparisons=comparisons,
    )


def _comparison_to_json(comparison: DatasetResidualErrorComparison) -> dict[str, JsonValue]:
    return {
        "dataset_id": comparison.dataset_id,
        "comparison": residual_error_comparison_report_to_json(comparison.comparison),
    }


def cift_meta_residual_suite_report_to_json(report: CiftMetaResidualSuiteReport) -> dict[str, JsonValue]:
    return {
        "source_model_id": report.source_model_id,
        "source_revision": report.source_revision,
        "source_selected_device": report.source_selected_device,
        "evaluation_strategy": report.evaluation_strategy,
        "fold_count": report.fold_count,
        "inner_fold_count": report.inner_fold_count,
        "random_seed": report.random_seed,
        "regularization_c": report.regularization_c,
        "max_iter": report.max_iter,
        "task_name": report.task_name,
        "method_name": report.method_name,
        "reference_feature_key": report.reference_feature_key,
        "candidate_feature_key": report.candidate_feature_key,
        "candidate_variant_id": report.candidate_variant_id,
        "candidate_source_feature_keys": list(report.candidate_source_feature_keys),
        "candidate_calibration_source_labels": list(report.candidate_calibration_source_labels),
        "dataset_count": report.dataset_count,
        "comparison_count": report.comparison_count,
        "reference_error_count": report.reference_error_count,
        "candidate_error_count": report.candidate_error_count,
        "fixed_error_count": report.fixed_error_count,
        "persistent_error_count": report.persistent_error_count,
        "introduced_error_count": report.introduced_error_count,
        "net_error_delta": report.net_error_delta,
        "comparisons": [_comparison_to_json(comparison) for comparison in report.comparisons],
    }


def write_cift_meta_residual_suite_json(path: Path, report: CiftMetaResidualSuiteReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cift_meta_residual_suite_report_to_json(report), file, indent=2)
        file.write("\n")


def _joined(values: tuple[str, ...]) -> str:
    return "`, `".join(values)


def _render_error_examples(
    title: str,
    comparisons: tuple[DatasetResidualErrorComparison, ...],
    error_kind: str,
) -> list[str]:
    lines = [f"## {title}", ""]
    rows: list[tuple[str, str, str, str, str, str]] = []
    for item in comparisons:
        if error_kind == "fixed":
            errors = item.comparison.fixed_errors
        elif error_kind == "persistent":
            errors = item.comparison.persistent_errors
        elif error_kind == "introduced":
            errors = item.comparison.introduced_errors
        else:
            raise BinaryTaskError(f"Unsupported residual error kind '{error_kind}'.")
        rows.extend(
            (
                item.dataset_id,
                error.example_id,
                error.family,
                error.true_label,
                error.reference_predicted_label,
                error.candidate_predicted_label,
            )
            for error in errors
        )

    if len(rows) == 0:
        lines.extend(["No examples.", ""])
        return lines

    lines.extend(
        [
            "| Dataset | Example | Family | True Label | Reference Prediction | Candidate Prediction |",
            "|---|---|---|---|---|---|",
        ]
    )
    for dataset_id, example_id, family, true_label, reference_label, candidate_label in rows:
        lines.append(
            f"| `{dataset_id}` | `{example_id}` | `{family}` | `{true_label}` | "
            f"`{reference_label}` | `{candidate_label}` |"
        )
    lines.append("")
    return lines


def render_cift_meta_residual_suite_markdown(report: CiftMetaResidualSuiteReport) -> str:
    lines = [
        "# CIFT Meta-Head Residual Suite",
        "",
        "## Source",
        "",
        f"- Evaluation strategy: `{report.evaluation_strategy}`",
        f"- Task: `{report.task_name}`",
        f"- Method: `{report.method_name}`",
        f"- Reference feature: `{report.reference_feature_key}`",
        f"- Candidate feature: `{report.candidate_feature_key}`",
        f"- Candidate variant: `{report.candidate_variant_id}`",
        f"- Candidate source features: `{_joined(report.candidate_source_feature_keys)}`",
        f"- Calibration source labels: `{_joined(report.candidate_calibration_source_labels)}`",
        f"- Dataset count: `{report.dataset_count}`",
        f"- Fold count: `{report.fold_count}`",
        f"- Inner fold count: `{report.inner_fold_count}`",
        "",
        "## Aggregate",
        "",
        "| Reference Errors | Candidate Errors | Fixed | Persistent | Introduced | Net Error Delta |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report.reference_error_count} | {report.candidate_error_count} | "
            f"{report.fixed_error_count} | {report.persistent_error_count} | "
            f"{report.introduced_error_count} | {report.net_error_delta} |"
        ),
        "",
        "## Dataset Comparisons",
        "",
        "| Dataset | Reference Errors | Candidate Errors | Fixed | Persistent | Introduced | Reference Accuracy | Candidate Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.comparisons:
        comparison = item.comparison
        lines.append(
            f"| `{item.dataset_id}` | {comparison.reference_error_count} | "
            f"{comparison.candidate_error_count} | {comparison.fixed_error_count} | "
            f"{comparison.persistent_error_count} | {comparison.introduced_error_count} | "
            f"{comparison.reference_accuracy:.4f} | {comparison.candidate_accuracy:.4f} |"
        )

    lines.extend(["", "## Family Deltas", ""])
    for item in report.comparisons:
        comparison = item.comparison
        lines.extend([f"### {item.dataset_id}", ""])
        if len(comparison.family_summaries) == 0:
            lines.extend(["No residual error changes.", ""])
            continue
        lines.extend(
            [
                "| Family | Fixed | Persistent | Introduced |",
                "|---|---:|---:|---:|",
            ]
        )
        for summary in comparison.family_summaries:
            lines.append(
                f"| `{summary.family}` | {summary.fixed_error_count} | "
                f"{summary.persistent_error_count} | {summary.introduced_error_count} |"
            )
        lines.append("")

    lines.extend(_render_error_examples("Fixed Errors", report.comparisons, "fixed"))
    lines.extend(_render_error_examples("Persistent Errors", report.comparisons, "persistent"))
    lines.extend(_render_error_examples("Introduced Errors", report.comparisons, "introduced"))
    return "\n".join(lines)


def write_cift_meta_residual_suite_markdown(path: Path, report: CiftMetaResidualSuiteReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cift_meta_residual_suite_markdown(report), encoding="utf-8")
