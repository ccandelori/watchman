from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_promotion_evidence_materializer import (  # noqa: E402
    DEFAULT_WORKFLOW_PROMOTION_EVIDENCE_ROLES,
    CiftPromotionEvidenceMaterializerConfig,
    CiftPromotionReportSource,
    cift_promotion_materializer_config_from_workflow_manifest,
    materialize_cift_promotion_evidence,
)


@dataclass(frozen=True)
class MaterializeCiftPromotionEvidenceCliConfig:
    bundle_path: Path
    repository_root: Path
    report_output_dir: Path
    evidence_output_path: Path
    evidence_id: str
    behavior_id: str
    behavior_description: str
    train_split_id: str
    calibration_split_id: str
    heldout_split_id: str
    sealed_holdout_split_id: str
    sealed_holdout_report_id: str
    metric_report_id: str
    metric_name: str
    metric_value: float
    metric_threshold: float
    calibration_report_id: str
    ablation_report_id: str
    ablation_delta: float
    ablation_delta_threshold: float
    patching_report_id: str
    failure_case_report_id: str
    runtime_prevention_report_id: str
    gateway_smoke_report_id: str
    lineage_report_id: str
    head_to_head_report_id: str | None
    report_sources: tuple[CiftPromotionReportSource, ...]
    created_at: str


@dataclass(frozen=True)
class MaterializeCiftPromotionEvidenceWorkflowCliConfig:
    repository_root: Path
    workflow_manifest_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize CIFT promotion report artifacts and write cift_promotion_evidence/v1."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--report-output-dir", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--evidence-id", required=True)
    parser.add_argument("--behavior-id", required=True)
    parser.add_argument("--behavior-description", required=True)
    parser.add_argument("--train-split-id", required=True)
    parser.add_argument("--calibration-split-id", required=True)
    parser.add_argument("--heldout-split-id", required=True)
    parser.add_argument("--sealed-holdout-split-id", required=True)
    parser.add_argument("--sealed-holdout-report-id", required=True)
    parser.add_argument("--metric-report-id", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--metric-value", required=True, type=float)
    parser.add_argument("--metric-threshold", required=True, type=float)
    parser.add_argument("--calibration-report-id", required=True)
    parser.add_argument("--ablation-report-id", required=True)
    parser.add_argument("--ablation-delta", required=True, type=float)
    parser.add_argument("--ablation-delta-threshold", required=True, type=float)
    parser.add_argument("--patching-report-id", required=True)
    parser.add_argument("--failure-case-report-id", required=True)
    parser.add_argument("--runtime-prevention-report-id", required=True)
    parser.add_argument("--gateway-smoke-report-id", required=True)
    parser.add_argument("--lineage-report-id", required=True)
    parser.add_argument("--head-to-head-report-id", required=False)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--report-source",
        required=True,
        action="append",
        help="Source report as report_id:schema_version:path. Repeat once for each required report.",
    )
    return parser


def _build_workflow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize CIFT promotion evidence from a certification workflow manifest."
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--workflow-manifest", required=True)
    return parser


def _parse_args(argv: Sequence[str]) -> MaterializeCiftPromotionEvidenceCliConfig:
    namespace = _build_parser().parse_args(argv)
    return MaterializeCiftPromotionEvidenceCliConfig(
        bundle_path=Path(str(namespace.bundle)),
        repository_root=Path(str(namespace.repository_root)),
        report_output_dir=Path(str(namespace.report_output_dir)),
        evidence_output_path=Path(str(namespace.evidence_output)),
        evidence_id=str(namespace.evidence_id),
        behavior_id=str(namespace.behavior_id),
        behavior_description=str(namespace.behavior_description),
        train_split_id=str(namespace.train_split_id),
        calibration_split_id=str(namespace.calibration_split_id),
        heldout_split_id=str(namespace.heldout_split_id),
        sealed_holdout_split_id=str(namespace.sealed_holdout_split_id),
        sealed_holdout_report_id=str(namespace.sealed_holdout_report_id),
        metric_report_id=str(namespace.metric_report_id),
        metric_name=str(namespace.metric_name),
        metric_value=float(namespace.metric_value),
        metric_threshold=float(namespace.metric_threshold),
        calibration_report_id=str(namespace.calibration_report_id),
        ablation_report_id=str(namespace.ablation_report_id),
        ablation_delta=float(namespace.ablation_delta),
        ablation_delta_threshold=float(namespace.ablation_delta_threshold),
        patching_report_id=str(namespace.patching_report_id),
        failure_case_report_id=str(namespace.failure_case_report_id),
        runtime_prevention_report_id=str(namespace.runtime_prevention_report_id),
        gateway_smoke_report_id=str(namespace.gateway_smoke_report_id),
        lineage_report_id=str(namespace.lineage_report_id),
        head_to_head_report_id=_optional_string(namespace.head_to_head_report_id),
        report_sources=_parse_report_sources(tuple(str(item) for item in namespace.report_source)),
        created_at=str(namespace.created_at),
    )


def _parse_workflow_args(argv: Sequence[str]) -> MaterializeCiftPromotionEvidenceWorkflowCliConfig:
    namespace = _build_workflow_parser().parse_args(argv)
    return MaterializeCiftPromotionEvidenceWorkflowCliConfig(
        repository_root=Path(str(namespace.repository_root)),
        workflow_manifest_path=Path(str(namespace.workflow_manifest)),
    )


def _parse_report_sources(values: tuple[str, ...]) -> tuple[CiftPromotionReportSource, ...]:
    sources: list[CiftPromotionReportSource] = []
    for value in values:
        parts = value.split(":", maxsplit=2)
        if len(parts) != 3 or any(part == "" for part in parts):
            raise ValueError("report-source must have shape report_id:schema_version:path.")
        sources.append(
            CiftPromotionReportSource(
                report_id=parts[0],
                schema_version=parts[1],
                source_path=Path(parts[2]),
            )
        )
    return tuple(sources)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _materializer_config(
    config: MaterializeCiftPromotionEvidenceCliConfig,
) -> CiftPromotionEvidenceMaterializerConfig:
    return CiftPromotionEvidenceMaterializerConfig(
        bundle_path=config.bundle_path,
        repository_root=config.repository_root,
        report_output_dir=config.report_output_dir,
        evidence_output_path=config.evidence_output_path,
        evidence_id=config.evidence_id,
        behavior_id=config.behavior_id,
        behavior_description=config.behavior_description,
        train_split_id=config.train_split_id,
        calibration_split_id=config.calibration_split_id,
        heldout_split_id=config.heldout_split_id,
        sealed_holdout_split_id=config.sealed_holdout_split_id,
        sealed_holdout_report_id=config.sealed_holdout_report_id,
        metric_report_id=config.metric_report_id,
        metric_name=config.metric_name,
        metric_value=config.metric_value,
        metric_threshold=config.metric_threshold,
        calibration_report_id=config.calibration_report_id,
        ablation_report_id=config.ablation_report_id,
        ablation_delta=config.ablation_delta,
        ablation_delta_threshold=config.ablation_delta_threshold,
        patching_report_id=config.patching_report_id,
        failure_case_report_id=config.failure_case_report_id,
        runtime_prevention_report_id=config.runtime_prevention_report_id,
        gateway_smoke_report_id=config.gateway_smoke_report_id,
        lineage_report_id=config.lineage_report_id,
        head_to_head_report_id=config.head_to_head_report_id,
        report_sources=config.report_sources,
        created_at=config.created_at,
    )


def run_materializer(config: MaterializeCiftPromotionEvidenceCliConfig) -> None:
    evidence = materialize_cift_promotion_evidence(_materializer_config(config))
    print(f"Wrote CIFT promotion evidence to {config.evidence_output_path}")
    print(f"Evidence ID: {evidence.evidence_id}")
    print(f"Metric: {evidence.metric_name}={evidence.metric_value:.6f}")
    print(f"Reports: {len(evidence.report_artifacts)}")


def run_workflow_materializer(config: MaterializeCiftPromotionEvidenceWorkflowCliConfig) -> None:
    materializer_config = cift_promotion_materializer_config_from_workflow_manifest(
        repository_root=config.repository_root,
        workflow_manifest_path=config.workflow_manifest_path,
        evidence_roles=DEFAULT_WORKFLOW_PROMOTION_EVIDENCE_ROLES,
    )
    evidence = materialize_cift_promotion_evidence(materializer_config)
    print(f"Wrote CIFT promotion evidence to {materializer_config.evidence_output_path}")
    print(f"Evidence ID: {evidence.evidence_id}")
    print(f"Metric: {evidence.metric_name}={evidence.metric_value:.6f}")
    print(f"Reports: {len(evidence.report_artifacts)}")


def main(argv: Sequence[str]) -> None:
    if _uses_workflow_manifest(argv):
        run_workflow_materializer(_parse_workflow_args(argv))
        return
    run_materializer(_parse_args(argv))


def _uses_workflow_manifest(argv: Sequence[str]) -> bool:
    return any(argument == "--workflow-manifest" or argument.startswith("--workflow-manifest=") for argument in argv)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
