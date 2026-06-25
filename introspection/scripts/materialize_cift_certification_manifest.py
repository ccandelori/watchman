from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_certification_workflow import (  # noqa: E402
    CiftCertificationEvidenceManifestConfig,
    build_cift_certification_evidence_manifest,
)


@dataclass(frozen=True)
class MaterializeCiftCertificationManifestCliConfig:
    certification_id: str
    repository_root: Path
    output_path: Path
    created_at: str
    behavior_id: str
    behavior_description: str
    requested_device: str
    prompt_renderer: str
    selected_choice_geometry: str
    selected_choice_readout_token_count: int
    dtype_name: str
    metric_threshold: float
    ablation_delta_threshold: float
    model_metadata_report_path: Path
    activation_artifact_path: Path
    linear_bundle_path: Path
    grouped_head_to_head_report_path: Path
    calibration_report_path: Path
    feature_ablation_report_path: Path
    patching_report_path: Path
    failure_cases_report_path: Path
    lineage_report_path: Path
    device_preflight_report_path: Path
    live_runtime_prevention_report_path: Path
    sealed_holdout_metric_path: Path
    gateway_smoke_report_path: Path
    paper_mlp_runtime_prevention_report_path: Path
    paper_mlp_sealed_holdout_metric_path: Path
    live_head_to_head_report_path: Path
    promotion_evidence_path: Path
    promoted_runtime_artifact_path: Path
    promotion_report_output_dir: Path
    evidence_chain_verification_report_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize an evidence-bound CIFT certification manifest.")
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--behavior-id", required=True)
    parser.add_argument("--behavior-description", required=True)
    parser.add_argument("--requested-device", required=True)
    parser.add_argument("--prompt-renderer", required=True)
    parser.add_argument("--selected-choice-geometry", required=True)
    parser.add_argument("--selected-choice-readout-token-count", required=True, type=int)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--metric-threshold", required=True, type=float)
    parser.add_argument("--ablation-delta-threshold", required=True, type=float)
    parser.add_argument("--model-metadata-report", required=True)
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--linear-bundle", required=True)
    parser.add_argument("--grouped-head-to-head-report", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--feature-ablation-report", required=True)
    parser.add_argument("--patching-report", required=True)
    parser.add_argument("--failure-cases-report", required=True)
    parser.add_argument("--lineage-report", required=True)
    parser.add_argument("--device-preflight-report", required=True)
    parser.add_argument("--live-runtime-prevention-report", required=True)
    parser.add_argument("--sealed-holdout-metric", required=True)
    parser.add_argument("--gateway-smoke-report", required=True)
    parser.add_argument("--paper-mlp-runtime-prevention-report", required=True)
    parser.add_argument("--paper-mlp-sealed-holdout-metric", required=True)
    parser.add_argument("--live-head-to-head-report", required=True)
    parser.add_argument("--promotion-evidence", required=True)
    parser.add_argument("--promoted-runtime-artifact", required=True)
    parser.add_argument("--promotion-report-output-dir", required=True)
    parser.add_argument("--evidence-chain-verification-report", required=True)
    return parser


def _parse_args(argv: Sequence[str]) -> MaterializeCiftCertificationManifestCliConfig:
    namespace = _build_parser().parse_args(argv)
    return MaterializeCiftCertificationManifestCliConfig(
        certification_id=str(namespace.certification_id),
        repository_root=Path(str(namespace.repository_root)),
        output_path=Path(str(namespace.output)),
        created_at=str(namespace.created_at),
        behavior_id=str(namespace.behavior_id),
        behavior_description=str(namespace.behavior_description),
        requested_device=str(namespace.requested_device),
        prompt_renderer=str(namespace.prompt_renderer),
        selected_choice_geometry=str(namespace.selected_choice_geometry),
        selected_choice_readout_token_count=_positive_int(
            raw_value=namespace.selected_choice_readout_token_count,
            field_name="--selected-choice-readout-token-count",
        ),
        dtype_name=str(namespace.dtype),
        metric_threshold=float(namespace.metric_threshold),
        ablation_delta_threshold=float(namespace.ablation_delta_threshold),
        model_metadata_report_path=Path(str(namespace.model_metadata_report)),
        activation_artifact_path=Path(str(namespace.activation_artifact)),
        linear_bundle_path=Path(str(namespace.linear_bundle)),
        grouped_head_to_head_report_path=Path(str(namespace.grouped_head_to_head_report)),
        calibration_report_path=Path(str(namespace.calibration_report)),
        feature_ablation_report_path=Path(str(namespace.feature_ablation_report)),
        patching_report_path=Path(str(namespace.patching_report)),
        failure_cases_report_path=Path(str(namespace.failure_cases_report)),
        lineage_report_path=Path(str(namespace.lineage_report)),
        device_preflight_report_path=Path(str(namespace.device_preflight_report)),
        live_runtime_prevention_report_path=Path(str(namespace.live_runtime_prevention_report)),
        sealed_holdout_metric_path=Path(str(namespace.sealed_holdout_metric)),
        gateway_smoke_report_path=Path(str(namespace.gateway_smoke_report)),
        paper_mlp_runtime_prevention_report_path=Path(str(namespace.paper_mlp_runtime_prevention_report)),
        paper_mlp_sealed_holdout_metric_path=Path(str(namespace.paper_mlp_sealed_holdout_metric)),
        live_head_to_head_report_path=Path(str(namespace.live_head_to_head_report)),
        promotion_evidence_path=Path(str(namespace.promotion_evidence)),
        promoted_runtime_artifact_path=Path(str(namespace.promoted_runtime_artifact)),
        promotion_report_output_dir=Path(str(namespace.promotion_report_output_dir)),
        evidence_chain_verification_report_path=Path(str(namespace.evidence_chain_verification_report)),
    )


def _positive_int(raw_value: object, field_name: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if raw_value < 1:
        raise ValueError(f"{field_name} must be positive.")
    return raw_value


def _manifest_config(
    config: MaterializeCiftCertificationManifestCliConfig,
) -> CiftCertificationEvidenceManifestConfig:
    return CiftCertificationEvidenceManifestConfig(
        certification_id=config.certification_id,
        repository_root=config.repository_root,
        created_at=config.created_at,
        behavior_id=config.behavior_id,
        behavior_description=config.behavior_description,
        requested_device=config.requested_device,
        prompt_renderer=config.prompt_renderer,
        selected_choice_geometry=config.selected_choice_geometry,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
        dtype_name=config.dtype_name,
        metric_threshold=config.metric_threshold,
        ablation_delta_threshold=config.ablation_delta_threshold,
        model_metadata_report_path=config.model_metadata_report_path,
        activation_artifact_path=config.activation_artifact_path,
        linear_bundle_path=config.linear_bundle_path,
        grouped_head_to_head_report_path=config.grouped_head_to_head_report_path,
        calibration_report_path=config.calibration_report_path,
        feature_ablation_report_path=config.feature_ablation_report_path,
        patching_report_path=config.patching_report_path,
        failure_cases_report_path=config.failure_cases_report_path,
        lineage_report_path=config.lineage_report_path,
        device_preflight_report_path=config.device_preflight_report_path,
        live_runtime_prevention_report_path=config.live_runtime_prevention_report_path,
        sealed_holdout_metric_path=config.sealed_holdout_metric_path,
        gateway_smoke_report_path=config.gateway_smoke_report_path,
        paper_mlp_runtime_prevention_report_path=config.paper_mlp_runtime_prevention_report_path,
        paper_mlp_sealed_holdout_metric_path=config.paper_mlp_sealed_holdout_metric_path,
        live_head_to_head_report_path=config.live_head_to_head_report_path,
        promotion_evidence_path=config.promotion_evidence_path,
        promoted_runtime_artifact_path=config.promoted_runtime_artifact_path,
        promotion_report_output_dir=config.promotion_report_output_dir,
        evidence_chain_verification_report_path=config.evidence_chain_verification_report_path,
    )


def run_cli(config: MaterializeCiftCertificationManifestCliConfig) -> None:
    manifest = build_cift_certification_evidence_manifest(_manifest_config(config))
    evidence_artifacts = manifest.get("required_evidence_artifacts")
    if not isinstance(evidence_artifacts, list):
        raise ValueError("CIFT certification manifest required_evidence_artifacts must be a list.")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote CIFT certification manifest to {config.output_path}")
    print(f"Certification ID: {manifest['certification_id']}")
    print(f"Evidence artifacts: {len(evidence_artifacts)}")


def main(argv: Sequence[str]) -> None:
    run_cli(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
