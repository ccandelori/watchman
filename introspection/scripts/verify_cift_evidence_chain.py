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
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_evidence_chain_verifier import (  # noqa: E402
    DEFAULT_WORKFLOW_EVIDENCE_ROLES,
    CiftEvidenceChainVerifierConfig,
    CiftEvidenceChainVerifierError,
    cift_evidence_chain_config_from_workflow_manifest,
    cift_evidence_chain_verification_report_to_json,
    verify_cift_evidence_chain,
)


@dataclass(frozen=True)
class VerifyCiftEvidenceChainCliConfig:
    repository_root: Path
    workflow_manifest_path: Path | None
    runtime_model_path: Path | None
    runtime_prevention_report_path: Path | None
    gateway_smoke_report_path: Path | None
    sealed_holdout_report_path: Path | None
    head_to_head_report_path: Path | None
    promotion_evidence_path: Path | None
    model_metadata_report_path: Path | None
    required_runtime_prevention_device: str | None
    expected_selected_choice_readout_token_count: int | None
    output_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify that CIFT runtime promotion evidence binds one detector.")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--workflow-manifest", required=False)
    parser.add_argument("--runtime-model", required=False)
    parser.add_argument("--runtime-prevention-report", required=False)
    parser.add_argument("--gateway-smoke-report", required=False)
    parser.add_argument("--sealed-holdout-report", required=False)
    parser.add_argument("--head-to-head-report", required=False)
    parser.add_argument("--promotion-evidence", required=False)
    parser.add_argument("--model-metadata-report", required=False)
    parser.add_argument("--required-runtime-prevention-device", required=False)
    parser.add_argument("--expected-selected-choice-readout-token-count", required=False, type=int)
    parser.add_argument("--output", required=True)
    return parser


def _parse_args(argv: Sequence[str]) -> VerifyCiftEvidenceChainCliConfig:
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    workflow_manifest = namespace.workflow_manifest
    required_manual_inputs = (
        ("--runtime-model", namespace.runtime_model),
        ("--runtime-prevention-report", namespace.runtime_prevention_report),
        ("--gateway-smoke-report", namespace.gateway_smoke_report),
        ("--sealed-holdout-report", namespace.sealed_holdout_report),
        ("--head-to-head-report", namespace.head_to_head_report),
        ("--promotion-evidence", namespace.promotion_evidence),
    )
    optional_manual_inputs = (
        ("--model-metadata-report", namespace.model_metadata_report),
        ("--required-runtime-prevention-device", namespace.required_runtime_prevention_device),
        (
            "--expected-selected-choice-readout-token-count",
            namespace.expected_selected_choice_readout_token_count,
        ),
    )
    if workflow_manifest is not None:
        mixed_inputs = tuple(
            flag for flag, value in (*required_manual_inputs, *optional_manual_inputs) if value is not None
        )
        if len(mixed_inputs) > 0:
            parser.error(f"--workflow-manifest cannot be combined with {', '.join(mixed_inputs)}.")
    else:
        missing_inputs = tuple(flag for flag, value in required_manual_inputs if value is None)
        if len(missing_inputs) > 0:
            parser.error(f"manual verification requires {', '.join(missing_inputs)}.")
    model_metadata_report = namespace.model_metadata_report
    return VerifyCiftEvidenceChainCliConfig(
        repository_root=Path(str(namespace.repository_root)),
        workflow_manifest_path=Path(str(workflow_manifest)) if workflow_manifest is not None else None,
        runtime_model_path=_optional_path(namespace.runtime_model),
        runtime_prevention_report_path=_optional_path(namespace.runtime_prevention_report),
        gateway_smoke_report_path=_optional_path(namespace.gateway_smoke_report),
        sealed_holdout_report_path=_optional_path(namespace.sealed_holdout_report),
        head_to_head_report_path=_optional_path(namespace.head_to_head_report),
        promotion_evidence_path=_optional_path(namespace.promotion_evidence),
        model_metadata_report_path=Path(str(model_metadata_report)) if model_metadata_report is not None else None,
        required_runtime_prevention_device=_optional_cli_string(namespace.required_runtime_prevention_device),
        expected_selected_choice_readout_token_count=_optional_positive_int(
            namespace.expected_selected_choice_readout_token_count,
            "expected_selected_choice_readout_token_count",
        ),
        output_path=Path(str(namespace.output)),
    )


def _verifier_config(config: VerifyCiftEvidenceChainCliConfig) -> CiftEvidenceChainVerifierConfig:
    if config.workflow_manifest_path is not None:
        return cift_evidence_chain_config_from_workflow_manifest(
            repository_root=config.repository_root,
            workflow_manifest_path=config.workflow_manifest_path,
            evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
        )
    return CiftEvidenceChainVerifierConfig(
        repository_root=config.repository_root,
        runtime_model_path=_required_path(config.runtime_model_path, "runtime_model_path"),
        runtime_prevention_report_path=_required_path(
            config.runtime_prevention_report_path,
            "runtime_prevention_report_path",
        ),
        gateway_smoke_report_path=_required_path(config.gateway_smoke_report_path, "gateway_smoke_report_path"),
        sealed_holdout_report_path=_required_path(config.sealed_holdout_report_path, "sealed_holdout_report_path"),
        head_to_head_report_path=_required_path(config.head_to_head_report_path, "head_to_head_report_path"),
        promotion_evidence_path=_required_path(config.promotion_evidence_path, "promotion_evidence_path"),
        model_metadata_report_path=config.model_metadata_report_path,
        required_runtime_prevention_device=config.required_runtime_prevention_device,
        expected_selected_choice_readout_token_count=config.expected_selected_choice_readout_token_count,
        workflow_artifacts_by_role=None,
    )


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _optional_cli_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text == "":
        raise CiftEvidenceChainVerifierError("required_runtime_prevention_device must not be empty.")
    return text


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CiftEvidenceChainVerifierError(f"{field_name} must be a positive integer.")
    return value


def _required_path(value: Path | None, field_name: str) -> Path:
    if value is None:
        raise CiftEvidenceChainVerifierError(f"{field_name} must be present.")
    return value


def run_cli(config: VerifyCiftEvidenceChainCliConfig) -> int:
    report = verify_cift_evidence_chain(_verifier_config(config))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(cift_evidence_chain_verification_report_to_json(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report.eligible:
        print(f"CIFT evidence chain verified: {report.runtime_model_path}")
        return 0
    print(f"CIFT evidence chain verification failed: {report.runtime_model_path}")
    for failed_requirement in report.failed_requirements:
        print(f"- {failed_requirement}")
    return 1


def main(argv: Sequence[str]) -> int:
    try:
        return run_cli(_parse_args(argv))
    except CiftEvidenceChainVerifierError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
