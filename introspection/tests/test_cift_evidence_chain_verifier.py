from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from aegis_introspection.cift_evidence_chain_verifier import (
    DEFAULT_WORKFLOW_EVIDENCE_ROLES,
    CiftEvidenceChainVerifierConfig,
    CiftEvidenceChainVerifierError,
    cift_evidence_chain_config_from_workflow_manifest,
    verify_cift_evidence_chain,
)
from aegis_introspection.cift_live_probe_competition import (
    CiftLiveProbeCompetitionConfig,
    CiftLiveProbeRun,
    cift_live_probe_competition_report_to_json,
    compare_cift_live_probe_candidates,
)
from aegis_introspection.cift_promotion_gate import (
    CiftPaperMethodContract,
    CiftPromotionEvidence,
    CiftPromotionReportArtifact,
    cift_promotion_evidence_to_json,
)
from aegis_introspection.cift_runtime_digest import cift_runtime_detector_sha256

from aegis.core.contracts import Action
from aegis.detectors.cift_runtime import CiftRuntimeLinearModel, cift_runtime_model_to_dict

_IMMUTABLE_MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


class CiftEvidenceChainVerifierTest(unittest.TestCase):
    def test_verifier_accepts_bound_runtime_evidence_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertTrue(report.eligible)
        self.assertEqual((), report.failed_requirements)
        self.assertEqual("synthetic-cift-runtime", report.model_bundle_id)
        self.assertEqual("synthetic-gateway-smoke-report", report.gateway_smoke_report_id)

    def test_verifier_accepts_bound_freeform_runtime_evidence_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
                feature_key="final_token_layer_12",
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertTrue(report.eligible)
        self.assertEqual((), report.failed_requirements)
        self.assertEqual("synthetic-cift-runtime", report.model_bundle_id)

    def test_verifier_rejects_mutable_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
                source_revision="main",
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "runtime model source_revision must be an immutable lowercase 40-character Git commit SHA "
            "or sha256:<64 lowercase hex digest>",
            report.failed_requirements,
        )

    def test_verifier_config_can_be_derived_from_certification_workflow_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            manifest_path = _write_workflow_manifest(root=root, fixture=fixture, omitted_role=None)

            config = cift_evidence_chain_config_from_workflow_manifest(
                repository_root=root,
                workflow_manifest_path=manifest_path,
                evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
            )
            report = verify_cift_evidence_chain(config)

        self.assertTrue(report.eligible)
        self.assertEqual((), report.failed_requirements)
        self.assertEqual("synthetic-cift-runtime", report.model_bundle_id)
        self.assertEqual("mps", report.required_runtime_prevention_device)

    def test_manifest_verifier_rejects_runtime_prevention_device_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="cpu",
            )
            manifest_path = _write_workflow_manifest(root=root, fixture=fixture, omitted_role=None)

            config = cift_evidence_chain_config_from_workflow_manifest(
                repository_root=root,
                workflow_manifest_path=manifest_path,
                evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
            )
            report = verify_cift_evidence_chain(config)

        self.assertFalse(report.eligible)
        self.assertEqual("mps", report.required_runtime_prevention_device)
        self.assertIn(
            "runtime_prevention_report selected_device must match workflow training.requested_device",
            report.failed_requirements,
        )

    def test_manifest_config_rejects_missing_required_evidence_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            manifest_path = _write_workflow_manifest(root=root, fixture=fixture, omitted_role="promotion_evidence")

            with self.assertRaisesRegex(CiftEvidenceChainVerifierError, "promotion_evidence"):
                cift_evidence_chain_config_from_workflow_manifest(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
                )

    def test_manifest_verifier_rejects_stale_gateway_smoke_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            manifest_path = _write_workflow_manifest(root=root, fixture=fixture, omitted_role=None)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for artifact in manifest["required_evidence_artifacts"]:
                if artifact["role"] == "linear_gateway_smoke":
                    artifact["sha256"] = "0" * 64
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(CiftEvidenceChainVerifierError, "linear_gateway_smoke.*sha256"):
                cift_evidence_chain_config_from_workflow_manifest(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
                )

    def test_manifest_verifier_rejects_gateway_smoke_schema_kind_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            manifest_path = _write_workflow_manifest(root=root, fixture=fixture, omitted_role=None)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for artifact in manifest["required_evidence_artifacts"]:
                if artifact["role"] == "linear_gateway_smoke":
                    artifact["artifact_kind"] = "runtime_model"
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(CiftEvidenceChainVerifierError, "linear_gateway_smoke.*artifact_kind"):
                cift_evidence_chain_config_from_workflow_manifest(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
                )

    def test_verifier_rejects_runtime_prevention_detector_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256="1" * 64,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "runtime_prevention_report selected_choice_runtime_model_detector_sha256 must match runtime model",
            report.failed_requirements,
        )

    def test_verifier_rejects_runtime_prevention_row_without_selected_choice_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            runtime_prevention = json.loads(fixture["runtime_prevention_report"].read_text(encoding="utf-8"))
            runtime_prevention["rows"][0]["window_selection_reason"] = "fallback_metadata"
            _write_json(fixture["runtime_prevention_report"], runtime_prevention)

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "runtime_prevention_report rows must have selected-choice metadata proof",
            report.failed_requirements,
        )

    def test_verifier_rejects_gateway_smoke_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            gateway_smoke = json.loads(fixture["gateway_smoke_report"].read_text(encoding="utf-8"))
            gateway_smoke["expected"]["sidecar_model_id"] = "Qwen/Qwen3-other"
            _write_json(fixture["gateway_smoke_report"], gateway_smoke)

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "gateway_smoke_report expected.sidecar_model_id must match Qwen/Qwen3-test",
            report.failed_requirements,
        )

    def test_verifier_rejects_gateway_smoke_device_mismatch_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            gateway_smoke = json.loads(fixture["gateway_smoke_report"].read_text(encoding="utf-8"))
            gateway_smoke["expected"]["sidecar_device"] = "cpu"
            gateway_smoke["checks"]["sidecar_feature_extraction"]["selected_device"] = "cpu"
            gateway_smoke["checks"]["benign_cift"]["extractor_selected_device"] = "cpu"
            gateway_smoke["checks"]["exfiltration_intent_prevention"]["extractor_selected_device"] = "cpu"
            _write_json(fixture["gateway_smoke_report"], gateway_smoke)
            manifest_path = _write_workflow_manifest(root=root, fixture=fixture, omitted_role=None)

            config = cift_evidence_chain_config_from_workflow_manifest(
                repository_root=root,
                workflow_manifest_path=manifest_path,
                evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
            )
            report = verify_cift_evidence_chain(config)

        self.assertFalse(report.eligible)
        self.assertIn(
            "gateway_smoke_report expected.sidecar_device must match workflow training.requested_device",
            report.failed_requirements,
        )
        self.assertIn(
            "gateway_smoke_report sidecar_feature_extraction.selected_device must match "
            "workflow training.requested_device",
            report.failed_requirements,
        )

    def test_verifier_rejects_gateway_smoke_without_pre_generation_exfiltration_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            gateway_smoke = json.loads(fixture["gateway_smoke_report"].read_text(encoding="utf-8"))
            gateway_smoke["checks"]["exfiltration_intent_prevention"]["final_action"] = "allow"
            gateway_smoke["checks"]["exfiltration_intent_prevention"]["cift_action"] = "allow"
            gateway_smoke["checks"]["exfiltration_intent_prevention"]["provider_status"] = "completed"
            gateway_smoke["checks"]["exfiltration_intent_prevention"]["provider_reason"] = None
            _write_json(fixture["gateway_smoke_report"], gateway_smoke)

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "gateway_smoke_report exfiltration_intent_prevention.final_action must be block or stronger",
            report.failed_requirements,
        )
        self.assertIn(
            "gateway_smoke_report exfiltration_intent_prevention.provider_status must be skipped",
            report.failed_requirements,
        )

    def test_verifier_rejects_gateway_smoke_benign_non_allow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            gateway_smoke = json.loads(fixture["gateway_smoke_report"].read_text(encoding="utf-8"))
            gateway_smoke["checks"]["benign_cift"]["final_action"] = "block"
            gateway_smoke["checks"]["benign_cift"]["cift_action"] = "block"
            gateway_smoke["checks"]["benign_cift"]["predicted_label"] = "exfiltration_intent"
            _write_json(fixture["gateway_smoke_report"], gateway_smoke)

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn("gateway_smoke_report benign_cift.final_action must be allow", report.failed_requirements)
        self.assertIn(
            "gateway_smoke_report benign_cift.predicted_label must not equal positive_label",
            report.failed_requirements,
        )

    def test_verifier_rejects_gateway_smoke_readout_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            gateway_smoke = json.loads(fixture["gateway_smoke_report"].read_text(encoding="utf-8"))
            gateway_smoke["checks"]["sidecar_feature_extraction"]["prompt_renderer"] = "legacy"
            gateway_smoke["checks"]["benign_cift"]["extractor_selected_choice_geometry"] = "legacy"
            _write_json(fixture["gateway_smoke_report"], gateway_smoke)

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "gateway_smoke_report sidecar_feature_extraction.prompt_renderer must match CIFT contract",
            report.failed_requirements,
        )
        self.assertIn(
            "gateway_smoke_report benign_cift.extractor_selected_choice_geometry must match semantic_indirection_v1",
            report.failed_requirements,
        )

    def test_verifier_rejects_gateway_smoke_selected_choice_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-report",
                runtime_prevention_selected_device="mps",
            )
            gateway_smoke = json.loads(fixture["gateway_smoke_report"].read_text(encoding="utf-8"))
            gateway_smoke["expected"]["selected_choice_readout_token_count"] = 1
            gateway_smoke["checks"]["sidecar_feature_extraction"]["selected_choice_readout_token_count"] = 1
            gateway_smoke["checks"]["benign_cift"]["extractor_selected_choice_readout_token_count"] = 1
            gateway_smoke["checks"]["exfiltration_intent_prevention"][
                "extractor_selected_choice_readout_token_count"
            ] = 1
            _write_json(fixture["gateway_smoke_report"], gateway_smoke)

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "gateway_smoke_report expected.selected_choice_readout_token_count must match CIFT contract",
            report.failed_requirements,
        )
        self.assertIn(
            "gateway_smoke_report sidecar_feature_extraction.selected_choice_readout_token_count must match "
            "CIFT contract",
            report.failed_requirements,
        )

    def test_verifier_rejects_gateway_smoke_selected_choice_float_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-report",
                runtime_prevention_selected_device="mps",
            )
            gateway_smoke = json.loads(fixture["gateway_smoke_report"].read_text(encoding="utf-8"))
            gateway_smoke["expected"]["selected_choice_readout_token_count"] = 4.0
            gateway_smoke["checks"]["sidecar_feature_extraction"]["selected_choice_readout_token_count"] = 4.0
            gateway_smoke["checks"]["benign_cift"]["extractor_selected_choice_readout_token_count"] = 4.0
            gateway_smoke["checks"]["exfiltration_intent_prevention"][
                "extractor_selected_choice_readout_token_count"
            ] = 4.0
            _write_json(fixture["gateway_smoke_report"], gateway_smoke)

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "gateway_smoke_report expected.selected_choice_readout_token_count must be a positive integer",
            report.failed_requirements,
        )
        self.assertIn(
            "gateway_smoke_report sidecar_feature_extraction.selected_choice_readout_token_count must be a "
            "positive integer",
            report.failed_requirements,
        )
        self.assertIn(
            "gateway_smoke_report benign_cift.extractor_selected_choice_readout_token_count must be a positive integer",
            report.failed_requirements,
        )

    def test_verifier_rejects_missing_required_runtime_prevention_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )

            report = verify_cift_evidence_chain(
                _config(root=root, fixture=fixture, required_runtime_prevention_device=None)
            )

        self.assertFalse(report.eligible)
        self.assertIn(
            "required_runtime_prevention_device must be present for CIFT evidence chain verification",
            report.failed_requirements,
        )

    def test_verifier_rejects_runtime_model_device_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
                source_selected_device="cpu",
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "runtime model source_selected_device must match required runtime prevention device",
            report.failed_requirements,
        )

    def test_verifier_accepts_generic_cpu_model_when_required_device_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="cpu",
                source_model_id="Local/Example-Hidden-State-Model",
                source_selected_device="cpu",
            )

            report = verify_cift_evidence_chain(
                _config(root=root, fixture=fixture, required_runtime_prevention_device="cpu")
            )

        self.assertTrue(report.eligible)
        self.assertEqual((), report.failed_requirements)
        self.assertEqual("Local/Example-Hidden-State-Model", report.source_model_id)

    def test_verifier_rejects_promotion_evidence_without_gateway_smoke_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
                include_gateway_smoke_artifact=False,
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "promotion evidence report_artifacts must include synthetic-gateway-smoke-report",
            report.failed_requirements,
        )

    def test_verifier_rejects_promotion_evidence_without_patching_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
                include_patching_artifact=False,
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "promotion evidence report_artifacts must include synthetic-patching-report",
            report.failed_requirements,
        )

    def test_workflow_verifier_rejects_promotion_artifact_drift_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="synthetic-sealed-holdout-report",
                runtime_prevention_selected_device="mps",
            )
            stale_gateway_path = root / "reports" / "stale_gateway_smoke.json"
            _write_json(
                stale_gateway_path,
                _support_report("synthetic-gateway-smoke-report", "aegis.proxy.cift_gateway_smoke/v1"),
            )
            promotion_evidence = json.loads(fixture["promotion_evidence"].read_text(encoding="utf-8"))
            for artifact in promotion_evidence["report_artifacts"]:
                if artifact["report_id"] == "synthetic-gateway-smoke-report":
                    artifact["path"] = str(stale_gateway_path.relative_to(root))
                    artifact["sha256"] = hashlib.sha256(stale_gateway_path.read_bytes()).hexdigest()
            _write_json(fixture["promotion_evidence"], promotion_evidence)
            manifest_path = _write_workflow_manifest(root=root, fixture=fixture, omitted_role=None)

            report = verify_cift_evidence_chain(
                cift_evidence_chain_config_from_workflow_manifest(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    evidence_roles=DEFAULT_WORKFLOW_EVIDENCE_ROLES,
                )
            )

        self.assertFalse(report.eligible)
        self.assertIn(
            "promotion evidence report_artifacts synthetic-gateway-smoke-report path must match workflow manifest "
            "role linear_gateway_smoke",
            report.failed_requirements,
        )

    def test_verifier_rejects_head_to_head_candidate_not_sourced_from_sealed_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(
                root=root,
                runtime_detector_sha256=None,
                candidate_source_report_id="stale-sealed-report",
                runtime_prevention_selected_device="mps",
            )

            report = verify_cift_evidence_chain(_config(root=root, fixture=fixture))

        self.assertFalse(report.eligible)
        self.assertIn(
            "head_to_head_report candidate_probe.source_report_id must match sealed holdout report",
            report.failed_requirements,
        )


def _config(
    root: Path,
    fixture: dict[str, Path],
    required_runtime_prevention_device: str | None = "mps",
) -> CiftEvidenceChainVerifierConfig:
    return CiftEvidenceChainVerifierConfig(
        repository_root=root,
        runtime_model_path=fixture["runtime_model"],
        runtime_prevention_report_path=fixture["runtime_prevention_report"],
        gateway_smoke_report_path=fixture["gateway_smoke_report"],
        sealed_holdout_report_path=fixture["sealed_holdout_report"],
        head_to_head_report_path=fixture["head_to_head_report"],
        promotion_evidence_path=fixture["promotion_evidence"],
        model_metadata_report_path=fixture["model_metadata_report"],
        required_runtime_prevention_device=required_runtime_prevention_device,
        expected_selected_choice_readout_token_count=4,
        workflow_artifacts_by_role=None,
    )


def _write_fixture(
    root: Path,
    runtime_detector_sha256: str | None,
    candidate_source_report_id: str,
    runtime_prevention_selected_device: str,
    include_gateway_smoke_artifact: bool = True,
    include_patching_artifact: bool = True,
    source_revision: str = _IMMUTABLE_MODEL_REVISION,
    source_model_id: str = "Qwen/Qwen3-test",
    source_selected_device: str = "mps",
    feature_key: str = "selected_choice_window_layer_21",
) -> dict[str, Path]:
    reports_root = root / "reports"
    reports_root.mkdir(parents=True)
    runtime_model = _runtime_model(
        source_revision=source_revision,
        source_model_id=source_model_id,
        source_selected_device=source_selected_device,
        feature_key=feature_key,
    )
    runtime_model_path = root / "runtime_model.json"
    runtime_model_path.write_text(json.dumps(cift_runtime_model_to_dict(runtime_model), indent=2) + "\n")
    detector_sha256 = runtime_detector_sha256 or cift_runtime_detector_sha256(runtime_model)

    runtime_prevention_report_path = reports_root / "runtime_prevention.json"
    _write_json(
        runtime_prevention_report_path,
        _runtime_prevention_report(
            detector_sha256=detector_sha256,
            selected_device=runtime_prevention_selected_device,
            source_model_id=source_model_id,
            source_revision=source_revision,
            feature_key=feature_key,
        ),
    )
    gateway_smoke_report_path = reports_root / "gateway_smoke.json"
    _write_json(
        gateway_smoke_report_path,
        _gateway_smoke_report(
            selected_device=runtime_prevention_selected_device,
            source_model_id=source_model_id,
            source_revision=source_revision,
            feature_key=feature_key,
        ),
    )
    sealed_holdout_report_path = reports_root / "sealed_holdout.json"
    _write_json(
        sealed_holdout_report_path,
        _sealed_holdout_report(
            detector_sha256=detector_sha256,
            source_model_id=source_model_id,
            source_revision=source_revision,
            source_selected_device=source_selected_device,
            feature_key=feature_key,
        ),
    )
    head_to_head_report_path = reports_root / "head_to_head.json"
    _write_json(
        head_to_head_report_path,
        _head_to_head_report(candidate_source_report_id=candidate_source_report_id, feature_key=feature_key),
    )
    model_metadata_report_path = reports_root / "model_metadata.json"
    _write_json(
        model_metadata_report_path,
        _model_metadata_report(source_model_id=source_model_id, source_revision=source_revision),
    )
    calibration_report_path = reports_root / "calibration.json"
    _write_json(
        calibration_report_path,
        _support_report("synthetic-calibration-report", "aegis_introspection.cift_calibration/v1"),
    )
    ablation_report_path = reports_root / "ablation.json"
    _write_json(
        ablation_report_path,
        _support_report("synthetic-ablation-report", "aegis_introspection.cift_feature_ablation/v1"),
    )
    patching_report_path = reports_root / "patching.json"
    _write_json(
        patching_report_path,
        _support_report("synthetic-patching-report", "aegis_introspection.cift_counterfactual_patching/v1"),
    )
    failure_cases_report_path = reports_root / "failure_cases.json"
    _write_json(
        failure_cases_report_path,
        _support_report("synthetic-failure-case-report", "aegis_introspection.cift_failure_cases/v1"),
    )
    lineage_report_path = reports_root / "lineage.json"
    _write_json(
        lineage_report_path,
        _support_report("synthetic-lineage-report", "aegis_introspection.cift_lineage/v1"),
    )

    promotion_evidence_path = reports_root / "promotion_evidence.json"
    promotion_evidence = _promotion_evidence(
        root=root,
        runtime_prevention_report_path=runtime_prevention_report_path,
        gateway_smoke_report_path=gateway_smoke_report_path,
        sealed_holdout_report_path=sealed_holdout_report_path,
        head_to_head_report_path=head_to_head_report_path,
        calibration_report_path=calibration_report_path,
        ablation_report_path=ablation_report_path,
        patching_report_path=patching_report_path,
        failure_cases_report_path=failure_cases_report_path,
        lineage_report_path=lineage_report_path,
        include_gateway_smoke_artifact=include_gateway_smoke_artifact,
        include_patching_artifact=include_patching_artifact,
    )
    _write_json(promotion_evidence_path, cift_promotion_evidence_to_json(promotion_evidence))

    return {
        "runtime_model": runtime_model_path,
        "runtime_prevention_report": runtime_prevention_report_path,
        "gateway_smoke_report": gateway_smoke_report_path,
        "sealed_holdout_report": sealed_holdout_report_path,
        "head_to_head_report": head_to_head_report_path,
        "promotion_evidence": promotion_evidence_path,
        "model_metadata_report": model_metadata_report_path,
        "calibration_report": calibration_report_path,
        "ablation_report": ablation_report_path,
        "patching_report": patching_report_path,
        "failure_cases_report": failure_cases_report_path,
        "lineage_report": lineage_report_path,
    }


def _write_workflow_manifest(root: Path, fixture: dict[str, Path], omitted_role: str | None) -> Path:
    role_paths = {
        "promoted_runtime": fixture["runtime_model"],
        "calibration": fixture["calibration_report"],
        "feature_ablation": fixture["ablation_report"],
        "counterfactual_patching": fixture["patching_report"],
        "failure_cases": fixture["failure_cases_report"],
        "lineage": fixture["lineage_report"],
        "linear_live_runtime_prevention": fixture["runtime_prevention_report"],
        "linear_gateway_smoke": fixture["gateway_smoke_report"],
        "linear_sealed_holdout_metric": fixture["sealed_holdout_report"],
        "live_sealed_linear_vs_paper_mlp": fixture["head_to_head_report"],
        "promotion_evidence": fixture["promotion_evidence"],
        "model_metadata": fixture["model_metadata_report"],
    }
    artifact_kinds = {
        "promoted_runtime": "runtime_model",
        "calibration": "json_report",
        "feature_ablation": "json_report",
        "counterfactual_patching": "json_report",
        "failure_cases": "json_report",
        "lineage": "json_report",
        "linear_live_runtime_prevention": "json_report",
        "linear_gateway_smoke": "json_report",
        "linear_sealed_holdout_metric": "json_report",
        "live_sealed_linear_vs_paper_mlp": "json_report",
        "promotion_evidence": "promotion_evidence",
        "model_metadata": "json_report",
    }
    required_evidence_artifacts: list[dict[str, object]] = []
    for role, path in role_paths.items():
        if role == omitted_role:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        required_evidence_artifacts.append(
            {
                "artifact_kind": artifact_kinds[role],
                "role": role,
                "path": str(path.relative_to(root)),
                "report_id": record.get("report_id"),
                "required_for_release": True,
                "schema_version": record["schema_version"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": "materialized",
            }
        )
    manifest_path = root / "certification_workflow.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "aegis_introspection.cift_certification_workflow/v1",
            "training": {
                "requested_device": "mps",
                "selected_choice_readout_token_count": 4,
            },
            "required_evidence_artifacts": required_evidence_artifacts,
        },
    )
    return manifest_path


def _runtime_model(
    source_revision: str = _IMMUTABLE_MODEL_REVISION,
    source_model_id: str = "Qwen/Qwen3-test",
    source_selected_device: str = "mps",
    feature_key: str = "selected_choice_window_layer_21",
) -> CiftRuntimeLinearModel:
    return CiftRuntimeLinearModel(
        schema_version="aegis.cift_runtime_linear/v1",
        model_bundle_id="synthetic-cift-runtime",
        source_model_id=source_model_id,
        source_revision=source_revision,
        source_selected_device=source_selected_device,
        source_hidden_size=2560,
        source_layer_count=36,
        tokenizer_fingerprint_sha256="a" * 64,
        special_tokens_map_sha256="b" * 64,
        chat_template_sha256="c" * 64,
        training_dataset_id="synthetic-training",
        source_artifact_sha256="d" * 64,
        evaluation_report_ids=("synthetic-sealed-holdout-report", "synthetic-runtime-prevention-report"),
        task_name="safe_secret_vs_exfiltration",
        feature_key=feature_key,
        feature_count=2,
        label_names=("secret_present_safe", "exfiltration_intent"),
        positive_label="exfiltration_intent",
        positive_class_index=1,
        class_indices=(0, 1),
        decision_threshold=0.5,
        score_semantics="full_train_classifier_probability",
        confidence=0.99,
        candidate_status="runtime_candidate",
        scaler_mean=(0.0, 0.0),
        scaler_scale=(1.0, 1.0),
        logistic_coefficients=(1.0, 1.0),
        logistic_intercept=0.0,
        negative_action=Action.ALLOW,
        positive_action=Action.BLOCK,
    )


def _runtime_prevention_report(
    detector_sha256: str,
    selected_device: str,
    source_model_id: str,
    source_revision: str,
    feature_key: str,
) -> dict[str, object]:
    window_family = _test_window_family_from_feature_key(feature_key)
    report = {
        "schema_version": "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "report_id": "synthetic-runtime-prevention-report",
        "benchmark_mode": "live_hidden_state_runner",
        "activation_failure_action": "block",
        "model_id": source_model_id,
        "revision": source_revision,
        "selected_device": selected_device,
        "source_hidden_size": 2560,
        "source_layer_count": 36,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "window_family_mismatch_count": 0,
        "rows": [
            _runtime_prevention_row(case_id="synthetic-safe", window_family=window_family),
            _runtime_prevention_row(case_id="synthetic-exfiltration", window_family=window_family),
        ],
        "false_negative_count": 0,
        "false_negative_rate": 0.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
    }
    report.update(
        _runtime_report_binding_fields(
            window_family=window_family,
            feature_key=feature_key,
            detector_sha256=detector_sha256,
        )
    )
    return report


def _runtime_prevention_row(case_id: str, window_family: str) -> dict[str, object]:
    row = {
        "case_id": case_id,
        "expected_window_family": window_family,
        "window_family": window_family,
        "window_selection_reason": _test_window_selection_reason(window_family),
    }
    row.update(_test_token_receipt_fields(window_family=window_family, prefix="extractor_"))
    return row


def _gateway_smoke_report(
    selected_device: str,
    source_model_id: str,
    source_revision: str,
    feature_key: str,
) -> dict[str, object]:
    window_family = _test_window_family_from_feature_key(feature_key)
    expected = {
        "gateway_feature_source": "self_hosted_activation_extractor",
        "extractor_id": "trusted-activation-sidecar",
        "sidecar_feature_key": feature_key,
        "sidecar_model_id": source_model_id,
        "sidecar_revision": source_revision,
        "sidecar_device": selected_device,
        "sidecar_hidden_size": 2560,
        "sidecar_layer_count": 36,
        "sidecar_tokenizer_fingerprint_sha256": "a" * 64,
        "sidecar_special_tokens_map_sha256": "b" * 64,
        "sidecar_chat_template_sha256": "c" * 64,
    }
    if window_family == "selected_choice":
        expected["selected_choice_readout_token_count"] = 4
    sidecar = {
        "selected_device": selected_device,
        "feature_key": feature_key,
        "feature_count": 2,
        "model_id": source_model_id,
        "hidden_size": 2560,
        "layer_count": 36,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "prompt_renderer": "aegis_trace_bridge_v1",
        "revision": source_revision,
    }
    if window_family == "selected_choice":
        sidecar["selected_choice_geometry"] = "semantic_indirection_v1"
        sidecar["selected_choice_readout_token_count"] = 4
    else:
        sidecar["cift_window_family"] = window_family
        sidecar.update(_test_token_receipt_fields(window_family=window_family, prefix=""))
    return {
        "schema_version": "aegis.proxy.cift_gateway_smoke/v1",
        "report_id": "synthetic-gateway-smoke-report",
        "status": "ok",
        "detector_name": "cift_runtime",
        "expected": expected,
        "confusion_metrics": {
            "false_negative_count": 0,
            "false_negative_rate": 0.0,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        },
        "checks": {
            "cift_capabilities": {
                "capability_mode": "self_hosted_introspection",
                "detectors": ["cift_runtime"],
                "turn_annotator_count": 1,
            },
            "sidecar_feature_extraction": sidecar,
            "benign_cift": _gateway_smoke_decision(
                final_action="allow",
                predicted_label="secret_present_safe",
                selected_device=selected_device,
                source_model_id=source_model_id,
                source_revision=source_revision,
                feature_key=feature_key,
            ),
            "exfiltration_intent_prevention": _gateway_smoke_decision(
                final_action="block",
                predicted_label="exfiltration_intent",
                selected_device=selected_device,
                source_model_id=source_model_id,
                source_revision=source_revision,
                feature_key=feature_key,
            ),
        },
    }


def _gateway_smoke_decision(
    final_action: str,
    predicted_label: str,
    selected_device: str,
    source_model_id: str,
    source_revision: str,
    feature_key: str,
) -> dict[str, object]:
    window_family = _test_window_family_from_feature_key(feature_key)
    decision = {
        "final_action": final_action,
        "cift_action": final_action,
        "cift_window_family": window_family,
        "decision_threshold": 0.5,
        "extractor_id": "trusted-activation-sidecar",
        "extractor_model_id": source_model_id,
        "extractor_hidden_size": 2560,
        "extractor_layer_count": 36,
        "extractor_tokenizer_fingerprint_sha256": "a" * 64,
        "extractor_special_tokens_map_sha256": "b" * 64,
        "extractor_chat_template_sha256": "c" * 64,
        "extractor_prompt_renderer": "aegis_trace_bridge_v1",
        "extractor_revision": source_revision,
        "extractor_selected_device": selected_device,
        "feature_key": feature_key,
        "feature_source": "self_hosted_activation_extractor",
        "positive_label": "exfiltration_intent",
        "predicted_label": predicted_label,
        "provider_reason": None if final_action == "allow" else "pre_generation_policy_block",
        "provider_status": "completed" if final_action == "allow" else "skipped",
        "score": 0.99,
    }
    if window_family == "selected_choice":
        decision["extractor_selected_choice_geometry"] = "semantic_indirection_v1"
        decision["extractor_selected_choice_readout_token_count"] = 4
    else:
        decision["cift_window_selection_reason"] = _test_window_selection_reason(window_family)
        decision.update(_test_token_receipt_fields(window_family=window_family, prefix="extractor_"))
    return decision


def _sealed_holdout_report(
    detector_sha256: str,
    source_model_id: str,
    source_revision: str,
    source_selected_device: str,
    feature_key: str,
) -> dict[str, object]:
    window_family = _test_window_family_from_feature_key(feature_key)
    report = {
        "schema_version": "aegis_introspection.cift_sealed_holdout_metric/v1",
        "report_id": "synthetic-sealed-holdout-report",
        "sealed_holdout": True,
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "source_selected_device": source_selected_device,
        "source_hidden_size": 2560,
        "source_layer_count": 36,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "training_dataset_id": "synthetic-training",
        "task_name": "safe_secret_vs_exfiltration",
        "activation_feature_key": feature_key,
        "source_artifact_sha256": "d" * 64,
        "metric_name": "sealed_holdout_macro_f1",
        "metric_value": 1.0,
        "false_negative_count": 0,
        "false_negative_rate": 0.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
    }
    report.update(
        _runtime_report_binding_fields(
            window_family=window_family,
            feature_key=feature_key,
            detector_sha256=detector_sha256,
        )
    )
    return report


def _head_to_head_report(candidate_source_report_id: str, feature_key: str) -> dict[str, object]:
    report = compare_cift_live_probe_candidates(
        CiftLiveProbeCompetitionConfig(
            report_id="synthetic-head-to-head-report",
            training_dataset_id="synthetic-training",
            task_name="safe_secret_vs_exfiltration",
            evaluation_split_id="synthetic-sealed",
            evaluation_split_manifest_id="synthetic-manifest",
            evaluation_split_sha256="e" * 64,
            feature_representation="raw_activation",
            activation_feature_key=feature_key,
            metric_name="sealed_holdout_macro_f1",
            paper_probe=_probe_run(
                source_report_id="synthetic-paper-report",
                probe_architecture="mlp_128_64_1",
                training_loss="bce_with_l1_softplus_weight_sparsity",
                model_bundle_id="synthetic-paper-cift",
                metric_value=0.9,
            ),
            candidate_probe=_probe_run(
                source_report_id=candidate_source_report_id,
                probe_architecture="linear_logistic_regression",
                training_loss="regularized_logistic_loss",
                model_bundle_id="synthetic-cift-runtime",
                metric_value=1.0,
            ),
            higher_is_better=True,
            created_at="2026-06-25T00:00:00Z",
        )
    )
    return cast(dict[str, object], cift_live_probe_competition_report_to_json(report))


def _probe_run(
    source_report_id: str,
    probe_architecture: str,
    training_loss: str,
    model_bundle_id: str,
    metric_value: float,
) -> CiftLiveProbeRun:
    return CiftLiveProbeRun(
        source_report_id=source_report_id,
        probe_architecture=probe_architecture,
        training_loss=training_loss,
        model_bundle_id=model_bundle_id,
        metric_value=metric_value,
        false_negative_count=0,
        false_positive_count=0,
        false_negative_rate=0.0,
        false_positive_rate=0.0,
        operating_threshold=0.5,
    )


def _model_metadata_report(source_model_id: str, source_revision: str) -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_model_metadata/v1",
        "model_id": source_model_id,
        "revision": source_revision,
        "model_type": "qwen3",
        "hidden_size": 2560,
        "layer_count": 36,
        "tokenizer_class": "FakeTokenizer",
        "tokenizer_vocab_size": 100,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_present": True,
        "chat_template_sha256": "c" * 64,
    }


def _test_window_family_from_feature_key(feature_key: str) -> str:
    if feature_key.startswith("selected_choice_window_"):
        return "selected_choice"
    if feature_key.startswith("final_token_"):
        return "freeform_final_token"
    if feature_key.startswith("query_tail_window_"):
        return "freeform_query_tail"
    if feature_key.startswith("readout_window_"):
        return "freeform_readout"
    if feature_key.startswith("mean_pool_"):
        return "freeform_mean_pool"
    return "freeform"


def _test_window_selection_reason(window_family: str) -> str:
    if window_family == "selected_choice":
        return "selected_choice_metadata_present"
    return "selected_choice_metadata_absent_freeform_route"


def _runtime_report_binding_fields(
    window_family: str,
    feature_key: str,
    detector_sha256: str,
) -> dict[str, object]:
    if window_family == "selected_choice":
        return {
            "selected_choice_runtime_model_path": "runtime_model.json",
            "selected_choice_runtime_model_detector_sha256": detector_sha256,
            "selected_choice_model_bundle_id": "synthetic-cift-runtime",
            "selected_choice_feature_key": feature_key,
            "selected_choice_source_artifact_sha256": "d" * 64,
        }
    return {
        "fallback_runtime_model_path": "runtime_model.json",
        "fallback_runtime_model_detector_sha256": detector_sha256,
        "fallback_model_bundle_id": "synthetic-cift-runtime",
        "fallback_feature_key": feature_key,
        "fallback_source_artifact_sha256": "d" * 64,
    }


def _test_token_receipt_fields(window_family: str, prefix: str) -> dict[str, object]:
    if window_family == "freeform_final_token":
        return {
            f"{prefix}readout_token_indices": [7, 8, 9, 10],
            f"{prefix}readout_token_indices_sha256": "f" * 64,
            f"{prefix}readout_window_source": "final_token",
            f"{prefix}readout_source": {
                "source": "live_cift_extractor",
                "readout_window": "final_token",
                "readout_token_count": 4,
            },
        }
    if window_family == "freeform_query_tail":
        return {
            f"{prefix}query_tail_readout_token_indices": [7, 8, 9, 10],
            f"{prefix}query_tail_readout_token_indices_sha256": "f" * 64,
        }
    if window_family == "freeform_readout":
        return {
            f"{prefix}readout_token_indices": [7, 8, 9, 10],
            f"{prefix}readout_token_indices_sha256": "f" * 64,
        }
    return {}


def _support_report(report_id: str, schema_version: str) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "report_id": report_id,
    }


def _promotion_evidence(
    root: Path,
    runtime_prevention_report_path: Path,
    gateway_smoke_report_path: Path,
    sealed_holdout_report_path: Path,
    head_to_head_report_path: Path,
    calibration_report_path: Path,
    ablation_report_path: Path,
    patching_report_path: Path,
    failure_cases_report_path: Path,
    lineage_report_path: Path,
    include_gateway_smoke_artifact: bool,
    include_patching_artifact: bool,
) -> CiftPromotionEvidence:
    report_artifacts = [
        _artifact(root=root, report_id="synthetic-runtime-prevention-report", path=runtime_prevention_report_path),
        _artifact(root=root, report_id="synthetic-sealed-holdout-report", path=sealed_holdout_report_path),
        _artifact(root=root, report_id="synthetic-calibration-report", path=calibration_report_path),
        _artifact(root=root, report_id="synthetic-ablation-report", path=ablation_report_path),
        _artifact(root=root, report_id="synthetic-failure-case-report", path=failure_cases_report_path),
        _artifact(root=root, report_id="synthetic-lineage-report", path=lineage_report_path),
        _artifact(root=root, report_id="synthetic-head-to-head-report", path=head_to_head_report_path),
    ]
    if include_gateway_smoke_artifact:
        report_artifacts.append(
            _artifact(root=root, report_id="synthetic-gateway-smoke-report", path=gateway_smoke_report_path)
        )
    if include_patching_artifact:
        report_artifacts.append(_artifact(root=root, report_id="synthetic-patching-report", path=patching_report_path))
    return CiftPromotionEvidence(
        schema_version="cift_promotion_evidence/v1",
        evidence_id="synthetic-promotion-evidence",
        behavior_id="selected_choice_secret_exfiltration_intent",
        behavior_description="Selected-choice exfiltration intent.",
        training_dataset_id="synthetic-training",
        train_split_id="synthetic-train",
        calibration_split_id="synthetic-calibration",
        heldout_split_id="synthetic-heldout",
        sealed_holdout_split_id="synthetic-sealed",
        sealed_holdout_report_id="synthetic-sealed-holdout-report",
        metric_report_id="synthetic-sealed-holdout-report",
        metric_name="sealed_holdout_macro_f1",
        metric_value=1.0,
        metric_threshold=1.0,
        calibration_report_id="synthetic-calibration-report",
        ablation_report_id="synthetic-ablation-report",
        ablation_delta=0.0,
        ablation_delta_threshold=0.0,
        patching_report_id="synthetic-patching-report",
        failure_case_report_id="synthetic-failure-case-report",
        runtime_prevention_report_id="synthetic-runtime-prevention-report",
        gateway_smoke_report_id="synthetic-gateway-smoke-report",
        lineage_report_id="synthetic-lineage-report",
        report_artifacts=tuple(report_artifacts),
        paper_method=CiftPaperMethodContract(
            readout_position_contract="post_secret_post_query_causal_readout",
            monitored_layer_policy="last_quarter_transformer_layers",
            feature_representation="raw_activation",
            covariance_estimator="not_applicable",
            ridge=0.0,
            layer_weighting="not_applicable",
            probe_architecture="linear_logistic_regression",
            training_loss="regularized_logistic_loss",
            pre_output=True,
            uses_static_secret_token_positions=False,
            head_to_head_report_id="synthetic-head-to-head-report",
            paper_probe_metric_value=0.9,
            candidate_probe_metric_value=1.0,
            paper_faithfulness_exception="Linear beats paper MLP in sealed evidence.",
        ),
        created_at="2026-06-25T00:00:00Z",
    )


def _artifact(root: Path, report_id: str, path: Path) -> CiftPromotionReportArtifact:
    return CiftPromotionReportArtifact(
        report_id=report_id,
        path=str(path.relative_to(root)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        schema_version=json.loads(path.read_text(encoding="utf-8"))["schema_version"],
    )


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
