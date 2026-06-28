from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from aegis_introspection.cift_certification_workflow_runner import (
    CiftCertificationWorkflowRunnerConfig,
    CiftCertificationWorkflowRunnerError,
    run_cift_certification_workflow,
)

from aegis.cift_contract import CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION

_IMMUTABLE_MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"


class CiftCertificationWorkflowRunnerTest(unittest.TestCase):
    def test_dry_run_resolves_template_inputs_from_evidence_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_template_inputs(root)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(manifest_path, _templated_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

            self.assertFalse(report.eligible)
            self.assertTrue(report.plan_eligible)
            self.assertTrue(report.evidence_eligible)
            self.assertFalse(report.certification_eligible)
            self.assertEqual("dry_run", report.mode)
            self.assertIn(
                "certification workflow run must be execute mode for release evidence",
                report.failed_requirements,
            )
            self.assertEqual(1, report.step_count)
            self.assertEqual("planned", report.steps[0].status)
            self.assertIn("0.91", report.steps[0].argv)
            self.assertIn("0.25", report.steps[0].argv)
            self.assertTrue(output_path.exists())

    def test_dry_run_defers_template_resolution_when_future_inputs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(manifest_path, _templated_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

            self.assertFalse(report.eligible)
            self.assertTrue(report.plan_eligible)
            self.assertTrue(report.evidence_eligible)
            self.assertFalse(report.certification_eligible)
            self.assertIn(
                "certification workflow run must be execute mode for release evidence",
                report.failed_requirements,
            )
            self.assertEqual("planned", report.steps[0].status)
            self.assertIn("${linear_sealed_holdout_metric.metric_value}", report.steps[0].argv)
            self.assertIn("template resolution deferred", report.steps[0].stderr_tail or "")

    def test_dry_run_requires_command_timeout_for_release_eligible_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_template_inputs(root)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(manifest_path, _templated_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=None,
                )
            )
            run_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertFalse(report.eligible)
        self.assertFalse(report.plan_eligible)
        self.assertTrue(report.evidence_eligible)
        self.assertIsNone(report.command_timeout_seconds)
        self.assertIn("certification workflow run requires command_timeout_seconds", report.failed_requirements)
        self.assertIsNone(run_record["command_timeout_seconds"])

    def test_dry_run_resolves_operator_supplied_template_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(manifest_path, _operator_templated_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={
                        "extractor_id": "trusted-activation-sidecar",
                        "gateway_base_url": "http://127.0.0.1:8000",
                        "gateway_model": "mock-model",
                        "sidecar_base_url": "http://127.0.0.1:9000",
                    },
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertTrue(report.plan_eligible)
        self.assertTrue(report.evidence_eligible)
        self.assertFalse(report.certification_eligible)
        self.assertIn(
            "certification workflow run must be execute mode for release evidence",
            report.failed_requirements,
        )
        self.assertEqual("planned", report.steps[0].status)
        self.assertIsNone(report.steps[0].stderr_tail)
        self.assertIn("trusted-activation-sidecar", report.steps[0].argv)
        self.assertIn("http://127.0.0.1:8000", report.steps[0].argv)
        self.assertIn("http://127.0.0.1:9000", report.steps[0].argv)
        self.assertIn("mock-model", report.steps[0].argv)

    def test_execute_fails_when_operator_supplied_template_values_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(manifest_path, _operator_templated_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.plan_eligible)
        self.assertEqual("failed", report.steps[0].status)
        self.assertIn("template context gateway_base_url", report.steps[0].stderr_tail or "")

    def test_operator_template_values_cannot_override_workflow_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(manifest_path, _operator_templated_manifest())

            with self.assertRaisesRegex(CiftCertificationWorkflowRunnerError, "may not override workflow context"):
                run_cift_certification_workflow(
                    CiftCertificationWorkflowRunnerConfig(
                        repository_root=root,
                        workflow_manifest_path=manifest_path,
                        output_path=output_path,
                        execute=False,
                        allow_sealed_holdout_execution=False,
                        overwrite_existing_outputs=False,
                        template_values={"workflow.training.candidate_feature_key": "wrong_layer"},
                        command_timeout_seconds=30.0,
                    )
                )

    def test_execute_runs_argv_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            produced_path = root / "runner-output.txt"
            _write_json(manifest_path, _executable_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

            self.assertTrue(report.eligible)
            self.assertTrue(report.plan_eligible)
            self.assertTrue(report.evidence_eligible)
            self.assertEqual("execute", report.mode)
            self.assertEqual(30.0, report.command_timeout_seconds)
            self.assertEqual("passed", report.steps[0].status)
            self.assertEqual(0, report.steps[0].returncode)
            self.assertEqual("ok", produced_path.read_text(encoding="utf-8"))
            run_record = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(30.0, run_record["command_timeout_seconds"])

    def test_execute_requires_command_timeout_before_running_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            produced_path = root / "runner-output.txt"
            _write_json(manifest_path, _executable_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=None,
                )
            )
            run_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertFalse(report.eligible)
        self.assertFalse(report.plan_eligible)
        self.assertIsNone(report.command_timeout_seconds)
        self.assertFalse(produced_path.exists())
        self.assertEqual("failed", report.steps[0].status)
        self.assertIn("execute mode requires command_timeout_seconds", report.failed_requirements)
        self.assertIsNone(run_record["command_timeout_seconds"])

    def test_execute_marks_step_failed_when_command_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            produced_path = root / "runner-output.txt"
            _write_json(manifest_path, _slow_executable_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=0.01,
                )
            )
            run_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertFalse(report.eligible)
        self.assertFalse(report.plan_eligible)
        self.assertEqual(0.01, report.command_timeout_seconds)
        self.assertFalse(produced_path.exists())
        self.assertEqual("failed", report.steps[0].status)
        self.assertIsNone(report.steps[0].returncode)
        self.assertIn("command timed out after", report.steps[0].stderr_tail or "")
        self.assertEqual(0.01, run_record["command_timeout_seconds"])

    def test_execute_skips_existing_outputs_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            produced_path = root / "runner-output.txt"
            produced_path.write_text("existing", encoding="utf-8")
            _write_json(
                manifest_path,
                _executable_manifest(
                    required_artifacts=[
                        _raw_artifact(
                            path="runner-output.txt",
                            sha256=hashlib.sha256(b"existing").hexdigest(),
                            artifact_kind="text_artifact",
                            schema_version=None,
                            report_id=None,
                        )
                    ]
                ),
            )

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

            self.assertTrue(report.eligible)
            self.assertTrue(report.plan_eligible)
            self.assertTrue(report.evidence_eligible)
            self.assertEqual("skipped", report.steps[0].status)
            self.assertEqual("existing", produced_path.read_text(encoding="utf-8"))

    def test_execute_rejects_unbound_existing_outputs_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            produced_path = root / "runner-output.txt"
            produced_path.write_text("existing", encoding="utf-8")
            _write_json(manifest_path, _executable_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.plan_eligible)
        self.assertEqual("failed", report.steps[0].status)
        self.assertIn("not manifest-bound", report.steps[0].stderr_tail or "")

    def test_execute_rejects_sealed_holdout_step_without_explicit_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(manifest_path, _sealed_executable_manifest())

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.plan_eligible)
        self.assertEqual("failed", report.steps[0].status)
        self.assertIn("sealed holdout execution", report.steps[0].stderr_tail or "")

    def test_dry_run_verifies_manifest_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_content = (
                json.dumps(
                    {
                        "report_id": "synthetic-report",
                        "schema_version": "synthetic/v1",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(
                manifest_path,
                _artifact_manifest(
                    sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                    schema_version="synthetic/v1",
                    report_id="synthetic-report",
                ),
            )

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertTrue(report.plan_eligible)
        self.assertTrue(report.evidence_eligible)
        self.assertFalse(report.certification_eligible)
        self.assertIn(
            "certification workflow run must be execute mode for release evidence",
            report.failed_requirements,
        )
        self.assertEqual(1, report.artifact_count)
        self.assertEqual("verified", report.artifacts[0].actual_status)
        self.assertEqual("synthetic/v1", report.artifacts[0].actual_schema_version)
        self.assertEqual("synthetic-report", report.artifacts[0].actual_report_id)

    def test_dry_run_rejects_manifest_artifact_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_path.write_text(
                '{"report_id":"synthetic-report","schema_version":"synthetic/v1"}\n',
                encoding="utf-8",
            )
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(
                manifest_path,
                _artifact_manifest(
                    sha256="0" * 64,
                    schema_version="synthetic/v1",
                    report_id="synthetic-report",
                ),
            )

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertEqual("failed", report.artifacts[0].actual_status)
        self.assertIn("artifact sha256 must match manifest", report.artifacts[0].failed_requirements)

    def test_evidence_bound_rejects_planned_artifact_without_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_path.write_text(
                '{"report_id":"synthetic-report","schema_version":"synthetic/v1"}\n',
                encoding="utf-8",
            )
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=None,
                schema_version="synthetic/v1",
                report_id="synthetic-report",
                status="planned",
            )
            manifest["status"] = "evidence_bound"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn("evidence-bound required artifact must be materialized", report.artifacts[0].failed_requirements)
        self.assertIn(
            "artifact synthetic_report is required for release but is not materialized",
            report.failed_requirements,
        )
        self.assertIn(
            "evidence-bound required artifact sha256 must be present",
            report.artifacts[0].failed_requirements,
        )

    def test_json_artifact_expected_identity_must_be_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_content = "{}\n"
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(
                manifest_path,
                _artifact_manifest(
                    sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                    schema_version="synthetic/v1",
                    report_id="synthetic-report",
                ),
            )

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn("artifact schema_version must be present", report.artifacts[0].failed_requirements)
        self.assertIn("artifact report_id must be present", report.artifacts[0].failed_requirements)

    def test_evidence_bound_rejects_runtime_prevention_device_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_content = (
                json.dumps(
                    {
                        "activation_failure_action": "block",
                        "benchmark_mode": "live_hidden_state_runner",
                        "false_negative_count": 0,
                        "false_negative_rate": 0.0,
                        "false_positive_count": 0,
                        "false_positive_rate": 0.0,
                        "report_id": "synthetic-report",
                        "rows": [
                            {
                                "expected_window_family": "selected_choice",
                                "window_family": "selected_choice",
                                "window_selection_reason": "selected_choice_metadata_present",
                            }
                        ],
                        "schema_version": "synthetic/v1",
                        "selected_device": "cpu",
                        "window_family_mismatch_count": 0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="synthetic/v1",
                report_id="synthetic-report",
                role="linear_live_runtime_prevention",
            )
            manifest["status"] = "evidence_bound"
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "runtime prevention selected_device must match requested_device",
            report.artifacts[0].failed_requirements,
        )

    def test_evidence_bound_rejects_device_preflight_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "device_preflight.json"
            artifact_content = (
                json.dumps(
                    {
                        "eligible": True,
                        "requested_device": "mps",
                        "schema_version": "aegis_introspection.device_preflight/v1",
                        "selected_device": "cpu",
                        "smoke_tensor_device": "cpu",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis_introspection.device_preflight/v1",
                report_id=None,
                role="device_preflight",
                path="device_preflight.json",
            )
            manifest["status"] = "evidence_bound"
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "device preflight selected_device must match workflow requested_device",
            report.artifacts[0].failed_requirements,
        )

    def test_runner_rejects_qwen3_4b_workflow_without_mps_training_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = _executable_manifest()
            manifest["model_identity"] = {"model_id": "Qwen/Qwen3-4B", "revision": _IMMUTABLE_MODEL_REVISION}
            training = cast(dict[str, object], manifest["training"])
            training["requested_device"] = "cpu"
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)
            output_path = root / "run.json"

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.plan_eligible)
        self.assertIn(
            "Qwen/Qwen3-4B certification workflow requires training.requested_device mps",
            report.failed_requirements,
        )

    def test_runner_rejects_mutable_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = _executable_manifest()
            manifest["model_identity"] = {"model_id": "Qwen/Qwen3-test", "revision": "main"}
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)
            output_path = root / "run.json"

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.plan_eligible)
        self.assertIn(
            "certification workflow model_identity.revision must be an immutable lowercase 40-character Git commit SHA "
            "or sha256:<64 lowercase hex digest>",
            report.failed_requirements,
        )

    def test_runner_does_not_execute_commands_when_model_revision_policy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = _executable_manifest()
            manifest["model_identity"] = {"model_id": "Qwen/Qwen3-test", "revision": "main"}
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)
            output_path = root / "run.json"

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )
            command_output_exists = (root / "runner-output.txt").exists()

        self.assertFalse(report.plan_eligible)
        self.assertFalse(command_output_exists)
        self.assertEqual("failed", report.steps[0].status)
        self.assertIn("workflow policy failed before execution", report.steps[0].stderr_tail or "")

    def test_runner_rejects_materialized_runtime_artifact_with_mutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "runtime.json"
            artifact_content = (
                json.dumps(
                    {
                        "schema_version": "aegis.cift_runtime_linear/v1",
                        "source_revision": "main",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(
                manifest_path,
                _artifact_manifest(
                    sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                    schema_version="aegis.cift_runtime_linear/v1",
                    report_id=None,
                    role="promoted_runtime",
                    path="runtime.json",
                    artifact_kind="runtime_model",
                ),
            )

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "runtime model source_revision must be an immutable lowercase 40-character Git commit SHA "
            "or sha256:<64 lowercase hex digest>",
            report.artifacts[0].failed_requirements,
        )

    def test_materialized_artifacts_in_planned_workflow_receive_semantic_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_content = (
                json.dumps(
                    {
                        "activation_failure_action": "block",
                        "benchmark_mode": "live_hidden_state_runner",
                        "false_negative_count": 0,
                        "false_negative_rate": 0.0,
                        "false_positive_count": 0,
                        "false_positive_rate": 0.0,
                        "report_id": "synthetic-report",
                        "rows": [
                            {
                                "expected_window_family": "selected_choice",
                                "window_family": "selected_choice",
                                "window_selection_reason": "selected_choice_metadata_present",
                            }
                        ],
                        "schema_version": "synthetic/v1",
                        "selected_device": "cpu",
                        "window_family_mismatch_count": 0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="synthetic/v1",
                report_id="synthetic-report",
                role="linear_live_runtime_prevention",
            )
            manifest["command_plan"] = [
                {
                    "step_id": "existing_runtime_prevention",
                    "evidence_item": "live_hidden_state_runtime_prevention",
                    "argv": ["python", "-c", "print('not reached')"],
                    "produces": ["evidence.json"],
                    "sealed_holdout_access": False,
                }
            ]
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertTrue(report.plan_eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertEqual("skipped", report.steps[0].status)
        self.assertIn(
            "runtime prevention selected_device must match requested_device",
            report.artifacts[0].failed_requirements,
        )

    def test_evidence_bound_rejects_cross_model_calibration_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "calibration.json"
            artifact_content = (
                json.dumps(
                    {
                        "activation_feature_key": "selected_choice_window_layer_21",
                        "positive_label": "exfiltration_intent",
                        "report_id": "synthetic-calibration",
                        "schema_version": "aegis_introspection.cift_calibration/v1",
                        "source_model_id": "Qwen/Qwen3-other",
                        "source_revision": _IMMUTABLE_MODEL_REVISION,
                        "source_selected_device": "mps",
                        "task_name": "safe_secret_vs_exfiltration",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis_introspection.cift_calibration/v1",
                report_id="synthetic-calibration",
                role="calibration",
                path="calibration.json",
            )
            manifest["status"] = "evidence_bound"
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            training["task_name"] = "safe_secret_vs_exfiltration"
            training["positive_label"] = "exfiltration_intent"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "calibration source_model_id must match workflow model_identity.model_id",
            report.artifacts[0].failed_requirements,
        )

    def test_evidence_bound_rejects_feature_ablation_candidate_feature_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "ablation.json"
            artifact_content = (
                json.dumps(
                    {
                        "baseline_feature_key": "readout_window_layer_21",
                        "report_id": "synthetic-ablation",
                        "schema_version": "aegis_introspection.cift_feature_ablation/v1",
                        "source_model_id": "Qwen/Qwen3-test",
                        "source_revision": _IMMUTABLE_MODEL_REVISION,
                        "source_selected_device": "mps",
                        "task_name": "safe_secret_vs_exfiltration",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis_introspection.cift_feature_ablation/v1",
                report_id="synthetic-ablation",
                role="feature_ablation",
                path="ablation.json",
            )
            manifest["status"] = "evidence_bound"
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            training["task_name"] = "safe_secret_vs_exfiltration"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=True,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "feature ablation baseline_feature_key must match workflow training.candidate_feature_key",
            report.artifacts[0].failed_requirements,
        )

    def test_evidence_bound_rejects_runtime_prevention_row_without_selected_choice_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_content = (
                json.dumps(
                    {
                        "activation_failure_action": "block",
                        "benchmark_mode": "live_hidden_state_runner",
                        "false_negative_count": 0,
                        "false_negative_rate": 0.0,
                        "false_positive_count": 0,
                        "false_positive_rate": 0.0,
                        "report_id": "synthetic-report",
                        "rows": [
                            {
                                "expected_window_family": "selected_choice",
                                "window_family": "selected_choice",
                                "window_selection_reason": "fallback_metadata",
                            }
                        ],
                        "schema_version": "synthetic/v1",
                        "selected_device": "mps",
                        "window_family_mismatch_count": 0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="synthetic/v1",
                report_id="synthetic-report",
                role="linear_live_runtime_prevention",
            )
            manifest["status"] = "evidence_bound"
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "runtime prevention rows must have route-specific extraction metadata proof",
            report.artifacts[0].failed_requirements,
        )

    def test_evidence_bound_accepts_freeform_final_token_runtime_route_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "evidence.json"
            artifact_content = json.dumps(_freeform_final_token_runtime_report(), sort_keys=True) + "\n"
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis_introspection.cift_live_window_selector_benchmark/v1",
                report_id="synthetic-final-token-runtime",
                role="linear_live_runtime_prevention",
            )
            manifest["status"] = "evidence_bound"
            training = manifest["training"]
            assert isinstance(training, dict)
            training["candidate_feature_key"] = "final_token_layer_12"
            training["requested_device"] = "mps"
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertTrue(report.evidence_eligible)
        self.assertTrue(report.artifacts[0].eligible)
        self.assertEqual((), report.artifacts[0].failed_requirements)

    def test_evidence_bound_accepts_freeform_final_token_gateway_route_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "gateway-smoke.json"
            artifact_content = json.dumps(_freeform_final_token_gateway_smoke_report(), sort_keys=True) + "\n"
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis.proxy.cift_gateway_smoke/v1",
                report_id="synthetic-final-token-gateway-smoke",
                role="linear_gateway_smoke",
                path="gateway-smoke.json",
            )
            manifest["status"] = "evidence_bound"
            manifest["model_identity"] = _model_identity_contract()
            training = manifest["training"]
            assert isinstance(training, dict)
            training["candidate_feature_key"] = "final_token_layer_12"
            training["requested_device"] = "mps"
            training["selected_choice_readout_token_count"] = 4
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertTrue(report.evidence_eligible)
        self.assertTrue(report.artifacts[0].eligible)
        self.assertEqual((), report.artifacts[0].failed_requirements)

    def test_evidence_bound_rejects_gateway_smoke_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "gateway-smoke.json"
            artifact_content = json.dumps(_gateway_smoke_report(model_id="Qwen/Qwen3-other", readout_count=1)) + "\n"
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis.proxy.cift_gateway_smoke/v1",
                report_id="synthetic-gateway-smoke",
                role="linear_gateway_smoke",
                path="gateway-smoke.json",
            )
            manifest["status"] = "evidence_bound"
            manifest["model_identity"] = {
                "chat_template_sha256": "d" * 64,
                "hidden_size": 2560,
                "layer_count": 36,
                "model_id": "Qwen/Qwen3-test",
                "revision": _IMMUTABLE_MODEL_REVISION,
                "special_tokens_map_sha256": "c" * 64,
                "tokenizer_fingerprint_sha256": "b" * 64,
            }
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            training["selected_choice_readout_token_count"] = 4
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "gateway smoke expected.sidecar_model_id must match workflow manifest contract",
            report.artifacts[0].failed_requirements,
        )
        self.assertIn(
            "gateway smoke expected.selected_choice_readout_token_count must match workflow manifest contract",
            report.artifacts[0].failed_requirements,
        )
        self.assertIn(
            "gateway smoke sidecar feature extraction selected_choice_readout_token_count must match workflow "
            "manifest contract",
            report.artifacts[0].failed_requirements,
        )

    def test_evidence_bound_rejects_gateway_smoke_without_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "gateway-smoke.json"
            gateway_smoke = _gateway_smoke_report(model_id="Qwen/Qwen3-test", readout_count=4)
            checks = gateway_smoke["checks"]
            assert isinstance(checks, dict)
            del checks["gateway_readiness"]
            artifact_content = json.dumps(gateway_smoke) + "\n"
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis.proxy.cift_gateway_smoke/v1",
                report_id="synthetic-gateway-smoke",
                role="linear_gateway_smoke",
                path="gateway-smoke.json",
            )
            manifest["status"] = "evidence_bound"
            manifest["model_identity"] = {
                "chat_template_sha256": "d" * 64,
                "hidden_size": 2560,
                "layer_count": 36,
                "model_id": "Qwen/Qwen3-test",
                "revision": _IMMUTABLE_MODEL_REVISION,
                "special_tokens_map_sha256": "c" * 64,
                "tokenizer_fingerprint_sha256": "b" * 64,
            }
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            training["selected_choice_readout_token_count"] = 4
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "gateway smoke gateway_readiness must be an object",
            report.artifacts[0].failed_requirements,
        )

    def test_evidence_bound_accepts_bootstrap_gateway_readiness_without_release_gate_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "gateway-smoke.json"
            gateway_smoke = _gateway_smoke_report(model_id="Qwen/Qwen3-test", readout_count=4)
            checks = gateway_smoke["checks"]
            assert isinstance(checks, dict)
            readiness = checks["gateway_readiness"]
            assert isinstance(readiness, dict)
            readiness["certification_mode"] = "gateway_smoke_bootstrap"
            readiness["certification_id"] = None
            readiness["release_gate_report_sha256"] = None
            artifact_content = json.dumps(gateway_smoke) + "\n"
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis.proxy.cift_gateway_smoke/v1",
                report_id="synthetic-gateway-smoke",
                role="linear_gateway_smoke",
                path="gateway-smoke.json",
            )
            manifest["status"] = "evidence_bound"
            manifest["model_identity"] = {
                "chat_template_sha256": "d" * 64,
                "hidden_size": 2560,
                "layer_count": 36,
                "model_id": "Qwen/Qwen3-test",
                "revision": _IMMUTABLE_MODEL_REVISION,
                "special_tokens_map_sha256": "c" * 64,
                "tokenizer_fingerprint_sha256": "b" * 64,
            }
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            training["selected_choice_readout_token_count"] = 4
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertEqual((), report.artifacts[0].failed_requirements)

    def test_evidence_bound_rejects_gateway_smoke_float_selected_choice_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "gateway-smoke.json"
            artifact_content = json.dumps(_gateway_smoke_report(model_id="Qwen/Qwen3-test", readout_count=4.0)) + "\n"
            artifact_path.write_text(artifact_content, encoding="utf-8")
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _artifact_manifest(
                sha256=hashlib.sha256(artifact_content.encode("utf-8")).hexdigest(),
                schema_version="aegis.proxy.cift_gateway_smoke/v1",
                report_id="synthetic-gateway-smoke",
                role="linear_gateway_smoke",
                path="gateway-smoke.json",
            )
            manifest["status"] = "evidence_bound"
            manifest["model_identity"] = {
                "chat_template_sha256": "d" * 64,
                "hidden_size": 2560,
                "layer_count": 36,
                "model_id": "Qwen/Qwen3-test",
                "revision": _IMMUTABLE_MODEL_REVISION,
                "special_tokens_map_sha256": "c" * 64,
                "tokenizer_fingerprint_sha256": "b" * 64,
            }
            training = manifest["training"]
            assert isinstance(training, dict)
            training["requested_device"] = "mps"
            training["selected_choice_readout_token_count"] = 4
            _write_json(manifest_path, manifest)

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertFalse(report.eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertIn(
            "gateway smoke expected.selected_choice_readout_token_count must be a positive integer",
            report.artifacts[0].failed_requirements,
        )
        self.assertIn(
            "gateway smoke sidecar feature extraction selected_choice_readout_token_count must be a positive integer",
            report.artifacts[0].failed_requirements,
        )

    def test_dry_run_with_planned_required_artifact_is_not_evidence_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            _write_json(
                manifest_path,
                _artifact_manifest(
                    sha256=None,
                    schema_version="synthetic/v1",
                    report_id="synthetic-report",
                    status="planned",
                ),
            )

            report = run_cift_certification_workflow(
                CiftCertificationWorkflowRunnerConfig(
                    repository_root=root,
                    workflow_manifest_path=manifest_path,
                    output_path=output_path,
                    execute=False,
                    allow_sealed_holdout_execution=False,
                    overwrite_existing_outputs=False,
                    template_values={},
                    command_timeout_seconds=30.0,
                )
            )

        self.assertTrue(report.plan_eligible)
        self.assertFalse(report.evidence_eligible)
        self.assertFalse(report.eligible)

    def test_rejects_duplicate_step_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "workflow.json"
            output_path = root / "run_report.json"
            manifest = _executable_manifest()
            command_plan = manifest["command_plan"]
            assert isinstance(command_plan, list)
            command_plan.append(command_plan[0])
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(CiftCertificationWorkflowRunnerError, "duplicate step_id"):
                run_cift_certification_workflow(
                    CiftCertificationWorkflowRunnerConfig(
                        repository_root=root,
                        workflow_manifest_path=manifest_path,
                        output_path=output_path,
                        execute=False,
                        allow_sealed_holdout_execution=False,
                        overwrite_existing_outputs=False,
                        template_values={},
                        command_timeout_seconds=30.0,
                    )
                )


def _write_template_inputs(root: Path) -> None:
    _write_json(root / "metric.json", {"metric_value": 0.91})
    _write_json(
        root / "ablation.json",
        {
            "variants": [
                {
                    "feature_key": "selected_choice_window_layer_21",
                    "macro_f1_mean": 0.70,
                },
                {
                    "feature_key": "combined_readout_window_layer_21",
                    "macro_f1_mean": 0.95,
                },
            ]
        },
    )


def _templated_manifest() -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_certification_workflow/v1",
        "certification_id": "synthetic_cift",
        "model_identity": {"model_id": "Qwen/Qwen3-test", "revision": _IMMUTABLE_MODEL_REVISION},
        "training": {
            "candidate_feature_key": "selected_choice_window_layer_21",
        },
        "command_plan": [
            {
                "step_id": "materialize_promotion_evidence",
                "evidence_item": "promotion_evidence_materialized",
                "argv_template": [
                    "python",
                    "script.py",
                    "--metric-value",
                    "${linear_sealed_holdout_metric.metric_value}",
                    "--ablation-delta",
                    "${feature_ablation_report.ablation_delta}",
                ],
                "template_inputs": [
                    {
                        "name": "linear_sealed_holdout_metric.metric_value",
                        "path": "metric.json",
                        "json_pointer": "/metric_value",
                    },
                    {
                        "name": "feature_ablation_report.ablation_delta",
                        "path": "ablation.json",
                        "derivation": "best_variant_macro_f1 - selected_candidate_feature_macro_f1",
                    },
                ],
                "produces": ["promotion.json"],
                "consumes": ["metric.json", "ablation.json"],
                "sealed_holdout_access": True,
            }
        ],
    }


def _operator_templated_manifest() -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_certification_workflow/v1",
        "certification_id": "synthetic_cift",
        "model_identity": {"model_id": "Qwen/Qwen3-test", "revision": _IMMUTABLE_MODEL_REVISION},
        "training": {
            "candidate_feature_key": "selected_choice_window_layer_21",
        },
        "command_plan": [
            {
                "step_id": "run_linear_gateway_smoke",
                "evidence_item": "live_gateway_sidecar_runtime_prevention",
                "argv_template": [
                    "aegis-proxy-cift-smoke",
                    "--url",
                    "${gateway_base_url}",
                    "--sidecar-url",
                    "${sidecar_base_url}",
                    "--gateway-model",
                    "${gateway_model}",
                    "--expected-extractor-id",
                    "${extractor_id}",
                ],
                "template_inputs": [
                    {
                        "name": "gateway_base_url",
                        "derivation": "operator-supplied running gateway base URL",
                    },
                    {
                        "name": "sidecar_base_url",
                        "derivation": "operator-supplied running CIFT activation sidecar base URL",
                    },
                    {
                        "name": "gateway_model",
                        "derivation": "operator-supplied gateway model name",
                    },
                    {
                        "name": "extractor_id",
                        "derivation": "operator-supplied trusted activation sidecar identifier",
                    },
                ],
                "produces": [],
                "consumes": [],
                "sealed_holdout_access": False,
            }
        ],
    }


def _executable_manifest(required_artifacts: list[dict[str, object]] | None = None) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "aegis_introspection.cift_certification_workflow/v1",
        "certification_id": "synthetic_cift",
        "model_identity": {"model_id": "Qwen/Qwen3-test", "revision": _IMMUTABLE_MODEL_REVISION},
        "training": {
            "candidate_feature_key": "selected_choice_window_layer_21",
        },
        "command_plan": [
            {
                "step_id": "write_file",
                "evidence_item": "model_metadata_discovered",
                "argv": [
                    "python",
                    "-c",
                    "from pathlib import Path\nPath('runner-output.txt').write_text('ok', encoding='utf-8')",
                ],
                "produces": ["runner-output.txt"],
                "sealed_holdout_access": False,
            }
        ],
    }
    if required_artifacts is not None:
        manifest["required_evidence_artifacts"] = required_artifacts
    return manifest


def _sealed_executable_manifest() -> dict[str, object]:
    manifest = _executable_manifest()
    command_plan = manifest["command_plan"]
    assert isinstance(command_plan, list)
    step = command_plan[0]
    assert isinstance(step, dict)
    step["sealed_holdout_access"] = True
    return manifest


def _slow_executable_manifest() -> dict[str, object]:
    manifest = _executable_manifest()
    command_plan = manifest["command_plan"]
    assert isinstance(command_plan, list)
    step = command_plan[0]
    assert isinstance(step, dict)
    step["argv"] = [
        "python",
        "-c",
        (
            "import time\n"
            "from pathlib import Path\n"
            "time.sleep(5)\n"
            "Path('runner-output.txt').write_text('late', encoding='utf-8')"
        ),
    ]
    return manifest


def _artifact_manifest(
    sha256: str | None,
    schema_version: str,
    report_id: str | None,
    status: str = "materialized",
    role: str = "synthetic_report",
    path: str = "evidence.json",
    artifact_kind: str = "json_report",
) -> dict[str, object]:
    return {
        "schema_version": "aegis_introspection.cift_certification_workflow/v1",
        "certification_id": "synthetic_cift",
        "model_identity": {"model_id": "Qwen/Qwen3-test", "revision": _IMMUTABLE_MODEL_REVISION},
        "training": {
            "candidate_feature_key": "selected_choice_window_layer_21",
        },
        "command_plan": [],
        "required_evidence_artifacts": [
            _raw_artifact(
                role=role,
                path=path,
                status=status,
                sha256=sha256,
                schema_version=schema_version,
                report_id=report_id,
                artifact_kind=artifact_kind,
            )
        ],
    }


def _raw_artifact(
    path: str,
    sha256: str | None,
    schema_version: str | None,
    report_id: str | None,
    role: str = "synthetic_report",
    status: str = "materialized",
    artifact_kind: str = "json_report",
) -> dict[str, object]:
    return {
        "role": role,
        "artifact_kind": artifact_kind,
        "path": path,
        "status": status,
        "required_for_release": True,
        "sha256": sha256,
        "schema_version": schema_version,
        "report_id": report_id,
    }


def _gateway_smoke_report(model_id: str, readout_count: object) -> dict[str, object]:
    return {
        "checks": {
            "benign_cift": _gateway_smoke_decision(
                final_action="allow",
                predicted_label="secret_present_safe",
                provider_reason=None,
                provider_status="completed",
                model_id=model_id,
                readout_count=readout_count,
            ),
            "cift_capabilities": {
                "capability_mode": "self_hosted_introspection",
                "detectors": ["cift_runtime"],
                "turn_annotator_count": 1,
            },
            "exfiltration_intent_prevention": _gateway_smoke_decision(
                final_action="block",
                predicted_label="exfiltration_intent",
                provider_reason="pre_generation_policy_block",
                provider_status="skipped",
                model_id=model_id,
                readout_count=readout_count,
            ),
            "gateway_readiness": _gateway_smoke_readiness(model_id=model_id, readout_count=readout_count),
            "sidecar_feature_extraction": {
                "chat_template_sha256": "d" * 64,
                "feature_count": 2560,
                "feature_vector_length": 2560,
                "feature_vector_sha256": "e" * 64,
                "feature_key": "selected_choice_window_layer_21",
                "hidden_size": 2560,
                "hidden_state_device_observed": "mps:0",
                "hidden_state_layer_count": 37,
                "input_device_observed": "mps:0",
                "layer_count": 36,
                "model_id": model_id,
                "prompt_renderer": "aegis.cift_prompt_renderer.trace_bridge/v1",
                "rendered_prompt_sha256": "f" * 64,
                "revision": _IMMUTABLE_MODEL_REVISION,
                "extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
                "selected_choice_geometry": "aegis.cift_selected_choice_geometry.semantic_indirection/v1",
                "selected_choice_readout_token_count": readout_count,
                "selected_choice_readout_token_indices": [11, 12, 13, 14],
                "selected_choice_readout_token_indices_sha256": _json_sha256([11, 12, 13, 14]),
                "selected_device": "mps",
                "special_tokens_map_sha256": "c" * 64,
                "tokenizer_fingerprint_sha256": "b" * 64,
            },
        },
        "confusion_metrics": {
            "false_negative_count": 0,
            "false_negative_rate": 0.0,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        },
        "detector_name": "cift_runtime",
        "expected": {
            "extractor_id": "trusted-activation-sidecar",
            "gateway_feature_source": "self_hosted_activation_extractor",
            "selected_choice_readout_token_count": readout_count,
            "sidecar_chat_template_sha256": "d" * 64,
            "sidecar_device": "mps",
            "sidecar_feature_key": "selected_choice_window_layer_21",
            "sidecar_hidden_size": 2560,
            "sidecar_layer_count": 36,
            "sidecar_model_id": model_id,
            "sidecar_revision": _IMMUTABLE_MODEL_REVISION,
            "sidecar_special_tokens_map_sha256": "c" * 64,
            "sidecar_tokenizer_fingerprint_sha256": "b" * 64,
        },
        "report_id": "synthetic-gateway-smoke",
        "schema_version": "aegis.proxy.cift_gateway_smoke/v1",
        "status": "ok",
    }


def _gateway_smoke_readiness(model_id: str, readout_count: object) -> dict[str, object]:
    return {
        "status": "ready",
        "capability_mode": "self_hosted_introspection",
        "certification_mode": "strict",
        "certification_id": "synthetic-certification",
        "runtime_model_sha256": "a" * 64,
        "release_gate_report_sha256": "b" * 64,
        "model_bundle_id": "synthetic-runtime-cift",
        "source_model_id": model_id,
        "source_revision": _IMMUTABLE_MODEL_REVISION,
        "source_selected_device": "mps",
        "feature_key": "selected_choice_window_layer_21",
        "feature_count": 2560,
        "feature_vector_length": 2560,
        "selected_choice_readout_token_count": readout_count,
        "observed_selected_choice_readout_token_count": readout_count,
        "extractor_id": "trusted-activation-sidecar",
        "extractor_feature_vector_sha256": "e" * 64,
        "extractor_rendered_prompt_sha256": "f" * 64,
        "extractor_hidden_state_device_observed": "mps:0",
        "extractor_input_device_observed": "mps:0",
    }


def _gateway_smoke_decision(
    final_action: str,
    predicted_label: str,
    provider_reason: str | None,
    provider_status: str,
    model_id: str,
    readout_count: object,
) -> dict[str, object]:
    return {
        "cift_action": final_action,
        "extractor_chat_template_sha256": "d" * 64,
        "extractor_extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
        "extractor_feature_vector_length": 2560,
        "extractor_feature_vector_sha256": "e" * 64,
        "extractor_hidden_size": 2560,
        "extractor_hidden_state_device_observed": "mps:0",
        "extractor_hidden_state_layer_count": 37,
        "extractor_id": "trusted-activation-sidecar",
        "extractor_input_device_observed": "mps:0",
        "extractor_layer_count": 36,
        "extractor_model_id": model_id,
        "extractor_rendered_prompt_sha256": "f" * 64,
        "extractor_revision": _IMMUTABLE_MODEL_REVISION,
        "extractor_selected_choice_readout_token_count": readout_count,
        "extractor_selected_choice_readout_token_indices": [11, 12, 13, 14],
        "extractor_selected_choice_readout_token_indices_sha256": _json_sha256([11, 12, 13, 14]),
        "extractor_selected_device": "mps",
        "extractor_special_tokens_map_sha256": "c" * 64,
        "extractor_tokenizer_fingerprint_sha256": "b" * 64,
        "feature_key": "selected_choice_window_layer_21",
        "feature_source": "self_hosted_activation_extractor",
        "final_action": final_action,
        "predicted_label": predicted_label,
        "provider_reason": provider_reason,
        "provider_status": provider_status,
    }


def _model_identity_contract() -> dict[str, object]:
    return {
        "chat_template_sha256": "d" * 64,
        "hidden_size": 2560,
        "layer_count": 36,
        "model_id": "Qwen/Qwen3-test",
        "revision": _IMMUTABLE_MODEL_REVISION,
        "special_tokens_map_sha256": "c" * 64,
        "tokenizer_fingerprint_sha256": "b" * 64,
    }


def _freeform_final_token_runtime_report() -> dict[str, object]:
    return {
        "activation_failure_action": "block",
        "benchmark_mode": "live_hidden_state_runner",
        "false_negative_count": 0,
        "false_negative_rate": 0.0,
        "false_positive_count": 0,
        "false_positive_rate": 0.0,
        "report_id": "synthetic-final-token-runtime",
        "revision": _IMMUTABLE_MODEL_REVISION,
        "rows": [
            _freeform_final_token_runtime_row(
                expected_label="benign",
                detector_action="allow",
                policy_action="allow",
                output_text_empty=False,
                provider_generation_skipped=False,
            ),
            _freeform_final_token_runtime_row(
                expected_label="secret_present_safe",
                detector_action="allow",
                policy_action="allow",
                output_text_empty=False,
                provider_generation_skipped=False,
            ),
            _freeform_final_token_runtime_row(
                expected_label="exfiltration_intent",
                detector_action="block",
                policy_action="block",
                output_text_empty=True,
                provider_generation_skipped=True,
            ),
        ],
        "schema_version": "aegis_introspection.cift_live_window_selector_benchmark/v1",
        "selected_device": "mps",
        "window_family_mismatch_count": 0,
    }


def _freeform_final_token_runtime_row(
    expected_label: str,
    detector_action: str,
    policy_action: str,
    output_text_empty: bool,
    provider_generation_skipped: bool,
) -> dict[str, object]:
    return {
        "capability_status": "active",
        "detector_action": detector_action,
        "expected_label": expected_label,
        "expected_window_family": "freeform_final_token",
        "model_forward_ms": 1.0,
        "output_text_empty": output_text_empty,
        "policy_action": policy_action,
        "provider_generation_skipped": provider_generation_skipped,
        "window_family": "freeform_final_token",
        "window_selection_reason": "selected_choice_metadata_absent_freeform_route",
        **_freeform_final_token_receipt_fields("extractor_"),
    }


def _freeform_final_token_gateway_smoke_report() -> dict[str, object]:
    return {
        "checks": {
            "benign_cift": _freeform_final_token_gateway_decision(
                final_action="allow",
                predicted_label="secret_present_safe",
                provider_reason=None,
                provider_status="completed",
            ),
            "cift_capabilities": {
                "capability_mode": "self_hosted_introspection",
                "detectors": ["cift_runtime"],
                "turn_annotator_count": 1,
            },
            "exfiltration_intent_prevention": _freeform_final_token_gateway_decision(
                final_action="block",
                predicted_label="exfiltration_intent",
                provider_reason="pre_generation_policy_block",
                provider_status="skipped",
            ),
            "gateway_readiness": {
                "status": "ready",
                "capability_mode": "self_hosted_introspection",
                "certification_mode": "gateway_smoke_bootstrap",
                "certification_id": None,
                "runtime_model_sha256": "a" * 64,
                "release_gate_report_sha256": None,
                "model_bundle_id": "synthetic-runtime-cift",
                "source_model_id": "Qwen/Qwen3-test",
                "source_revision": _IMMUTABLE_MODEL_REVISION,
                "source_selected_device": "mps",
                "feature_key": "final_token_layer_12",
                "feature_count": 2560,
                "feature_vector_length": 2560,
                "cift_window_family": "freeform_final_token",
                "extractor_id": "trusted-activation-sidecar",
                "extractor_feature_vector_sha256": "e" * 64,
                "extractor_rendered_prompt_sha256": "f" * 64,
                "extractor_hidden_state_device_observed": "mps:0",
                "extractor_input_device_observed": "mps:0",
            },
            "sidecar_feature_extraction": {
                "chat_template_sha256": "d" * 64,
                "cift_window_family": "freeform_final_token",
                "feature_count": 2560,
                "feature_key": "final_token_layer_12",
                "feature_vector_length": 2560,
                "feature_vector_sha256": "e" * 64,
                "hidden_size": 2560,
                "hidden_state_device_observed": "mps:0",
                "hidden_state_layer_count": 37,
                "input_device_observed": "mps:0",
                "layer_count": 36,
                "model_id": "Qwen/Qwen3-test",
                "prompt_renderer": "aegis.cift_prompt_renderer.trace_bridge/v1",
                "revision": _IMMUTABLE_MODEL_REVISION,
                "selected_device": "mps",
                "special_tokens_map_sha256": "c" * 64,
                "tokenizer_fingerprint_sha256": "b" * 64,
                **_freeform_final_token_receipt_fields(""),
            },
        },
        "confusion_metrics": {
            "false_negative_count": 0,
            "false_negative_rate": 0.0,
            "false_positive_count": 0,
            "false_positive_rate": 0.0,
        },
        "detector_name": "cift_runtime",
        "expected": {
            "cift_window_family": "freeform_final_token",
            "extractor_id": "trusted-activation-sidecar",
            "gateway_feature_source": "self_hosted_activation_extractor",
            "selected_choice_readout_token_count": 4,
            "sidecar_chat_template_sha256": "d" * 64,
            "sidecar_device": "mps",
            "sidecar_feature_key": "final_token_layer_12",
            "sidecar_hidden_size": 2560,
            "sidecar_layer_count": 36,
            "sidecar_model_id": "Qwen/Qwen3-test",
            "sidecar_revision": _IMMUTABLE_MODEL_REVISION,
            "sidecar_special_tokens_map_sha256": "c" * 64,
            "sidecar_tokenizer_fingerprint_sha256": "b" * 64,
        },
        "report_id": "synthetic-final-token-gateway-smoke",
        "schema_version": "aegis.proxy.cift_gateway_smoke/v1",
        "status": "ok",
    }


def _freeform_final_token_gateway_decision(
    final_action: str,
    predicted_label: str,
    provider_reason: str | None,
    provider_status: str,
) -> dict[str, object]:
    return {
        "cift_action": final_action,
        "cift_window_family": "freeform_final_token",
        "cift_window_selection_reason": "selected_choice_metadata_absent_freeform_route",
        "extractor_chat_template_sha256": "d" * 64,
        "extractor_hidden_size": 2560,
        "extractor_id": "trusted-activation-sidecar",
        "extractor_layer_count": 36,
        "extractor_model_id": "Qwen/Qwen3-test",
        "extractor_revision": _IMMUTABLE_MODEL_REVISION,
        "extractor_selected_device": "mps",
        "extractor_special_tokens_map_sha256": "c" * 64,
        "extractor_tokenizer_fingerprint_sha256": "b" * 64,
        "feature_key": "final_token_layer_12",
        "feature_source": "self_hosted_activation_extractor",
        "final_action": final_action,
        "predicted_label": predicted_label,
        "provider_reason": provider_reason,
        "provider_status": provider_status,
        **_freeform_final_token_receipt_fields("extractor_"),
    }


def _freeform_final_token_receipt_fields(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
        f"{prefix}feature_vector_length": 2560,
        f"{prefix}feature_vector_sha256": "e" * 64,
        f"{prefix}hidden_state_device_observed": "mps:0",
        f"{prefix}hidden_state_layer_count": 37,
        f"{prefix}input_device_observed": "mps:0",
        f"{prefix}readout_source": {
            "readout_token_count": 1,
            "readout_window": "final_token",
            "source": "sidecar_freeform",
        },
        f"{prefix}readout_token_indices": [8],
        f"{prefix}readout_token_indices_sha256": _json_sha256([8]),
        f"{prefix}readout_window_source": "final_token",
        f"{prefix}rendered_prompt_sha256": "f" * 64,
    }


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    unittest.main()
