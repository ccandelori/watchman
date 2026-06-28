from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aegis_introspection.cift_deployment_env import (
    CiftDeploymentEnvConfig,
    CiftDeploymentEnvError,
    materialize_cift_deployment_env,
    run_deployment_env_cli,
)
from aegis_introspection.cift_release_gate import CiftReleaseGateReport


class CiftDeploymentEnvTest(unittest.TestCase):
    def test_materialize_env_requires_passing_release_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)

            with (
                patch(
                    "aegis_introspection.cift_deployment_env.evaluate_cift_release_gate",
                    return_value=CiftReleaseGateReport(
                        runtime_model_path=fixture["runtime"],
                        model_bundle_id="synthetic-runtime",
                        candidate_status="runtime_candidate",
                        required_runtime_prevention_device="mps",
                        evidence_mode="certification_bound",
                        eligible=False,
                        diagnostic_eligible=False,
                        failed_requirements=("gateway smoke failed",),
                    ),
                ),
                self.assertRaisesRegex(CiftDeploymentEnvError, "gateway smoke failed"),
            ):
                materialize_cift_deployment_env(_config(root=root, fixture=fixture))

    def test_materialize_env_emits_strict_shell_exports_without_secret_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            config = _config(root=root, fixture=fixture)

            with patch(
                "aegis_introspection.cift_deployment_env.evaluate_cift_release_gate",
                return_value=CiftReleaseGateReport(
                    runtime_model_path=fixture["runtime"],
                    model_bundle_id="synthetic-runtime",
                    candidate_status="runtime_candidate",
                    required_runtime_prevention_device="mps",
                    evidence_mode="certification_bound",
                    eligible=True,
                    diagnostic_eligible=False,
                    failed_requirements=(),
                ),
            ) as release_gate:
                env_text = materialize_cift_deployment_env(config)
                release_gate_report_sha256 = hashlib.sha256(config.release_gate_report_path.read_bytes()).hexdigest()
                manifest_sha256 = hashlib.sha256(fixture["manifest"].read_bytes()).hexdigest()
                report_sha256 = hashlib.sha256(fixture["report"].read_bytes()).hexdigest()

        self.assertIn("export AEGIS_CIFT_PROFILE=self_hosted_window_selector", env_text)
        self.assertIn("export AEGIS_CIFT_CERTIFICATION_MODE=strict", env_text)
        self.assertIn("export AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH=models/runtime.json", env_text)
        self.assertIn("export AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256=" + manifest_sha256, env_text)
        self.assertIn("export AEGIS_CIFT_CERTIFICATION_REPORT_SHA256=" + report_sha256, env_text)
        self.assertIn("export AEGIS_CIFT_RELEASE_GATE_REPORT_PATH=reports/release_gate.json", env_text)
        self.assertIn("export AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256=" + release_gate_report_sha256, env_text)
        self.assertIn("export AEGIS_CIFT_REQUIRED_DEVICE=mps", env_text)
        self.assertIn("export AEGIS_CIFT_EXTRACTOR_BASE_URL=http://127.0.0.1:9000", env_text)
        self.assertIn(
            'export AEGIS_CIFT_EXTRACTOR_API_KEY="${CIFT_SIDE_SECRET:?set CIFT_SIDE_SECRET}"',
            env_text,
        )
        self.assertNotIn("sidecar-token", env_text)
        release_config = release_gate.call_args.args[0]
        self.assertEqual(manifest_sha256, release_config.certification_manifest_sha256)
        self.assertEqual(report_sha256, release_config.certification_report_sha256)
        self.assertFalse(release_config.allow_embedded_artifact_only)

    def test_materialize_env_emits_freeform_route_exports_for_freeform_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root, feature_key="final_token_layer_12")
            config = _config(root=root, fixture=fixture)

            with patch(
                "aegis_introspection.cift_deployment_env.evaluate_cift_release_gate",
                return_value=CiftReleaseGateReport(
                    runtime_model_path=fixture["runtime"],
                    model_bundle_id="synthetic-runtime",
                    candidate_status="runtime_candidate",
                    required_runtime_prevention_device="mps",
                    evidence_mode="certification_bound",
                    eligible=True,
                    diagnostic_eligible=False,
                    failed_requirements=(),
                ),
            ):
                env_text = materialize_cift_deployment_env(config)
                release_gate_report_sha256 = hashlib.sha256(config.release_gate_report_path.read_bytes()).hexdigest()
                manifest_sha256 = hashlib.sha256(fixture["manifest"].read_bytes()).hexdigest()
                report_sha256 = hashlib.sha256(fixture["report"].read_bytes()).hexdigest()

        self.assertIn("export AEGIS_CIFT_FREEFORM_MODEL_PATH=models/runtime.json", env_text)
        self.assertIn("export AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH=models/selected_choice_runtime.json", env_text)
        self.assertIn("export AEGIS_CIFT_FREEFORM_CERTIFICATION_MANIFEST_SHA256=" + manifest_sha256, env_text)
        self.assertIn("export AEGIS_CIFT_FREEFORM_CERTIFICATION_REPORT_SHA256=" + report_sha256, env_text)
        self.assertIn("export AEGIS_CIFT_FREEFORM_RELEASE_GATE_REPORT_PATH=reports/release_gate.json", env_text)
        self.assertIn("export AEGIS_CIFT_FREEFORM_RELEASE_GATE_REPORT_SHA256=" + release_gate_report_sha256, env_text)
        self.assertNotIn("export AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256=", env_text)

    def test_materialize_env_writes_release_gate_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            config = _config(root=root, fixture=fixture)

            with patch(
                "aegis_introspection.cift_deployment_env.evaluate_cift_release_gate",
                return_value=CiftReleaseGateReport(
                    runtime_model_path=fixture["runtime"],
                    model_bundle_id="synthetic-runtime",
                    candidate_status="runtime_candidate",
                    required_runtime_prevention_device="mps",
                    evidence_mode="certification_bound",
                    eligible=True,
                    diagnostic_eligible=False,
                    failed_requirements=(),
                ),
            ):
                materialize_cift_deployment_env(config)
            payload = json.loads(config.release_gate_report_path.read_text(encoding="utf-8"))

        self.assertEqual("aegis_introspection.cift_release_gate/v1", payload["schema_version"])
        self.assertTrue(payload["production_release_eligible"])
        self.assertEqual([], payload["failed_requirements"])

    def test_materialize_env_rejects_shell_unsafe_api_key_env_var_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            config = replace(
                _config(root=root, fixture=fixture),
                extractor_api_key_env_var='X}"; touch /tmp/pwn; #',
            )

            with (
                patch("aegis_introspection.cift_deployment_env.evaluate_cift_release_gate") as release_gate,
                self.assertRaisesRegex(CiftDeploymentEnvError, "shell-safe environment variable name"),
            ):
                materialize_cift_deployment_env(config)

        release_gate.assert_not_called()

    def test_materialize_env_rejects_non_finite_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            config = replace(
                _config(root=root, fixture=fixture),
                extractor_timeout_seconds=float("nan"),
            )

            with (
                patch("aegis_introspection.cift_deployment_env.evaluate_cift_release_gate") as release_gate,
                self.assertRaisesRegex(CiftDeploymentEnvError, "finite positive number"),
            ):
                materialize_cift_deployment_env(config)

        release_gate.assert_not_called()

    def test_cli_writes_env_file_only_after_strict_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_fixture(root)
            output_path = root / "reports" / "deployment_env.sh"
            release_gate_report_path = root / "reports" / "release_gate.json"

            with patch(
                "aegis_introspection.cift_deployment_env.evaluate_cift_release_gate",
                return_value=CiftReleaseGateReport(
                    runtime_model_path=fixture["runtime"],
                    model_bundle_id="synthetic-runtime",
                    candidate_status="runtime_candidate",
                    required_runtime_prevention_device="mps",
                    evidence_mode="certification_bound",
                    eligible=True,
                    diagnostic_eligible=False,
                    failed_requirements=(),
                ),
            ):
                exit_code = run_deployment_env_cli(
                    (
                        str(fixture["runtime"]),
                        "--repository-root",
                        str(root),
                        "--certification-manifest",
                        str(fixture["manifest"]),
                        "--certification-report",
                        str(fixture["report"]),
                        "--certification-artifact-root",
                        str(root),
                        "--required-device",
                        "mps",
                        "--expected-detector-name",
                        "cift_runtime",
                        "--expected-extractor-id",
                        "trusted-activation-sidecar",
                        "--expected-feature-source",
                        "self_hosted_activation_extractor",
                        "--expected-selected-choice-readout-token-count",
                        "4",
                        "--extractor-base-url",
                        "http://127.0.0.1:9000",
                        "--extractor-timeout-seconds",
                        "30.0",
                        "--extractor-api-key-env-var",
                        "CIFT_SIDE_SECRET",
                        "--release-gate-report-output",
                        str(release_gate_report_path),
                        "--output",
                        str(output_path),
                    )
                )

            self.assertEqual(0, exit_code)
            env_text = output_path.read_text(encoding="utf-8")
            self.assertIn("export AEGIS_CIFT_CERTIFICATION_MODE=strict", env_text)
            self.assertIn("export AEGIS_CIFT_REQUIRED_DEVICE=mps", env_text)
            self.assertIn("export AEGIS_CIFT_RELEASE_GATE_REPORT_PATH=reports/release_gate.json", env_text)
            self.assertIn("export AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256=", env_text)
            self.assertIn(
                'export AEGIS_CIFT_EXTRACTOR_API_KEY="${CIFT_SIDE_SECRET:?set CIFT_SIDE_SECRET}"',
                env_text,
            )
            self.assertTrue(release_gate_report_path.is_file())


def _write_fixture(root: Path, feature_key: str = "selected_choice_window_layer_21") -> dict[str, Path]:
    paths = {
        "runtime": root / "models" / "runtime.json",
        "selected_choice_runtime": root / "models" / "selected_choice_runtime.json",
        "manifest": root / "reports" / "manifest.json",
        "report": root / "reports" / "run.json",
    }
    paths["runtime"].parent.mkdir(parents=True)
    paths["manifest"].parent.mkdir(parents=True)
    paths["runtime"].write_text(json.dumps({"feature_key": feature_key}) + "\n", encoding="utf-8")
    if feature_key.startswith("final_token_"):
        paths["selected_choice_runtime"].write_text(
            json.dumps({"feature_key": "selected_choice_window_layer_21"}) + "\n",
            encoding="utf-8",
        )
        paths["manifest"].write_text(
            json.dumps(
                {
                    "training": {
                        "selected_choice_runtime_model_path": "models/selected_choice_runtime.json",
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        paths["manifest"].write_text("manifest", encoding="utf-8")
    paths["report"].write_text("report", encoding="utf-8")
    return paths


def _config(root: Path, fixture: dict[str, Path]) -> CiftDeploymentEnvConfig:
    return CiftDeploymentEnvConfig(
        runtime_model_path=fixture["runtime"],
        repository_root=root,
        certification_manifest_path=fixture["manifest"],
        certification_report_path=fixture["report"],
        certification_artifact_root=root,
        required_device="mps",
        expected_detector_name="cift_runtime",
        expected_extractor_id="trusted-activation-sidecar",
        expected_feature_source="self_hosted_activation_extractor",
        expected_selected_choice_readout_token_count=4,
        extractor_base_url="http://127.0.0.1:9000",
        extractor_timeout_seconds=30.0,
        extractor_api_key_env_var="CIFT_SIDE_SECRET",
        release_gate_report_path=root / "reports" / "release_gate.json",
    )
