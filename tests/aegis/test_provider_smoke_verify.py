from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis.proxy.provider_smoke_verify import (
    ProviderSmokeEvidenceConfig,
    ProviderSmokeEvidenceError,
    parse_args,
    verify_provider_smoke_evidence,
)


def test_provider_smoke_verify_parses_required_artifact_paths_and_markers() -> None:
    config = parse_args(
        (
            "--preflight",
            "introspection/data/reports/preflight.json",
            "--smoke",
            "introspection/data/reports/smoke.json",
            "--provider-request-log",
            "introspection/data/reports/provider.jsonl",
            "--audit-jsonl",
            "introspection/data/reports/audit.jsonl",
            "--output",
            "introspection/data/reports/verification.json",
            "--forbidden-marker",
            "ghp_secret",
            "--forbidden-marker",
            "fake-",
        )
    )

    assert config == ProviderSmokeEvidenceConfig(
        preflight_path=Path("introspection/data/reports/preflight.json"),
        smoke_path=Path("introspection/data/reports/smoke.json"),
        provider_request_log_path=Path("introspection/data/reports/provider.jsonl"),
        audit_jsonl_path=Path("introspection/data/reports/audit.jsonl"),
        output_path=Path("introspection/data/reports/verification.json"),
        forbidden_markers=("ghp_secret", "fake-"),
    )


def test_provider_smoke_verify_accepts_loopback_real_provider_evidence(tmp_path: Path) -> None:
    paths = _write_valid_evidence(tmp_path)

    report = verify_provider_smoke_evidence(
        ProviderSmokeEvidenceConfig(
            preflight_path=paths["preflight"],
            smoke_path=paths["smoke"],
            provider_request_log_path=paths["provider_request_log"],
            audit_jsonl_path=paths["audit_jsonl"],
            output_path=None,
            forbidden_markers=("ghp_secret", "fake-"),
        )
    )

    assert report["schema_version"] == "aegis.provider_smoke_evidence/v1"
    assert report["status"] == "pass"
    assert report["evidence_scope"] == "local_loopback_openai_compatible_adapter"
    assert report["external_provider_evidence"] is False
    check_statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert check_statuses == {
        "provider_preflight": "pass",
        "gateway_smoke_report": "pass",
        "gateway_readiness": "pass",
        "gateway_capabilities": "pass",
        "benign_provider_completion": "pass",
        "egress_guard_pre_provider_block": "pass",
        "mock_only_probe_skip": "pass",
        "provider_request_receipts": "pass",
        "audit_trace_receipts": "pass",
        "artifact_redaction": "pass",
    }
    artifacts = report["input_artifacts"]
    assert isinstance(artifacts, dict)
    smoke = artifacts["smoke"]
    assert isinstance(smoke, dict)
    assert isinstance(smoke["sha256"], str)


def test_provider_smoke_verify_fails_when_provider_receipt_saw_forbidden_marker(tmp_path: Path) -> None:
    paths = _write_valid_evidence(tmp_path)
    _write_jsonl(
        paths["provider_request_log"],
        (
            {
                "schema_version": "aegis.loopback_openai_provider_request/v1",
                "method": "POST",
                "path": "/v1/chat/completions",
                "authorization_status": "matched_expected",
                "forbidden_substring_count": 2,
                "forbidden_substring_present": True,
                "message_count": 2,
                "request_body_bytes": 412,
                "request_body_sha256": "0" * 64,
            },
        ),
    )

    report = verify_provider_smoke_evidence(
        ProviderSmokeEvidenceConfig(
            preflight_path=paths["preflight"],
            smoke_path=paths["smoke"],
            provider_request_log_path=paths["provider_request_log"],
            audit_jsonl_path=paths["audit_jsonl"],
            output_path=None,
            forbidden_markers=("ghp_secret", "fake-"),
        )
    )

    assert report["status"] == "fail"
    check_statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert check_statuses["provider_request_receipts"] == "fail"


def test_provider_smoke_verify_fails_when_audit_runtime_evidence_is_missing(tmp_path: Path) -> None:
    paths = _write_valid_evidence(tmp_path)
    audit_records = list(_audit_records())
    audit_records[1].pop("runtime_evidence")
    _write_jsonl(paths["audit_jsonl"], tuple(audit_records))

    report = verify_provider_smoke_evidence(
        ProviderSmokeEvidenceConfig(
            preflight_path=paths["preflight"],
            smoke_path=paths["smoke"],
            provider_request_log_path=paths["provider_request_log"],
            audit_jsonl_path=paths["audit_jsonl"],
            output_path=None,
            forbidden_markers=("ghp_secret", "fake-"),
        )
    )

    assert report["status"] == "fail"
    check_statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert check_statuses["audit_trace_receipts"] == "fail"


def test_provider_smoke_verify_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    paths = _write_valid_evidence(tmp_path)
    paths["preflight"].write_text('{"schema_version":"one","schema_version":"two"}\n', encoding="utf-8")

    with pytest.raises(ProviderSmokeEvidenceError, match="duplicate JSON key"):
        verify_provider_smoke_evidence(
            ProviderSmokeEvidenceConfig(
                preflight_path=paths["preflight"],
                smoke_path=paths["smoke"],
                provider_request_log_path=paths["provider_request_log"],
                audit_jsonl_path=paths["audit_jsonl"],
                output_path=None,
                forbidden_markers=("ghp_secret",),
            )
        )


def _write_valid_evidence(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "preflight": tmp_path / "preflight.json",
        "smoke": tmp_path / "smoke.json",
        "provider_request_log": tmp_path / "provider.jsonl",
        "audit_jsonl": tmp_path / "audit.jsonl",
    }
    _write_json(
        paths["preflight"],
        {
            "schema_version": "aegis.provider_preflight/v1",
            "ready": True,
            "status": "ready",
            "require_real_provider": True,
            "provider_kind": "openai_compatible",
            "provider_name": "openai_compatible",
            "mock_controls_enabled": False,
            "network_access": "not_attempted",
        },
    )
    _write_json(paths["smoke"], _smoke_report())
    _write_jsonl(
        paths["provider_request_log"],
        (
            {
                "schema_version": "aegis.loopback_openai_provider_request/v1",
                "method": "POST",
                "path": "/v1/chat/completions",
                "authorization_status": "matched_expected",
                "forbidden_substring_count": 2,
                "forbidden_substring_present": False,
                "message_count": 2,
                "request_body_bytes": 412,
                "request_body_sha256": "0" * 64,
            },
        ),
    )
    _write_jsonl(paths["audit_jsonl"], _audit_records())
    return paths


def _smoke_report() -> dict[str, object]:
    return {
        "status": "ok",
        "base_url": "http://127.0.0.1:8773",
        "provider_mode": "real-provider",
        "checks": {
            "gateway_readiness": {
                "status": "ready",
                "provider_name": "openai_compatible",
                "provider_mock_controls_enabled": False,
                "dp_honey_status": "ready",
                "cift_capability_mode": "black_box",
            },
            "capabilities": {
                "provider_mode": "real-provider",
                "mock_response_modes": [],
            },
            "benign_chat": {
                "final_action": "allow",
                "dp_honey_status": "active",
                "credential_slot_status": "honeytoken_substituted",
                "provider_status": "completed",
                "stage_evidence": [{"stage": "provider", "status": "completed"}],
            },
            "provider_egress_guard_block": {
                "final_action": "block",
                "guard_action": "block",
                "guard_reason": "blocked_sensitive_value_before_provider_egress",
                "provider_status": "skipped",
                "provider_reason": "pre_generation_policy_block",
                "stage_evidence": [{"stage": "provider", "status": "skipped"}],
            },
            "encoded_canary_leak": {"status": "skipped"},
            "metadata_slot_canary_leak": {"status": "skipped"},
            "nimbus_partial_leak": {"status": "skipped"},
        },
    }


def _audit_records() -> tuple[dict[str, object], ...]:
    return (
        {
            "trace_id": "smoke-benign-trace",
            "policy_decision": {"final_action": "allow"},
            "model_response_metadata": {"provider": "openai_compatible"},
            "detector_results": [],
            "runtime_evidence": {
                "schema_version": "aegis.audit_runtime_evidence/v1",
                "policy_mode": "severity",
                "final_action": "allow",
                "provider_state": {"provider": "openai_compatible", "status": "completed"},
                "credential_slot_status": "honeytoken_substituted",
                "detector_versions": {"provider_egress_guard": "provider-egress-guard-v1"},
                "detector_latency_ms": {"provider_egress_guard": 0.0},
                "artifact_hashes": {},
                "cift": {},
                "fail_closed_events": [],
                "latency_ms": 0.1,
            },
        },
        {
            "trace_id": "smoke-egress-guard-trace",
            "policy_decision": {"final_action": "block"},
            "model_response_metadata": {"provider": "skipped", "reason": "pre_generation_policy_block"},
            "detector_results": [
                {
                    "detector_name": "provider_egress_guard",
                    "recommended_action": "block",
                    "evidence": {"reason": "blocked_sensitive_value_before_provider_egress"},
                }
            ],
            "runtime_evidence": {
                "schema_version": "aegis.audit_runtime_evidence/v1",
                "policy_mode": "severity",
                "final_action": "block",
                "provider_state": {
                    "provider": "skipped",
                    "status": "skipped",
                    "reason": "pre_generation_policy_block",
                },
                "credential_slot_status": "real_secret_present",
                "detector_versions": {"provider_egress_guard": "provider-egress-guard-v1"},
                "detector_latency_ms": {"provider_egress_guard": 0.0},
                "artifact_hashes": {},
                "cift": {},
                "fail_closed_events": [
                    {
                        "kind": "pre_generation_policy_block",
                        "final_action": "block",
                        "provider_status": "skipped",
                        "triggered_detectors": ["provider_egress_guard"],
                    }
                ],
                "latency_ms": 0.1,
            },
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
