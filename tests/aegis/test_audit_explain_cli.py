from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aegis.audit.explain_cli import (
    AuditExplainCliConfig,
    AuditExplainCliError,
    explain_audit_jsonl,
    main,
    parse_args,
    render_audit_explanation_json,
)
from aegis.audit.jsonl import find_jsonl_audit_record, read_jsonl_audit_records


def test_explains_audit_jsonl_by_trace_id(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    _write_jsonl(audit_path, (_audit_record(trace_id="trace-a", session_id="session-a", turn_index=0),))

    explanation = explain_audit_jsonl(
        AuditExplainCliConfig(
            input_path=audit_path,
            output_path=None,
            trace_id="trace-a",
            session_id=None,
        )
    )

    assert explanation["schema_version"] == "aegis.audit_explain/v1"
    assert explanation["trace_id"] == "trace-a"
    assert explanation["runtime_evidence"]["schema_version"] == "aegis.audit_runtime_evidence/v1"
    assert explanation["runtime_evidence"]["policy_mode"] == "severity"
    assert explanation["runtime_evidence"]["provider_state"]["status"] == "skipped"
    assert explanation["runtime_evidence"]["provider_state"]["reason"] == "pre_generation_policy_block"
    assert explanation["runtime_evidence"]["detector_versions"]["nimbus"] == "canary-v0"
    assert explanation["runtime_evidence"]["fail_closed_events"] == [
        {
            "kind": "pre_generation_policy_block",
            "provider_status": "skipped",
            "final_action": "block",
            "triggered_detectors": ["provider_egress_guard"],
        }
    ]
    assert explanation["stage_timeline"] == [
        {"stage": "normalize", "status": "ok"},
        {
            "stage": "dp_honey",
            "status": "active",
            "canary_count": 1,
            "credential_slot_status": "honeytoken_substituted",
            "credential_needed_count": 1,
            "honeytoken_substituted_count": 1,
            "real_secret_present_count": 0,
        },
        {"stage": "cift", "status": "unavailable", "detectors": ["activation_unavailable"], "actions": ["allow"]},
        {
            "stage": "provider_egress_guard",
            "status": "active",
            "detectors": ["provider_egress_guard"],
            "actions": ["block"],
        },
        {
            "stage": "provider",
            "status": "skipped",
            "provider": "skipped",
            "model_id": "mock-model",
            "reason": "pre_generation_policy_block",
        },
        {"stage": "canary", "status": "not_configured", "detectors": []},
        {"stage": "nimbus", "status": "unavailable", "detectors": ["nimbus"], "actions": ["allow"]},
        {"stage": "policy", "status": "decided", "final_action": "block", "reason": "test_policy_block"},
        {"stage": "audit", "status": "written"},
    ]


def test_explains_newest_audit_jsonl_record_by_session_id(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    _write_jsonl(
        audit_path,
        (
            _audit_record(trace_id="trace-old", session_id="session-a", turn_index=0),
            _audit_record(trace_id="trace-new", session_id="session-a", turn_index=1),
        ),
    )

    explanation = explain_audit_jsonl(
        AuditExplainCliConfig(
            input_path=audit_path,
            output_path=None,
            trace_id=None,
            session_id="session-a",
        )
    )

    assert explanation["trace_id"] == "trace-new"
    assert explanation["turn_index"] == 1


def test_audit_jsonl_lookup_helpers_validate_inputs(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    _write_jsonl(audit_path, (_audit_record(trace_id="trace-a", session_id="session-a", turn_index=0),))

    records = read_jsonl_audit_records(audit_path)
    found = find_jsonl_audit_record(audit_path, trace_id="trace-a", session_id=None)

    assert len(records) == 1
    assert found is not None
    assert found["session_id"] == "session-a"
    with pytest.raises(ValueError, match="trace_id or session_id"):
        find_jsonl_audit_record(audit_path, trace_id=None, session_id=None)


def test_parse_args_requires_trace_or_session() -> None:
    with pytest.raises(AuditExplainCliError, match="trace-id or --session-id"):
        parse_args(("--input", "audit.jsonl"))


def test_explain_audit_jsonl_reports_not_found(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    _write_jsonl(audit_path, (_audit_record(trace_id="trace-a", session_id="session-a", turn_index=0),))

    with pytest.raises(AuditExplainCliError, match="No audit record matched"):
        explain_audit_jsonl(
            AuditExplainCliConfig(
                input_path=audit_path,
                output_path=None,
                trace_id="trace-missing",
                session_id=None,
            )
        )


def test_cli_writes_explanation_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_path = tmp_path / "audit.jsonl"
    output_path = tmp_path / "explain.json"
    _write_jsonl(audit_path, (_audit_record(trace_id="trace-a", session_id="session-a", turn_index=0),))
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-audit-explain",
            "--input",
            str(audit_path),
            "--trace-id",
            "trace-a",
            "--output",
            str(output_path),
        ),
    )

    main()

    stdout_payload = json.loads(capsys.readouterr().out)
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == output_payload
    assert output_payload["trace_id"] == "trace-a"
    assert output_payload["artifacts"]["critic_version"] == "canary-v0"


def test_render_audit_explanation_json_is_stable(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    _write_jsonl(audit_path, (_audit_record(trace_id="trace-a", session_id="session-a", turn_index=0),))
    explanation = explain_audit_jsonl(
        AuditExplainCliConfig(
            input_path=audit_path,
            output_path=None,
            trace_id="trace-a",
            session_id=None,
        )
    )

    rendered = render_audit_explanation_json(explanation)

    assert json.loads(rendered)["schema_version"] == "aegis.audit_explain/v1"
    assert rendered.endswith("\n")


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records), encoding="utf-8")


def _audit_record(trace_id: str, session_id: str, turn_index: int) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "created_at": "2026-06-25T00:00:00+00:00",
        "latency_ms": 1.25,
        "normalized_turn": {
            "trace_id": trace_id,
            "session_id": session_id,
            "turn_index": turn_index,
            "capability_mode": "black_box",
            "model": {
                "provider": "mock",
                "model_id": "mock-model",
                "revision": None,
                "selected_device": None,
            },
            "messages": [{"role": "user", "content": "[REDACTED_SENSITIVE]"}],
            "tool_calls": [{"name": "external_ticket", "arguments": {"redacted": True}}],
            "sensitive_spans": [],
            "metadata": {
                "aegis_credential_slot_detection": {
                    "status": "honeytoken_substituted",
                    "credential_needed_count": 1,
                    "honeytoken_substituted_count": 1,
                    "real_secret_present_count": 0,
                },
                "dp_honey_canary_count": 1,
            },
        },
        "detector_results": [
            {
                "detector_name": "activation_unavailable",
                "component": "cift",
                "score": 0.0,
                "confidence": 1.0,
                "recommended_action": "allow",
                "capability_required": "self_hosted_introspection",
                "capability_status": "unavailable",
                "evidence": {"reason": "black_box_mode"},
                "latency_ms": 0.0,
            },
            {
                "detector_name": "provider_egress_guard",
                "component": "tool_scanner",
                "score": 1.0,
                "confidence": 1.0,
                "recommended_action": "block",
                "capability_required": None,
                "capability_status": "active",
                "evidence": {
                    "reason": "blocked_sensitive_value_before_provider_egress",
                    "checked_span_count": 1,
                },
                "latency_ms": 0.0,
            },
            {
                "detector_name": "nimbus",
                "component": "nimbus",
                "score": 0.0,
                "confidence": 0.8,
                "recommended_action": "allow",
                "capability_required": None,
                "capability_status": "unavailable",
                "evidence": {"critic_version": "canary-v0", "reason": "no_secret_context_handle"},
                "latency_ms": 0.0,
            },
        ],
        "policy_decision": {
            "final_action": "block",
            "reason": "test_policy_block",
            "triggered_detectors": ["provider_egress_guard"],
            "risk_score": 1.0,
            "sanitized_output": "",
        },
        "model_response_metadata": {
            "provider": "skipped",
            "reason": "pre_generation_policy_block",
            "model_id": "mock-model",
        },
    }
