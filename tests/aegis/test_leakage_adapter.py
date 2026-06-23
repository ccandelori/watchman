import hashlib

import pytest

from aegis.core.contracts import JsonValue, SensitiveSpan
from aegis.core.leakage import LeakageLabel, LeakageTraceError, build_leakage_trace


def _make_span(kind: str, handle: str | None) -> SensitiveSpan:
    metadata: dict[str, JsonValue] = {"handle": handle} if handle is not None else {}
    return SensitiveSpan(
        kind=kind,
        source="",
        char_start=None,
        char_end=None,
        token_start=None,
        token_end=None,
        identifier=handle,
        metadata=metadata,
    )


def test_build_leakage_trace_from_spans() -> None:
    spans = (
        _make_span("credential", "cred-1"),
        _make_span("honeytoken", "honey-1"),
    )
    trace = build_leakage_trace(
        session_id="s1",
        turn_index=0,
        output_text="out",
        sensitive_spans=spans,
        metadata={},
    )
    assert len(trace.protected_context_refs) == 2
    assert trace.protected_context_refs[0].kind.value == "credential"
    assert trace.protected_context_refs[1].kind.value == "honeytoken"


def test_build_leakage_trace_from_metadata() -> None:
    raw_output = "response containing fake secret value"
    trace = build_leakage_trace(
        session_id="s1",
        turn_index=0,
        output_text=raw_output,
        sensitive_spans=(),
        metadata={"run": "test", "secret_context_handle": "meta-secret", "unsafe_note": "drop-me"},
    )
    assert len(trace.protected_context_refs) == 1
    assert trace.protected_context_refs[0].handle == "meta-secret"
    assert trace.label == LeakageLabel.UNKNOWN
    assert trace.metadata == {"run": "test"}
    assert trace.output_char_count == len(raw_output)
    assert trace.output_sha256 == hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    assert raw_output not in str(trace.to_dict())


def test_build_leakage_trace_omits_sensitive_span_without_handle() -> None:
    trace = build_leakage_trace(
        session_id="s1",
        turn_index=0,
        output_text="response",
        sensitive_spans=(_make_span("credential", None),),
        metadata={},
    )

    assert trace.protected_context_refs == ()


def test_build_leakage_trace_rejects_secret_like_span_handle() -> None:
    with pytest.raises(LeakageTraceError, match="protected span kind 'credential' handle"):
        build_leakage_trace(
            session_id="s1",
            turn_index=0,
            output_text="response",
            sensitive_spans=(_make_span("credential", "ghp_abcdefghijklmnopqrstuvwxyz1234567890"),),
            metadata={},
        )


def test_build_leakage_trace_rejects_secret_like_allowed_metadata() -> None:
    with pytest.raises(LeakageTraceError, match=r"metadata\.trace_id"):
        build_leakage_trace(
            session_id="s1",
            turn_index=0,
            output_text="response",
            sensitive_spans=(),
            metadata={"trace_id": "sk-abcdefghijklmnop1234567890"},
        )


def test_build_leakage_trace_rejects_nested_allowed_metadata() -> None:
    with pytest.raises(LeakageTraceError, match=r"metadata\.trace_id must be a scalar"):
        build_leakage_trace(
            session_id="s1",
            turn_index=0,
            output_text="response",
            sensitive_spans=(),
            metadata={"trace_id": {"unsafe": "nested"}},
        )
