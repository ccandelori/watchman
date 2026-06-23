from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aegis.audit.leakage_trace import LeakageTraceWriter
from aegis.core.leakage import (
    LeakageLabel,
    LeakageTrace,
    ProtectedContextKind,
    ProtectedContextRef,
    ProtectedContextSource,
)


def test_protected_context_ref_to_dict() -> None:
    ref = ProtectedContextRef(
        handle="h1",
        kind=ProtectedContextKind.CREDENTIAL,
        source=ProtectedContextSource.DP_HONEY,
        representation_ref="emb-42",
    )
    d = ref.to_dict()
    assert d["handle"] == "h1"
    assert d["kind"] == "credential"
    assert d["source"] == "dp_honey"
    assert d["representation_ref"] == "emb-42"


def test_leakage_trace_to_dict() -> None:
    ref = ProtectedContextRef(
        handle="h1",
        kind=ProtectedContextKind.HONEYTOKEN,
        source=ProtectedContextSource.DP_HONEY,
        representation_ref=None,
    )
    trace = LeakageTrace(
        session_id="s1",
        turn_index=3,
        output_sha256=hashlib.sha256(b"some output").hexdigest(),
        output_char_count=len("some output"),
        protected_context_refs=(ref,),
        negative_context_refs=(),
        label=LeakageLabel.NO_LEAK,
        metadata={"run": "test"},
    )
    d = trace.to_dict()

    refs = d["protected_context_refs"]
    assert isinstance(refs, list)
    assert len(refs) == 1
    assert d["output_char_count"] == 11
    assert d["output_sha256"] == hashlib.sha256(b"some output").hexdigest()
    assert "some output" not in str(d)


def test_leakage_trace_writer_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "leakage.jsonl"
    leaked_output = "sk-live-fake-leakage-value"
    ref = ProtectedContextRef(
        handle="h1",
        kind=ProtectedContextKind.CREDENTIAL,
        source=ProtectedContextSource.DP_HONEY,
        representation_ref=None,
    )
    trace = LeakageTrace(
        session_id="s1",
        turn_index=0,
        output_sha256=hashlib.sha256(leaked_output.encode("utf-8")).hexdigest(),
        output_char_count=len(leaked_output),
        protected_context_refs=(ref,),
        negative_context_refs=(),
        label=LeakageLabel.EXACT_LEAK,
        metadata={},
    )

    with LeakageTraceWriter(path, "w") as w:
        w.write(trace)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["label"] == "exact_leak"
    assert loaded["protected_context_refs"][0]["handle"] == "h1"
    assert loaded["output_char_count"] == len(leaked_output)
    assert leaked_output not in lines[0]
