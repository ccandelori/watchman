from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import TypeAlias

from aegis.core.contracts import JsonValue, SensitiveSpan
from aegis.core.sensitive_context import metadata_secret_context_handle, sensitive_span_handle

TraceMetadataScalar: TypeAlias = str | int | float | bool | None


class ProtectedContextKind(StrEnum):
    CREDENTIAL = "credential"
    HONEYTOKEN = "honeytoken"
    SECRET = "secret"
    CANARY = "canary"


class ProtectedContextSource(StrEnum):
    DP_HONEY = "dp_honey"
    CREDENTIAL_BROKER = "credential_broker"
    TRACE_FIXTURE = "trace_fixture"


class LeakageLabel(StrEnum):
    NO_LEAK = "no_leak"
    PARTIAL_LEAK = "partial_leak"
    EXACT_LEAK = "exact_leak"
    UNKNOWN = "unknown"


class LeakageTraceError(ValueError):
    """Raised when a leakage trace cannot be represented safely."""


@dataclass(frozen=True)
class ProtectedContextRef:
    handle: str
    kind: ProtectedContextKind
    source: ProtectedContextSource
    representation_ref: str | None

    def __post_init__(self) -> None:
        if self.handle == "":
            raise LeakageTraceError("protected context handle must not be empty.")
        _validate_non_secret_reference(value=self.handle, field_name="protected context handle")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "handle": self.handle,
            "kind": self.kind.value,
            "source": self.source.value,
            "representation_ref": self.representation_ref,
        }


@dataclass(frozen=True)
class LeakageTrace:
    session_id: str
    turn_index: int
    output_sha256: str
    output_char_count: int
    protected_context_refs: tuple[ProtectedContextRef, ...]
    negative_context_refs: tuple[ProtectedContextRef, ...]
    label: LeakageLabel
    metadata: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if self.session_id == "":
            raise LeakageTraceError("session_id must not be empty.")
        if self.turn_index < 0:
            raise LeakageTraceError("turn_index must be non-negative.")
        if self.output_char_count < 0:
            raise LeakageTraceError("output_char_count must be non-negative.")
        if self.output_sha256 == "":
            raise LeakageTraceError("output_sha256 must not be empty.")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "output_sha256": self.output_sha256,
            "output_char_count": self.output_char_count,
            "protected_context_refs": [ref.to_dict() for ref in self.protected_context_refs],
            "negative_context_refs": [ref.to_dict() for ref in self.negative_context_refs],
            "label": self.label.value,
            "metadata": self.metadata,
        }


_TRACE_METADATA_ALLOWLIST = frozenset(
    {
        "assignment_id",
        "family",
        "label",
        "run",
        "run_id",
        "scenario_id",
        "task_id",
        "trace_id",
    }
)
_SECRET_LIKE_PATTERNS = (
    re.compile(r"\{\{CREDENTIAL:[^}]+}}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
)


def _kind_from_span(span: SensitiveSpan) -> ProtectedContextKind:
    if span.kind == ProtectedContextKind.CREDENTIAL.value:
        return ProtectedContextKind.CREDENTIAL
    if span.kind == ProtectedContextKind.HONEYTOKEN.value:
        return ProtectedContextKind.HONEYTOKEN
    if span.kind == ProtectedContextKind.SECRET.value:
        return ProtectedContextKind.SECRET
    if span.kind == ProtectedContextKind.CANARY.value:
        return ProtectedContextKind.CANARY
    return ProtectedContextKind.SECRET


def _source_from_span(span: SensitiveSpan) -> ProtectedContextSource:
    source = span.metadata.get("source")
    if source == ProtectedContextSource.DP_HONEY.value:
        return ProtectedContextSource.DP_HONEY
    if source == ProtectedContextSource.CREDENTIAL_BROKER.value:
        return ProtectedContextSource.CREDENTIAL_BROKER
    if span.source == ProtectedContextSource.DP_HONEY.value:
        return ProtectedContextSource.DP_HONEY
    if span.source == ProtectedContextSource.CREDENTIAL_BROKER.value:
        return ProtectedContextSource.CREDENTIAL_BROKER
    return ProtectedContextSource.TRACE_FIXTURE


def _safe_trace_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    safe_metadata: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        if key not in _TRACE_METADATA_ALLOWLIST:
            continue
        safe_metadata[key] = _safe_trace_metadata_value(key=key, value=value)
    return safe_metadata


def _safe_trace_metadata_value(key: str, value: JsonValue) -> TraceMetadataScalar:
    if isinstance(value, str):
        _validate_non_secret_reference(value=value, field_name=f"metadata.{key}")
        return value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise LeakageTraceError(f"metadata.{key} must be a finite number.")
        return value
    raise LeakageTraceError(f"metadata.{key} must be a scalar audit identifier.")


def _validate_non_secret_reference(value: str, field_name: str) -> None:
    for pattern in _SECRET_LIKE_PATTERNS:
        if pattern.search(value) is not None:
            raise LeakageTraceError(f"{field_name} must be an opaque non-secret handle.")


def build_leakage_trace(
    session_id: str,
    turn_index: int,
    output_text: str,
    sensitive_spans: tuple[SensitiveSpan, ...],
    metadata: dict[str, JsonValue],
) -> LeakageTrace:
    """Convert runtime turn data into a handle-based leakage trace."""
    protected_refs: list[ProtectedContextRef] = []

    for span in sensitive_spans:
        if span.kind in {
            ProtectedContextKind.CREDENTIAL.value,
            ProtectedContextKind.HONEYTOKEN.value,
            ProtectedContextKind.SECRET.value,
            ProtectedContextKind.CANARY.value,
        }:
            handle = sensitive_span_handle(span)
            if handle is None:
                continue
            _validate_non_secret_reference(value=handle, field_name=f"protected span kind '{span.kind}' handle")
            protected_refs.append(
                ProtectedContextRef(
                    handle=handle,
                    kind=_kind_from_span(span),
                    source=_source_from_span(span),
                    representation_ref=None,
                )
            )

    metadata_handle = metadata_secret_context_handle(metadata)
    if metadata_handle is not None:
        _validate_non_secret_reference(value=metadata_handle, field_name="metadata.secret_context_handle")
        protected_refs.append(
            ProtectedContextRef(
                handle=metadata_handle,
                kind=ProtectedContextKind.SECRET,
                source=ProtectedContextSource.TRACE_FIXTURE,
                representation_ref=None,
            )
        )

    return LeakageTrace(
        session_id=session_id,
        turn_index=turn_index,
        output_sha256=_sha256_text(output_text),
        output_char_count=len(output_text),
        protected_context_refs=tuple(protected_refs),
        negative_context_refs=(),
        label=LeakageLabel.UNKNOWN,
        metadata=_safe_trace_metadata(metadata),
    )


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
