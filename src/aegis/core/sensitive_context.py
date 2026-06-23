from __future__ import annotations

from collections.abc import Mapping

from aegis.core.contracts import JsonValue, SensitiveSpan

SECRET_CONTEXT_HANDLE_METADATA_KEY = "secret_context_handle"
SENSITIVE_SPAN_HANDLE_METADATA_KEY = "handle"


def sensitive_span_handle(span: SensitiveSpan) -> str | None:
    metadata_handle = span.metadata.get(SENSITIVE_SPAN_HANDLE_METADATA_KEY)
    if isinstance(metadata_handle, str) and metadata_handle != "":
        return metadata_handle
    if span.identifier is not None and span.identifier != "":
        return span.identifier
    return None


def first_sensitive_span_handle(spans: tuple[SensitiveSpan, ...], kinds: tuple[str, ...]) -> str | None:
    allowed_kinds = frozenset(kinds)
    for span in spans:
        if span.kind not in allowed_kinds:
            continue
        handle = sensitive_span_handle(span)
        if handle is not None:
            return handle
    return None


def metadata_secret_context_handle(metadata: Mapping[str, JsonValue]) -> str | None:
    metadata_handle = metadata.get(SECRET_CONTEXT_HANDLE_METADATA_KEY)
    if isinstance(metadata_handle, str) and metadata_handle != "":
        return metadata_handle
    return None
