from __future__ import annotations

CIFT_FEATURE_EXTRACT_REQUEST_SCHEMA_VERSION = "aegis.cift_feature_extract_request/v1"
CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION = "aegis.cift_feature_extract_response/v1"
CIFT_MODEL_ATTESTATION_SCHEMA_VERSION = "aegis.cift_model_attestation/v1"
CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION = "aegis.cift_extraction_receipt/v1"
CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1 = "aegis_trace_bridge_v1"
CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1 = "semantic_indirection_v1"

_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")


def is_cift_immutable_model_revision(revision: str) -> bool:
    if _is_lowercase_hex_revision(revision=revision, length=40):
        return True
    if not revision.startswith("sha256:"):
        return False
    return _is_lowercase_hex_revision(revision=revision.removeprefix("sha256:"), length=64)


def _is_lowercase_hex_revision(revision: str, length: int) -> bool:
    if len(revision) != length:
        return False
    return all(character in _LOWERCASE_HEX_DIGITS for character in revision)
