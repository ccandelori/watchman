from __future__ import annotations

CIFT_FEATURE_EXTRACT_REQUEST_SCHEMA_VERSION = "aegis.cift_feature_extract_request/v1"
CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION = "aegis.cift_feature_extract_response/v1"
CIFT_MODEL_ATTESTATION_SCHEMA_VERSION = "aegis.cift_model_attestation/v1"
CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION = "aegis.cift_extraction_receipt/v1"
CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1 = "aegis_trace_bridge_v1"
CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1 = "semantic_indirection_v1"
CIFT_SUPPORT_STATE_UNSUPPORTED = "unsupported"
CIFT_SUPPORT_STATE_DISCOVERED = "discovered"
CIFT_SUPPORT_STATE_HIDDEN_STATE_CAPABLE = "hidden-state capable"
CIFT_SUPPORT_STATE_CALIBRATION_READY = "calibration-ready"
CIFT_SUPPORT_STATE_CERTIFIED = "certified"
CIFT_SUPPORT_STATE_RUNTIME_ENFORCEABLE = "runtime-enforceable"
CIFT_SUPPORT_STATE_FAILED_CERTIFICATION = "failed certification"
CIFT_SUPPORT_STATES = (
    CIFT_SUPPORT_STATE_UNSUPPORTED,
    CIFT_SUPPORT_STATE_DISCOVERED,
    CIFT_SUPPORT_STATE_HIDDEN_STATE_CAPABLE,
    CIFT_SUPPORT_STATE_CALIBRATION_READY,
    CIFT_SUPPORT_STATE_CERTIFIED,
    CIFT_SUPPORT_STATE_RUNTIME_ENFORCEABLE,
    CIFT_SUPPORT_STATE_FAILED_CERTIFICATION,
)

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
