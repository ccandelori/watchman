from __future__ import annotations

import base64
import codecs
import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from aegis.core.contracts import Action, CapabilityStatus, DetectorComponent, DetectorResult, JsonValue, NormalizedTurn
from aegis.core.orchestrator import ModelResponse

_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{12,}={0,2}")
_HEX_RUN = re.compile(r"[0-9a-fA-F]{16,}")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")
_LEET_MAP: dict[str, str] = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7", "b": "8"}


class CanaryDetectorError(ValueError):
    """Raised when canary detector configuration is invalid."""


@dataclass(frozen=True)
class CanaryRecord:
    canary_id: str
    credential_type: str
    value: str
    sha256: str
    source: str
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class CanaryMatch:
    canary_id: str
    credential_type: str
    sha256: str
    source: str
    char_start: int
    char_end: int
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class EncodedCanaryMatch:
    canary_id: str
    credential_type: str
    sha256: str
    source: str
    encoding: str
    exact: bool
    fragment_ratio: float
    char_start: int | None
    char_end: int | None
    metadata: dict[str, JsonValue]


class InMemoryCanaryRegistry:
    def __init__(self, records: tuple[CanaryRecord, ...]) -> None:
        self._records = _validate_records(records)

    def records(self) -> tuple[CanaryRecord, ...]:
        return self._records


class TextCanaryDetector:
    def __init__(self, detector_name: str, registry: InMemoryCanaryRegistry) -> None:
        if detector_name == "":
            raise CanaryDetectorError("detector_name must not be empty.")
        self.detector_name = detector_name
        self._registry = registry

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        if model_response is None:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.TEXT_CANARY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.DEGRADED,
                evidence={
                    "reason": "model_response_required",
                    "session_id": turn.session_id,
                },
                latency_ms=0.0,
            )

        matches = _scan_text_for_canaries(text=model_response.output_text, records=self._registry.records())
        if len(matches) == 0:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.TEXT_CANARY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.ACTIVE,
                evidence={
                    "reason": "no_canary_leak_detected",
                    "session_id": turn.session_id,
                    "match_count": 0,
                    "matches": [],
                },
                latency_ms=0.0,
            )

        match_values: list[JsonValue] = [_match_to_dict(match) for match in matches]
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.TEXT_CANARY,
            score=1.0,
            confidence=1.0,
            recommended_action=Action.ESCALATE,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": "registered_canary_leak_detected",
                "session_id": turn.session_id,
                "match_count": len(matches),
                "matches": match_values,
            },
            latency_ms=0.0,
        )


class EncodedCanaryDetector:
    def __init__(
        self,
        detector_name: str,
        registry: InMemoryCanaryRegistry,
        partial_match_threshold: float,
    ) -> None:
        if detector_name == "":
            raise CanaryDetectorError("detector_name must not be empty.")
        _validate_probability(partial_match_threshold, "partial_match_threshold")
        self.detector_name = detector_name
        self._registry = registry
        self._partial_match_threshold = partial_match_threshold

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        if model_response is None:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.TEXT_CANARY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.DEGRADED,
                evidence={
                    "reason": "model_response_required",
                    "session_id": turn.session_id,
                },
                latency_ms=0.0,
            )

        matches = _scan_text_for_encoded_canaries(
            text=model_response.output_text,
            records=self._registry.records(),
            partial_match_threshold=self._partial_match_threshold,
        )
        if len(matches) == 0:
            return DetectorResult(
                detector_name=self.detector_name,
                component=DetectorComponent.TEXT_CANARY,
                score=0.0,
                confidence=1.0,
                recommended_action=Action.ALLOW,
                capability_required=None,
                capability_status=CapabilityStatus.ACTIVE,
                evidence={
                    "reason": "no_encoded_canary_leak_detected",
                    "session_id": turn.session_id,
                    "match_count": 0,
                    "matches": [],
                },
                latency_ms=0.0,
            )

        score = max(match.fragment_ratio for match in matches)
        exact_match_found = any(match.exact for match in matches)
        recommended_action = Action.ESCALATE if exact_match_found else Action.SANITIZE
        reason = "encoded_canary_leak_detected" if exact_match_found else "partial_canary_overlap_detected"
        match_values: list[JsonValue] = [_encoded_match_to_dict(match) for match in matches]
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.TEXT_CANARY,
            score=score,
            confidence=1.0,
            recommended_action=recommended_action,
            capability_required=None,
            capability_status=CapabilityStatus.ACTIVE,
            evidence={
                "reason": reason,
                "session_id": turn.session_id,
                "match_count": len(matches),
                "matches": match_values,
            },
            latency_ms=0.0,
        )


class NoopCanaryDetector:
    detector_name = "noop_canary"

    def evaluate(self, turn: NormalizedTurn, model_response: ModelResponse | None) -> DetectorResult:
        return DetectorResult(
            detector_name=self.detector_name,
            component=DetectorComponent.DP_HONEY,
            score=0.0,
            confidence=1.0,
            recommended_action=Action.ALLOW,
            capability_required=None,
            capability_status=CapabilityStatus.DEGRADED,
            evidence={
                "reason": "canary_registry_not_configured",
                "session_id": turn.session_id,
            },
            latency_ms=0.0,
        )


def canary_sha256(value: str) -> str:
    if value == "":
        raise CanaryDetectorError("canary value must not be empty.")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_records(records: tuple[CanaryRecord, ...]) -> tuple[CanaryRecord, ...]:
    seen_ids: set[str] = set()
    for record in records:
        _validate_record(record)
        if record.canary_id in seen_ids:
            raise CanaryDetectorError(f"duplicate canary_id '{record.canary_id}'.")
        seen_ids.add(record.canary_id)
    return records


def _validate_record(record: CanaryRecord) -> None:
    for field_name, value in (
        ("canary_id", record.canary_id),
        ("credential_type", record.credential_type),
        ("value", record.value),
        ("sha256", record.sha256),
        ("source", record.source),
    ):
        if value == "":
            raise CanaryDetectorError(f"CanaryRecord field '{field_name}' must not be empty.")
    expected_hash = canary_sha256(record.value)
    if record.sha256 != expected_hash:
        raise CanaryDetectorError(f"CanaryRecord '{record.canary_id}' has sha256 that does not match its value.")


def _validate_probability(value: float, field_name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise CanaryDetectorError(f"{field_name} must be in [0.0, 1.0].")


def _scan_text_for_canaries(text: str, records: tuple[CanaryRecord, ...]) -> tuple[CanaryMatch, ...]:
    matches: list[CanaryMatch] = []
    for record in records:
        start_index = text.find(record.value)
        while start_index != -1:
            end_index = start_index + len(record.value)
            matches.append(
                CanaryMatch(
                    canary_id=record.canary_id,
                    credential_type=record.credential_type,
                    sha256=record.sha256,
                    source=record.source,
                    char_start=start_index,
                    char_end=end_index,
                    metadata=record.metadata,
                )
            )
            start_index = text.find(record.value, end_index)
    return tuple(matches)


def _scan_text_for_encoded_canaries(
    text: str,
    records: tuple[CanaryRecord, ...],
    partial_match_threshold: float,
) -> tuple[EncodedCanaryMatch, ...]:
    matches: list[EncodedCanaryMatch] = []
    for record in records:
        if record.value in text:
            continue
        match = _match_encoded_canary(text=text, record=record, partial_match_threshold=partial_match_threshold)
        if match is not None:
            matches.append(match)
    return tuple(matches)


def _match_encoded_canary(
    text: str,
    record: CanaryRecord,
    partial_match_threshold: float,
) -> EncodedCanaryMatch | None:
    for encoding_name, encoded_value in _encoded_canary_forms(record.value):
        start_index = text.find(encoded_value)
        if encoded_value != "" and start_index != -1:
            return _encoded_match(
                record=record,
                encoding=encoding_name,
                exact=True,
                fragment_ratio=1.0,
                char_start=start_index,
                char_end=start_index + len(encoded_value),
            )

    decoded_blob_match = _match_decoded_blob(text=text, record=record)
    if decoded_blob_match is not None:
        return decoded_blob_match

    normalized_value = _alnum(record.value)
    normalized_text = _alnum(text)
    if normalized_value != "" and normalized_value in normalized_text:
        return _encoded_match(
            record=record,
            encoding="fragmentation",
            exact=True,
            fragment_ratio=1.0,
            char_start=None,
            char_end=None,
        )

    fragment_ratio = _longest_overlap_ratio(token=record.value, text=text)
    if fragment_ratio >= partial_match_threshold:
        return _encoded_match(
            record=record,
            encoding="partial",
            exact=False,
            fragment_ratio=fragment_ratio,
            char_start=None,
            char_end=None,
        )
    return None


def _encoded_canary_forms(value: str) -> tuple[tuple[str, str], ...]:
    return (
        ("base64", base64.b64encode(value.encode("utf-8")).decode("utf-8")),
        ("hex", value.encode("utf-8").hex()),
        ("rot13", codecs.encode(value, "rot_13")),
        ("leet", _leet_encode(value)),
        ("reverse", value[::-1]),
    )


def _match_decoded_blob(text: str, record: CanaryRecord) -> EncodedCanaryMatch | None:
    for regex, decoder_name in ((_BASE64_RUN, "decoded_base64_blob"), (_HEX_RUN, "decoded_hex_blob")):
        for match in regex.finditer(text):
            decoded_value = _decode_blob(match.group(), decoder_name)
            if decoded_value is not None and record.value in decoded_value:
                return _encoded_match(
                    record=record,
                    encoding=decoder_name,
                    exact=True,
                    fragment_ratio=1.0,
                    char_start=match.start(),
                    char_end=match.end(),
                )
    return None


def _decode_blob(value: str, decoder_name: str) -> str | None:
    if decoder_name == "decoded_base64_blob":
        try:
            return base64.b64decode(value + "=" * (-len(value) % 4)).decode("utf-8", "ignore")
        except ValueError:
            return None
    if decoder_name == "decoded_hex_blob":
        try:
            return bytes.fromhex(value).decode("utf-8", "ignore")
        except ValueError:
            return None
    raise CanaryDetectorError(f"unsupported decoder '{decoder_name}'.")


def _encoded_match(
    record: CanaryRecord,
    encoding: str,
    exact: bool,
    fragment_ratio: float,
    char_start: int | None,
    char_end: int | None,
) -> EncodedCanaryMatch:
    return EncodedCanaryMatch(
        canary_id=record.canary_id,
        credential_type=record.credential_type,
        sha256=record.sha256,
        source=record.source,
        encoding=encoding,
        exact=exact,
        fragment_ratio=fragment_ratio,
        char_start=char_start,
        char_end=char_end,
        metadata=record.metadata,
    )


def _leet_encode(value: str) -> str:
    return "".join(_LEET_MAP.get(character.lower(), character) for character in value)


def _alnum(value: str) -> str:
    return _NON_ALNUM.sub("", value)


def _longest_overlap_ratio(token: str, text: str) -> float:
    normalized_token = _alnum(token)
    normalized_text = _alnum(text)
    if normalized_token == "":
        return 0.0
    match = SequenceMatcher(None, normalized_token, normalized_text).find_longest_match(
        0,
        len(normalized_token),
        0,
        len(normalized_text),
    )
    return match.size / len(normalized_token)


def _match_to_dict(match: CanaryMatch) -> dict[str, JsonValue]:
    return {
        "canary_id": match.canary_id,
        "credential_type": match.credential_type,
        "sha256": match.sha256,
        "source": match.source,
        "char_start": match.char_start,
        "char_end": match.char_end,
        "metadata": match.metadata,
    }


def _encoded_match_to_dict(match: EncodedCanaryMatch) -> dict[str, JsonValue]:
    return {
        "canary_id": match.canary_id,
        "credential_type": match.credential_type,
        "sha256": match.sha256,
        "source": match.source,
        "encoding": match.encoding,
        "exact": match.exact,
        "fragment_ratio": match.fragment_ratio,
        "char_start": match.char_start,
        "char_end": match.char_end,
        "metadata": match.metadata,
    }
