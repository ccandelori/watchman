from __future__ import annotations

import re
import secrets
import string
from collections.abc import Callable
from dataclasses import dataclass

from aegis.core.contracts import JsonValue, Message, SensitiveSpan
from aegis.detectors.canary import CanaryRecord, InMemoryCanaryRegistry, canary_sha256

HoneytokenGenerator = Callable[[str, str], str]
HoneytokenMetadataProvider = Callable[[str, str, str], dict[str, JsonValue]]

_PLACEHOLDER_PATTERN = re.compile(r"\{\{CREDENTIAL:([^:}]+):([^}]+)\}\}")


class HoneytokenLedgerError(ValueError):
    """Raised when honeytoken ledger configuration or input is invalid."""


@dataclass(frozen=True)
class Honeytoken:
    slot_name: str
    credential_type: str
    value: str
    canary_id: str
    sha256: str
    source: str
    turn_planted: int
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class HoneytokenInjectionResult:
    messages: tuple[Message, ...]
    sensitive_spans: tuple[SensitiveSpan, ...]
    canary_records: tuple[CanaryRecord, ...]

    def canary_registry(self) -> InMemoryCanaryRegistry:
        return InMemoryCanaryRegistry(records=self.canary_records)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "sensitive_spans": [span.to_dict() for span in self.sensitive_spans],
            "canary_records": [_canary_record_summary(record) for record in self.canary_records],
        }


class HoneytokenLedger:
    def __init__(
        self,
        session_id: str,
        generator: HoneytokenGenerator,
        source: str = "dp_honey_lite",
        metadata_provider: HoneytokenMetadataProvider | None = None,
    ) -> None:
        if session_id == "":
            raise HoneytokenLedgerError("session_id must not be empty.")
        if source == "":
            raise HoneytokenLedgerError("source must not be empty.")
        self.session_id = session_id
        self.source = source
        self._generator = generator
        self._metadata_provider = metadata_provider
        self._honeytokens_by_slot: dict[str, Honeytoken] = {}
        self._real_secrets_by_slot: dict[str, tuple[str, str]] = {}

    def register_real_secret(self, slot_name: str, credential_type: str, value: str) -> None:
        _validate_slot(slot_name=slot_name, credential_type=credential_type)
        if value == "":
            raise HoneytokenLedgerError("real secret value must not be empty.")
        self._real_secrets_by_slot[slot_name] = (credential_type, value)

    def plant(self, slot_name: str, credential_type: str, turn_index: int) -> Honeytoken:
        _validate_slot(slot_name=slot_name, credential_type=credential_type)
        if turn_index < 0:
            raise HoneytokenLedgerError("turn_index must be non-negative.")
        existing = self._honeytokens_by_slot.get(slot_name)
        if existing is not None:
            return existing
        value = self._generator(slot_name, credential_type)
        if value == "":
            raise HoneytokenLedgerError("honeytoken generator returned an empty value.")
        canary_id = _canary_id(session_id=self.session_id, slot_name=slot_name)
        metadata = _copy_metadata(
            self._metadata_provider(slot_name, credential_type, value) if self._metadata_provider is not None else {}
        )
        honeytoken = Honeytoken(
            slot_name=slot_name,
            credential_type=credential_type,
            value=value,
            canary_id=canary_id,
            sha256=canary_sha256(value),
            source=self.source,
            turn_planted=turn_index,
            metadata=metadata,
        )
        self._honeytokens_by_slot[slot_name] = honeytoken
        return honeytoken

    def substitute_real_secrets(self, text: str, turn_index: int) -> tuple[str, tuple[Honeytoken, ...]]:
        substituted_text = text
        planted: list[Honeytoken] = []
        for slot_name, secret_pair in self._real_secrets_by_slot.items():
            credential_type, value = secret_pair
            if value not in substituted_text:
                continue
            honeytoken = self.plant(slot_name=slot_name, credential_type=credential_type, turn_index=turn_index)
            substituted_text = substituted_text.replace(value, honeytoken.value)
            planted.append(honeytoken)
        return substituted_text, tuple(planted)

    def canary_records(self) -> tuple[CanaryRecord, ...]:
        return tuple(_canary_record(honeytoken) for honeytoken in self._honeytokens_by_slot.values())


def inject_honeytokens(
    messages: tuple[Message, ...],
    ledger: HoneytokenLedger,
    turn_index: int,
) -> HoneytokenInjectionResult:
    if turn_index < 0:
        raise HoneytokenLedgerError("turn_index must be non-negative.")
    transformed_messages: list[Message] = []
    sensitive_spans: list[SensitiveSpan] = []
    seen_span_keys: set[tuple[str, int | None, int | None]] = set()

    for message in messages:
        transformed_content, placeholder_spans = _replace_placeholders(
            content=message.content,
            ledger=ledger,
            turn_index=turn_index,
        )
        transformed_content, scrubbed_tokens = ledger.substitute_real_secrets(
            text=transformed_content,
            turn_index=turn_index,
        )
        transformed_messages.append(Message(role=message.role, content=transformed_content))
        for span in placeholder_spans + _spans_for_scrubbed_tokens(
            content=transformed_content,
            honeytokens=scrubbed_tokens,
        ):
            span_key = (span.identifier or "", span.char_start, span.char_end)
            if span_key in seen_span_keys:
                continue
            sensitive_spans.append(span)
            seen_span_keys.add(span_key)

    return HoneytokenInjectionResult(
        messages=tuple(transformed_messages),
        sensitive_spans=tuple(sensitive_spans),
        canary_records=ledger.canary_records(),
    )


def default_honeytoken_generator(slot_name: str, credential_type: str) -> str:
    _validate_slot(slot_name=slot_name, credential_type=credential_type)
    alphabet = string.ascii_letters + string.digits
    body = "".join(secrets.choice(alphabet) for _ in range(24))
    prefix = _prefix_for_credential_type(credential_type)
    return f"{prefix}{body}"


def _replace_placeholders(
    content: str,
    ledger: HoneytokenLedger,
    turn_index: int,
) -> tuple[str, tuple[SensitiveSpan, ...]]:
    parts: list[str] = []
    spans: list[SensitiveSpan] = []
    cursor = 0
    for match in _PLACEHOLDER_PATTERN.finditer(content):
        slot_name = match.group(1)
        credential_type = match.group(2)
        honeytoken = ledger.plant(slot_name=slot_name, credential_type=credential_type, turn_index=turn_index)
        parts.append(content[cursor : match.start()])
        char_start = sum(len(part) for part in parts)
        parts.append(honeytoken.value)
        char_end = char_start + len(honeytoken.value)
        spans.append(_sensitive_span(honeytoken=honeytoken, char_start=char_start, char_end=char_end))
        cursor = match.end()
    parts.append(content[cursor:])
    return "".join(parts), tuple(spans)


def _spans_for_scrubbed_tokens(content: str, honeytokens: tuple[Honeytoken, ...]) -> tuple[SensitiveSpan, ...]:
    spans: list[SensitiveSpan] = []
    for honeytoken in honeytokens:
        start_index = content.find(honeytoken.value)
        while start_index != -1:
            spans.append(
                _sensitive_span(
                    honeytoken=honeytoken,
                    char_start=start_index,
                    char_end=start_index + len(honeytoken.value),
                )
            )
            start_index = content.find(honeytoken.value, start_index + len(honeytoken.value))
    return tuple(spans)


def _sensitive_span(honeytoken: Honeytoken, char_start: int | None, char_end: int | None) -> SensitiveSpan:
    metadata: dict[str, JsonValue] = {
        "slot_name": honeytoken.slot_name,
        "credential_type": honeytoken.credential_type,
        "sha256": honeytoken.sha256,
        "turn_planted": honeytoken.turn_planted,
    }
    metadata.update(_copy_metadata(honeytoken.metadata))
    return SensitiveSpan(
        kind="honeytoken",
        source=honeytoken.source,
        char_start=char_start,
        char_end=char_end,
        token_start=None,
        token_end=None,
        identifier=honeytoken.canary_id,
        metadata=metadata,
    )


def _canary_record(honeytoken: Honeytoken) -> CanaryRecord:
    metadata: dict[str, JsonValue] = {
        "slot_name": honeytoken.slot_name,
        "turn_planted": honeytoken.turn_planted,
    }
    metadata.update(_copy_metadata(honeytoken.metadata))
    return CanaryRecord(
        canary_id=honeytoken.canary_id,
        credential_type=honeytoken.credential_type,
        value=honeytoken.value,
        sha256=honeytoken.sha256,
        source=honeytoken.source,
        metadata=metadata,
    )


def _canary_record_summary(record: CanaryRecord) -> dict[str, JsonValue]:
    return {
        "canary_id": record.canary_id,
        "credential_type": record.credential_type,
        "sha256": record.sha256,
        "source": record.source,
        "metadata": _copy_metadata(record.metadata),
    }


def _copy_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _copy_json_value(value) for key, value in metadata.items()}


def _copy_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value


def _canary_id(session_id: str, slot_name: str) -> str:
    safe_session = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)
    safe_slot = re.sub(r"[^A-Za-z0-9_-]", "_", slot_name)
    return f"hny_{safe_session}_{safe_slot}"


def _validate_slot(slot_name: str, credential_type: str) -> None:
    if slot_name == "":
        raise HoneytokenLedgerError("slot_name must not be empty.")
    if credential_type == "":
        raise HoneytokenLedgerError("credential_type must not be empty.")


def _prefix_for_credential_type(credential_type: str) -> str:
    prefixes: dict[str, str] = {
        "aws_access_key": "AKIA",
        "github_pat": "ghp_",
        "oauth_token": "ya29.",
        "openai_key": "sk-",
        "stripe_key": "sk_live_",
    }
    return prefixes.get(credential_type, "hny_")
