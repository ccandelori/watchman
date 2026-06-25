from __future__ import annotations

import base64
import hashlib
import re

from aegis.core.contracts import NormalizedTurn
from aegis.core.orchestrator import ModelResponse

SUPPORTED_MOCK_RESPONSE_MODES = frozenset(
    (
        "default",
        "echo_last_user",
        "leak_first_honeytoken",
        "base64_first_honeytoken",
        "partial_first_honeytoken",
    )
)

_HONEYTOKEN_PATTERN = re.compile(r"(?:AKIA|ghp_|ya29\.|sk_live_|sk-|hny_)[A-Za-z0-9._-]{8,}")
_TRAILING_HONEYTOKEN_PUNCTUATION = ".,;:"


class MockModelProvider:
    def __init__(self, default_content: str) -> None:
        self._default_content = default_content

    def generate(self, turn: NormalizedTurn) -> ModelResponse:
        content = turn.metadata.get("mock_response")
        if isinstance(content, str) and content != "":
            return ModelResponse(output_text=content, metadata={"provider": "mock", "mode": "static"})

        mode = _mock_response_mode(turn)
        output_text = _generate_mode_output(turn=turn, mode=mode, default_content=self._default_content)
        return ModelResponse(output_text=output_text, metadata={"provider": "mock", "mode": mode})


def _mock_response_mode(turn: NormalizedTurn) -> str:
    mode = turn.metadata.get("mock_response_mode")
    if not isinstance(mode, str) or mode == "":
        return "default"
    if mode not in SUPPORTED_MOCK_RESPONSE_MODES:
        raise ValueError(f"unsupported mock_response_mode '{mode}'.")
    return mode


def _generate_mode_output(turn: NormalizedTurn, mode: str, default_content: str) -> str:
    if mode == "default":
        return default_content
    if mode == "echo_last_user":
        return _last_user_message_content(turn)

    honeytoken = _first_honeytoken(turn)
    if honeytoken is None:
        return default_content
    if mode == "leak_first_honeytoken":
        return f"leaked={honeytoken}"
    if mode == "base64_first_honeytoken":
        encoded = base64.b64encode(honeytoken.encode("utf-8")).decode("utf-8")
        return f"leaked_base64={encoded}"
    if mode == "partial_first_honeytoken":
        return f"leaked_partial={honeytoken[: max(4, len(honeytoken) // 2)]}"
    raise ValueError(f"unsupported mock_response_mode '{mode}'.")


def _last_user_message_content(turn: NormalizedTurn) -> str:
    for message in reversed(turn.messages):
        if message.role == "user":
            return message.content
    return ""


def _first_honeytoken(turn: NormalizedTurn) -> str | None:
    known_hashes = _known_honeytoken_hashes(turn)
    if len(known_hashes) == 0:
        return None
    for message in turn.messages:
        for match in _HONEYTOKEN_PATTERN.finditer(message.content):
            candidate = _registered_honeytoken_candidate(value=match.group(0), known_hashes=known_hashes)
            if candidate is not None:
                return candidate
    return None


def _known_honeytoken_hashes(turn: NormalizedTurn) -> frozenset[str]:
    hashes: set[str] = set()
    for span in turn.sensitive_spans:
        if span.kind != "honeytoken":
            continue
        sha256 = span.metadata.get("sha256")
        if isinstance(sha256, str) and sha256 != "":
            hashes.add(sha256)
    return frozenset(hashes)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _registered_honeytoken_candidate(value: str, known_hashes: frozenset[str]) -> str | None:
    candidate = value
    while candidate != "":
        if _sha256(candidate) in known_hashes:
            return candidate
        if candidate[-1] not in _TRAILING_HONEYTOKEN_PUNCTUATION:
            return None
        candidate = candidate[:-1]
    return None
