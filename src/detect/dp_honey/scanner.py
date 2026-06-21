"""Registry-driven secret scanner and auto-decoy helpers.

The scanner derives candidate patterns from each scannable ``FormatSpec`` and
confirms matches with the spec's own ``validate()`` method. Per SAFE-1, findings
never store, log, or return the matched secret value.
"""

from __future__ import annotations

import math
import random
import re
import string
from functools import cache

from .bigram import generate_honeytokens
from .formats import get_format, list_formats
from .grammar import Checksum, FormatSpec, Literal, Variable

_BOUNDARY_BEFORE = r"(?<![A-Za-z0-9_./+-])"
_BOUNDARY_AFTER = r"(?![A-Za-z0-9_/+-])"
_CHECKSUM_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}
_UNKNOWN_FORMAT = "unknown-token"
_UNKNOWN_TOKEN_RE = re.compile(_BOUNDARY_BEFORE + r"[A-Za-z0-9][A-Za-z0-9._~+/\-]{23,}={0,2}" + _BOUNDARY_AFTER)


def _segment_pattern(spec: FormatSpec) -> str:
    parts: list[str] = []
    for segment in spec.segments:
        if isinstance(segment, Literal):
            parts.append(re.escape(segment.text))
        elif isinstance(segment, Variable):
            parts.append(f"[{re.escape(segment.alphabet)}]{{{segment.length}}}")
        elif isinstance(segment, Checksum):
            parts.append(f"[{_CHECKSUM_ALPHABET}]{{{segment.length}}}")
    return "".join(parts)


@cache
def detection_pattern(slug: str) -> re.Pattern[str]:
    """Return the compiled detection regex for a registered format slug."""
    spec = get_format(slug)
    return re.compile(_BOUNDARY_BEFORE + _segment_pattern(spec) + _BOUNDARY_AFTER)


def scan(text: str) -> list[dict[str, int | str]]:
    """Return ``{format, start, end, confidence}`` findings for *text*.

    SAFE-1: matched values are used only as local validation candidates and are
    never included in returned findings.
    """
    raw: list[dict[str, int | str]] = []
    for spec in _scannable_specs():
        checksummed = _has_checksum(spec)
        for match in detection_pattern(spec.slug).finditer(text):
            candidate = match.group(0)
            if not spec.validate(candidate):
                continue
            raw.append(
                {
                    "format": spec.slug,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": "high" if checksummed else "medium",
                }
            )
    raw.extend(_unknown_findings(text))
    return _dedupe(raw)


def auto_decoy(text: str, *, seed: int = 0) -> dict[str, object]:
    """Scan *text*, generate one matching decoy per finding, and swap spans."""
    findings = scan(text)
    decoys: list[str] = []
    for index, finding in enumerate(findings):
        original = text[int(finding["start"]) : int(finding["end"])]
        decoy = original
        attempt = 0
        while decoy == original and attempt < 1000:
            sample_seed = seed + (index * 1000) + attempt
            decoy = _decoy_for_finding(original, str(finding["format"]), sample_seed)
            attempt += 1
        decoys.append(decoy)

    swapped = text
    replacements = zip(findings, decoys, strict=True)
    for finding, decoy in sorted(replacements, key=lambda pair: int(pair[0]["start"]), reverse=True):
        start = int(finding["start"])
        end = int(finding["end"])
        swapped = swapped[:start] + decoy + swapped[end:]

    return {"findings": findings, "decoys": decoys, "swapped_text": swapped}


def _scannable_specs() -> list[FormatSpec]:
    return [spec for spec in list_formats() if spec.scannable]


def _has_checksum(spec: FormatSpec) -> bool:
    return any(isinstance(segment, Checksum) for segment in spec.segments)


def _unknown_findings(text: str) -> list[dict[str, int | str]]:
    findings: list[dict[str, int | str]] = []
    for match in _UNKNOWN_TOKEN_RE.finditer(text):
        candidate = match.group(0)
        if not _is_unknown_secret_like(candidate):
            continue
        findings.append(
            {
                "format": _UNKNOWN_FORMAT,
                "start": match.start(),
                "end": match.end(),
                "confidence": "low",
            }
        )
    return findings


def _is_unknown_secret_like(token: str) -> bool:
    core = token.rstrip("=")
    if len(core) < 24:
        return False
    classes = sum(
        (
            any(char.islower() for char in core),
            any(char.isupper() for char in core),
            any(char.isdigit() for char in core),
            any(char in "._~+/-" for char in core),
        )
    )
    entropy = _shannon_entropy(core)
    prefix_len = _unknown_prefix_length(core)
    if prefix_len and len(core) - prefix_len >= 16 and classes >= 2:
        return entropy >= 3.0
    return len(core) >= 32 and classes >= 2 and entropy >= 3.3


def _shannon_entropy(text: str) -> float:
    counts = {char: text.count(char) for char in set(text)}
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _unknown_prefix_length(token: str) -> int:
    search_window = token[: min(len(token), 24)]
    last = max(search_window.rfind("_"), search_window.rfind("-"))
    if last < 1:
        return 0
    prefix = token[: last + 1]
    suffix = token[last + 1 :]
    if len(suffix) < 16 or not any(char.isalpha() for char in prefix):
        return 0
    return last + 1


def _decoy_for_finding(original: str, fmt: str, seed: int) -> str:
    if fmt == _UNKNOWN_FORMAT:
        return _generate_unknown_decoy(original, seed)
    return generate_honeytokens(fmt, count=1, sample_seed=seed)[0]


def _generate_unknown_decoy(original: str, seed: int) -> str:
    rng = random.Random(seed)
    prefix_len = _unknown_prefix_length(original.rstrip("="))
    chars: list[str] = []
    for index, char in enumerate(original):
        if index < prefix_len or char in "._~+/-=":
            chars.append(char)
        elif char.islower():
            chars.append(rng.choice(string.ascii_lowercase))
        elif char.isupper():
            chars.append(rng.choice(string.ascii_uppercase))
        elif char.isdigit():
            chars.append(rng.choice(string.digits))
        else:
            chars.append(rng.choice(string.ascii_letters + string.digits))
    return "".join(chars)


def _dedupe(findings: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    ordered = sorted(
        findings,
        key=lambda finding: (
            -_CONFIDENCE_RANK[str(finding["confidence"])],
            -(int(finding["end"]) - int(finding["start"])),
            int(finding["start"]),
        ),
    )
    kept: list[dict[str, int | str]] = []
    for finding in ordered:
        start = int(finding["start"])
        end = int(finding["end"])
        if any(start < int(kept_finding["end"]) and int(kept_finding["start"]) < end for kept_finding in kept):
            continue
        kept.append(finding)
    return sorted(kept, key=lambda finding: int(finding["start"]))
