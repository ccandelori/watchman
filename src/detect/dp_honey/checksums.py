"""Checksum algorithms for structurally-faithful formats.

These make a decoy pass a provider's structural checksum check. A
checksum-valid decoy is still non-functional: the provider has no record of it,
so it authenticates nothing. No real cryptography is used here.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _to_base62(number: int) -> str:
    if number == 0:
        return _BASE62[0]
    digits: list[str] = []
    while number:
        number, remainder = divmod(number, 62)
        digits.append(_BASE62[remainder])
    return "".join(reversed(digits))


def github_crc32_base62(body: str, *, length: int = 6) -> str:
    """Return GitHub's base62-encoded CRC32 checksum for a random body.

    Confirmed in Task A0 against GitHub's public token-format writeup and the
    independent therootcompany/base62-token.js implementation/test fixtures.
    GitHub classic tokens checksum only the entropy/body after the underscore,
    not the ``ghp_``/``gho_`` prefix.
    """
    checksum = zlib.crc32(body.encode("ascii")) & 0xFFFFFFFF
    encoded = _to_base62(checksum)
    return encoded.rjust(length, _BASE62[0])[:length]


CHECKSUM_ALGORITHMS: dict[str, Callable[..., str]] = {
    "github-crc32-base62": github_crc32_base62,
}


def compute_checksum(algorithm: str, body: str, *, length: int) -> str:
    """Resolve *algorithm* from the registry and compute the checksum for *body*."""
    return CHECKSUM_ALGORITHMS[algorithm](body, length=length)
