"""Declarative grammar primitives for shape-only honeytoken formats.

A :class:`FormatSpec` is an ordered sequence of :class:`Literal` and
:class:`Variable` segments:

* :class:`Literal` -- fixed text (prefixes, separators, PEM armor). Never sampled.
* :class:`Variable` -- a fixed-length run over an explicit character alphabet.
  These are the *only* parts the DP bigram model samples.

Keeping formats declarative (data, not callbacks) makes the registry auditable,
lets validation be one shared cursor walk, and lets the README matrix and the
JSON model artifacts derive from a single source of truth.
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .errors import FormatRepairError

# --- Shared character alphabets -------------------------------------------------
UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase
DIGITS = string.digits
ALNUM = UPPER + LOWER + DIGITS
UPPER_DIGITS = UPPER + DIGITS
BASE64 = ALNUM + "+/"
BASE64URL = ALNUM + "-_"
PASSWORD = ALNUM + "!@#$%^&*()-_=+"
HEX = "0123456789abcdef"


def canonical_json(obj: object) -> str:
    """Serialize *obj* to a stable, compact JSON string (for hashing/equality)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class Literal:
    """A fixed text segment (prefix, separator, or armor). Never sampled."""

    text: str

    def to_dict(self) -> dict:
        return {"kind": "literal", "text": self.text}


@dataclass(frozen=True)
class Variable:
    """A fixed-length variable segment drawn from an explicit alphabet."""

    name: str
    alphabet: str
    length: int

    def to_dict(self) -> dict:
        return {
            "kind": "variable",
            "name": self.name,
            "alphabet": self.alphabet,
            "length": self.length,
        }


@dataclass(frozen=True)
class Checksum:
    """A computed segment derived from preceding variable payload.

    ``algorithm`` is an id resolved through :mod:`detect.dp_honey.checksums`.
    The value is never sampled by the bigram model; it is derived at assembly
    time and verified at validation time. Only the id is serialized.
    """

    name: str
    length: int
    algorithm: str

    def to_dict(self) -> dict:
        return {
            "kind": "checksum",
            "name": self.name,
            "length": self.length,
            "algorithm": self.algorithm,
        }


Segment = Literal | Variable | Checksum


@dataclass(frozen=True)
class FormatSpec:
    """A declarative, shape-only specification for one secret family.

    ``provider_valid`` is always ``False`` for this package: outputs are
    format-compatible decoys, never real credentials. ``extra_predicate`` is an
    optional whole-token constraint (e.g. password class requirements) that the
    per-character sampler cannot guarantee, so it is enforced via bounded repair.
    It is intentionally *not* serialized into artifacts -- it is behavior keyed by
    ``slug``, while the snapshot/hash covers the serializable structure.
    """

    slug: str
    name: str
    description: str
    category: str
    segments: tuple
    safety_note: str
    provider_valid: bool = False
    scannable: bool = True
    extra_predicate: Callable[[str], bool] | None = None

    def variable_segments(self) -> list[Variable]:
        return [s for s in self.segments if isinstance(s, Variable)]

    def variable_alphabet(self) -> str:
        """Sorted union of every variable segment's alphabet (the model alphabet)."""
        chars: set[str] = set()
        for seg in self.variable_segments():
            chars.update(seg.alphabet)
        return "".join(sorted(chars))

    def assemble(self, variables: list[str]) -> str:
        """Interleave fixed literals with sampled *variables* into a full token."""
        from .checksums import compute_checksum

        out: list[str] = []
        checksum_input: list[str] = []
        vi = 0
        for seg in self.segments:
            if isinstance(seg, Literal):
                out.append(seg.text)
            elif isinstance(seg, Variable):
                value = variables[vi]
                out.append(value)
                checksum_input.append(value)
                vi += 1
            else:
                out.append(compute_checksum(seg.algorithm, "".join(checksum_input), length=seg.length))
        return "".join(out)

    def extract_variables(self, token: str) -> list[str] | None:
        """Parse *token* against the spec; return the variable chunks or ``None``.

        Returns ``None`` when any literal, length, or charset constraint fails, or
        when the token has trailing characters. This is the structural half of
        validation and is reused by training to recover the variable stream.
        """
        from .checksums import compute_checksum

        pos = 0
        variables: list[str] = []
        checksum_input: list[str] = []
        for seg in self.segments:
            if isinstance(seg, Literal):
                if not token.startswith(seg.text, pos):
                    return None
                pos += len(seg.text)
            elif isinstance(seg, Variable):
                chunk = token[pos : pos + seg.length]
                if len(chunk) != seg.length:
                    return None
                if any(c not in seg.alphabet for c in chunk):
                    return None
                variables.append(chunk)
                checksum_input.append(chunk)
                pos += seg.length
            else:
                chunk = token[pos : pos + seg.length]
                if len(chunk) != seg.length:
                    return None
                expected = compute_checksum(seg.algorithm, "".join(checksum_input), length=seg.length)
                if chunk != expected:
                    return None
                pos += seg.length
        if pos != len(token):
            return None
        return variables

    def validate(self, token: str) -> bool:
        """True iff *token* matches the structural spec and any extra predicate."""
        if self.extract_variables(token) is None:
            return False
        return self.extra_predicate is None or self.extra_predicate(token)

    def random_example(self, rng: np.random.Generator, max_attempts: int = 1000) -> str:
        """Generate one uniform-random, spec-valid synthetic example.

        Variable segments are filled uniformly at random; if an ``extra_predicate``
        is present we retry until it is satisfied (bounded), raising
        :class:`FormatRepairError` if it never is.
        """
        for _ in range(max_attempts):
            variables: list[str] = []
            for seg in self.variable_segments():
                idx = rng.integers(0, len(seg.alphabet), size=seg.length)
                variables.append("".join(seg.alphabet[int(i)] for i in idx))
            token = self.assemble(variables)
            if self.extra_predicate is None or self.extra_predicate(token):
                return token
        raise FormatRepairError(
            f"could not generate a spec-valid example for {self.slug!r} within {max_attempts} attempts"
        )

    def synthetic_corpus(self, size: int, rng: np.random.Generator) -> list[str]:
        """Generate *size* synthetic, spec-valid examples for training."""
        return [self.random_example(rng) for _ in range(size)]

    def to_snapshot(self) -> dict:
        """Serializable structural snapshot (the artifact's format identity)."""
        return {
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "provider_valid": self.provider_valid,
            "safety_note": self.safety_note,
            "segments": [seg.to_dict() for seg in self.segments],
        }

    def spec_hash(self) -> str:
        """Stable ``sha256:`` hash of the canonical snapshot for drift detection."""
        digest = hashlib.sha256(canonical_json(self.to_snapshot()).encode("utf-8"))
        return "sha256:" + digest.hexdigest()
