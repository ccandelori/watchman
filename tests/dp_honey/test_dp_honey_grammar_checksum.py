"""Tests for the computed Checksum grammar segment."""

from __future__ import annotations

import json

import numpy as np
import pytest

from detect.dp_honey.checksums import github_crc32_base62
from detect.dp_honey.grammar import ALNUM, Checksum, FormatSpec, Literal, Variable


def _spec() -> FormatSpec:
    return FormatSpec(
        slug="t-cksum",
        name="t",
        description="checksum test format",
        category="test",
        segments=(
            Literal("ghp_"),
            Variable("body", ALNUM, 30),
            Checksum("crc", 6, "github-crc32-base62"),
        ),
        safety_note="synthetic test format; not a real credential",
    )


def test_checksum_segment_excluded_from_variable_segments():
    spec = _spec()
    assert [s.name for s in spec.variable_segments()] == ["body"]


def test_assemble_appends_correct_checksum_and_validates():
    spec = _spec()
    token = spec.random_example(np.random.default_rng(0))
    assert spec.validate(token)
    body = token[len("ghp_") : len("ghp_") + 30]
    assert token.endswith(github_crc32_base62(body, length=6))


def test_wrong_checksum_fails_validation():
    spec = _spec()
    token = spec.random_example(np.random.default_rng(1))
    broken = token[:-1] + ("0" if token[-1] != "0" else "1")
    assert not spec.validate(broken)


def test_checksum_segment_serializes_algorithm_id_only():
    spec = _spec()
    snap = spec.to_snapshot()
    seg = snap["segments"][-1]
    assert seg == {
        "kind": "checksum",
        "name": "crc",
        "length": 6,
        "algorithm": "github-crc32-base62",
    }
    json.loads(json.dumps(snap))


def test_unknown_algorithm_raises_on_assemble():
    spec = FormatSpec(
        slug="t-bad",
        name="t",
        description="t",
        category="test",
        segments=(Literal("x_"), Variable("b", ALNUM, 4), Checksum("c", 6, "nope")),
        safety_note="t",
    )
    with pytest.raises(KeyError):
        spec.random_example(np.random.default_rng(0))


def test_scannable_defaults_true_and_is_overridable():
    base = FormatSpec(
        slug="s",
        name="s",
        description="s",
        category="t",
        segments=(Literal("x_"), Variable("b", ALNUM, 4)),
        safety_note="t",
    )
    assert base.scannable is True
    off = FormatSpec(
        slug="s2",
        name="s",
        description="s",
        category="t",
        segments=(Variable("b", ALNUM, 4),),
        safety_note="t",
        scannable=False,
    )
    assert off.scannable is False
