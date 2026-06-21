"""The committed golden fixture as a teammate-facing integration contract (U8).

The fixture (`tests/dp_honey/fixtures/dp_honey/golden_model.json`) is the one
model artifact committed to the repo. These tests prove a teammate can inspect,
validate, and generate from it without retraining, that its provenance is clearly
synthetic, and that registry drift against it is detected rather than silently
accepted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detect.dp_honey import load_model, model_io
from detect.dp_honey.__main__ import main
from detect.dp_honey.errors import FormatSpecMismatchError
from detect.dp_honey.grammar import UPPER_DIGITS, FormatSpec, Literal, Variable

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "dp_honey" / "golden_model.json"


def test_golden_fixture_sampling_is_deterministic():
    model_a = load_model(GOLDEN)
    model_b = load_model(GOLDEN)
    tokens = model_a.sample(5, seed=123)
    assert tokens == model_b.sample(5, seed=123)
    assert all(model_a.format_spec.validate(token) for token in tokens)


def test_golden_fixture_metadata_is_synthetic_only():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["safety"]["synthetic_only"] is True
    assert data["safety"]["provider_valid"] is False
    assert "not" in data["safety"]["note"].lower()
    assert data["format"]["spec_snapshot"]["provider_valid"] is False


def test_golden_fixture_drift_is_detected(monkeypatch):
    # Simulate the registry spec changing out from under the saved artifact: a
    # same-slug format with a different body length yields a different spec hash.
    drifted = FormatSpec(
        slug="aws-access-key-id",
        name="AWS Access Key ID",
        description="drifted",
        category="cloud-key",
        segments=(Literal("AKIA"), Variable("body", UPPER_DIGITS, 17)),  # 17 != 16
        safety_note="drifted spec for the drift test",
    )
    monkeypatch.setattr(model_io, "get_format", lambda slug: drifted)
    with pytest.raises(FormatSpecMismatchError):
        load_model(GOLDEN)


def test_demo_commands_run_against_the_fixture(capsys):
    fixture = str(GOLDEN)
    assert main(["inspect-model", "--model", fixture]) == 0
    assert main(["validate", "--model", fixture]) == 0
    assert main(["generate", "--model", fixture, "--count", "5", "--seed", "1"]) == 0
    assert main(["report", "--model", fixture, "--count", "50", "--seed", "1"]) == 0
    # The demo produced real output on stdout (tokens + a JSON report).
    assert capsys.readouterr().out.strip()
