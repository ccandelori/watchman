"""Tests for realism / sanity metrics (U6)."""

from __future__ import annotations

import math

import pytest

from detect.dp_honey import build_model, compute_report
from detect.dp_honey.errors import CountLimitError
from detect.dp_honey.realism import REPORT_MAX, enforce_count_limit

EXPECTED_FIELDS = {
    "format",
    "count",
    "unique_count",
    "validity_rate",
    "duplicate_rate",
    "char_entropy_bits",
    "avg_log_likelihood",
    "debug",
    "safety",
}


def _model():
    return build_model("github-ghp", epsilon=1.0, clip=1.0, corpus_size=80, train_seed=4)


def test_report_includes_all_expected_fields():
    model = _model()
    tokens = model.sample(20, seed=1)
    report = compute_report(tokens, model)
    assert set(report) >= EXPECTED_FIELDS
    assert report["format"] == "github-ghp"
    assert report["count"] == 20
    assert math.isfinite(report["char_entropy_bits"])
    assert math.isfinite(report["avg_log_likelihood"])


def test_generated_batch_is_fully_valid():
    model = _model()
    tokens = model.sample(15, seed=2)
    report = compute_report(tokens, model)
    assert report["validity_rate"] == 1.0
    # log-probabilities are <= 0 (probabilities <= 1)
    assert report["avg_log_likelihood"] <= 0.0


def test_duplicate_rate_empty_and_single():
    model = _model()
    empty = compute_report([], model)
    assert empty["count"] == 0
    assert empty["duplicate_rate"] == 0.0
    assert empty["char_entropy_bits"] == 0.0
    assert empty["avg_log_likelihood"] == 0.0

    single = compute_report(model.sample(1, seed=3), model)
    assert single["count"] == 1
    assert single["duplicate_rate"] == 0.0


def test_entropy_handles_repeated_characters_without_error():
    # A degenerate batch of identical AWS-secret-shaped tokens (all 'A').
    model = build_model("aws-secret-access-key", epsilon=1.0, clip=1.0, corpus_size=10, train_seed=0)
    tokens = ["A" * 40, "A" * 40]
    report = compute_report(tokens, model)
    assert report["char_entropy_bits"] == 0.0  # one symbol -> zero entropy, no division error
    assert report["duplicate_rate"] == 0.5


def test_report_is_deterministic_for_fixed_inputs():
    model = _model()
    tokens = model.sample(10, seed=5)
    assert compute_report(tokens, model) == compute_report(tokens, model)


@pytest.mark.parametrize("bad_count", [0, -1, REPORT_MAX + 1])
def test_enforce_count_limit_rejects_out_of_range(bad_count):
    with pytest.raises(CountLimitError):
        enforce_count_limit(bad_count, maximum=REPORT_MAX, label="count")


def test_enforce_count_limit_accepts_in_range():
    enforce_count_limit(1, maximum=REPORT_MAX)
    enforce_count_limit(REPORT_MAX, maximum=REPORT_MAX)  # boundary is allowed


def test_report_metadata_is_synthetic_only():
    model = _model()
    report = compute_report(model.sample(5, seed=6), model)
    safety = report["safety"]
    assert safety["synthetic_only"] is True
    assert safety["provider_valid"] is False
    assert "not" in safety["note"].lower()


def test_validity_rate_reflects_invalid_tokens():
    model = _model()
    batch = [*model.sample(5, seed=1), "not-a-valid-token", "also-bad"]
    report = compute_report(batch, model)
    assert report["count"] == 7
    assert report["validity_rate"] == pytest.approx(5 / 7)
    # Likelihood is computed over the valid tokens only and stays finite.
    assert math.isfinite(report["avg_log_likelihood"])


def test_char_entropy_exact_value():
    model = _model()
    # Two equally likely symbols across the batch -> exactly 1 bit.
    assert compute_report(["AB"], model)["char_entropy_bits"] == pytest.approx(1.0)
