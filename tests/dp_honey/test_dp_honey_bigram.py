"""Tests for DP bigram training, sampling, and bounded repair (U3)."""

from __future__ import annotations

import numpy as np
import pytest

from detect.dp_honey import (
    BigramHoneytokenModel,
    build_model,
    generate_honeytokens,
    get_format,
    train_model,
)
from detect.dp_honey.bigram import START
from detect.dp_honey.errors import (
    EmptyCorpusError,
    FormatRepairError,
    InvalidPrivacyParameter,
)
from detect.dp_honey.formats import REGISTRY_VERSION
from detect.dp_honey.grammar import FormatSpec, Variable


def _tiny_spec(predicate=None) -> FormatSpec:
    """A minimal 2-letter format used to make counting/clipping observable."""
    return FormatSpec(
        slug="t-tiny",
        name="tiny",
        description="tiny test format",
        category="test",
        segments=(Variable("v", "AB", 4),),
        safety_note="synthetic test format; not a real credential",
        extra_predicate=predicate,
    )


def test_train_then_sample_produces_valid_tokens():
    spec = get_format("github-ghp")
    tokens = generate_honeytokens(spec, count=8, sample_seed=1, train_seed=2)
    assert len(tokens) == 8
    assert all(spec.validate(t) for t in tokens)


def test_same_seeds_produce_identical_sequence():
    a = generate_honeytokens("github-ghp", count=5, sample_seed=7, train_seed=3)
    b = generate_honeytokens("github-ghp", count=5, sample_seed=7, train_seed=3)
    assert a == b


def test_different_sample_seed_changes_output():
    a = generate_honeytokens("github-ghp", count=5, sample_seed=7, train_seed=3)
    c = generate_honeytokens("github-ghp", count=5, sample_seed=8, train_seed=3)
    assert a != c  # collision is astronomically unlikely for a 36-char alnum body


def test_epsilon_changes_the_transition_table():
    spec = get_format("aws-secret-access-key")
    corpus = spec.synthetic_corpus(50, np.random.default_rng(0))
    low = train_model(spec, corpus, epsilon=0.1, clip=1.0, train_seed=42)
    high = train_model(spec, corpus, epsilon=10.0, clip=1.0, train_seed=42)
    assert low.transitions != high.transitions


def test_clipping_limits_a_repeated_bigram():
    # "AAAB": bigram A->A appears twice, A->B once. With clip=1 the repeated
    # A->A is capped, so P(A|A) drops relative to an unclipped (clip=3) fit.
    # epsilon is huge so Laplace noise is negligible and the effect is clipping.
    spec = _tiny_spec()
    corpus = ["AAAB"]
    clipped = train_model(spec, corpus, epsilon=1e6, clip=1.0, train_seed=0)
    unclipped = train_model(spec, corpus, epsilon=1e6, clip=3.0, train_seed=0)
    assert clipped.transitions["A"]["A"] < unclipped.transitions["A"]["A"]


@pytest.mark.parametrize("epsilon", [0.0, -1.0])
def test_nonpositive_epsilon_raises(epsilon):
    spec = get_format("github-ghp")
    corpus = spec.synthetic_corpus(5, np.random.default_rng(0))
    with pytest.raises(InvalidPrivacyParameter):
        train_model(spec, corpus, epsilon=epsilon, clip=1.0, train_seed=0)


@pytest.mark.parametrize("clip", [0.0, -2.0])
def test_nonpositive_clip_raises(clip):
    spec = get_format("github-ghp")
    corpus = spec.synthetic_corpus(5, np.random.default_rng(0))
    with pytest.raises(InvalidPrivacyParameter):
        train_model(spec, corpus, epsilon=1.0, clip=clip, train_seed=0)


def test_empty_corpus_raises():
    spec = get_format("github-ghp")
    with pytest.raises(EmptyCorpusError):
        train_model(spec, [], epsilon=1.0, clip=1.0, train_seed=0)


def test_impossible_repair_raises_after_attempt_limit():
    spec = _tiny_spec(predicate=lambda token: False)  # never satisfiable
    model = train_model(spec, ["AAAA"], epsilon=1.0, clip=1.0, train_seed=0)
    with pytest.raises(FormatRepairError):
        model.sample(1, seed=0, max_repair_attempts=5)


def test_zero_mass_rows_use_uniform_fallback():
    # An empty transition table forces the uniform fallback at every step.
    spec = get_format("aws-secret-access-key")
    model = BigramHoneytokenModel(
        format_spec=spec,
        alphabet=spec.variable_alphabet(),
        transitions={},
        epsilon=1.0,
        clip=1.0,
        corpus_size=0,
        train_seed=0,
        registry_version=REGISTRY_VERSION,
        spec_hash=spec.spec_hash(),
    )
    tokens = model.sample(3, seed=1)
    assert len(tokens) == 3
    assert all(spec.validate(t) for t in tokens)
    # Fallback is deterministic in the seed too.
    assert model.sample(3, seed=1) == tokens


def test_transition_rows_are_normalized():
    spec = get_format("stripe-sk-live")
    model = build_model(spec, epsilon=1.0, clip=1.0, corpus_size=100, train_seed=11)
    for state, row in model.transitions.items():
        assert abs(sum(row.values()) - 1.0) < 1e-6, state
    assert START in model.transitions  # the first-char distribution must exist


def test_generate_honeytokens_with_prebuilt_model_matches_sample():
    model = build_model("github-ghp", epsilon=1.0, clip=1.0, corpus_size=30, train_seed=4)
    tokens = generate_honeytokens(model=model, count=3, sample_seed=9)
    assert tokens == model.sample(3, seed=9)
    spec = get_format("github-ghp")
    assert all(spec.validate(t) for t in tokens)


def test_generate_honeytokens_requires_a_source():
    with pytest.raises(ValueError):
        generate_honeytokens()


def test_negative_count_raises_value_error():
    model = build_model("github-ghp", corpus_size=10, train_seed=0)
    with pytest.raises(ValueError):
        model.sample(-1, seed=0)
