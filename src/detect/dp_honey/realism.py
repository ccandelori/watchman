"""Lightweight realism / sanity metrics for generated honeytoken batches.

These metrics help demo and audit a batch without overclaiming: they are sanity
checks for a *synthetic, shape-only* generator, **not** evidence that the tokens
are computationally indistinguishable from real credentials. Every number is
deterministic for fixed inputs (no randomness here) and JSON-friendly.

Metrics:

* ``validity_rate``      -- fraction of tokens that pass the format spec.
* ``duplicate_rate``     -- ``1 - unique/count`` (0.0 for empty/singleton batches).
* ``char_entropy_bits``  -- pooled character-level Shannon entropy across the batch.
* ``avg_log_likelihood`` -- mean per-variable-character log-probability (nats) under
                            the model's own sampling distribution.
"""

from __future__ import annotations

import math
from collections import Counter

from .bigram import START, BigramHoneytokenModel
from .errors import CountLimitError

# Practical cap for `report`: metrics require materializing the whole batch.
REPORT_MAX = 5000

# Floor for per-character probability when scoring EXTERNAL tokens the model
# assigns ~0 mass; self-generated tokens never hit it. Keeps log-likelihood finite.
LOG_PROB_FLOOR = 1e-12

SAFETY_NOTE = (
    "Sanity metrics for a synthetic, shape-only generator. NOT proof that tokens "
    "are indistinguishable from real credentials, and not derived from real secrets."
)


def enforce_count_limit(count: int, *, maximum: int, label: str = "count") -> None:
    """Raise :class:`CountLimitError` unless ``1 <= count <= maximum``.

    Callers run this *before* any expensive generation so oversized requests fail
    fast instead of materializing a huge batch.
    """
    if count < 1:
        raise CountLimitError(f"{label} must be >= 1, got {count}")
    if count > maximum:
        raise CountLimitError(f"{label} must be <= {maximum}, got {count}")


def compute_report(tokens: list[str], model: BigramHoneytokenModel) -> dict[str, object]:
    """Compute the realism report for *tokens* generated/validated against *model*."""
    spec = model.format_spec
    count = len(tokens)
    unique_count = len(set(tokens))
    valid = sum(1 for token in tokens if spec.validate(token))

    return {
        "format": spec.slug,
        "count": count,
        "unique_count": unique_count,
        "validity_rate": (valid / count) if count else 0.0,
        "duplicate_rate": (1.0 - unique_count / count) if count else 0.0,
        "char_entropy_bits": _pooled_char_entropy_bits(tokens),
        "avg_log_likelihood": _mean_char_log_likelihood(model, tokens),
        "debug": {
            "epsilon": model.epsilon,
            "clip": model.clip,
            "corpus_size": model.corpus_size,
            "train_seed": model.train_seed,
            "registry_version": model.registry_version,
            "spec_hash": model.spec_hash,
            "alphabet_size": len(model.alphabet),
        },
        "safety": {"synthetic_only": True, "provider_valid": False, "note": SAFETY_NOTE},
    }


def _pooled_char_entropy_bits(tokens: list[str]) -> float:
    """Shannon entropy (bits) of the pooled character distribution; 0.0 if empty."""
    counter: Counter[str] = Counter()
    total = 0
    for token in tokens:
        counter.update(token)
        total += len(token)
    if total == 0:
        return 0.0
    entropy = 0.0
    for occurrences in counter.values():
        probability = occurrences / total
        entropy -= probability * math.log2(probability)
    return entropy


def _mean_char_log_likelihood(model: BigramHoneytokenModel, tokens: list[str]) -> float:
    """Mean per-variable-character log-probability (nats) under the model.

    Uses the model's *effective sampling distribution* (masked-and-renormalized
    rows, with uniform fallback) so a self-generated token always has positive
    probability at each step. A tiny floor guards against external invalid tokens.
    """
    spec = model.format_spec
    segments = spec.variable_segments()
    total_log = 0.0
    total_chars = 0
    for token in tokens:
        variables = spec.extract_variables(token)
        if variables is None:
            continue  # invalid tokens are reflected in validity_rate, not likelihood
        for chunk, segment in zip(variables, segments, strict=True):
            prev = START
            for char in chunk:
                probability = _char_probability(model, prev, char, segment.alphabet)
                total_log += math.log(max(probability, LOG_PROB_FLOOR))
                total_chars += 1
                prev = char
    return (total_log / total_chars) if total_chars else 0.0


def _char_probability(model: BigramHoneytokenModel, state: str, char: str, alphabet: str) -> float:
    """The model's effective P(char | state) restricted to *alphabet*.

    Must mirror ``BigramHoneytokenModel._sample_char``'s masking + R8 uniform
    fallback so the likelihood matches the sampler's actual distribution.
    """
    row = model.transitions.get(state, {})
    masked_sum = sum(row.get(symbol, 0.0) for symbol in alphabet)
    if masked_sum > 0.0:
        return row.get(char, 0.0) / masked_sum
    return 1.0 / len(alphabet)  # uniform fallback over the segment alphabet
