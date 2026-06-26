"""DP-noised character bigram model: training, sampling, and bounded repair.

The model learns, per format, a transition table ``P(next_char | prev_char)`` over
the *variable* characters of synthetic, format-valid examples. Training applies a
differential-privacy sketch to the bigram counts:

1. **clip**      -- bound each example's contribution to each cell by ``clip``
                    (per-cell sensitivity becomes ``clip``);
2. **Laplace**   -- add ``Laplace(scale = clip / epsilon)`` noise to every cell;
3. **project**   -- clamp negative noisy counts to zero;
4. **normalize** -- divide each row by its (nonnegative) sum.

This is a research-grade DP *sketch* over bigram counts -- ``epsilon`` is the
per-cell budget. It is **not** a formally audited end-to-end DP guarantee, and the
plan deliberately keeps that claim narrow.

Sampling fills each variable segment left to right, masking each step to the
segment's allowed alphabet (a hard constraint). Transition states are
segment-aware so digit-only fields do not contaminate later alphanumeric fields.
Rows with no positive mass fall back to a deterministic uniform draw over the
segment alphabet (R8). A bounded repair loop re-samples a whole token until it
satisfies the spec (including any ``extra_predicate``), raising
:class:`FormatRepairError` if it cannot.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .errors import EmptyCorpusError, FormatRepairError, InvalidPrivacyParameter
from .formats import REGISTRY_VERSION, get_format
from .grammar import FormatSpec, Variable

# Sentinel "previous state" marking the first character of a variable segment.
START = "<START>"
SEGMENT_START_PREFIX = "<SEGMENT_START"
SEGMENT_CHAR_PREFIX = "<SEGMENT_CHAR"

DEFAULT_EPSILON = 1.0
DEFAULT_CLIP = 1.0
DEFAULT_CORPUS_SIZE = 2000
DEFAULT_TRAIN_SEED = 0
DEFAULT_SAMPLE_SEED = 0
DEFAULT_MAX_REPAIR_ATTEMPTS = 1000


@dataclass
class BigramHoneytokenModel:
    """A trained, DP-noised bigram model for one shape-only format.

    ``transitions`` maps a previous state to ``{next_char: probability}``. New
    models use segment-aware states, while the sampler can still read older
    artifacts that only contain ``START`` and character rows. Rows with no
    positive mass are omitted; sampling falls back to a uniform draw over the
    active segment alphabet for those.

    The stable reuse surface is :meth:`sample`, :meth:`isample`, and
    ``format_slug``. ``transitions``/``alphabet`` are implementation detail -- the
    JSON artifact, not this in-memory object, is the cross-team contract.
    """

    format_spec: FormatSpec
    alphabet: str
    transitions: dict[str, dict[str, float]]
    epsilon: float
    clip: float
    corpus_size: int
    train_seed: int
    registry_version: str
    spec_hash: str

    @property
    def format_slug(self) -> str:
        return self.format_spec.slug

    def sample(
        self,
        count: int,
        *,
        seed: int = DEFAULT_SAMPLE_SEED,
        max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
    ) -> list[str]:
        """Return *count* spec-valid synthetic tokens (deterministic in *seed*)."""
        return list(self.isample(count, seed=seed, max_repair_attempts=max_repair_attempts))

    def isample(
        self,
        count: int,
        *,
        seed: int = DEFAULT_SAMPLE_SEED,
        max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
    ) -> Iterator[str]:
        """Yield *count* spec-valid synthetic tokens one at a time (for streaming)."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        rng = np.random.default_rng(seed)
        for _ in range(count):
            yield self._sample_one(rng, max_repair_attempts)

    def _sample_one(self, rng: np.random.Generator, max_repair_attempts: int) -> str:
        spec = self.format_spec
        segments = spec.variable_segments()
        for _ in range(max_repair_attempts):
            variables = [self._sample_segment(index, seg, rng) for index, seg in enumerate(segments)]
            candidate = spec.assemble(variables)
            if spec.validate(candidate):
                return candidate
        raise FormatRepairError(
            f"could not sample a spec-valid {spec.slug!r} token within {max_repair_attempts} repair attempts"
        )

    def _sample_segment(self, segment_index: int, seg: Variable, rng: np.random.Generator) -> str:
        chars: list[str] = []
        state = segment_start_state(segment_index, seg)
        fallback_state = START
        for _ in range(seg.length):
            ch = self._sample_char(state, seg.alphabet, rng, fallback_state=fallback_state)
            chars.append(ch)
            state = segment_char_state(segment_index, seg, ch)
            fallback_state = ch
        return "".join(chars)

    def _sample_char(
        self,
        state: str,
        seg_alphabet: str,
        rng: np.random.Generator,
        *,
        fallback_state: str | None,
    ) -> str:
        row = self.transitions.get(state, {})
        if not row and fallback_state is not None:
            row = self.transitions.get(fallback_state, {})
        symbols = [c for c in seg_alphabet if row.get(c, 0.0) > 0.0]
        if symbols:
            weights = np.array([row[c] for c in symbols], dtype=float)
            weights /= weights.sum()
            return symbols[int(rng.choice(len(symbols), p=weights))]
        # Zero-mass row -> deterministic uniform fallback over the segment alphabet (R8).
        return seg_alphabet[int(rng.integers(0, len(seg_alphabet)))]


def train_model(
    spec: FormatSpec,
    corpus: list[str],
    *,
    epsilon: float = DEFAULT_EPSILON,
    clip: float = DEFAULT_CLIP,
    train_seed: int = DEFAULT_TRAIN_SEED,
    corpus_size: int | None = None,
) -> BigramHoneytokenModel:
    """Train a DP bigram model from an explicit, format-valid *corpus*.

    ``epsilon`` and ``clip`` must be strictly positive. Every corpus example must
    parse against *spec* (it must be a format-valid token); a foreign example is a
    programming error and raises :class:`ValueError`.
    """
    if epsilon <= 0:
        raise InvalidPrivacyParameter(f"epsilon must be > 0, got {epsilon}")
    if clip <= 0:
        raise InvalidPrivacyParameter(f"clip must be > 0, got {clip}")
    if not corpus:
        raise EmptyCorpusError("training corpus is empty")

    alphabet = spec.variable_alphabet()
    states = sorted(transition_states_for_format(spec))
    row_index = {state: i for i, state in enumerate(states)}
    col_index = {char: j for j, char in enumerate(alphabet)}

    counts = np.zeros((len(states), len(alphabet)), dtype=float)
    for example in corpus:
        variables = spec.extract_variables(example)
        if variables is None:
            raise ValueError(f"corpus example does not match format {spec.slug!r}: {example!r}")
        example_counts = np.zeros_like(counts)
        for segment_index, (chunk, segment) in enumerate(zip(variables, spec.variable_segments(), strict=True)):
            prev = segment_start_state(segment_index, segment)
            for char in chunk:
                example_counts[row_index[prev], col_index[char]] += 1.0
                prev = segment_char_state(segment_index, segment, char)
        # Per-example, per-cell clipping bounds sensitivity to `clip`.
        np.minimum(example_counts, clip, out=example_counts)
        counts += example_counts

    # Laplace mechanism, then nonnegative projection.
    noise_rng = np.random.default_rng(train_seed)
    noisy = counts + noise_rng.laplace(loc=0.0, scale=clip / epsilon, size=counts.shape)
    np.maximum(noisy, 0.0, out=noisy)

    # Normalize rows; omit zero-mass rows (sampling uses the uniform fallback).
    transitions: dict[str, dict[str, float]] = {}
    row_sums = noisy.sum(axis=1)
    for state, i in row_index.items():
        total = float(row_sums[i])
        if total > 0.0:
            probs = noisy[i] / total
            transitions[state] = {alphabet[j]: float(probs[j]) for j in range(len(alphabet)) if probs[j] > 0.0}

    return BigramHoneytokenModel(
        format_spec=spec,
        alphabet=alphabet,
        transitions=transitions,
        epsilon=float(epsilon),
        clip=float(clip),
        corpus_size=int(corpus_size) if corpus_size is not None else len(corpus),
        train_seed=int(train_seed),
        registry_version=REGISTRY_VERSION,
        spec_hash=spec.spec_hash(),
    )


def segment_start_state(segment_index: int, segment: Variable) -> str:
    return f"{SEGMENT_START_PREFIX}:{segment_index}:{segment.name}>"


def segment_char_state(segment_index: int, segment: Variable, char: str) -> str:
    return f"{SEGMENT_CHAR_PREFIX}:{segment_index}:{segment.name}:{char}>"


def transition_states_for_format(spec: FormatSpec) -> set[str]:
    states = {START, *spec.variable_alphabet()}
    for segment_index, segment in enumerate(spec.variable_segments()):
        states.add(segment_start_state(segment_index, segment))
        for char in segment.alphabet:
            states.add(segment_char_state(segment_index, segment, char))
    return states


def build_model(
    fmt: str | FormatSpec,
    *,
    epsilon: float = DEFAULT_EPSILON,
    clip: float = DEFAULT_CLIP,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
    train_seed: int = DEFAULT_TRAIN_SEED,
) -> BigramHoneytokenModel:
    """Convenience: generate a synthetic corpus for *fmt* and train on it.

    The corpus stream is a spawned child of ``train_seed`` and the Laplace-noise
    stream (in :func:`train_model`) is seeded from ``train_seed`` directly. The two
    RNG states are independent, so the whole build is deterministic in
    ``train_seed`` without correlating the two sources of randomness.
    """
    spec = fmt if isinstance(fmt, FormatSpec) else get_format(fmt)
    corpus_seed = np.random.SeedSequence(train_seed).spawn(1)[0]
    corpus = spec.synthetic_corpus(corpus_size, np.random.default_rng(corpus_seed))
    return train_model(spec, corpus, epsilon=epsilon, clip=clip, train_seed=train_seed, corpus_size=corpus_size)


def generate_honeytokens(
    fmt: str | FormatSpec | None = None,
    *,
    model: BigramHoneytokenModel | None = None,
    count: int = 1,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    epsilon: float = DEFAULT_EPSILON,
    clip: float = DEFAULT_CLIP,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
    train_seed: int = DEFAULT_TRAIN_SEED,
    max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
) -> list[str]:
    """High-level entry point: produce *count* shape-only synthetic honeytokens.

    Provide either *fmt* (a slug or :class:`FormatSpec`, trained on the fly) or a
    pre-built/loaded *model*. Outputs are synthetic and never real credentials.
    """
    if model is None:
        if fmt is None:
            raise ValueError("provide either `fmt` (to train) or `model` (pre-built)")
        model = build_model(fmt, epsilon=epsilon, clip=clip, corpus_size=corpus_size, train_seed=train_seed)
    return model.sample(count, seed=sample_seed, max_repair_attempts=max_repair_attempts)
