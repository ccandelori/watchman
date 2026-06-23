from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from aegis.detectors.nimbus import NimbusCriticInput, NimbusCriticScore

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_BASE64_CANDIDATE_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{12,}={0,2}\b")


class NimbusLearnedCriticError(ValueError):
    """Raised when learned NIMBUS critic inputs are invalid."""


class NimbusSecretContextResolver(Protocol):
    def resolve_candidates(self, critic_input: NimbusCriticInput) -> NimbusResolvedContextCandidates | None:
        """Resolve the positive and negative secret contexts for one turn."""


@dataclass(frozen=True)
class NimbusResolvedContext:
    context_id: str
    context_text: str

    def __post_init__(self) -> None:
        _require_non_empty(self.context_id, "context_id")
        _require_non_empty(self.context_text, "context_text")


@dataclass(frozen=True)
class NimbusResolvedContextCandidates:
    true_context: NimbusResolvedContext
    negative_contexts: tuple[NimbusResolvedContext, ...]

    def __post_init__(self) -> None:
        negative_ids = tuple(context.context_id for context in self.negative_contexts)
        if self.true_context.context_id in negative_ids:
            raise NimbusLearnedCriticError("true context must not appear in negative contexts.")
        if len(set(negative_ids)) != len(negative_ids):
            raise NimbusLearnedCriticError("negative context ids must be unique.")


@dataclass(frozen=True)
class NimbusLearnedInfoNCEModel:
    model_id: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    negative_count: int

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        if self.feature_names != ("output_token_overlap", "decoded_output_token_overlap", "state_token_overlap"):
            raise NimbusLearnedCriticError("feature_names do not match the learned NIMBUS v0 scorer.")
        if len(self.weights) != len(self.feature_names):
            raise NimbusLearnedCriticError("weights length must match feature_names length.")
        for weight in self.weights:
            _validate_non_negative_finite(weight, "weights entry")
        if self.negative_count != 16:
            raise NimbusLearnedCriticError("negative_count must be 16.")


class LearnedNimbusInfoNCECritic:
    def __init__(self, model: NimbusLearnedInfoNCEModel, resolver: NimbusSecretContextResolver) -> None:
        self._model = model
        self._resolver = resolver

    def score_turn(self, critic_input: NimbusCriticInput) -> NimbusCriticScore:
        candidates = self._resolver.resolve_candidates(critic_input)
        if candidates is None:
            return NimbusCriticScore(
                estimated_leakage_bits=0.0,
                confidence=0.0,
                evidence={
                    "critic_kind": "learned_infonce",
                    "model_id": self._model.model_id,
                    "reason": "no_resolved_secret_contexts",
                    "candidate_count": 0,
                },
            )
        _validate_candidates(candidates, self._model)
        scores = _candidate_scores(self._model, critic_input, candidates)
        positive_probability = _positive_probability(scores)
        nce_loss_bits = -math.log2(positive_probability)
        estimated_leakage_bits = max(0.0, math.log2(len(scores)) - nce_loss_bits)
        if estimated_leakage_bits < 1e-12:
            estimated_leakage_bits = 0.0
        return NimbusCriticScore(
            estimated_leakage_bits=estimated_leakage_bits,
            confidence=positive_probability,
            evidence={
                "critic_kind": "learned_infonce",
                "model_id": self._model.model_id,
                "candidate_count": len(scores),
                "positive_context_id_sha256": _sha256(candidates.true_context.context_id),
                "positive_probability": positive_probability,
                "nce_loss_bits": nce_loss_bits,
                "estimated_leakage_bits": estimated_leakage_bits,
                "positive_rank": _positive_rank(scores),
                "feature_names": list(self._model.feature_names),
            },
        )


def _validate_candidates(candidates: NimbusResolvedContextCandidates, model: NimbusLearnedInfoNCEModel) -> None:
    if len(candidates.negative_contexts) != model.negative_count:
        raise NimbusLearnedCriticError(f"learned NIMBUS critic requires {model.negative_count} negative contexts.")


def _candidate_scores(
    model: NimbusLearnedInfoNCEModel,
    critic_input: NimbusCriticInput,
    candidates: NimbusResolvedContextCandidates,
) -> tuple[float, ...]:
    contexts = (candidates.true_context, *candidates.negative_contexts)
    return tuple(_score_context(model, critic_input, context) for context in contexts)


def _score_context(
    model: NimbusLearnedInfoNCEModel,
    critic_input: NimbusCriticInput,
    context: NimbusResolvedContext,
) -> float:
    features = _features_for_context(critic_input, context)
    return sum(weight * feature for weight, feature in zip(model.weights, features, strict=True))


def _features_for_context(
    critic_input: NimbusCriticInput,
    context: NimbusResolvedContext,
) -> tuple[float, float, float]:
    context_tokens = _tokens(context.context_text)
    output_tokens = _tokens(critic_input.output_text)
    decoded_tokens = _decoded_output_tokens(critic_input.output_text)
    state_tokens = _tokens(" ".join(message.content for message in critic_input.messages))
    return (
        float(len(output_tokens & context_tokens)),
        float(len(decoded_tokens & context_tokens)),
        float(len(state_tokens & context_tokens)),
    )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _decoded_output_tokens(text: str) -> set[str]:
    decoded_texts: list[str] = []
    for candidate in _BASE64_CANDIDATE_PATTERN.findall(text):
        decoded = _decode_base64_candidate(candidate)
        if decoded is not None:
            decoded_texts.append(decoded)
    return _tokens(" ".join(decoded_texts))


def _decode_base64_candidate(candidate: str) -> str | None:
    padded = candidate + ("=" * ((4 - len(candidate) % 4) % 4))
    try:
        decoded = base64.b64decode(padded.encode("ascii"), validate=True)
        text = decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not _looks_like_text(text):
        return None
    return text


def _looks_like_text(text: str) -> bool:
    if text == "":
        return False
    printable_count = sum(1 for character in text if character.isprintable())
    return printable_count / len(text) >= 0.9


def _positive_probability(scores: tuple[float, ...]) -> float:
    max_score = max(scores)
    exp_scores = tuple(math.exp(score - max_score) for score in scores)
    denominator = sum(exp_scores)
    if denominator <= 0.0:
        raise NimbusLearnedCriticError("softmax denominator must be positive.")
    return exp_scores[0] / denominator


def _positive_rank(scores: tuple[float, ...]) -> int:
    positive_score = scores[0]
    stronger_or_tied_count = sum(1 for score in scores[1:] if score >= positive_score)
    return stronger_or_tied_count + 1


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_non_empty(value: str, field_name: str) -> None:
    if value.strip() == "":
        raise NimbusLearnedCriticError(f"{field_name} must not be empty.")


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise NimbusLearnedCriticError(f"{field_name} must be finite and non-negative.")
