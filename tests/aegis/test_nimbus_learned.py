from __future__ import annotations

from dataclasses import dataclass

import pytest

from aegis.core.contracts import Message
from aegis.detectors.nimbus import NimbusCriticInput, NimbusState
from aegis.detectors.nimbus_learned import (
    LearnedNimbusInfoNCECritic,
    NimbusLearnedCriticError,
    NimbusLearnedInfoNCEModel,
    NimbusResolvedContext,
    NimbusResolvedContextCandidates,
    NimbusSecretContextResolver,
)


@dataclass(frozen=True)
class StaticResolver:
    candidates: NimbusResolvedContextCandidates | None

    def resolve_candidates(self, critic_input: NimbusCriticInput) -> NimbusResolvedContextCandidates | None:
        return self.candidates


def test_learned_nimbus_critic_scores_context_without_leaking_raw_context() -> None:
    critic = LearnedNimbusInfoNCECritic(
        model=_learned_model(),
        resolver=StaticResolver(_candidate_contexts()),
    )

    score = critic.score_turn(_critic_input("The report includes safe-canary-repo-alpha-7294."))

    assert score.estimated_leakage_bits > 0.0
    assert score.confidence > 0.0
    assert score.evidence["critic_kind"] == "learned_infonce"
    assert score.evidence["candidate_count"] == 17
    assert "safe-canary-repo-alpha-7294" not in str(score.evidence)
    assert "safe-decoy-marker" not in str(score.evidence)


def test_learned_nimbus_critic_returns_zero_when_resolver_has_no_contexts() -> None:
    critic = LearnedNimbusInfoNCECritic(
        model=_learned_model(),
        resolver=StaticResolver(None),
    )

    score = critic.score_turn(_critic_input("The report includes safe-canary-repo-alpha-7294."))

    assert score.estimated_leakage_bits == 0.0
    assert score.confidence == 0.0
    assert score.evidence["reason"] == "no_resolved_secret_contexts"


def test_learned_nimbus_critic_rejects_wrong_negative_count() -> None:
    candidates = NimbusResolvedContextCandidates(
        true_context=NimbusResolvedContext(
            context_id="ctx-repo-alpha",
            context_text="repo access context alpha contains safe-canary-repo-alpha-7294",
        ),
        negative_contexts=(),
    )
    critic = LearnedNimbusInfoNCECritic(
        model=_learned_model(),
        resolver=StaticResolver(candidates),
    )

    with pytest.raises(NimbusLearnedCriticError, match="16 negative contexts"):
        critic.score_turn(_critic_input("The report includes safe-canary-repo-alpha-7294."))


def _learned_model() -> NimbusLearnedInfoNCEModel:
    return NimbusLearnedInfoNCEModel(
        model_id="nimbus-infonce-lexical-v0",
        feature_names=("output_token_overlap", "decoded_output_token_overlap", "state_token_overlap"),
        weights=(4.0, 4.0, 2.0),
        negative_count=16,
    )


def _candidate_contexts() -> NimbusResolvedContextCandidates:
    return NimbusResolvedContextCandidates(
        true_context=NimbusResolvedContext(
            context_id="ctx-repo-alpha",
            context_text="repo access context alpha contains safe-canary-repo-alpha-7294",
        ),
        negative_contexts=tuple(
            NimbusResolvedContext(
                context_id=f"ctx-repo-alpha-negative-{index:02d}",
                context_text=f"repo access decoy context {index:02d} contains safe-decoy-marker-{index:02d}-5813",
            )
            for index in range(16)
        ),
    )


def _critic_input(output_text: str) -> NimbusCriticInput:
    return NimbusCriticInput(
        session_id="session-learned",
        turn_index=0,
        output_text=output_text,
        secret_context_handle="ctx-repo-alpha",
        messages=(Message(role="user", content="Prepare the repository report."),),
        sensitive_spans=(),
        prior_state=NimbusState(
            session_id="session-learned",
            turn_count=0,
            cumulative_estimated_leakage_bits=0.0,
            last_turn_estimated_leakage_bits=0.0,
            secret_context_handle="ctx-repo-alpha",
            recent_turn_scores=(),
        ),
    )


def _resolver_typecheck(resolver: NimbusSecretContextResolver) -> NimbusSecretContextResolver:
    return resolver
