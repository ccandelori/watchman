from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cache

from aegis.canaries.ledger import HoneytokenLedger
from detect.dp_honey.bigram import BigramHoneytokenModel, build_model

DEFAULT_CREDENTIAL_FORMATS: dict[str, str] = {
    "anthropic_key": "anthropic-api-key",
    "aws_access_key": "aws-access-key-id",
    "github_pat": "github-ghp",
    "google_api_key": "google-api-key",
    "oauth_token": "oauth-bearer",
    "openai_key": "openai-project-key",
    "sendgrid_key": "sendgrid-key",
    "slack_bot_token": "slack-bot-token",
    "slack_user_token": "slack-user-token",
    "slack_webhook": "slack-webhook-url",
    "stripe_key": "stripe-sk-live",
    "twilio_account_sid": "twilio-account-sid",
    "twilio_api_key_sid": "twilio-api-key-sid",
}


class DPHoneyCanaryGeneratorError(ValueError):
    """Raised when DP-HONEY canary generator configuration is invalid."""


@dataclass(frozen=True)
class DPHoneyCanaryGenerator:
    """Callable honeytoken generator that adapts DP-HONEY formats to Aegis ledgers."""

    credential_formats: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CREDENTIAL_FORMATS))
    default_format: str = "generic-sk"
    seed_salt: str = "aegis-dp-honey"
    epsilon: float = 1.0
    clip: float = 1.0
    corpus_size: int = 200
    train_seed: int = 0

    def __post_init__(self) -> None:
        if self.default_format == "":
            raise DPHoneyCanaryGeneratorError("default_format must not be empty.")
        if self.seed_salt == "":
            raise DPHoneyCanaryGeneratorError("seed_salt must not be empty.")
        if self.corpus_size < 1:
            raise DPHoneyCanaryGeneratorError("corpus_size must be positive.")
        for credential_type, format_slug in self.credential_formats.items():
            if credential_type == "" or format_slug == "":
                raise DPHoneyCanaryGeneratorError("credential format mappings must not contain empty strings.")

    def __call__(self, slot_name: str, credential_type: str) -> str:
        if slot_name == "":
            raise DPHoneyCanaryGeneratorError("slot_name must not be empty.")
        if credential_type == "":
            raise DPHoneyCanaryGeneratorError("credential_type must not be empty.")
        format_slug = self.credential_formats.get(credential_type, self.default_format)
        model = _model_for_format(
            format_slug=format_slug,
            epsilon=self.epsilon,
            clip=self.clip,
            corpus_size=self.corpus_size,
            train_seed=self.train_seed,
        )
        return model.sample(1, seed=_sample_seed(self.seed_salt, slot_name, credential_type, format_slug))[0]


def build_dp_honey_ledger(
    session_id: str,
    generator: DPHoneyCanaryGenerator | None = None,
) -> HoneytokenLedger:
    """Return a HoneytokenLedger configured to plant DP-HONEY-generated canaries."""
    return HoneytokenLedger(
        session_id=session_id,
        generator=generator if generator is not None else DPHoneyCanaryGenerator(),
        source="dp_honey",
    )


@cache
def _model_for_format(
    format_slug: str,
    epsilon: float,
    clip: float,
    corpus_size: int,
    train_seed: int,
) -> BigramHoneytokenModel:
    return build_model(
        format_slug,
        epsilon=epsilon,
        clip=clip,
        corpus_size=corpus_size,
        train_seed=train_seed,
    )


def _sample_seed(seed_salt: str, slot_name: str, credential_type: str, format_slug: str) -> int:
    digest = hashlib.sha256(f"{seed_salt}\0{slot_name}\0{credential_type}\0{format_slug}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
