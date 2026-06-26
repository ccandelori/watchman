from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Literal

from aegis.canaries.ledger import HoneytokenLedger
from aegis.core.contracts import JsonValue
from detect.dp_honey.bigram import DEFAULT_CORPUS_SIZE, BigramHoneytokenModel, build_model

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
class DPHoneyFormatSelection:
    """Resolved DP-HONEY format for an Aegis credential type."""

    credential_type: str
    format_slug: str
    mapping_source: Literal["explicit", "default"]


@dataclass(frozen=True)
class DPHoneyModelProvenance:
    """Model settings and registry identity used for a generated canary."""

    epsilon: float
    clip: float
    corpus_size: int
    train_seed: int
    registry_version: str
    spec_hash: str


@dataclass(frozen=True)
class DPHoneySeedProvenance:
    """Deterministic sample seed and derivation label for a generated canary."""

    sample_seed: int
    derivation: str


@dataclass(frozen=True)
class GeneratedDPHoneyCanary:
    """DP-HONEY canary value plus format, model, and seed provenance."""

    value: str
    format_selection: DPHoneyFormatSelection
    model_provenance: DPHoneyModelProvenance
    seed_provenance: DPHoneySeedProvenance

    def metadata(self) -> dict[str, JsonValue]:
        return _metadata_from_provenance(
            format_selection=self.format_selection,
            model_provenance=self.model_provenance,
            seed_provenance=self.seed_provenance,
        )


@dataclass(frozen=True)
class DPHoneyCanaryGenerator:
    """Callable honeytoken generator that adapts DP-HONEY formats to Aegis ledgers."""

    credential_formats: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CREDENTIAL_FORMATS))
    default_format: str = "generic-sk"
    seed_salt: str = "aegis-dp-honey"
    epsilon: float = 1.0
    clip: float = 1.0
    corpus_size: int = DEFAULT_CORPUS_SIZE
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
        return self.generate(slot_name=slot_name, credential_type=credential_type).value

    def generate(self, slot_name: str, credential_type: str) -> GeneratedDPHoneyCanary:
        """Return a DP-HONEY canary with format and seed provenance."""
        if slot_name == "":
            raise DPHoneyCanaryGeneratorError("slot_name must not be empty.")
        if credential_type == "":
            raise DPHoneyCanaryGeneratorError("credential_type must not be empty.")
        selection = resolve_dp_honey_format(
            credential_formats=self.credential_formats,
            default_format=self.default_format,
            credential_type=credential_type,
        )
        model = _model_for_format(
            format_slug=selection.format_slug,
            epsilon=self.epsilon,
            clip=self.clip,
            corpus_size=self.corpus_size,
            train_seed=self.train_seed,
        )
        seed = _sample_seed(self.seed_salt, slot_name, credential_type, selection.format_slug)
        return GeneratedDPHoneyCanary(
            value=model.sample(1, seed=seed)[0],
            format_selection=selection,
            model_provenance=_model_provenance(model),
            seed_provenance=DPHoneySeedProvenance(
                sample_seed=seed,
                derivation="sha256(seed_salt, slot_name, credential_type, format_slug)",
            ),
        )

    def metadata_for(self, slot_name: str, credential_type: str, value: str) -> dict[str, JsonValue]:
        """Return DP-HONEY metadata for a ledger-generated value without storing the value."""
        if value == "":
            raise DPHoneyCanaryGeneratorError("value must not be empty.")
        generated = self.generate(slot_name=slot_name, credential_type=credential_type)
        if generated.value != value:
            raise DPHoneyCanaryGeneratorError(
                "value does not match deterministic DP-HONEY canary for slot_name and credential_type."
            )
        return generated.metadata()

    def for_session(self, session_id: str) -> DPHoneyCanaryGenerator:
        """Return a generator whose deterministic seed space is scoped to a session."""
        if session_id == "":
            raise DPHoneyCanaryGeneratorError("session_id must not be empty.")
        return replace(self, seed_salt=f"{self.seed_salt}\0session:{session_id}")


def build_dp_honey_ledger(
    session_id: str,
    generator: DPHoneyCanaryGenerator | None = None,
) -> HoneytokenLedger:
    """Return a HoneytokenLedger configured to plant DP-HONEY-generated canaries."""
    session_generator = (generator if generator is not None else DPHoneyCanaryGenerator()).for_session(session_id)
    return HoneytokenLedger(
        session_id=session_id,
        generator=session_generator,
        source="dp_honey",
        metadata_provider=session_generator.metadata_for,
    )


def resolve_dp_honey_format(
    credential_formats: dict[str, str],
    default_format: str,
    credential_type: str,
) -> DPHoneyFormatSelection:
    """Resolve an Aegis credential type to a DP-HONEY format slug."""
    if credential_type in credential_formats:
        return DPHoneyFormatSelection(
            credential_type=credential_type,
            format_slug=credential_formats[credential_type],
            mapping_source="explicit",
        )
    return DPHoneyFormatSelection(
        credential_type=credential_type,
        format_slug=default_format,
        mapping_source="default",
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


def _model_provenance(model: BigramHoneytokenModel) -> DPHoneyModelProvenance:
    return DPHoneyModelProvenance(
        epsilon=model.epsilon,
        clip=model.clip,
        corpus_size=model.corpus_size,
        train_seed=model.train_seed,
        registry_version=model.registry_version,
        spec_hash=model.spec_hash,
    )


def _metadata_from_provenance(
    format_selection: DPHoneyFormatSelection,
    model_provenance: DPHoneyModelProvenance,
    seed_provenance: DPHoneySeedProvenance,
) -> dict[str, JsonValue]:
    return {
        "dp_honey": {
            "credential_type": format_selection.credential_type,
            "format_slug": format_selection.format_slug,
            "mapping_source": format_selection.mapping_source,
            "seed_derivation": seed_provenance.derivation,
            "epsilon": model_provenance.epsilon,
            "clip": model_provenance.clip,
            "corpus_size": model_provenance.corpus_size,
            "train_seed": model_provenance.train_seed,
            "registry_version": model_provenance.registry_version,
            "spec_hash": model_provenance.spec_hash,
        }
    }
