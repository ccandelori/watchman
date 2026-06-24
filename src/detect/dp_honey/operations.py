"""Typed operation requests for DP-HONEY generation, reports, and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bigram import BigramHoneytokenModel, build_model
from .model_io import load_model, save_model
from .realism import REPORT_MAX, compute_report, enforce_count_limit

GENERATE_MAX = 10000

_SAFETY = {
    "synthetic_only": True,
    "provider_valid": False,
    "note": "Synthetic, shape-only honeytokens. Not real, valid, or usable credentials.",
}


@dataclass(frozen=True)
class FormatModelSource:
    """Train a model from a format and explicit privacy/training settings."""

    format_slug: str
    epsilon: float
    clip: float
    corpus_size: int
    train_seed: int

    def build(self) -> BigramHoneytokenModel:
        return build_model(
            self.format_slug,
            epsilon=self.epsilon,
            clip=self.clip,
            corpus_size=self.corpus_size,
            train_seed=self.train_seed,
        )


@dataclass(frozen=True)
class ModelArtifactSource:
    """Load a model from an already validated JSON artifact path."""

    path: Path

    def build(self) -> BigramHoneytokenModel:
        return load_model(self.path)


ModelSource = FormatModelSource | ModelArtifactSource


@dataclass(frozen=True)
class GenerateRequest:
    """Typed request for synthetic honeytoken generation."""

    source: ModelSource
    count: int
    sample_seed: int
    max_repair_attempts: int


@dataclass(frozen=True)
class ReportRequest:
    """Typed request for a generated-token realism report."""

    source: ModelSource
    count: int
    sample_seed: int
    max_repair_attempts: int


@dataclass(frozen=True)
class TrainRequest:
    """Typed request for building and saving a model artifact."""

    format_slug: str
    output_path: Path
    epsilon: float
    clip: float
    corpus_size: int
    train_seed: int
    force: bool


@dataclass(frozen=True)
class GenerateResult:
    """Generated token batch plus the format it came from."""

    tokens: tuple[str, ...]
    format_slug: str

    def to_dict(self) -> dict[str, object]:
        return {"tokens": list(self.tokens), "format": self.format_slug, "safety": dict(_SAFETY)}


@dataclass(frozen=True)
class TrainResult:
    """Saved artifact metadata for train operations."""

    path: Path
    format_slug: str
    epsilon: float
    clip: float
    corpus_size: int
    train_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "saved": self.path.name,
            "format": self.format_slug,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "corpus_size": self.corpus_size,
            "train_seed": self.train_seed,
        }


def generate_tokens(request: GenerateRequest) -> GenerateResult:
    """Generate a bounded batch from a typed request."""
    enforce_count_limit(request.count, maximum=GENERATE_MAX, label="count")
    model = request.source.build()
    tokens = tuple(
        model.sample(
            request.count,
            seed=request.sample_seed,
            max_repair_attempts=request.max_repair_attempts,
        )
    )
    return GenerateResult(tokens=tokens, format_slug=model.format_slug)


def run_report_request(request: ReportRequest) -> dict[str, object]:
    """Generate a bounded batch and compute a realism report."""
    enforce_count_limit(request.count, maximum=REPORT_MAX, label="count")
    model = request.source.build()
    tokens = model.sample(
        request.count,
        seed=request.sample_seed,
        max_repair_attempts=request.max_repair_attempts,
    )
    return compute_report(tokens, model)


def train_to_artifact(request: TrainRequest) -> TrainResult:
    """Train and save a model artifact from a typed request."""
    model = build_model(
        request.format_slug,
        epsilon=request.epsilon,
        clip=request.clip,
        corpus_size=request.corpus_size,
        train_seed=request.train_seed,
    )
    path = save_model(model, request.output_path, force=request.force)
    return TrainResult(
        path=path,
        format_slug=model.format_slug,
        epsilon=model.epsilon,
        clip=model.clip,
        corpus_size=model.corpus_size,
        train_seed=model.train_seed,
    )
