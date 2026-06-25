"""Tests for typed DP-HONEY operation request modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from detect.dp_honey import get_format, load_model
from detect.dp_honey.errors import CountLimitError
from detect.dp_honey.operations import (
    FormatModelSource,
    GenerateRequest,
    ModelArtifactSource,
    ReportRequest,
    TrainRequest,
    generate_tokens,
    run_report_request,
    train_to_artifact,
)


def test_generate_request_from_format_executes_with_typed_source() -> None:
    request = GenerateRequest(
        source=FormatModelSource(
            format_slug="github-ghp",
            epsilon=1.0,
            clip=1.0,
            corpus_size=20,
            train_seed=3,
        ),
        count=3,
        sample_seed=7,
        max_repair_attempts=1000,
    )

    result = generate_tokens(request)

    assert result.format_slug == "github-ghp"
    assert len(result.tokens) == 3
    assert all(get_format("github-ghp").validate(token) for token in result.tokens)
    assert result.to_dict()["safety"]["provider_valid"] is False


def test_generate_request_rejects_oversized_count_before_model_load(tmp_path: Path) -> None:
    request = GenerateRequest(
        source=ModelArtifactSource(path=tmp_path / "missing.json"),
        count=10_001,
        sample_seed=1,
        max_repair_attempts=1000,
    )

    with pytest.raises(CountLimitError):
        generate_tokens(request)


def test_report_request_uses_same_typed_source_shape() -> None:
    request = ReportRequest(
        source=FormatModelSource(
            format_slug="github-ghp",
            epsilon=1.0,
            clip=1.0,
            corpus_size=20,
            train_seed=0,
        ),
        count=10,
        sample_seed=1,
        max_repair_attempts=1000,
    )

    report = run_report_request(request)

    assert report["format"] == "github-ghp"
    assert report["safety"]["provider_valid"] is False


def test_report_request_rejects_report_count_limit() -> None:
    request = ReportRequest(
        source=FormatModelSource(
            format_slug="github-ghp",
            epsilon=1.0,
            clip=1.0,
            corpus_size=20,
            train_seed=0,
        ),
        count=5001,
        sample_seed=1,
        max_repair_attempts=1000,
    )

    with pytest.raises(CountLimitError):
        run_report_request(request)


def test_train_request_writes_loadable_artifact(tmp_path: Path) -> None:
    request = TrainRequest(
        format_slug="github-ghp",
        output_path=tmp_path / "model.json",
        epsilon=1.0,
        clip=1.0,
        corpus_size=20,
        train_seed=2,
        force=False,
    )

    result = train_to_artifact(request)

    assert result.path == tmp_path / "model.json"
    assert result.format_slug == "github-ghp"
    assert load_model(result.path).format_slug == "github-ghp"
