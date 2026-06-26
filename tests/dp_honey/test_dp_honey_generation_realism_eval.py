from __future__ import annotations

import json

import pytest

from detect.dp_honey.__main__ import main
from detect.dp_honey.generation_realism_eval import (
    GENERATION_REALISM_EVAL_MAX_PER_FORMAT,
    GENERATION_REALISM_EVAL_SCHEMA_VERSION,
    DPHoneyGenerationRealismEvalConfig,
    DPHoneyGenerationRealismEvalError,
    build_generation_realism_eval_report,
)


def test_generation_realism_eval_reports_aggregate_metrics_without_raw_values() -> None:
    report = build_generation_realism_eval_report(DPHoneyGenerationRealismEvalConfig(count_per_format=3, seed=11))
    rendered = json.dumps(report, sort_keys=True)

    assert report["schema_version"] == GENERATION_REALISM_EVAL_SCHEMA_VERSION
    assert report["status"] == "bounded_generated_vs_reference_sanity_metrics"
    assert report["count_per_format"] == 3
    assert report["format_count"] >= report["scannable_format_count"]
    assert report["all_generated_tokens_valid"] is True
    assert report["all_reference_tokens_valid"] is True
    assert report["all_metrics_finite"] is True
    assert report["bounded_sanity_gate_passed"] is True
    assert report["paper_faithful_statistical_distinguisher"] is False
    assert report["audit_safety"]["raw_secret_values_in_report"] is False
    assert len(report["format_metrics"]) == report["format_count"]
    assert report["format_metrics"][0]["generated_count"] == 3
    assert report["format_metrics"][0]["reference_count"] == 3
    assert "ghp_" not in rendered
    assert "xoxb-" not in rendered


def test_generation_realism_eval_rejects_zero_count() -> None:
    with pytest.raises(DPHoneyGenerationRealismEvalError, match="count_per_format"):
        DPHoneyGenerationRealismEvalConfig(count_per_format=0, seed=1)


def test_generation_realism_eval_rejects_oversized_count() -> None:
    with pytest.raises(DPHoneyGenerationRealismEvalError, match=str(GENERATION_REALISM_EVAL_MAX_PER_FORMAT)):
        DPHoneyGenerationRealismEvalConfig(count_per_format=GENERATION_REALISM_EVAL_MAX_PER_FORMAT + 1, seed=1)


def test_cli_eval_realism_writes_report_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "dp-honey-generation-realism-eval.json"

    assert (
        main(
            [
                "eval-realism",
                "--count-per-format",
                "2",
                "--seed",
                "3",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert file_payload["schema_version"] == GENERATION_REALISM_EVAL_SCHEMA_VERSION
    assert file_payload["bounded_sanity_gate_passed"] is True
