from __future__ import annotations

import json

import pytest

from detect.dp_honey.__main__ import main
from detect.dp_honey.statistical_distinguisher_eval import (
    REQUIRED_TESTS,
    STATISTICAL_DISTINGUISHER_EVAL_MAX_PER_FORMAT,
    STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION,
    DPHoneyStatisticalDistinguisherEvalConfig,
    DPHoneyStatisticalDistinguisherEvalError,
    build_statistical_distinguisher_eval_report,
)


def test_statistical_distinguisher_eval_reports_required_suite_without_raw_values() -> None:
    report = build_statistical_distinguisher_eval_report(
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=3,
            test_count_per_format=3,
            seed=11,
            alpha=0.1,
        )
    )
    rendered = json.dumps(report, sort_keys=True)
    suite = report["statistical_distinguisher_suite"]

    assert report["schema_version"] == STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION
    assert report["status"] == "statistical_distinguisher_suite_evaluated"
    assert report["raw_values_serialized"] is False
    assert report["required_tests"] == list(REQUIRED_TESTS)
    assert set(suite) == set(REQUIRED_TESTS)
    assert report["audit_safety"]["raw_secret_values_in_report"] is False
    assert "ghp_" not in rendered
    assert "xoxb-" not in rendered
    assert "sk_live_" not in rendered


def test_statistical_distinguisher_eval_declared_flag_matches_required_test_statuses() -> None:
    report = build_statistical_distinguisher_eval_report(
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=2,
            test_count_per_format=2,
            seed=7,
            alpha=0.1,
        )
    )
    suite = report["statistical_distinguisher_suite"]
    passed = all(suite[test_name]["status"] == "passed" for test_name in REQUIRED_TESTS)

    assert report["all_required_tests_passed"] is passed
    assert report["synthetic_registry_statistical_distinguisher_passed"] is passed
    assert report["paper_faithful_statistical_distinguisher"] is False
    assert report["reference_source"] == "same_format_uniform_synthetic_holdout"


def test_statistical_distinguisher_eval_rejects_invalid_config() -> None:
    with pytest.raises(DPHoneyStatisticalDistinguisherEvalError, match="train_count_per_format"):
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=0,
            test_count_per_format=1,
            seed=1,
            alpha=0.1,
        )
    with pytest.raises(DPHoneyStatisticalDistinguisherEvalError, match="test_count_per_format"):
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=1,
            test_count_per_format=STATISTICAL_DISTINGUISHER_EVAL_MAX_PER_FORMAT + 1,
            seed=1,
            alpha=0.1,
        )
    with pytest.raises(DPHoneyStatisticalDistinguisherEvalError, match="alpha"):
        DPHoneyStatisticalDistinguisherEvalConfig(
            train_count_per_format=1,
            test_count_per_format=1,
            seed=1,
            alpha=1.0,
        )


def test_cli_eval_statistical_distinguishers_writes_report_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "dp-honey-statistical-distinguisher-eval.json"

    assert (
        main(
            [
                "eval-statistical-distinguishers",
                "--train-count-per-format",
                "2",
                "--test-count-per-format",
                "2",
                "--alpha",
                "0.1",
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
    assert file_payload["schema_version"] == STATISTICAL_DISTINGUISHER_EVAL_SCHEMA_VERSION
    assert set(file_payload["statistical_distinguisher_suite"]) == set(REQUIRED_TESTS)
