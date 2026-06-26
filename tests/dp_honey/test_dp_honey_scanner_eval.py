from __future__ import annotations

import json

import pytest

from detect.dp_honey.__main__ import main
from detect.dp_honey.scanner_eval import (
    SCANNER_EVAL_SCHEMA_VERSION,
    DPHoneyScannerEvalConfig,
    DPHoneyScannerEvalError,
    build_scanner_eval_report,
)


def test_scanner_eval_report_emits_separate_fn_fp_without_raw_values() -> None:
    report = build_scanner_eval_report(
        DPHoneyScannerEvalConfig(
            positive_per_format=3,
            seed=7,
            target_alpha=0.01,
            negative_count=25,
            calibration_count=30,
        )
    )
    rendered = json.dumps(report, sort_keys=True)
    counts = report["counts"]
    calibration = report["conformal_calibration"]

    assert report["schema_version"] == SCANNER_EVAL_SCHEMA_VERSION
    assert report["target_alpha"] == 0.01
    assert report["target_coverage"] == 0.99
    assert report["positive_example_count"] == report["scannable_format_count"] * 3
    assert report["negative_example_count"] == 25
    assert calibration["calibration_benign_count"] == 30
    assert counts["true_positive"] + counts["false_negative"] == report["positive_example_count"]
    assert counts["true_negative"] + counts["false_positive"] == report["negative_example_count"]
    assert "false_positive_rate" in report
    assert "false_negative_rate" in report
    assert calibration["implemented"] is True
    assert calibration["status"] == "split_conformal_confidence_threshold"
    assert calibration["target_alpha"] == 0.01
    assert calibration["target_coverage"] == 0.99
    assert calibration["threshold"] == 0.35
    assert calibration["recommended_min_confidence"] == "medium"
    assert report["audit_safety"]["raw_secret_values_in_report"] is False
    assert "ghp_" not in rendered
    assert "xoxb-" not in rendered


def test_scanner_eval_rejects_zero_positive_count() -> None:
    with pytest.raises(DPHoneyScannerEvalError, match="positive_per_format"):
        DPHoneyScannerEvalConfig(
            positive_per_format=0,
            seed=1,
            target_alpha=0.1,
            negative_count=10,
            calibration_count=10,
        )


def test_scanner_eval_rejects_invalid_target_alpha() -> None:
    with pytest.raises(DPHoneyScannerEvalError, match="target_alpha"):
        DPHoneyScannerEvalConfig(
            positive_per_format=1,
            seed=1,
            target_alpha=1.0,
            negative_count=10,
            calibration_count=10,
        )


def test_scanner_eval_rejects_invalid_negative_counts() -> None:
    with pytest.raises(DPHoneyScannerEvalError, match="negative_count"):
        DPHoneyScannerEvalConfig(
            positive_per_format=1,
            seed=1,
            target_alpha=0.01,
            negative_count=0,
            calibration_count=10,
        )
    with pytest.raises(DPHoneyScannerEvalError, match="calibration_count"):
        DPHoneyScannerEvalConfig(
            positive_per_format=1,
            seed=1,
            target_alpha=0.01,
            negative_count=10,
            calibration_count=0,
        )


def test_cli_eval_scanner_writes_report_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "dp-honey-scanner-eval.json"

    assert main(
        [
            "eval-scanner",
            "--positive-per-format",
            "2",
            "--target-alpha",
            "0.01",
            "--negative-count",
            "20",
            "--calibration-count",
            "20",
            "--seed",
            "3",
            "--output",
            str(output_path),
        ]
    ) == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert file_payload["schema_version"] == SCANNER_EVAL_SCHEMA_VERSION
    assert file_payload["conformal_calibration"]["implemented"] is True
    assert file_payload["counts"]["false_negative"] == 0
