from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aegis.core.contracts import Action
from aegis.replay.nimbus_eval import (
    NIMBUS_EVAL_LABELS_SCHEMA_VERSION,
    NimbusEvalError,
    evaluate_nimbus_redteam_jsonl,
    load_nimbus_eval_labels_json,
    main,
    render_nimbus_eval_json,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPOSITORY_ROOT / "tests" / "aegis" / "fixtures" / "nimbus_redteam"
SANITIZED_EXTERNAL_FIXTURE = FIXTURE_DIR / "external_runner_sanitized_v1.jsonl"
SANITIZED_EXTERNAL_LABELS = FIXTURE_DIR / "eval_labels_v1.json"


def test_evaluates_sanitized_fixture_with_separate_false_positive_and_false_negative_rates() -> None:
    report = evaluate_nimbus_redteam_jsonl(
        input_path=SANITIZED_EXTERNAL_FIXTURE,
        labels_path=SANITIZED_EXTERNAL_LABELS,
        positive_action_threshold=Action.WARN,
    )

    rendered = report.to_dict()

    assert rendered["critic_status"] == "deterministic_beta"
    assert rendered["critic_kind"] == "canary"
    assert rendered["paper_faithful_learned_critic"] is False
    assert rendered["scenario_count"] == 2
    assert rendered["positive_label_count"] == 1
    assert rendered["negative_label_count"] == 1
    assert rendered["true_positive"] == 1
    assert rendered["true_negative"] == 1
    assert rendered["false_positive"] == 0
    assert rendered["false_negative"] == 0
    assert rendered["false_positive_rate"] == 0.0
    assert rendered["false_negative_rate"] == 0.0
    assert {row["scenario_name"]: row["outcome"] for row in rendered["rows"]} == {
        "external_multi_turn_drip": "true_positive",
        "external_benign": "true_negative",
    }


def test_evaluates_false_negative_when_labeled_leakage_has_no_nimbus_action(tmp_path: Path) -> None:
    labels_path = _write_labels(
        tmp_path,
        (
            ("external_multi_turn_drip", True),
            ("external_benign", True),
        ),
    )

    report = evaluate_nimbus_redteam_jsonl(
        input_path=SANITIZED_EXTERNAL_FIXTURE,
        labels_path=labels_path,
        positive_action_threshold=Action.WARN,
    )

    rendered = report.to_dict()

    assert rendered["true_positive"] == 1
    assert rendered["false_negative"] == 1
    assert rendered["false_negative_rate"] == 0.5
    assert {row["scenario_name"]: row["outcome"] for row in rendered["rows"]}["external_benign"] == "false_negative"


def test_evaluates_false_positive_when_nimbus_action_is_labeled_benign(tmp_path: Path) -> None:
    labels_path = _write_labels(
        tmp_path,
        (
            ("external_multi_turn_drip", False),
            ("external_benign", False),
        ),
    )

    report = evaluate_nimbus_redteam_jsonl(
        input_path=SANITIZED_EXTERNAL_FIXTURE,
        labels_path=labels_path,
        positive_action_threshold=Action.WARN,
    )

    rendered = report.to_dict()

    assert rendered["false_positive"] == 1
    assert rendered["true_negative"] == 1
    assert rendered["false_positive_rate"] == 0.5
    assert {row["scenario_name"]: row["outcome"] for row in rendered["rows"]}[
        "external_multi_turn_drip"
    ] == "false_positive"


def test_block_threshold_counts_only_block_or_higher_action() -> None:
    report = evaluate_nimbus_redteam_jsonl(
        input_path=SANITIZED_EXTERNAL_FIXTURE,
        labels_path=SANITIZED_EXTERNAL_LABELS,
        positive_action_threshold=Action.BLOCK,
    )

    rendered = report.to_dict()

    assert rendered["positive_action_threshold"] == "block"
    assert rendered["true_positive"] == 1
    assert rendered["false_negative"] == 0


def test_rejects_missing_and_unlabeled_scenarios(tmp_path: Path) -> None:
    missing_labels_path = _write_labels(tmp_path, (("external_missing", True),))
    unlabeled_path = _write_labels(tmp_path, (("external_multi_turn_drip", True),))

    with pytest.raises(NimbusEvalError, match="Missing NIMBUS summary"):
        evaluate_nimbus_redteam_jsonl(
            input_path=SANITIZED_EXTERNAL_FIXTURE,
            labels_path=missing_labels_path,
            positive_action_threshold=Action.WARN,
        )

    with pytest.raises(NimbusEvalError, match="unlabeled scenario"):
        evaluate_nimbus_redteam_jsonl(
            input_path=SANITIZED_EXTERNAL_FIXTURE,
            labels_path=unlabeled_path,
            positive_action_threshold=Action.WARN,
        )


def test_rejects_duplicate_or_credential_shaped_labels(tmp_path: Path) -> None:
    duplicate_path = tmp_path / "duplicate-labels.json"
    duplicate_path.write_text(
        json.dumps(
            {
                "schema_version": NIMBUS_EVAL_LABELS_SCHEMA_VERSION,
                "labels": [
                    {"scenario_name": "external_benign", "leakage_expected": False},
                    {"scenario_name": "external_benign", "leakage_expected": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    unsafe_path = _write_labels(tmp_path, (("sk_unsafe_label", False),))

    with pytest.raises(NimbusEvalError, match="duplicate scenario label"):
        load_nimbus_eval_labels_json(duplicate_path)

    with pytest.raises(NimbusEvalError, match="credential-shaped"):
        load_nimbus_eval_labels_json(unsafe_path)


def test_cli_writes_eval_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "nimbus-eval.json"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "aegis-nimbus-eval",
            "--input",
            str(SANITIZED_EXTERNAL_FIXTURE),
            "--labels",
            str(SANITIZED_EXTERNAL_LABELS),
            "--output",
            str(output_path),
        ),
    )

    main()

    stdout_report = json.loads(capsys.readouterr().out)
    saved_report = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_report == saved_report
    assert saved_report["schema_version"] == "aegis.nimbus_eval/v1"
    assert saved_report["false_positive_rate"] == 0.0
    assert saved_report["false_negative_rate"] == 0.0


def test_rendered_eval_json_rejects_non_finite_json_values() -> None:
    report = evaluate_nimbus_redteam_jsonl(
        input_path=SANITIZED_EXTERNAL_FIXTURE,
        labels_path=SANITIZED_EXTERNAL_LABELS,
        positive_action_threshold=Action.WARN,
    )

    rendered = render_nimbus_eval_json(report)

    assert json.loads(rendered)["schema_version"] == "aegis.nimbus_eval/v1"


def _write_labels(tmp_path: Path, labels: tuple[tuple[str, bool], ...]) -> Path:
    path = tmp_path / f"labels-{len(tuple(tmp_path.iterdir()))}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": NIMBUS_EVAL_LABELS_SCHEMA_VERSION,
                "labels": [
                    {"scenario_name": scenario_name, "leakage_expected": leakage_expected}
                    for scenario_name, leakage_expected in labels
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
