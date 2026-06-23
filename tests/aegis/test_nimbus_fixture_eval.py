from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aegis.core.contracts import Action
from aegis.replay.nimbus_fixture import main, write_nimbus_fixture_results_jsonl
from aegis.replay.nimbus_redteam import (
    load_nimbus_redteam_metrics_jsonl,
    summarize_nimbus_redteam_metrics,
)
from aegis.replay.nimbus_report import NimbusReportConfig, NimbusReportFormat, render_report


def test_nimbus_fixture_results_are_parseable_without_raw_prompts_or_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "nimbus-fixtures.jsonl"

    write_nimbus_fixture_results_jsonl(output_path)

    raw_output = output_path.read_text(encoding="utf-8")
    records = tuple(json.loads(line) for line in raw_output.splitlines())
    metrics = load_nimbus_redteam_metrics_jsonl(output_path)
    summaries = summarize_nimbus_redteam_metrics(metrics)
    summaries_by_name = {summary.scenario_name: summary for summary in summaries}

    assert {record["scenario_name"] for record in records} == {"benign", "partial_drip", "encoded_leak"}
    assert all("request" not in turn for record in records for turn in record["turn_results"])
    assert all("assistant_content" not in turn for record in records for turn in record["turn_results"])
    assert "{{CREDENTIAL:" not in raw_output
    assert "ghp_" not in raw_output
    assert "sk_live_" not in raw_output
    assert summaries_by_name["partial_drip"].nimbus_action_progression == (
        Action.WARN,
        Action.SANITIZE,
        Action.BLOCK,
    )
    assert summaries_by_name["encoded_leak"].public_canary_triggered is True
    assert summaries_by_name["benign"].final_action_progression == (Action.ALLOW,)


def test_nimbus_fixture_cli_writes_reportable_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "fixture-results.jsonl"
    monkeypatch.setattr(sys, "argv", ("aegis-nimbus-fixtures", "--output", str(output_path)))

    main()

    report = render_report(NimbusReportConfig(input_path=output_path, output_format=NimbusReportFormat.MARKDOWN))

    assert output_path.exists()
    assert "partial_drip" in report
    assert "warn -> sanitize -> block" in report
