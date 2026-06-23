from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis.replay.nimbus_redteam import (
    NimbusRedteamParseError,
    load_nimbus_redteam_metrics_jsonl,
    render_nimbus_redteam_markdown,
    summarize_nimbus_redteam_metrics,
)
from aegis.replay.nimbus_report import NimbusReportConfig, NimbusReportFormat, main, render_report


def test_loads_multi_turn_nimbus_metrics(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, (_multi_turn_drip_record(),))

    metrics = load_nimbus_redteam_metrics_jsonl(path)

    assert tuple(metric.turn_index for metric in metrics) == (0, 1, 2)
    assert tuple(metric.final_action.value for metric in metrics) == ("warn", "sanitize", "block")
    assert tuple(metric.nimbus_action.value for metric in metrics) == ("warn", "sanitize", "block")
    assert tuple(metric.cumulative_leakage_bits for metric in metrics) == pytest.approx((0.385185, 0.77037, 1.155555))
    assert metrics[0].triggered_detector_names == ("nimbus",)
    assert metrics[0].public_canary_triggered is False
    assert all("hny_raw_value_should_not_render" not in str(metric.to_dict()) for metric in metrics)


def test_summarizes_nimbus_action_progression_without_raw_output(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, (_multi_turn_drip_record(), _benign_record()))

    summaries = summarize_nimbus_redteam_metrics(load_nimbus_redteam_metrics_jsonl(path))
    markdown = render_nimbus_redteam_markdown(summaries)

    assert "multi_turn_drip" in markdown
    assert "warn -> sanitize -> block" in markdown
    assert "benign" in markdown
    assert "allow" in markdown
    assert "public canary detectors" in markdown
    assert "NIMBUS critic evidence" in markdown
    assert "hny_raw_value_should_not_render" not in markdown


def test_public_canary_trigger_is_reported_from_component(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    detector_results = _required_list(first_turn["detector_results"])
    encoded = dict(_required_dict(detector_results[0]))
    encoded["triggered"] = True
    encoded["component"] = "text_canary"
    encoded["recommended_action"] = "escalate"
    first_turn["detector_results"] = (encoded, *detector_results[1:])
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "public-canary.jsonl"
    _write_jsonl(path, (record,))

    metrics = load_nimbus_redteam_metrics_jsonl(path)
    summaries = summarize_nimbus_redteam_metrics(metrics)
    markdown = render_nimbus_redteam_markdown(summaries)

    assert metrics[0].public_canary_triggered is True
    assert summaries[0].public_canary_triggered is True
    assert (
        "| multi_turn_drip | 3 | warn -> sanitize -> block | warn -> sanitize -> block | 1.15556 | 1 | yes |"
        in markdown
    )


def test_missing_nimbus_detector_result_fails_with_turn_context(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    first_turn["detector_results"] = [
        {"name": "encoded_canary", "triggered": False, "evidence": {}},
    ]
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "missing-nimbus.jsonl"
    _write_jsonl(path, (record,))

    with pytest.raises(NimbusRedteamParseError, match=r"multi_turn_drip.*turn 0.*nimbus"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_bad_budget_fraction_fails_with_field_context(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    detector_results = _required_list(first_turn["detector_results"])
    nimbus = dict(_required_dict(detector_results[1]))
    evidence = dict(_required_dict(nimbus["evidence"]))
    evidence["budget_fraction"] = "not-numeric"
    nimbus["evidence"] = evidence
    first_turn["detector_results"] = (detector_results[0], nimbus)
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "bad-budget.jsonl"
    _write_jsonl(path, (record,))

    with pytest.raises(NimbusRedteamParseError, match="budget_fraction"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_rejects_non_finite_numeric_json(tmp_path: Path) -> None:
    path = tmp_path / "nan.jsonl"
    path.write_text('{"scenario_name":"nan_case","turn_results":[NaN]}\n', encoding="utf-8")

    with pytest.raises(NimbusRedteamParseError, match="non-standard JSON constant"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text('{"scenario_name":"a","scenario_name":"b","turn_results":[]}\n', encoding="utf-8")

    with pytest.raises(NimbusRedteamParseError, match="duplicate JSON object key"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_rejects_non_object_json_line(tmp_path: Path) -> None:
    path = tmp_path / "array.jsonl"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(NimbusRedteamParseError, match=r"Line 1.*expected a JSON object"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_rejects_missing_top_level_turn_results(tmp_path: Path) -> None:
    path = tmp_path / "missing-turns.jsonl"
    _write_jsonl(path, ({"scenario_name": "missing_turns"},))

    with pytest.raises(NimbusRedteamParseError, match="turn_results"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_rejects_empty_turn_results(tmp_path: Path) -> None:
    path = tmp_path / "empty-turns.jsonl"
    _write_jsonl(path, ({"scenario_name": "empty_turns", "turn_results": []},))

    with pytest.raises(NimbusRedteamParseError, match=r"turn_results.*must not be empty"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_rejects_unsafe_scenario_identifier(tmp_path: Path) -> None:
    record = _benign_record()
    record["scenario_name"] = "ghp_rawSecretShouldNotRender"
    path = tmp_path / "unsafe-name.jsonl"
    _write_jsonl(path, (record,))

    with pytest.raises(NimbusRedteamParseError, match="credential-shaped"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_accepts_aegis_detector_name_shape(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    detector_results = _required_list(first_turn["detector_results"])
    encoded = {"detector_name": "encoded_canary", "recommended_action": "allow", "evidence": {}}
    nimbus = dict(_required_dict(detector_results[1]))
    nimbus["detector_name"] = nimbus.pop("name")
    nimbus["recommended_action"] = nimbus.pop("action")
    first_turn["detector_results"] = (encoded, nimbus)
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "aegis-shape.jsonl"
    _write_jsonl(path, (record,))

    metrics = load_nimbus_redteam_metrics_jsonl(path)

    assert metrics[0].nimbus_action.value == "warn"


@pytest.mark.parametrize(
    ("budget_fraction", "expected_action"),
    (
        (0.29, "allow"),
        (0.3, "warn"),
        (0.6, "sanitize"),
        (0.9, "block"),
    ),
)
def test_infers_nimbus_action_from_budget_thresholds(
    tmp_path: Path,
    budget_fraction: float,
    expected_action: str,
) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    second_turn = dict(_required_dict(turn_results[1]))
    second_turn["policy_decision"] = {
        "final_action": expected_action,
        "reason": f"policy_{expected_action}",
        "triggered_detectors": [] if expected_action == "allow" else ["nimbus"],
    }
    detector_results = _required_list(second_turn["detector_results"])
    nimbus = dict(_required_dict(detector_results[1]))
    nimbus.pop("action")
    evidence = dict(_required_dict(nimbus["evidence"]))
    evidence["budget_fraction"] = budget_fraction
    evidence["cumulative_estimated_leakage_bits"] = budget_fraction
    nimbus["evidence"] = evidence
    second_turn["detector_results"] = (detector_results[0], nimbus)
    record["turn_results"] = (turn_results[0], second_turn, turn_results[2])
    path = tmp_path / "threshold-inference.jsonl"
    _write_jsonl(path, (record,))

    metrics = load_nimbus_redteam_metrics_jsonl(path)

    assert metrics[1].nimbus_action.value == expected_action


def test_missing_nimbus_action_without_thresholds_fails(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    detector_results = _required_list(first_turn["detector_results"])
    nimbus = dict(_required_dict(detector_results[1]))
    nimbus.pop("action")
    evidence = dict(_required_dict(nimbus["evidence"]))
    evidence.pop("warn_threshold")
    nimbus["evidence"] = evidence
    first_turn["detector_results"] = (detector_results[0], nimbus)
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "missing-action.jsonl"
    _write_jsonl(path, (record,))

    with pytest.raises(NimbusRedteamParseError, match="threshold evidence"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_missing_leakage_bits_for_non_allow_nimbus_fails(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    detector_results = _required_list(first_turn["detector_results"])
    nimbus = dict(_required_dict(detector_results[1]))
    evidence = dict(_required_dict(nimbus["evidence"]))
    evidence.pop("turn_estimated_leakage_bits")
    nimbus["evidence"] = evidence
    first_turn["detector_results"] = (detector_results[0], nimbus)
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "missing-leakage.jsonl"
    _write_jsonl(path, (record,))

    with pytest.raises(NimbusRedteamParseError, match="turn_estimated_leakage_bits"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_bool_budget_fraction_is_rejected(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    detector_results = _required_list(first_turn["detector_results"])
    nimbus = dict(_required_dict(detector_results[1]))
    evidence = dict(_required_dict(nimbus["evidence"]))
    evidence["budget_fraction"] = True
    nimbus["evidence"] = evidence
    first_turn["detector_results"] = (detector_results[0], nimbus)
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "bool-budget.jsonl"
    _write_jsonl(path, (record,))

    with pytest.raises(NimbusRedteamParseError, match="budget_fraction"):
        load_nimbus_redteam_metrics_jsonl(path)


def test_markdown_escapes_scenario_name_cells(tmp_path: Path) -> None:
    record = _benign_record()
    record["scenario_name"] = "safe.scenario"
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, (record,))

    summaries = summarize_nimbus_redteam_metrics(load_nimbus_redteam_metrics_jsonl(path))
    markdown = render_nimbus_redteam_markdown(summaries)

    assert "safe.scenario" in markdown


def test_report_renderer_outputs_json(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, (_multi_turn_drip_record(),))

    output = render_report(NimbusReportConfig(input_path=path, output_format=NimbusReportFormat.JSON))
    decoded = json.loads(output)

    assert decoded[0]["scenario_name"] == "multi_turn_drip"
    assert decoded[0]["nimbus_action_progression"] == ["warn", "sanitize", "block"]
    assert "hny_raw_value_should_not_render" not in output


def test_report_renderer_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, (_benign_record(),))

    with pytest.raises(NimbusRedteamParseError, match="Unsupported output format"):
        render_report(NimbusReportConfig(input_path=path, output_format="xml"))  # type: ignore[arg-type]


def test_main_writes_markdown_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, (_benign_record(),))
    monkeypatch.setattr("sys.argv", ("aegis-nimbus-report", "--input", str(path)))

    main()

    captured = capsys.readouterr()
    assert "# NIMBUS Redteam Report" in captured.out
    assert captured.err == ""


def test_main_exits_nonzero_for_parse_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ("aegis-nimbus-report", "--input", str(path)))

    with pytest.raises(SystemExit) as exc_info:
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "expected a JSON object" in captured.err


def test_nested_aegis_metadata_detector_results_are_supported(tmp_path: Path) -> None:
    record = _multi_turn_drip_record()
    turn_results = _required_list(record["turn_results"])
    first_turn = dict(_required_dict(turn_results[0]))
    first_turn["aegis_metadata"] = {
        "detector_results": first_turn.pop("detector_results"),
        "policy_decision": first_turn.pop("policy_decision"),
    }
    record["turn_results"] = (first_turn, *turn_results[1:])
    path = tmp_path / "nested-aegis.jsonl"
    _write_jsonl(path, (record,))

    metrics = load_nimbus_redteam_metrics_jsonl(path)

    assert metrics[0].nimbus_action.value == "warn"
    assert metrics[0].triggered_detector_names == ("nimbus",)


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def _multi_turn_drip_record() -> dict[str, object]:
    return {
        "run_id": "run-multi",
        "scenario_name": "multi_turn_drip",
        "target_url": "http://127.0.0.1:8765",
        "started_at": "2026-06-23T00:00:00Z",
        "finished_at": "2026-06-23T00:00:01Z",
        "passed": True,
        "turn_results": [
            _turn_result(turn_index=0, final_action="warn", cumulative_bits=0.385185, budget_fraction=0.385185),
            _turn_result(turn_index=1, final_action="sanitize", cumulative_bits=0.77037, budget_fraction=0.77037),
            _turn_result(turn_index=2, final_action="block", cumulative_bits=1.155555, budget_fraction=1.0),
        ],
        "failures": [],
        "raw_response": {},
    }


def _benign_record() -> dict[str, object]:
    turn = _turn_result(turn_index=0, final_action="allow", cumulative_bits=0.0, budget_fraction=0.0)
    detector_results = _required_list(turn["detector_results"])
    nimbus = dict(_required_dict(detector_results[1]))
    nimbus["score"] = 0.0
    nimbus["capability_status"] = "unavailable"
    nimbus["evidence"] = {"capability_reason": "no_secret_context_handle", "turn_index": 0}
    turn["detector_results"] = (detector_results[0], nimbus)
    return {
        "run_id": "run-benign",
        "scenario_name": "benign",
        "target_url": "http://127.0.0.1:8765",
        "started_at": "2026-06-23T00:00:00Z",
        "finished_at": "2026-06-23T00:00:01Z",
        "passed": True,
        "turn_results": [turn],
        "failures": [],
        "raw_response": {},
    }


def _turn_result(
    turn_index: int,
    final_action: str,
    cumulative_bits: float,
    budget_fraction: float,
) -> dict[str, object]:
    return {
        "turn_index": turn_index,
        "request": {"role": "user", "content": "redacted"},
        "response_status": 200,
        "assistant_content": "hny_raw_value_should_not_render",
        "aegis_metadata": {},
        "detector_results": [
            {"name": "encoded_canary", "triggered": False, "evidence": {}},
            {
                "name": "nimbus",
                "triggered": final_action != "allow",
                "action": final_action,
                "evidence": {
                    "reason": f"nimbus_leakage_budget_{final_action}",
                    "turn_estimated_leakage_bits": 0.385185 if final_action != "allow" else 0.0,
                    "cumulative_estimated_leakage_bits": cumulative_bits,
                    "budget_fraction": budget_fraction,
                    "warn_threshold": 0.3,
                    "sanitize_threshold": 0.6,
                    "block_threshold": 0.9,
                    "critic_evidence": {
                        "critic_kind": "canary",
                        "partial_match_count": 1 if final_action != "allow" else 0,
                        "estimated_leakage_bits": 0.385185 if final_action != "allow" else 0.0,
                    },
                },
            },
        ],
        "policy_decision": {
            "final_action": final_action,
            "reason": f"policy_{final_action}",
            "triggered_detectors": [] if final_action == "allow" else ["nimbus"],
        },
        "latency_ms": 5,
    }


def _required_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _required_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value
