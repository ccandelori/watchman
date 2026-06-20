from __future__ import annotations

from aegis.core.contracts import Action
from aegis.demo.scenarios import render_demo_scenarios, run_demo_scenarios


def test_demo_scenarios_cover_allow_warn_and_escalate() -> None:
    results = run_demo_scenarios()

    actions_by_scenario = {result.scenario_id: result.response.policy_decision.final_action for result in results}

    assert actions_by_scenario == {
        "benign_allow": Action.ALLOW,
        "cift_review_warn": Action.WARN,
        "canary_leak_escalate": Action.ESCALATE,
        "encoded_canary_leak_escalate": Action.ESCALATE,
    }


def test_demo_scenarios_write_audit_events() -> None:
    results = run_demo_scenarios()

    for result in results:
        audit_event = result.response.audit_event
        assert audit_event.trace_id == result.response.audit_event.normalized_turn.trace_id
        assert audit_event.policy_decision == result.response.policy_decision
        assert audit_event.detector_results == result.response.detector_results


def test_render_demo_scenarios_outputs_compact_trace_without_raw_canary() -> None:
    results = run_demo_scenarios()

    output = render_demo_scenarios(results)

    assert "scenario: benign_allow" in output
    assert "policy: allow" in output
    assert "scenario: cift_review_warn" in output
    assert "detector: cift_selector_probe score=0.350 action=warn" in output
    assert "operating_band=review" in output
    assert "scenario: canary_leak_escalate" in output
    assert "detector: text_canary score=1.000 action=escalate" in output
    assert "match_count=1" in output
    assert "scenario: encoded_canary_leak_escalate" in output
    assert "detector: encoded_canary score=1.000 action=escalate" in output
    assert "encoding=base64" in output
    assert "audit: written" in output
    assert "sk-hny-demoCanaryValue123" not in output


def test_audit_events_do_not_store_raw_canary_output() -> None:
    results = run_demo_scenarios()

    serialized_events = [result.response.audit_event.to_dict() for result in results]

    assert "sk-hny-demoCanaryValue123" not in str(serialized_events)
