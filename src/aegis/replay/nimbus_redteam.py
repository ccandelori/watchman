from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

from aegis.core.contracts import Action, DetectorComponent, JsonValue

PUBLIC_CANARY_DETECTOR_NAMES = frozenset(("text_canary", "encoded_canary"))
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
CREDENTIAL_LIKE_PREFIXES = ("ghp_", "github_pat_", "sk_", "hny_", "aws_", "AKIA")


class NimbusRedteamParseError(ValueError):
    """Raised when redteam results cannot be converted into NIMBUS metrics."""


@dataclass(frozen=True)
class NimbusTurnMetric:
    scenario_name: str
    turn_index: int
    final_action: Action
    nimbus_action: Action
    turn_leakage_bits: float
    cumulative_leakage_bits: float
    budget_fraction: float
    triggered_detector_names: tuple[str, ...]
    public_canary_triggered: bool
    nimbus_reason: str
    critic_kind: str | None
    partial_match_count: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "scenario_name": self.scenario_name,
            "turn_index": self.turn_index,
            "final_action": self.final_action.value,
            "nimbus_action": self.nimbus_action.value,
            "turn_leakage_bits": self.turn_leakage_bits,
            "cumulative_leakage_bits": self.cumulative_leakage_bits,
            "budget_fraction": self.budget_fraction,
            "triggered_detector_names": list(self.triggered_detector_names),
            "public_canary_triggered": self.public_canary_triggered,
            "nimbus_reason": self.nimbus_reason,
            "critic_kind": self.critic_kind,
            "partial_match_count": self.partial_match_count,
        }


@dataclass(frozen=True)
class NimbusScenarioSummary:
    scenario_name: str
    turn_count: int
    final_action_progression: tuple[Action, ...]
    nimbus_action_progression: tuple[Action, ...]
    max_cumulative_leakage_bits: float
    final_budget_fraction: float
    triggered_detector_names: tuple[str, ...]
    public_canary_triggered: bool
    partial_leak_turn_count: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "scenario_name": self.scenario_name,
            "turn_count": self.turn_count,
            "final_action_progression": [action.value for action in self.final_action_progression],
            "nimbus_action_progression": [action.value for action in self.nimbus_action_progression],
            "max_cumulative_leakage_bits": self.max_cumulative_leakage_bits,
            "final_budget_fraction": self.final_budget_fraction,
            "triggered_detector_names": list(self.triggered_detector_names),
            "public_canary_triggered": self.public_canary_triggered,
            "partial_leak_turn_count": self.partial_leak_turn_count,
        }


def load_nimbus_redteam_metrics_jsonl(path: Path) -> tuple[NimbusTurnMetric, ...]:
    records = _read_jsonl(path)
    metrics: list[NimbusTurnMetric] = []
    for line_number, record in records:
        metrics.extend(_metrics_from_record(record, line_number))
    if len(metrics) == 0:
        raise NimbusRedteamParseError(f"No NIMBUS turn metrics found in {path}.")
    return tuple(metrics)


def summarize_nimbus_redteam_metrics(metrics: Iterable[NimbusTurnMetric]) -> tuple[NimbusScenarioSummary, ...]:
    grouped: dict[str, list[NimbusTurnMetric]] = {}
    for metric in metrics:
        grouped.setdefault(metric.scenario_name, []).append(metric)
    if len(grouped) == 0:
        raise NimbusRedteamParseError("No NIMBUS turn metrics were supplied.")

    summaries: list[NimbusScenarioSummary] = []
    for scenario_name, scenario_metrics in grouped.items():
        ordered_metrics = sorted(scenario_metrics, key=lambda metric: metric.turn_index)
        summaries.append(_summary_from_metrics(scenario_name, tuple(ordered_metrics)))
    return tuple(summaries)


def render_nimbus_redteam_markdown(summaries: Sequence[NimbusScenarioSummary]) -> str:
    if len(summaries) == 0:
        raise NimbusRedteamParseError("No NIMBUS summaries were supplied.")

    lines = [
        "# NIMBUS Redteam Report",
        "",
        "| Scenario | Turns | Policy progression | NIMBUS progression | Max cumulative bits | "
        "Final budget | Public canary triggered |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(_summary_markdown_row(summary))

    lines.extend(
        (
            "",
            "## Detector Notes",
            "",
            "- Public canary detectors are immediate post-output checks with their own thresholds.",
            "- NIMBUS critic evidence can accumulate partial leakage even when public canary detectors do not trigger.",
        )
    )
    return "\n".join(lines) + "\n"


def summaries_to_json(summaries: Sequence[NimbusScenarioSummary]) -> str:
    return json.dumps([summary.to_dict() for summary in summaries], allow_nan=False, indent=2, sort_keys=True) + "\n"


def _summary_markdown_row(summary: NimbusScenarioSummary) -> str:
    cells = (
        _markdown_table_cell(summary.scenario_name),
        str(summary.turn_count),
        _action_progression(summary.final_action_progression),
        _action_progression(summary.nimbus_action_progression),
        _format_float(summary.max_cumulative_leakage_bits),
        _format_float(summary.final_budget_fraction),
        "yes" if summary.public_canary_triggered else "no",
    )
    return "| " + " | ".join(cells) + " |"


def _markdown_table_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _read_jsonl(path: Path) -> tuple[tuple[int, Mapping[str, object]], ...]:
    records: list[tuple[int, Mapping[str, object]]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = _loads_json_line(line, line_number)
            records.append((line_number, _as_mapping(decoded, f"Line {line_number}")))
    if len(records) == 0:
        raise NimbusRedteamParseError(f"No JSONL records found in {path}.")
    return tuple(records)


def _loads_json_line(line: str, line_number: int) -> object:
    try:
        return json.loads(
            line,
            object_pairs_hook=_json_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise NimbusRedteamParseError(f"Line {line_number}: invalid JSON: {exc}") from exc


def _json_object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON object key '{key}'")
        record[key] = value
    return record


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant '{value}'")


def _metrics_from_record(record: Mapping[str, object], line_number: int) -> tuple[NimbusTurnMetric, ...]:
    scenario_name = _safe_identifier(_required_string(record, "scenario_name", f"Line {line_number}"), "scenario_name")
    turn_results = _required_list(record, "turn_results", f"Line {line_number} scenario '{scenario_name}'")
    if len(turn_results) == 0:
        raise NimbusRedteamParseError(
            f"Line {line_number} scenario '{scenario_name}': field 'turn_results' must not be empty."
        )
    metrics: list[NimbusTurnMetric] = []
    for turn_result in turn_results:
        turn_record = _as_mapping(turn_result, f"Line {line_number} scenario '{scenario_name}' turn")
        metrics.append(_metric_from_turn(scenario_name, turn_record, line_number))
    return tuple(metrics)


def _metric_from_turn(
    scenario_name: str,
    turn: Mapping[str, object],
    line_number: int,
) -> NimbusTurnMetric:
    turn_index = _required_int(turn, "turn_index", f"Line {line_number} scenario '{scenario_name}'")
    context = f"Line {line_number} scenario '{scenario_name}' turn {turn_index}"
    detector_results = _turn_detector_results(turn, context)
    detector_records = tuple(_as_mapping(detector, f"{context} detector_result") for detector in detector_results)
    nimbus_detector = _find_detector(detector_records, "nimbus", context)
    evidence = _as_mapping(nimbus_detector.get("evidence"), f"{context} nimbus evidence")
    critic_evidence = _optional_mapping(evidence.get("critic_evidence"), f"{context} nimbus critic_evidence")
    nimbus_action = _detector_action(nimbus_detector, evidence, context)

    return NimbusTurnMetric(
        scenario_name=scenario_name,
        turn_index=turn_index,
        final_action=_policy_action(turn, context),
        nimbus_action=nimbus_action,
        turn_leakage_bits=_nimbus_non_negative_float(
            detector=nimbus_detector,
            evidence=evidence,
            field_name="turn_estimated_leakage_bits",
            action=nimbus_action,
            context=context,
        ),
        cumulative_leakage_bits=_nimbus_non_negative_float(
            detector=nimbus_detector,
            evidence=evidence,
            field_name="cumulative_estimated_leakage_bits",
            action=nimbus_action,
            context=context,
        ),
        budget_fraction=_nimbus_budget_fraction(
            detector=nimbus_detector,
            evidence=evidence,
            action=nimbus_action,
            context=context,
        ),
        triggered_detector_names=_policy_triggered_detector_names(turn, context),
        public_canary_triggered=_public_canary_triggered(detector_records, context),
        nimbus_reason=_optional_string(evidence, "reason", context) or "",
        critic_kind=_optional_string(critic_evidence, "critic_kind", context),
        partial_match_count=_optional_int(critic_evidence, "partial_match_count", context) or 0,
    )


def _find_detector(
    detector_records: tuple[Mapping[str, object], ...],
    detector_name: str,
    context: str,
) -> Mapping[str, object]:
    for detector in detector_records:
        if _detector_name(detector, context) == detector_name:
            return detector
    raise NimbusRedteamParseError(f"{context}: detector_results must include detector '{detector_name}'.")


def _detector_name(detector: Mapping[str, object], context: str) -> str:
    name = detector.get("name")
    if isinstance(name, str) and name != "":
        return _safe_identifier(name, "detector name")
    detector_name = detector.get("detector_name")
    if isinstance(detector_name, str) and detector_name != "":
        return _safe_identifier(detector_name, "detector_name")
    raise NimbusRedteamParseError(f"{context}: detector result must include 'name' or 'detector_name'.")


def _detector_action(detector: Mapping[str, object], evidence: Mapping[str, object], context: str) -> Action:
    action = detector.get("action")
    if isinstance(action, str) and action != "":
        return _action_from_string(action, f"{context} nimbus action")
    recommended_action = detector.get("recommended_action")
    if isinstance(recommended_action, str) and recommended_action != "":
        return _action_from_string(recommended_action, f"{context} nimbus recommended_action")
    inferred_action = _budget_action_from_evidence(evidence, context)
    if inferred_action is not None:
        return inferred_action
    raise NimbusRedteamParseError(
        f"{context}: nimbus detector must include action/recommended_action or threshold evidence."
    )


def _policy_action(turn: Mapping[str, object], context: str) -> Action:
    for policy_decision in _policy_decision_candidates(turn, context):
        if "final_action" not in policy_decision:
            continue
        final_action = _required_string(policy_decision, "final_action", f"{context} policy_decision")
        return _action_from_string(final_action, f"{context} policy final_action")
    raise NimbusRedteamParseError(f"{context}: policy_decision must include final_action.")


def _policy_triggered_detector_names(turn: Mapping[str, object], context: str) -> tuple[str, ...]:
    names: list[str] = []
    for policy_decision in _policy_decision_candidates(turn, context):
        triggered = policy_decision.get("triggered_detectors")
        if not isinstance(triggered, list):
            continue
        for value in triggered:
            if not isinstance(value, str):
                raise NimbusRedteamParseError(f"{context}: triggered_detectors entries must be strings.")
            name = _safe_identifier(value, "triggered detector name")
            if name not in names:
                names.append(name)
    return tuple(names)


def _turn_detector_results(turn: Mapping[str, object], context: str) -> list[object]:
    detector_results = turn.get("detector_results")
    if isinstance(detector_results, list):
        return detector_results

    aegis_metadata = _optional_mapping(turn.get("aegis_metadata"), f"{context} aegis_metadata")
    nested_detector_results = aegis_metadata.get("detector_results")
    if isinstance(nested_detector_results, list):
        return nested_detector_results

    raise NimbusRedteamParseError(f"{context}: field 'detector_results' must be a list.")


def _policy_decision_candidates(turn: Mapping[str, object], context: str) -> tuple[Mapping[str, object], ...]:
    candidates: list[Mapping[str, object]] = []
    policy_decision = turn.get("policy_decision")
    if isinstance(policy_decision, dict):
        candidates.append(cast(Mapping[str, object], policy_decision))

    aegis_metadata = _optional_mapping(turn.get("aegis_metadata"), f"{context} aegis_metadata")
    nested_policy = aegis_metadata.get("policy_decision")
    if isinstance(nested_policy, dict):
        candidates.append(cast(Mapping[str, object], nested_policy))
    return tuple(candidates)


def _public_canary_triggered(detector_records: tuple[Mapping[str, object], ...], context: str) -> bool:
    for detector in detector_records:
        name = _detector_name(detector, context)
        component = _detector_component(detector, context)
        is_public_canary = component == DetectorComponent.TEXT_CANARY or name in PUBLIC_CANARY_DETECTOR_NAMES
        if is_public_canary and _detector_triggered(detector, context):
            return True
    return False


def _detector_component(detector: Mapping[str, object], context: str) -> DetectorComponent | None:
    value = detector.get("component")
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise NimbusRedteamParseError(f"{context}: detector component must be a non-empty string.")
    try:
        return DetectorComponent(value)
    except ValueError as exc:
        raise NimbusRedteamParseError(f"{context}: unsupported detector component '{value}'.") from exc


def _detector_triggered(detector: Mapping[str, object], context: str) -> bool:
    triggered = detector.get("triggered")
    if isinstance(triggered, bool):
        return triggered
    action = detector.get("action")
    if isinstance(action, str) and action != "":
        return _action_from_string(action, f"{context} detector action") != Action.ALLOW
    recommended_action = detector.get("recommended_action")
    if isinstance(recommended_action, str) and recommended_action != "":
        return _action_from_string(recommended_action, f"{context} detector recommended_action") != Action.ALLOW
    score = detector.get("score")
    if _is_number(score):
        return float(score) > 0.0
    return False


def _budget_action_from_evidence(evidence: Mapping[str, object], context: str) -> Action | None:
    if "budget_fraction" not in evidence:
        return None
    if "warn_threshold" not in evidence or "sanitize_threshold" not in evidence or "block_threshold" not in evidence:
        return None

    budget_fraction = _required_probability_float(evidence, "budget_fraction", context)
    warn_threshold = _required_probability_float(evidence, "warn_threshold", context)
    sanitize_threshold = _required_probability_float(evidence, "sanitize_threshold", context)
    block_threshold = _required_probability_float(evidence, "block_threshold", context)
    if budget_fraction >= block_threshold:
        return Action.BLOCK
    if budget_fraction >= sanitize_threshold:
        return Action.SANITIZE
    if budget_fraction >= warn_threshold:
        return Action.WARN
    return Action.ALLOW


def _nimbus_non_negative_float(
    detector: Mapping[str, object],
    evidence: Mapping[str, object],
    field_name: str,
    action: Action,
    context: str,
) -> float:
    if field_name in evidence:
        return _required_non_negative_float(evidence, field_name, context)
    if action == Action.ALLOW and _detector_score(detector, context) == 0.0:
        return 0.0
    return _required_non_negative_float(evidence, field_name, context)


def _nimbus_budget_fraction(
    detector: Mapping[str, object],
    evidence: Mapping[str, object],
    action: Action,
    context: str,
) -> float:
    if "budget_fraction" in evidence:
        return _required_probability_float(evidence, "budget_fraction", context)
    if action == Action.ALLOW and _detector_score(detector, context) == 0.0:
        return 0.0
    return _required_probability_float(evidence, "budget_fraction", context)


def _detector_score(detector: Mapping[str, object], context: str) -> float:
    value = detector.get("score")
    if not _is_number(value):
        raise NimbusRedteamParseError(f"{context}: detector score must be numeric.")
    score = float(value)
    if not math.isfinite(score):
        raise NimbusRedteamParseError(f"{context}: detector score must be finite.")
    return score


def _summary_from_metrics(
    scenario_name: str,
    metrics: tuple[NimbusTurnMetric, ...],
) -> NimbusScenarioSummary:
    triggered_names: list[str] = []
    for metric in metrics:
        for name in metric.triggered_detector_names:
            if name not in triggered_names:
                triggered_names.append(name)
    return NimbusScenarioSummary(
        scenario_name=scenario_name,
        turn_count=len(metrics),
        final_action_progression=tuple(metric.final_action for metric in metrics),
        nimbus_action_progression=tuple(metric.nimbus_action for metric in metrics),
        max_cumulative_leakage_bits=max(metric.cumulative_leakage_bits for metric in metrics),
        final_budget_fraction=metrics[-1].budget_fraction,
        triggered_detector_names=tuple(triggered_names),
        public_canary_triggered=any(metric.public_canary_triggered for metric in metrics),
        partial_leak_turn_count=sum(1 for metric in metrics if metric.partial_match_count > 0),
    )


def _action_progression(actions: tuple[Action, ...]) -> str:
    return " -> ".join(action.value for action in actions)


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _action_from_string(value: str, context: str) -> Action:
    try:
        return Action(value)
    except ValueError as exc:
        raise NimbusRedteamParseError(f"{context}: unsupported action '{value}'.") from exc


def _safe_identifier(value: str, field_name: str) -> str:
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        raise NimbusRedteamParseError(
            f"field '{field_name}' must be a safe identifier: ASCII letters, digits, '.', '_', ':', or '-'."
        )
    if _looks_credential_like(value):
        raise NimbusRedteamParseError(f"field '{field_name}' must not contain credential-shaped material.")
    return value


def _looks_credential_like(value: str) -> bool:
    return value.startswith(CREDENTIAL_LIKE_PREFIXES) or "{{CREDENTIAL:" in value.upper()


def _as_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise NimbusRedteamParseError(f"{description}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _optional_mapping(value: object, description: str) -> Mapping[str, object]:
    if value is None:
        return {}
    return _as_mapping(value, description)


def _required_string(record: Mapping[str, object], field_name: str, context: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be a non-empty string.")
    return value


def _optional_string(record: Mapping[str, object], field_name: str, context: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be null or a non-empty string.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, context: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be an integer.")
    return value


def _optional_int(record: Mapping[str, object], field_name: str, context: str) -> int | None:
    value = record.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be null or an integer.")
    return value


def _required_float(record: Mapping[str, object], field_name: str, context: str) -> float:
    value = record.get(field_name)
    if not _is_number(value):
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be finite.")
    return numeric


def _required_non_negative_float(record: Mapping[str, object], field_name: str, context: str) -> float:
    value = _required_float(record, field_name, context)
    if value < 0.0:
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be non-negative.")
    return value


def _required_probability_float(record: Mapping[str, object], field_name: str, context: str) -> float:
    value = _required_float(record, field_name, context)
    if value < 0.0 or value > 1.0:
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be between 0.0 and 1.0.")
    return value


def _required_list(record: Mapping[str, object], field_name: str, context: str) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise NimbusRedteamParseError(f"{context}: field '{field_name}' must be a list.")
    return value


def _is_number(value: object) -> TypeGuard[int | float]:
    return not isinstance(value, bool) and isinstance(value, int | float)
