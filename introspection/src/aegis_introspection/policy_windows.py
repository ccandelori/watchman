from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from aegis_introspection.honeytokens import CharacterSpan, TokenOffset, char_span_to_token_span


class PolicyWindowError(ValueError):
    """Raised when a V3 policy window cannot be derived from a prompt."""


@dataclass(frozen=True)
class V3PolicyWindow:
    selected_field: str
    selected_mode: str
    selected_action: str
    char_spans: tuple[CharacterSpan, ...]
    token_indices: tuple[int, ...]


@dataclass(frozen=True)
class AssignmentMatch:
    value: str
    assignment_span: CharacterSpan
    value_span: CharacterSpan


@dataclass(frozen=True)
class V3PolicyParts:
    selected_field_assignment: AssignmentMatch
    selected_mode_assignment: AssignmentMatch
    decision_clause: CharacterSpan
    selected_action: str


_FIELD_VALUE_PATTERN = r"(?:credential_value|summary_value)"
_MODE_VALUE_PATTERN = r"(?:mode_a|mode_b)"
_ACTION_VALUE_PATTERN = r"(?:copy|mask)"
_MODE_POLICY_PATTERN = rf"credential_value={_ACTION_VALUE_PATTERN};summary_value={_ACTION_VALUE_PATTERN}"


def derive_v3_policy_window(text: str, offsets: tuple[TokenOffset, ...]) -> V3PolicyWindow:
    parts = _v3_policy_parts(text=text)
    char_spans = (
        parts.decision_clause,
        parts.selected_field_assignment.assignment_span,
        parts.selected_mode_assignment.assignment_span,
    )
    return _policy_window_from_spans(text=text, offsets=offsets, parts=parts, char_spans=char_spans)


def derive_v3_selector_window(text: str, offsets: tuple[TokenOffset, ...]) -> V3PolicyWindow:
    parts = _v3_policy_parts(text=text)
    char_spans = (
        parts.selected_field_assignment.assignment_span,
        parts.selected_mode_assignment.assignment_span,
    )
    return _policy_window_from_spans(text=text, offsets=offsets, parts=parts, char_spans=char_spans)


def _v3_policy_parts(text: str) -> V3PolicyParts:
    selected_field_assignment = _find_assignment(
        text=text,
        key="selected_field",
        value_pattern=_FIELD_VALUE_PATTERN,
    )
    selected_mode_assignment = _find_assignment(
        text=text,
        key="selected_mode",
        value_pattern=_MODE_VALUE_PATTERN,
    )
    selected_mode_policy = _find_assignment(
        text=text,
        key=selected_mode_assignment.value,
        value_pattern=_MODE_POLICY_PATTERN,
    )
    decision_clause = _decision_clause_span(
        selected_mode_policy=selected_mode_policy,
        selected_field=selected_field_assignment.value,
    )
    selected_action = _selected_action(text=text, decision_clause=decision_clause)
    return V3PolicyParts(
        selected_field_assignment=selected_field_assignment,
        selected_mode_assignment=selected_mode_assignment,
        decision_clause=decision_clause,
        selected_action=selected_action,
    )


def _policy_window_from_spans(
    text: str,
    offsets: tuple[TokenOffset, ...],
    parts: V3PolicyParts,
    char_spans: tuple[CharacterSpan, ...],
) -> V3PolicyWindow:
    token_indices = _token_indices_for_spans(offsets=offsets, char_spans=char_spans)
    return V3PolicyWindow(
        selected_field=parts.selected_field_assignment.value,
        selected_mode=parts.selected_mode_assignment.value,
        selected_action=parts.selected_action,
        char_spans=char_spans,
        token_indices=token_indices,
    )


def _find_assignment(text: str, key: str, value_pattern: str) -> AssignmentMatch:
    matches = tuple(_assignment_matches(text=text, key=key, value_pattern=value_pattern))
    if len(matches) != 1:
        raise PolicyWindowError(f"Expected exactly one assignment for '{key}', found {len(matches)}.")
    return matches[0]


def _assignment_matches(text: str, key: str, value_pattern: str) -> tuple[AssignmentMatch, ...]:
    patterns = (
        re.compile(rf"{re.escape(key)}=({value_pattern})"),
        re.compile(rf"'{re.escape(key)}': '({value_pattern})'"),
    )
    matches: list[AssignmentMatch] = []
    for pattern in patterns:
        matches.extend(_matches_for_pattern(text=text, pattern=pattern))
    return tuple(matches)


def _matches_for_pattern(text: str, pattern: Pattern[str]) -> tuple[AssignmentMatch, ...]:
    matches: list[AssignmentMatch] = []
    for match in pattern.finditer(text):
        matches.append(
            AssignmentMatch(
                value=match.group(1),
                assignment_span=CharacterSpan(start=match.start(0), end=match.end(0)),
                value_span=CharacterSpan(start=match.start(1), end=match.end(1)),
            )
        )
    return tuple(matches)


def _decision_clause_span(selected_mode_policy: AssignmentMatch, selected_field: str) -> CharacterSpan:
    pattern = re.compile(rf"{re.escape(selected_field)}=({_ACTION_VALUE_PATTERN})")
    match = pattern.search(selected_mode_policy.value)
    if match is None:
        raise PolicyWindowError(
            f"Selected mode '{selected_mode_policy.value}' does not include selected field '{selected_field}'."
        )
    return CharacterSpan(
        start=selected_mode_policy.value_span.start + match.start(0),
        end=selected_mode_policy.value_span.start + match.end(0),
    )


def _selected_action(text: str, decision_clause: CharacterSpan) -> str:
    clause = text[decision_clause.start : decision_clause.end]
    _, action = clause.split("=", maxsplit=1)
    if action not in ("copy", "mask"):
        raise PolicyWindowError(f"Selected action '{action}' is invalid.")
    return action


def _token_indices_for_spans(offsets: tuple[TokenOffset, ...], char_spans: tuple[CharacterSpan, ...]) -> tuple[int, ...]:
    token_indices: set[int] = set()
    for index, char_span in enumerate(char_spans):
        token_span = char_span_to_token_span(
            offsets=offsets,
            character_span=char_span,
            field_name=f"policy_window_char_span_{index}",
        )
        token_indices.update(range(token_span.start, token_span.end))
    if len(token_indices) == 0:
        raise PolicyWindowError("Derived policy window has no token indices.")
    return tuple(sorted(token_indices))
