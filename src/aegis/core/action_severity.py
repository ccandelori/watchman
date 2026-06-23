from __future__ import annotations

from aegis.core.contracts import Action

_ACTION_SEVERITY: dict[Action, int] = {
    Action.ALLOW: 0,
    Action.WARN: 1,
    Action.SANITIZE: 2,
    Action.BLOCK: 3,
    Action.ESCALATE: 4,
}


def action_severity(action: Action) -> int:
    return _ACTION_SEVERITY[action]


def highest_action(actions: tuple[Action, ...]) -> Action:
    if len(actions) == 0:
        raise ValueError("actions must not be empty.")
    return max(actions, key=action_severity)
