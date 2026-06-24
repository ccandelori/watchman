from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewaySmokeContract:
    reset_session_ids: tuple[str, ...]
    check_names: tuple[str, ...]
    runtime_trace_stages: tuple[str, ...]


def gateway_smoke_contract() -> GatewaySmokeContract:
    return GatewaySmokeContract(
        reset_session_ids=("smoke-session", "smoke-seeded-session", "smoke-partial-session"),
        check_names=(
            "health",
            "capabilities",
            "benign_chat",
            "encoded_canary_leak",
            "seeded_canary_leak",
            "nimbus_partial_leak",
            "audit_recent",
        ),
        runtime_trace_stages=(
            "normalize",
            "dp_honey",
            "cift",
            "provider_egress_guard",
            "provider",
            "canary",
            "nimbus",
            "policy",
            "audit",
        ),
    )
