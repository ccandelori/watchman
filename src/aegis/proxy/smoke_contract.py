from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewaySmokeContract:
    reset_session_ids: tuple[str, ...]
    check_names: tuple[str, ...]
    runtime_trace_stages: tuple[str, ...]


def gateway_smoke_contract() -> GatewaySmokeContract:
    return GatewaySmokeContract(
        reset_session_ids=(
            "smoke-session",
            "smoke-cift-session",
            "smoke-slot-session",
            "smoke-tool-session",
            "smoke-partial-session",
        ),
        check_names=(
            "health",
            "gateway_readiness",
            "capabilities",
            "benign_chat",
            "adversarial_benign_no_block",
            "ambiguous_protected_workflow",
            "cift_pre_generation_block",
            "provider_egress_guard_block",
            "tool_argument_canary_leak",
            "encoded_canary_leak",
            "metadata_slot_canary_leak",
            "nimbus_partial_leak",
            "audit_recent",
            "audit_explain",
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
