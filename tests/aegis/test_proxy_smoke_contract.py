from __future__ import annotations

from aegis.proxy.smoke_contract import gateway_smoke_contract


def test_gateway_smoke_contract_names_checks_resets_and_runtime_stages() -> None:
    contract = gateway_smoke_contract()

    assert contract.reset_session_ids == ("smoke-session", "smoke-seeded-session", "smoke-partial-session")
    assert contract.check_names == (
        "health",
        "capabilities",
        "benign_chat",
        "encoded_canary_leak",
        "seeded_canary_leak",
        "nimbus_partial_leak",
        "audit_recent",
    )
    assert contract.runtime_trace_stages == (
        "normalize",
        "dp_honey",
        "cift",
        "provider_egress_guard",
        "provider",
        "canary",
        "nimbus",
        "policy",
        "audit",
    )
