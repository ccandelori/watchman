"""Session destruction tests for Nimbus."""

from aegis.core.contracts import JsonValue
from aegis.proxy.mock_app import create_default_proxy


def _nimbus_turn_count(response: dict[str, JsonValue]) -> int:
    aegis_payload = response["aegis"]
    if not isinstance(aegis_payload, dict):
        raise AssertionError("response.aegis must be an object.")
    detector_results = aegis_payload.get("detector_results")
    if not isinstance(detector_results, list):
        raise AssertionError("response.aegis.detector_results must be a list.")

    for detector_result in detector_results:
        if not isinstance(detector_result, dict):
            continue
        if detector_result.get("detector_name") != "nimbus":
            continue
        evidence = detector_result.get("evidence")
        if not isinstance(evidence, dict):
            raise AssertionError("nimbus evidence must be an object.")
        turn_count = evidence.get("turn_count")
        if not isinstance(turn_count, int):
            raise AssertionError("nimbus evidence.turn_count must be an integer.")
        return turn_count

    raise AssertionError("nimbus detector result not found.")


def test_nimbus_state_destroyed_on_session_end() -> None:
    proxy = create_default_proxy()

    session_id = "destroy-test-session"
    secret_handle = "secret-to-destroy"

    body: dict[str, JsonValue] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"session_id": session_id, "secret_context_handle": secret_handle},
    }
    status, first_response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 200
    assert _nimbus_turn_count(first_response) == 1

    proxy.destroy_session(session_id)

    status2, response2 = proxy.handle("POST", "/v1/chat/completions", body)
    assert status2 == 200
    assert _nimbus_turn_count(response2) == 1
