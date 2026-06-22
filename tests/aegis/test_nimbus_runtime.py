"""Runtime integration tests for NIMBUS."""

from aegis.proxy.mock_app import create_default_proxy


def test_default_proxy_emits_nimbus_result():
    proxy = create_default_proxy()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"secret_context_handle": "secret-123"},
    }
    status, response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 200

    detector_results = response["aegis"]["detector_results"]
    nimbus_results = [r for r in detector_results if r["detector_name"] == "nimbus"]
    assert len(nimbus_results) >= 1
    assert any(r["capability_status"] == "active" for r in nimbus_results)


def test_nimbus_unavailable_without_secret_handle():
    proxy = create_default_proxy()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    status, response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 200

    detector_results = response["aegis"]["detector_results"]
    nimbus_results = [r for r in detector_results if r["detector_name"] == "nimbus"]
    assert any(r["capability_status"] == "unavailable" for r in nimbus_results)


def test_nimbus_active_with_metadata_handle():
    proxy = create_default_proxy()
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"secret_context_handle": "secret-xyz"},
    }
    status, response = proxy.handle("POST", "/v1/chat/completions", body)
    assert status == 200

    # Response-level check
    detector_results = response["aegis"]["detector_results"]
    nimbus_results = [r for r in detector_results if r["detector_name"] == "nimbus"]
    assert any(r["capability_status"] == "active" for r in nimbus_results)

    # Audit-level check (nested detector_results)
    audit_status, audit_payload = proxy.handle("GET", "/audit/recent", {})
    assert audit_status == 200

    audit_events = audit_payload["events"]
    assert len(audit_events) >= 1

    # Find an audit event that contains NIMBUS results
    nimbus_audit_results = []
    for event in audit_events:
        detector_results = event.get("detector_results")
        if isinstance(detector_results, list):
            for result in detector_results:
                if isinstance(result, dict) and result.get("detector_name") == "nimbus":
                    nimbus_audit_results.append(result)

    assert len(nimbus_audit_results) >= 1
    assert any(r.get("capability_status") == "active" for r in nimbus_audit_results)

    # Ensure the actual secret handle is not leaked in evidence
    for result in nimbus_audit_results:
        assert "secret-xyz" not in str(result.get("evidence", {}))
