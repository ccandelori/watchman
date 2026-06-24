import unittest

from aegis.audit.memory import InMemoryAuditSink
from aegis.proxy.mock_app import MockProxyApp, create_default_proxy


class MockProxyAppTest(unittest.TestCase):
    def test_health_route_returns_ok(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(method="GET", path="/health", body={})

        self.assertEqual(200, status)
        self.assertEqual({"status": "ok"}, payload)

    def test_chat_completions_route_returns_openai_compatible_response_and_audit(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"session_id": "session-1", "trace_id": "trace-1"},
            },
        )
        audit_status, audit_payload = proxy.handle(method="GET", path="/audit/recent", body={})

        self.assertEqual(200, status)
        self.assertEqual("chat.completion", payload["object"])
        self.assertEqual("assistant", payload["choices"][0]["message"]["role"])
        aegis_metadata = payload["aegis"]
        self.assertEqual("aegis.proxy.chat_completion/v1", aegis_metadata["schema_version"])
        self.assertEqual("trace-1", aegis_metadata["trace_id"])
        self.assertEqual("session-1", aegis_metadata["session_id"])
        self.assertEqual(1, aegis_metadata["turn_index"])
        self.assertEqual("black_box", aegis_metadata["capability_mode"])
        self.assertEqual(len(aegis_metadata["detector_results"]), aegis_metadata["detector_count"])
        self.assertEqual(200, audit_status)
        self.assertEqual(1, len(audit_payload["events"]))
        audit_event = audit_payload["events"][0]
        self.assertEqual("trace-1", audit_event["trace_id"])
        self.assertIn("turn_summary", audit_event)
        self.assertIn("detector_results", audit_event)
        self.assertIn("policy_decision", audit_event)
        self.assertNotIn("normalized_turn", audit_event)

    def test_audit_recent_does_not_echo_raw_messages_or_metadata_values(self) -> None:
        proxy = create_default_proxy()
        private_marker = "redacted-marker-123456789"

        proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": f"please summarize {private_marker}"}],
                "metadata": {
                    "session_id": "session-1",
                    "trace_id": "trace-secret",
                    "mock_response": f"assistant output {private_marker}",
                    "operator_note": f"private note {private_marker}",
                },
            },
        )

        audit_status, audit_payload = proxy.handle(method="GET", path="/audit/recent", body={})

        self.assertEqual(200, audit_status)
        audit_event = audit_payload["events"][0]
        audit_text = str(audit_payload)
        self.assertEqual("trace-secret", audit_event["trace_id"])
        self.assertEqual("session-1", audit_event["session_id"])
        self.assertEqual(1, audit_event["turn_summary"]["message_count"])
        self.assertEqual(["user"], audit_event["turn_summary"]["message_roles"])
        self.assertNotIn(private_marker, audit_text)
        self.assertNotIn("please summarize", audit_text)
        self.assertNotIn("mock_response", audit_text)
        self.assertNotIn("operator_note", audit_text)

    def test_chat_completions_route_normalizes_tool_calls_without_audit_argument_echo(self) -> None:
        proxy = create_default_proxy()
        private_marker = "redacted-marker-246813579"

        proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "prepare a tool call"}],
                "metadata": {"session_id": "session-tools", "trace_id": "trace-tools"},
                "tool_calls": [
                    {
                        "name": "send_slack_message",
                        "arguments": {"channel": "#ir", "text": private_marker},
                    }
                ],
            },
        )

        audit_status, audit_payload = proxy.handle(method="GET", path="/audit/recent", body={})

        self.assertEqual(200, audit_status)
        audit_event = audit_payload["events"][0]
        self.assertEqual(1, audit_event["turn_summary"]["tool_call_count"])
        self.assertEqual(["send_slack_message"], audit_event["turn_summary"]["tool_call_names"])
        self.assertNotIn(private_marker, str(audit_payload))
        self.assertNotIn("#ir", str(audit_payload))

    def test_chat_completions_route_rejects_malformed_tool_calls(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "tool_calls": [{"name": "send_slack_message", "arguments": "bad"}],
            },
        )

        self.assertEqual(400, status)
        self.assertIn("tool_calls", payload["error"])

    def test_chat_completions_route_rejects_invalid_messages(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={"model": "mock-model", "messages": []},
        )

        self.assertEqual(400, status)
        self.assertIn("messages", payload["error"])

    def test_chat_completions_route_rejects_non_object_body(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(method="POST", path="/v1/chat/completions", body=[])

        self.assertEqual(400, status)
        self.assertIn("request body", payload["error"])
        self.assertNotIn("[]", str(payload))

    def test_chat_completions_route_rejects_bool_turn_index(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"turn_index": True},
            },
        )

        self.assertEqual(400, status)
        self.assertIn("turn_index", payload["error"])

    def test_chat_completions_route_rejects_negative_turn_index(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"turn_index": -1},
            },
        )

        self.assertEqual(400, status)
        self.assertIn("turn_index", payload["error"])

    def test_chat_completions_route_sanitizes_runtime_exceptions(self) -> None:
        proxy = MockProxyApp(
            runtime=FailingRuntime(),
            audit_sink=InMemoryAuditSink(),
            test_canary_registry=None,
        )
        private_marker = "redacted-marker-987654321"

        status, payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": private_marker}],
                "metadata": {"trace_id": "trace-runtime-error"},
            },
        )

        self.assertEqual(500, status)
        self.assertEqual("internal proxy error", payload["error"])
        self.assertEqual("trace-runtime-error", payload["aegis"]["trace_id"])
        self.assertNotIn(private_marker, str(payload))

    def test_seed_canary_route_registers_tool_egress_scanner_without_returning_value(self) -> None:
        proxy = create_default_proxy()
        private_marker = "redacted-marker-314159265"

        seed_status, seed_payload = proxy.handle(
            method="POST",
            path="/test/seed-canary",
            body={
                "canary_id": "hny_test_tool",
                "credential_type": "api_key",
                "value": private_marker,
                "metadata": {"scenario": "proxy_test"},
            },
        )
        chat_status, chat_payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "send status"}],
                "tool_calls": [
                    {
                        "name": "send_slack_message",
                        "arguments": {"text": f"forward {private_marker}"},
                    }
                ],
            },
        )

        self.assertEqual(200, seed_status)
        self.assertEqual("hny_test_tool", seed_payload["canary"]["canary_id"])
        self.assertIn("sha256", seed_payload["canary"])
        self.assertNotIn(private_marker, str(seed_payload))
        self.assertEqual(200, chat_status)
        policy = chat_payload["aegis"]["policy_decision"]
        self.assertEqual("escalate", policy["final_action"])
        self.assertIn("tool_call_canary", policy["triggered_detectors"])
        self.assertNotIn(private_marker, str(chat_payload))

    def test_seed_canary_route_rejects_invalid_seed_without_echoing_value(self) -> None:
        proxy = create_default_proxy()
        private_marker = "redacted-marker-271828182"

        status, payload = proxy.handle(
            method="POST",
            path="/test/seed-canary",
            body={"canary_id": "hny_bad", "credential_type": "api_key", "value": private_marker, "metadata": []},
        )

        self.assertEqual(400, status)
        self.assertIn("metadata", payload["error"])
        self.assertNotIn(private_marker, str(payload))

    def test_test_reset_clears_audit_history_and_seeded_canaries(self) -> None:
        proxy = create_default_proxy()
        private_marker = "redacted-marker-161803398"

        proxy.handle(
            method="POST",
            path="/test/seed-canary",
            body={"canary_id": "hny_reset", "credential_type": "api_key", "value": private_marker},
        )
        proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "send status"}],
                "tool_calls": [{"name": "send_slack_message", "arguments": {"text": private_marker}}],
            },
        )

        reset_status, reset_payload = proxy.handle(method="POST", path="/test/reset", body={})
        audit_status, audit_payload = proxy.handle(method="GET", path="/audit/recent", body={})
        chat_status, chat_payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "send status"}],
                "tool_calls": [{"name": "send_slack_message", "arguments": {"text": private_marker}}],
            },
        )

        self.assertEqual(200, reset_status)
        self.assertEqual({"status": "reset"}, reset_payload)
        self.assertEqual(200, audit_status)
        self.assertEqual([], audit_payload["events"])
        self.assertEqual(200, chat_status)
        self.assertEqual("allow", chat_payload["aegis"]["policy_decision"]["final_action"])
        self.assertNotIn(private_marker, str(chat_payload))


class FailingRuntime:
    def evaluate_turn(self, request: object) -> object:
        raise RuntimeError("private runtime failure redacted-marker-987654321")


if __name__ == "__main__":
    unittest.main()
