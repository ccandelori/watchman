import unittest

from aegis.proxy.mock_app import create_default_proxy


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

    def test_chat_completions_route_rejects_invalid_messages(self) -> None:
        proxy = create_default_proxy()

        status, payload = proxy.handle(
            method="POST",
            path="/v1/chat/completions",
            body={"model": "mock-model", "messages": []},
        )

        self.assertEqual(400, status)
        self.assertIn("messages", payload["error"])


if __name__ == "__main__":
    unittest.main()
