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
        self.assertEqual("trace-1", audit_payload["events"][0]["trace_id"])

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
