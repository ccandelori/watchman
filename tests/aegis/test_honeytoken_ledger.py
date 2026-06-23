from __future__ import annotations

import base64
import unittest

from aegis.audit.memory import InMemoryAuditSink
from aegis.canaries.ledger import HoneytokenLedger, HoneytokenLedgerError, inject_honeytokens
from aegis.core.contracts import Action, CapabilityMode, Message, ModelInfo, NormalizedTurn
from aegis.core.orchestrator import AegisRuntime, ModelResponse, RuntimeRequest
from aegis.detectors.canary import EncodedCanaryDetector, TextCanaryDetector
from aegis.policy.engine import SeverityPolicyEngine


class EchoEncodedProvider:
    def generate(self, turn: NormalizedTurn) -> ModelResponse:
        canary_value = turn.messages[0].content.split("key: ")[1].split(".")[0]
        encoded_value = base64.b64encode(canary_value.encode()).decode("utf-8")
        return ModelResponse(output_text=f"encoded={encoded_value}", metadata={"provider": "encoded_echo"})


def _generator(slot_name: str, credential_type: str) -> str:
    return f"sk-hny-{slot_name}-{credential_type}-0001"


def _messages() -> tuple[Message, ...]:
    return (
        Message(role="system", content="Use key: {{CREDENTIAL:api_key:openai_key}}."),
        Message(role="user", content="Summarize the request."),
    )


class HoneytokenLedgerTest(unittest.TestCase):
    def test_injection_replaces_placeholders_and_registers_canary_record(self) -> None:
        ledger = HoneytokenLedger(session_id="session-ledger", generator=_generator)

        result = inject_honeytokens(
            messages=_messages(),
            ledger=ledger,
            turn_index=2,
        )

        self.assertEqual("Use key: sk-hny-api_key-openai_key-0001.", result.messages[0].content)
        self.assertEqual(_messages()[0].content, "Use key: {{CREDENTIAL:api_key:openai_key}}.")
        self.assertEqual(1, len(result.canary_records))
        self.assertEqual("hny_session-ledger_api_key", result.canary_records[0].canary_id)
        self.assertEqual("openai_key", result.canary_records[0].credential_type)
        self.assertEqual("sk-hny-api_key-openai_key-0001", result.canary_records[0].value)
        self.assertEqual(1, len(result.sensitive_spans))
        self.assertEqual("honeytoken", result.sensitive_spans[0].kind)
        self.assertEqual("dp_honey_lite", result.sensitive_spans[0].source)
        self.assertEqual("hny_session-ledger_api_key", result.sensitive_spans[0].identifier)
        self.assertEqual("openai_key", result.sensitive_spans[0].metadata["credential_type"])
        self.assertEqual(result.canary_records[0].sha256, result.sensitive_spans[0].metadata["sha256"])

    def test_real_secret_scrub_substitutes_honeytoken_without_auditing_raw_secret(self) -> None:
        ledger = HoneytokenLedger(session_id="session-ledger", generator=_generator)
        ledger.register_real_secret(slot_name="api_key", credential_type="openai_key", value="sk-real-secret-123")

        result = inject_honeytokens(
            messages=(
                Message(role="system", content="Use key: sk-real-secret-123."),
                Message(role="user", content="Do not leak it."),
            ),
            ledger=ledger,
            turn_index=0,
        )

        self.assertEqual("Use key: sk-hny-api_key-openai_key-0001.", result.messages[0].content)
        self.assertEqual(1, len(result.canary_records))
        self.assertNotIn("sk-real-secret-123", str(result.to_dict()))

    def test_repeated_slot_reuses_existing_honeytoken(self) -> None:
        ledger = HoneytokenLedger(session_id="session-ledger", generator=_generator)

        first = inject_honeytokens(messages=_messages(), ledger=ledger, turn_index=0)
        second = inject_honeytokens(messages=_messages(), ledger=ledger, turn_index=1)

        self.assertEqual(first.canary_records[0].value, second.canary_records[0].value)
        self.assertEqual(1, len(ledger.canary_records()))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(HoneytokenLedgerError, "session_id"):
            HoneytokenLedger(session_id="", generator=_generator)

    def test_runtime_detects_encoded_leak_from_injected_honeytoken(self) -> None:
        ledger = HoneytokenLedger(session_id="session-ledger", generator=_generator)
        injection = inject_honeytokens(messages=_messages(), ledger=ledger, turn_index=0)
        registry = injection.canary_registry()
        audit_sink = InMemoryAuditSink()
        runtime = AegisRuntime(
            turn_annotators=(),
            pre_generation_detectors=(),
            post_generation_detectors=(
                TextCanaryDetector(detector_name="text_canary", registry=registry),
                EncodedCanaryDetector(
                    detector_name="encoded_canary",
                    registry=registry,
                    partial_match_threshold=0.8,
                ),
            ),
            session_detectors=(),
            policy_engine=SeverityPolicyEngine(),
            audit_sink=audit_sink,
            model_provider=EchoEncodedProvider(),
        )
        request = RuntimeRequest(
            trace_id="trace-ledger",
            session_id="session-ledger",
            turn_index=0,
            capability_mode=CapabilityMode.BLACK_BOX,
            model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
            messages=injection.messages,
            tool_calls=(),
            sensitive_spans=injection.sensitive_spans,
            metadata={"canary_ids": [record.canary_id for record in injection.canary_records]},
        )

        response = runtime.evaluate_turn(request)

        self.assertEqual(Action.ESCALATE, response.policy_decision.final_action)
        self.assertEqual(("encoded_canary",), response.policy_decision.triggered_detectors)
        self.assertEqual(2, len(response.detector_results))
        self.assertNotIn("{{CREDENTIAL", str(response.audit_event.to_dict()))
        self.assertNotIn("sk-real-secret-123", str(response.audit_event.to_dict()))


if __name__ == "__main__":
    unittest.main()
