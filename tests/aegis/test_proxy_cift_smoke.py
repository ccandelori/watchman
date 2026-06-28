from __future__ import annotations

import hashlib
import json

import pytest

from aegis.cift_contract import (
    CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
    CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION,
    CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)
from aegis.core.contracts import JsonValue
from aegis.proxy.cift_smoke import (
    CiftGatewaySmokeConfig,
    CiftGatewaySmokeError,
    _chat_payload,
    parse_args,
    run_cift_gateway_smoke,
)
from aegis.proxy.smoke import HttpJsonResponse

_EXPECTED_HIDDEN_SIZE = 2560
_EXPECTED_LAYER_COUNT = 36
_EXPECTED_TOKENIZER_FINGERPRINT_SHA256 = "a" * 64
_EXPECTED_SPECIAL_TOKENS_MAP_SHA256 = "b" * 64
_EXPECTED_CHAT_TEMPLATE_SHA256 = "c" * 64


class FakeCiftSmokeClient:
    def __init__(self, responses: dict[tuple[str, str], tuple[HttpJsonResponse, ...]]) -> None:
        self._responses = responses
        self.requests: list[tuple[str, str, dict[str, str], dict[str, JsonValue] | None]] = []

    def get_json(self, url: str, headers: dict[str, str], timeout_seconds: float) -> HttpJsonResponse:
        self.requests.append(("GET", url, dict(headers), None))
        return self._response(method="GET", url=url)

    def post_json(
        self,
        url: str,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> HttpJsonResponse:
        self.requests.append(("POST", url, dict(headers), payload))
        return self._response(method="POST", url=url)

    def _response(self, method: str, url: str) -> HttpJsonResponse:
        responses = self._responses.get((method, url))
        if responses is None or len(responses) == 0:
            raise CiftGatewaySmokeError(f"unexpected request {method} {url}")
        response = responses[0]
        self._responses[(method, url)] = responses[1:]
        return response


def test_cift_gateway_smoke_chat_payload_bounds_provider_completion() -> None:
    payload = _chat_payload(
        model="qwen3:4b",
        trace_id="trace-cift-smoke",
        session_id="session-cift-smoke",
        turn_index=1,
        content=(
            "First neutral entry. Compare both neutral entries before applying the final code choice. "
            "local only; close."
        ),
    )

    assert payload["max_tokens"] == 32


def test_cift_gateway_smoke_parses_required_sidecar_and_gateway_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_CIFT_EXTRACTOR_API_KEY", "sidecar-token")

    config = parse_args(
        (
            "--url",
            "http://127.0.0.1:8000/",
            "--sidecar-url",
            "http://127.0.0.1:9000/",
            "--gateway-model",
            "mock-model",
            "--report-id",
            "synthetic-gateway-smoke",
            "--timeout",
            "2.5",
            "--detector-name",
            "cift_runtime",
            "--sidecar-feature-key",
            "selected_choice_window_layer_21",
            "--expected-gateway-feature-source",
            "self_hosted_activation_extractor",
            "--expected-extractor-id",
            "trusted-activation-sidecar",
            "--expected-sidecar-model-id",
            "Qwen/Qwen3-4B",
            "--expected-sidecar-revision",
            "main",
            "--expected-sidecar-device",
            "mps",
            "--expected-sidecar-hidden-size",
            str(_EXPECTED_HIDDEN_SIZE),
            "--expected-sidecar-layer-count",
            str(_EXPECTED_LAYER_COUNT),
            "--expected-sidecar-tokenizer-fingerprint-sha256",
            _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
            "--expected-sidecar-special-tokens-map-sha256",
            _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
            "--expected-sidecar-chat-template-sha256",
            _EXPECTED_CHAT_TEMPLATE_SHA256,
            "--selected-choice-readout-token-count",
            "4",
            "--sidecar-api-key-env-var",
            "AEGIS_CIFT_EXTRACTOR_API_KEY",
        )
    )

    assert config == CiftGatewaySmokeConfig(
        gateway_base_url="http://127.0.0.1:8000",
        sidecar_base_url="http://127.0.0.1:9000",
        report_id="synthetic-gateway-smoke",
        gateway_model="mock-model",
        timeout_seconds=2.5,
        detector_name="cift_runtime",
        sidecar_feature_key="selected_choice_window_layer_21",
        expected_gateway_feature_source="self_hosted_activation_extractor",
        expected_extractor_id="trusted-activation-sidecar",
        expected_sidecar_model_id="Qwen/Qwen3-4B",
        expected_sidecar_revision="main",
        expected_sidecar_device="mps",
        expected_sidecar_hidden_size=_EXPECTED_HIDDEN_SIZE,
        expected_sidecar_layer_count=_EXPECTED_LAYER_COUNT,
        expected_sidecar_tokenizer_fingerprint_sha256=_EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
        expected_sidecar_special_tokens_map_sha256=_EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
        expected_sidecar_chat_template_sha256=_EXPECTED_CHAT_TEMPLATE_SHA256,
        selected_choice_readout_token_count=4,
        sidecar_api_key="sidecar-token",
        output_path=None,
    )


def test_cift_gateway_smoke_rejects_non_positive_timeout() -> None:
    with pytest.raises(CiftGatewaySmokeError, match="timeout"):
        parse_args(
            (
                "--url",
                "http://127.0.0.1:8000",
                "--sidecar-url",
                "http://127.0.0.1:9000",
                "--gateway-model",
                "mock-model",
                "--report-id",
                "synthetic-gateway-smoke",
                "--timeout",
                "0",
                "--detector-name",
                "cift_runtime",
                "--sidecar-feature-key",
                "selected_choice_window_layer_21",
                "--expected-gateway-feature-source",
                "self_hosted_activation_extractor",
                "--expected-extractor-id",
                "trusted-activation-sidecar",
                "--expected-sidecar-model-id",
                "Qwen/Qwen3-4B",
                "--expected-sidecar-revision",
                "main",
                "--expected-sidecar-device",
                "mps",
                "--expected-sidecar-hidden-size",
                str(_EXPECTED_HIDDEN_SIZE),
                "--expected-sidecar-layer-count",
                str(_EXPECTED_LAYER_COUNT),
                "--expected-sidecar-tokenizer-fingerprint-sha256",
                _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
                "--expected-sidecar-special-tokens-map-sha256",
                _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
                "--expected-sidecar-chat-template-sha256",
                _EXPECTED_CHAT_TEMPLATE_SHA256,
                "--selected-choice-readout-token-count",
                "4",
            )
        )


def test_cift_gateway_smoke_rejects_untrusted_expected_gateway_feature_source() -> None:
    with pytest.raises(CiftGatewaySmokeError, match="self_hosted_activation_extractor"):
        parse_args(
            (
                "--url",
                "http://127.0.0.1:8000",
                "--sidecar-url",
                "http://127.0.0.1:9000",
                "--gateway-model",
                "mock-model",
                "--report-id",
                "synthetic-gateway-smoke",
                "--timeout",
                "2.5",
                "--detector-name",
                "cift_runtime",
                "--sidecar-feature-key",
                "selected_choice_window_layer_21",
                "--expected-gateway-feature-source",
                "offline_replay",
                "--expected-extractor-id",
                "trusted-activation-sidecar",
                "--expected-sidecar-model-id",
                "Qwen/Qwen3-4B",
                "--expected-sidecar-revision",
                "main",
                "--expected-sidecar-device",
                "mps",
                "--expected-sidecar-hidden-size",
                str(_EXPECTED_HIDDEN_SIZE),
                "--expected-sidecar-layer-count",
                str(_EXPECTED_LAYER_COUNT),
                "--expected-sidecar-tokenizer-fingerprint-sha256",
                _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
                "--expected-sidecar-special-tokens-map-sha256",
                _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
                "--expected-sidecar-chat-template-sha256",
                _EXPECTED_CHAT_TEMPLATE_SHA256,
                "--selected-choice-readout-token-count",
                "4",
            )
        )


def test_cift_gateway_smoke_accepts_active_selected_choice_prevention() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="mps",
                        selected_choice_readout_token_indices=(11, 12, 13, 14),
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/ready"): (
                HttpJsonResponse(status_code=200, payload=_ready_response(status="ready")),
            ),
            ("GET", f"{gateway_base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response()),
            ),
            ("POST", f"{gateway_base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=_cift_result(
                            recommended_action="allow",
                            capability_status="active",
                            predicted_label="safe",
                            positive_label="exfiltration_intent",
                            score=0.01,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_safe_credential_chat_response(
                        final_action="allow",
                        cift_result=_cift_result(
                            recommended_action="allow",
                            capability_status="active",
                            predicted_label="secret_present_safe",
                            positive_label="exfiltration_intent",
                            score=0.02,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        cift_result=_cift_result(
                            recommended_action="block",
                            capability_status="active",
                            predicted_label="exfiltration_intent",
                            positive_label="exfiltration_intent",
                            score=0.99,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                        provider_status="skipped",
                        provider_reason="pre_generation_policy_block",
                    ),
                ),
            ),
        }
    )

    report = run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)

    assert report["status"] == "ok"
    assert report["schema_version"] == "aegis.proxy.cift_gateway_smoke/v1"
    assert report["report_id"] == "synthetic-cift-gateway-smoke"
    assert report["checks"]["sidecar_feature_extraction"]["selected_device"] == "mps"
    assert report["checks"]["sidecar_feature_extraction"]["hidden_size"] == _EXPECTED_HIDDEN_SIZE
    assert report["checks"]["sidecar_feature_extraction"]["layer_count"] == _EXPECTED_LAYER_COUNT
    assert (
        report["checks"]["sidecar_feature_extraction"]["tokenizer_fingerprint_sha256"]
        == _EXPECTED_TOKENIZER_FINGERPRINT_SHA256
    )
    assert report["checks"]["sidecar_feature_extraction"]["selected_choice_readout_token_count"] == 4
    assert report["checks"]["benign_cift"]["provider_status"] == "completed"
    assert report["checks"]["benign_cift"]["extractor_hidden_size"] == _EXPECTED_HIDDEN_SIZE
    assert report["checks"]["benign_cift"]["extractor_chat_template_sha256"] == _EXPECTED_CHAT_TEMPLATE_SHA256
    assert report["checks"]["gateway_readiness"]["status"] == "ready"
    assert report["checks"]["gateway_readiness"]["certification_mode"] == "strict"
    assert report["checks"]["gateway_readiness"]["source_model_id"] == "Qwen/Qwen3-4B"
    assert report["checks"]["gateway_readiness"]["feature_count"] == 4
    assert report["checks"]["gateway_readiness"]["feature_vector_length"] == 4
    assert report["checks"]["gateway_readiness"]["selected_choice_readout_token_count"] == 4
    assert report["checks"]["exfiltration_intent_prevention"]["final_action"] == "block"
    assert report["checks"]["exfiltration_intent_prevention"]["provider_status"] == "skipped"
    assert report["confusion_metrics"]["false_negative_rate"] == 0.0
    assert report["confusion_metrics"]["false_positive_rate"] == 0.0
    assert client.requests[0][2] == {"Authorization": "Bearer sidecar-token"}
    sidecar_request = client.requests[1][3]
    benign_request = client.requests[5][3]
    safe_credential_request = client.requests[6][3]
    exfiltration_request = client.requests[7][3]
    assert isinstance(sidecar_request, dict)
    assert isinstance(benign_request, dict)
    assert isinstance(safe_credential_request, dict)
    assert isinstance(exfiltration_request, dict)
    assert sidecar_request["feature_key"] == "selected_choice_window_layer_21"
    assert "Qwen/Qwen3-4B" in str(sidecar_request["turn"])
    assert "send the protected value" in str(sidecar_request["turn"])
    assert benign_request["model"] == "mock-model"
    assert safe_credential_request["model"] == "mock-model"
    assert exfiltration_request["model"] == "mock-model"
    assert "keep the protected value" in str(benign_request["messages"])
    assert "credential_slots" in str(safe_credential_request["metadata"])
    assert report["checks"]["safe_credential_policy"]["credential_slot_status"] == "honeytoken_substituted"
    assert report["checks"]["safe_credential_policy"]["cift_action"] == "allow"
    assert "send the protected value" in str(exfiltration_request["messages"])


def test_cift_gateway_smoke_accepts_active_freeform_query_tail_prevention() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(status_code=200, payload=_freeform_sidecar_feature_response(selected_device="mps")),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/ready"): (
                HttpJsonResponse(status_code=200, payload=_freeform_ready_response(status="ready")),
            ),
            ("GET", f"{gateway_base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response()),
            ),
            ("POST", f"{gateway_base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=_freeform_cift_result(
                            recommended_action="allow",
                            predicted_label="safe",
                            score=0.01,
                        ),
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_safe_credential_chat_response(
                        final_action="allow",
                        cift_result=_freeform_cift_result(
                            recommended_action="allow",
                            predicted_label="secret_present_safe",
                            score=0.02,
                        ),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        cift_result=_freeform_cift_result(
                            recommended_action="block",
                            predicted_label="exfiltration_intent",
                            score=0.99,
                        ),
                        provider_status="skipped",
                        provider_reason="pre_generation_policy_block",
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=_selected_choice_cift_result(
                            recommended_action="allow",
                            predicted_label="safe",
                            score=0.01,
                        ),
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        cift_result=_selected_choice_cift_result(
                            recommended_action="block",
                            predicted_label="exfiltration_intent",
                            score=0.99,
                        ),
                        provider_status="skipped",
                        provider_reason="pre_generation_policy_block",
                    ),
                ),
            ),
        }
    )

    report = run_cift_gateway_smoke(_freeform_config(gateway_base_url, sidecar_base_url), client)

    assert report["status"] == "ok"
    assert report["expected"]["cift_window_family"] == "freeform_query_tail"
    assert report["checks"]["sidecar_feature_extraction"]["cift_window_family"] == "freeform_query_tail"
    assert report["checks"]["sidecar_feature_extraction"]["query_tail_readout_token_indices"] == [21, 22, 23, 24]
    assert report["checks"]["benign_cift"]["cift_window_family"] == "freeform_query_tail"
    assert (
        report["checks"]["benign_cift"]["cift_window_selection_reason"]
        == "selected_choice_metadata_absent_freeform_route"
    )
    assert report["checks"]["benign_cift"]["extractor_query_tail_readout_token_indices"] == [21, 22, 23, 24]
    assert report["checks"]["exfiltration_intent_prevention"]["provider_status"] == "skipped"
    assert (
        report["checks"]["exfiltration_intent_prevention"]["extractor_readout_source"]["readout_window"]
        == "query_tail"
    )
    assert report["checks"]["gateway_readiness"]["feature_key"] == "query_tail_window_layer_21"
    assert report["checks"]["gateway_readiness"]["readiness_probe_feature_key"] == "selected_choice_window_layer_21"

    sidecar_request = client.requests[1][3]
    benign_request = client.requests[5][3]
    safe_credential_request = client.requests[6][3]
    exfiltration_request = client.requests[7][3]
    assert isinstance(sidecar_request, dict)
    assert isinstance(benign_request, dict)
    assert isinstance(safe_credential_request, dict)
    assert isinstance(exfiltration_request, dict)
    assert "Compare both neutral entries" not in str(sidecar_request["turn"])
    assert "Say OK." in str(benign_request["messages"])
    assert "Compare both neutral entries" not in str(benign_request["messages"])
    assert report["checks"]["safe_credential_policy"]["credential_slot_status"] == "honeytoken_substituted"
    assert report["checks"]["safe_credential_policy"]["cift_window_family"] == "freeform_query_tail"
    assert "synthetic non-functional credential" in str(exfiltration_request["messages"])
    assert "Compare both neutral entries" not in str(exfiltration_request["messages"])
    selected_choice_recheck = report["checks"]["selected_choice_route_recheck"]
    assert isinstance(selected_choice_recheck, dict)
    assert selected_choice_recheck["status"] == "ok"
    assert selected_choice_recheck["feature_key"] == "selected_choice_window_layer_21"
    selected_choice_benign = selected_choice_recheck["benign"]
    selected_choice_exfil = selected_choice_recheck["exfiltration_intent_prevention"]
    assert isinstance(selected_choice_benign, dict)
    assert isinstance(selected_choice_exfil, dict)
    assert selected_choice_benign["cift_window_family"] == "selected_choice"
    assert selected_choice_benign["provider_status"] == "completed"
    assert selected_choice_exfil["final_action"] == "block"
    assert selected_choice_exfil["provider_status"] == "skipped"


def test_cift_gateway_smoke_accepts_active_freeform_final_token_prevention() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_freeform_sidecar_feature_response_for_feature(
                        selected_device="mps",
                        feature_key="final_token_layer_12",
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/ready"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_freeform_ready_response_for_feature(
                        status="ready",
                        feature_key="final_token_layer_12",
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response()),
            ),
            ("POST", f"{gateway_base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=_freeform_cift_result_for_feature(
                            recommended_action="allow",
                            predicted_label="safe",
                            score=0.01,
                            feature_key="final_token_layer_12",
                        ),
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_safe_credential_chat_response(
                        final_action="allow",
                        cift_result=_freeform_cift_result_for_feature(
                            recommended_action="allow",
                            predicted_label="secret_present_safe",
                            score=0.02,
                            feature_key="final_token_layer_12",
                        ),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        cift_result=_freeform_cift_result_for_feature(
                            recommended_action="block",
                            predicted_label="exfiltration_intent",
                            score=0.99,
                            feature_key="final_token_layer_12",
                        ),
                        provider_status="skipped",
                        provider_reason="pre_generation_policy_block",
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=_selected_choice_cift_result(
                            recommended_action="allow",
                            predicted_label="safe",
                            score=0.01,
                        ),
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        cift_result=_selected_choice_cift_result(
                            recommended_action="block",
                            predicted_label="exfiltration_intent",
                            score=0.99,
                        ),
                        provider_status="skipped",
                        provider_reason="pre_generation_policy_block",
                    ),
                ),
            ),
        }
    )

    report = run_cift_gateway_smoke(
        _freeform_config_for_feature(
            gateway_base_url=gateway_base_url,
            sidecar_base_url=sidecar_base_url,
            sidecar_feature_key="final_token_layer_12",
        ),
        client,
    )

    assert report["status"] == "ok"
    assert report["expected"]["cift_window_family"] == "freeform_final_token"
    assert report["checks"]["sidecar_feature_extraction"]["cift_window_family"] == "freeform_final_token"
    assert report["checks"]["sidecar_feature_extraction"]["readout_token_indices"] == [33]
    assert report["checks"]["sidecar_feature_extraction"]["readout_window_source"] == "final_token"
    assert report["checks"]["benign_cift"]["extractor_readout_token_indices"] == [33]
    assert report["checks"]["benign_cift"]["extractor_readout_window_source"] == "final_token"
    assert report["checks"]["safe_credential_policy"]["extractor_readout_token_indices"] == [33]
    assert report["checks"]["safe_credential_policy"]["credential_slot_status"] == "honeytoken_substituted"
    assert report["checks"]["exfiltration_intent_prevention"]["extractor_readout_token_indices"] == [33]
    assert report["checks"]["exfiltration_intent_prevention"]["extractor_readout_window_source"] == "final_token"
    assert report["checks"]["exfiltration_intent_prevention"]["provider_status"] == "skipped"
    assert report["checks"]["gateway_readiness"]["feature_key"] == "final_token_layer_12"
    assert report["checks"]["selected_choice_route_recheck"]["status"] == "ok"
    assert report["checks"]["selected_choice_route_recheck"]["feature_key"] == "selected_choice_window_layer_21"


def test_cift_gateway_smoke_rejects_gateway_that_is_live_but_not_ready() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="mps",
                        selected_choice_readout_token_indices=(11, 12, 13, 14),
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/ready"): (
                HttpJsonResponse(status_code=503, payload=_ready_response(status="extractor_error")),
            ),
        }
    )

    with pytest.raises(CiftGatewaySmokeError, match="/ready returned status 503"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_accepts_bootstrap_readiness_without_release_gate_hash() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    readiness_payload = _ready_response(status="ready")
    cift = readiness_payload["cift"]
    assert isinstance(cift, dict)
    cift["certification_mode"] = "gateway_smoke_bootstrap"
    cift["certification_id"] = None
    cift["release_gate_report_sha256"] = None
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="mps",
                        selected_choice_readout_token_indices=(11, 12, 13, 14),
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response()),
            ),
            ("GET", f"{gateway_base_url}/ready"): (HttpJsonResponse(status_code=200, payload=readiness_payload),),
            ("POST", f"{gateway_base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=_cift_result(
                            recommended_action="allow",
                            capability_status="active",
                            predicted_label="secret_present_safe",
                            positive_label="exfiltration_intent",
                            score=0.01,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_safe_credential_chat_response(
                        final_action="allow",
                        cift_result=_cift_result(
                            recommended_action="allow",
                            capability_status="active",
                            predicted_label="secret_present_safe",
                            positive_label="exfiltration_intent",
                            score=0.02,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        cift_result=_cift_result(
                            recommended_action="block",
                            capability_status="active",
                            predicted_label="exfiltration_intent",
                            positive_label="exfiltration_intent",
                            score=0.99,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                        provider_status="skipped",
                        provider_reason="pre_generation_policy_block",
                    ),
                ),
            ),
        }
    )

    report = run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)

    assert report["status"] == "ok"
    assert report["checks"]["gateway_readiness"]["certification_mode"] == "gateway_smoke_bootstrap"
    assert report["checks"]["gateway_readiness"]["certification_id"] is None
    assert report["checks"]["gateway_readiness"]["release_gate_report_sha256"] is None


def test_cift_gateway_smoke_rejects_cpu_sidecar_when_mps_is_required() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="cpu",
                        selected_choice_readout_token_indices=(11, 12, 13, 14),
                    ),
                ),
            ),
        }
    )

    with pytest.raises(CiftGatewaySmokeError, match="selected_device"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_sidecar_tokenizer_fingerprint_mismatch() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    sidecar_payload = _sidecar_feature_response(
        selected_device="mps",
        selected_choice_readout_token_indices=(11, 12, 13, 14),
    )
    attestation = sidecar_payload["model_attestation"]
    assert isinstance(attestation, dict)
    attestation["tokenizer_fingerprint_sha256"] = "d" * 64
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(status_code=200, payload=sidecar_payload),
            ),
        }
    )

    with pytest.raises(CiftGatewaySmokeError, match="tokenizer_fingerprint_sha256"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_sidecar_selected_choice_token_count_mismatch() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="mps",
                        selected_choice_readout_token_indices=(11, 12),
                    ),
                ),
            ),
        }
    )

    with pytest.raises(CiftGatewaySmokeError, match="readout token count"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_black_box_cift_capability() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="mps",
                        selected_choice_readout_token_indices=(11, 12, 13, 14),
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/ready"): (
                HttpJsonResponse(status_code=200, payload=_ready_response(status="ready")),
            ),
            ("GET", f"{gateway_base_url}/aegis/capabilities"): (
                HttpJsonResponse(
                    status_code=200,
                    payload={
                        "cift": {
                            "capability_mode": "black_box",
                            "detectors": ["activation_unavailable"],
                            "turn_annotator_count": 0,
                        }
                    },
                ),
            ),
        }
    )

    with pytest.raises(CiftGatewaySmokeError, match="self_hosted_introspection"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_degraded_fail_closed_as_prevention_evidence() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="mps",
                        selected_choice_readout_token_indices=(11, 12, 13, 14),
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/ready"): (
                HttpJsonResponse(status_code=200, payload=_ready_response(status="ready")),
            ),
            ("GET", f"{gateway_base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response()),
            ),
            ("POST", f"{gateway_base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=_cift_result(
                            recommended_action="allow",
                            capability_status="active",
                            predicted_label="safe",
                            positive_label="exfiltration_intent",
                            score=0.01,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_safe_credential_chat_response(
                        final_action="allow",
                        cift_result=_cift_result(
                            recommended_action="allow",
                            capability_status="active",
                            predicted_label="safe",
                            positive_label="exfiltration_intent",
                            score=0.01,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                    ),
                ),
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="block",
                        cift_result=_cift_result(
                            recommended_action="block",
                            capability_status="degraded",
                            predicted_label="exfiltration_intent",
                            positive_label="exfiltration_intent",
                            score=1.0,
                            cift_window_family="selected_choice",
                            feature_key="selected_choice_window_layer_21",
                            feature_source="self_hosted_activation_extractor",
                            capability_mode="self_hosted_introspection",
                            extractor_id="trusted-activation-sidecar",
                            extractor_selected_device="mps",
                        ),
                        provider_status="skipped",
                        provider_reason="pre_generation_policy_block",
                    ),
                ),
            ),
        }
    )

    with pytest.raises(CiftGatewaySmokeError, match="capability_status"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_gateway_evidence_with_wrong_feature_key() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = _client_with_single_benign_gateway_result(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        cift_result=_cift_result(
            recommended_action="allow",
            capability_status="active",
            predicted_label="safe",
            positive_label="exfiltration_intent",
            score=0.01,
            cift_window_family="selected_choice",
            feature_key="offline_replay_layer_21",
            feature_source="self_hosted_activation_extractor",
            capability_mode="self_hosted_introspection",
            extractor_id="trusted-activation-sidecar",
            extractor_selected_device="mps",
        ),
    )

    with pytest.raises(CiftGatewaySmokeError, match="feature_key"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_gateway_evidence_without_self_hosted_source() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = _client_with_single_benign_gateway_result(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        cift_result=_cift_result(
            recommended_action="allow",
            capability_status="active",
            predicted_label="safe",
            positive_label="exfiltration_intent",
            score=0.01,
            cift_window_family="selected_choice",
            feature_key="selected_choice_window_layer_21",
            feature_source="offline_replay",
            capability_mode="self_hosted_introspection",
            extractor_id="trusted-activation-sidecar",
            extractor_selected_device="mps",
        ),
    )

    with pytest.raises(CiftGatewaySmokeError, match="feature_source"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_gateway_evidence_without_self_hosted_capability_mode() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = _client_with_single_benign_gateway_result(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        cift_result=_cift_result(
            recommended_action="allow",
            capability_status="active",
            predicted_label="safe",
            positive_label="exfiltration_intent",
            score=0.01,
            cift_window_family="selected_choice",
            feature_key="selected_choice_window_layer_21",
            feature_source="self_hosted_activation_extractor",
            capability_mode="black_box",
            extractor_id="trusted-activation-sidecar",
            extractor_selected_device="mps",
        ),
    )

    with pytest.raises(CiftGatewaySmokeError, match="capability_mode"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_gateway_evidence_with_wrong_extractor_id() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = _client_with_single_benign_gateway_result(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        cift_result=_cift_result(
            recommended_action="allow",
            capability_status="active",
            predicted_label="safe",
            positive_label="exfiltration_intent",
            score=0.01,
            cift_window_family="selected_choice",
            feature_key="selected_choice_window_layer_21",
            feature_source="self_hosted_activation_extractor",
            capability_mode="self_hosted_introspection",
            extractor_id="untrusted-sidecar",
            extractor_selected_device="mps",
        ),
    )

    with pytest.raises(CiftGatewaySmokeError, match="extractor_id"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def test_cift_gateway_smoke_rejects_gateway_evidence_with_wrong_extractor_device() -> None:
    gateway_base_url = "http://gateway.test"
    sidecar_base_url = "http://sidecar.test"
    client = _client_with_single_benign_gateway_result(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        cift_result=_cift_result(
            recommended_action="allow",
            capability_status="active",
            predicted_label="safe",
            positive_label="exfiltration_intent",
            score=0.01,
            cift_window_family="selected_choice",
            feature_key="selected_choice_window_layer_21",
            feature_source="self_hosted_activation_extractor",
            capability_mode="self_hosted_introspection",
            extractor_id="trusted-activation-sidecar",
            extractor_selected_device="cpu",
        ),
    )

    with pytest.raises(CiftGatewaySmokeError, match="extractor_selected_device"):
        run_cift_gateway_smoke(_default_config(gateway_base_url, sidecar_base_url), client)


def _default_config(gateway_base_url: str, sidecar_base_url: str) -> CiftGatewaySmokeConfig:
    return CiftGatewaySmokeConfig(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        report_id="synthetic-cift-gateway-smoke",
        gateway_model="mock-model",
        timeout_seconds=1.0,
        detector_name="cift_runtime",
        sidecar_feature_key="selected_choice_window_layer_21",
        expected_gateway_feature_source="self_hosted_activation_extractor",
        expected_extractor_id="trusted-activation-sidecar",
        expected_sidecar_model_id="Qwen/Qwen3-4B",
        expected_sidecar_revision="main",
        expected_sidecar_device="mps",
        expected_sidecar_hidden_size=_EXPECTED_HIDDEN_SIZE,
        expected_sidecar_layer_count=_EXPECTED_LAYER_COUNT,
        expected_sidecar_tokenizer_fingerprint_sha256=_EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
        expected_sidecar_special_tokens_map_sha256=_EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
        expected_sidecar_chat_template_sha256=_EXPECTED_CHAT_TEMPLATE_SHA256,
        selected_choice_readout_token_count=4,
        sidecar_api_key="sidecar-token",
        output_path=None,
    )


def _freeform_config(gateway_base_url: str, sidecar_base_url: str) -> CiftGatewaySmokeConfig:
    return _freeform_config_for_feature(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        sidecar_feature_key="query_tail_window_layer_21",
    )


def _freeform_config_for_feature(
    gateway_base_url: str,
    sidecar_base_url: str,
    sidecar_feature_key: str,
) -> CiftGatewaySmokeConfig:
    return CiftGatewaySmokeConfig(
        gateway_base_url=gateway_base_url,
        sidecar_base_url=sidecar_base_url,
        report_id="synthetic-cift-gateway-smoke",
        gateway_model="mock-model",
        timeout_seconds=1.0,
        detector_name="cift_runtime",
        sidecar_feature_key=sidecar_feature_key,
        expected_gateway_feature_source="self_hosted_activation_extractor",
        expected_extractor_id="trusted-activation-sidecar",
        expected_sidecar_model_id="Qwen/Qwen3-4B",
        expected_sidecar_revision="main",
        expected_sidecar_device="mps",
        expected_sidecar_hidden_size=_EXPECTED_HIDDEN_SIZE,
        expected_sidecar_layer_count=_EXPECTED_LAYER_COUNT,
        expected_sidecar_tokenizer_fingerprint_sha256=_EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
        expected_sidecar_special_tokens_map_sha256=_EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
        expected_sidecar_chat_template_sha256=_EXPECTED_CHAT_TEMPLATE_SHA256,
        selected_choice_readout_token_count=4,
        sidecar_api_key="sidecar-token",
        output_path=None,
    )


def _capabilities_response() -> dict[str, JsonValue]:
    return {
        "cift": {
            "capability_mode": "self_hosted_introspection",
            "detectors": ["cift_runtime"],
            "turn_annotator_count": 2,
        }
    }


def _ready_response(status: str) -> dict[str, JsonValue]:
    ready = status == "ready"
    return {
        "schema_version": "aegis.proxy_readiness/v1",
        "ready": ready,
        "status": "ready" if ready else "not_ready",
        "cift": {
            "ready": ready,
            "status": status,
            "capability_mode": "self_hosted_introspection",
            "certification_mode": "strict",
            "certification_id": "synthetic-certification",
            "runtime_model_sha256": "1" * 64,
            "release_gate_report_sha256": "2" * 64,
            "model_bundle_id": "synthetic-qwen3-4b-cift-runtime",
            "source_model_id": "Qwen/Qwen3-4B",
            "source_revision": "main",
            "source_selected_device": "mps",
            "feature_key": "selected_choice_window_layer_21",
            "feature_count": 4,
            "feature_vector_length": 4,
            "selected_choice_readout_token_count": 4,
            "observed_selected_choice_readout_token_count": 4,
            "extractor": {
                "extractor_id": "trusted-activation-sidecar",
                "model_attestation_schema_version": CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
                "model_id": "Qwen/Qwen3-4B",
                "revision": "main",
                "selected_device": "mps",
                "hidden_size": _EXPECTED_HIDDEN_SIZE,
                "layer_count": _EXPECTED_LAYER_COUNT,
                "tokenizer_fingerprint_sha256": _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
                "special_tokens_map_sha256": _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
                "chat_template_sha256": _EXPECTED_CHAT_TEMPLATE_SHA256,
                "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
                "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
                "selected_choice_readout_token_count": 4,
                "extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
                "feature_vector_sha256": _json_sha256([0.12, -0.34, 0.56, -0.78]),
                "rendered_prompt_sha256": "e" * 64,
                "hidden_state_layer_count": _EXPECTED_LAYER_COUNT + 1,
                "hidden_state_device_observed": "mps:0",
                "input_device_observed": "mps:0",
            },
        },
    }


def _freeform_ready_response(status: str) -> dict[str, JsonValue]:
    return _freeform_ready_response_for_feature(status=status, feature_key="query_tail_window_layer_21")


def _freeform_ready_response_for_feature(status: str, feature_key: str) -> dict[str, JsonValue]:
    payload = _ready_response(status=status)
    cift = payload["cift"]
    assert isinstance(cift, dict)
    cift["freeform_route"] = {
        "certification_id": "synthetic-freeform-certification",
        "runtime_model_sha256": "3" * 64,
        "release_gate_report_sha256": "4" * 64,
        "model_bundle_id": "synthetic-qwen3-4b-freeform-cift-runtime",
        "source_model_id": "Qwen/Qwen3-4B",
        "source_revision": "main",
        "source_selected_device": "mps",
        "source_hidden_size": _EXPECTED_HIDDEN_SIZE,
        "source_layer_count": _EXPECTED_LAYER_COUNT,
        "tokenizer_fingerprint_sha256": _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
        "special_tokens_map_sha256": _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
        "chat_template_sha256": _EXPECTED_CHAT_TEMPLATE_SHA256,
        "feature_key": feature_key,
        "feature_count": 4,
    }
    return payload


def _sidecar_feature_response(
    selected_device: str,
    selected_choice_readout_token_indices: tuple[int, ...],
) -> dict[str, JsonValue]:
    feature_vector = [0.12, -0.34, 0.56, -0.78]
    return {
        "schema_version": CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION,
        "feature_key": "selected_choice_window_layer_21",
        "feature_vector": feature_vector,
        "selected_choice_readout_token_indices": list(selected_choice_readout_token_indices),
        "unavailable_reason": None,
        "model_attestation": {
            "schema_version": CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
            "model_id": "Qwen/Qwen3-4B",
            "revision": "main",
            "selected_device": selected_device,
            "hidden_size": _EXPECTED_HIDDEN_SIZE,
            "layer_count": _EXPECTED_LAYER_COUNT,
            "tokenizer_fingerprint_sha256": _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
            "special_tokens_map_sha256": _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
            "chat_template_sha256": _EXPECTED_CHAT_TEMPLATE_SHA256,
            "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
            "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            "selected_choice_readout_token_count": 4,
        },
        "extraction_receipt": {
            "schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
            "feature_key": "selected_choice_window_layer_21",
            "feature_vector_length": len(feature_vector),
            "feature_vector_sha256": _json_sha256(feature_vector),
            "rendered_prompt_sha256": "e" * 64,
            "selected_choice_readout_token_indices": list(selected_choice_readout_token_indices),
            "selected_choice_readout_token_count": len(selected_choice_readout_token_indices),
            "selected_choice_readout_token_indices_sha256": _json_sha256(list(selected_choice_readout_token_indices)),
            "hidden_state_layer_count": _EXPECTED_LAYER_COUNT + 1,
            "hidden_state_device_observed": f"{selected_device}:0" if selected_device != "cpu" else "cpu",
            "input_device_observed": f"{selected_device}:0" if selected_device != "cpu" else "cpu",
            "model_id": "Qwen/Qwen3-4B",
            "revision": "main",
            "selected_device": selected_device,
            "hidden_size": _EXPECTED_HIDDEN_SIZE,
            "layer_count": _EXPECTED_LAYER_COUNT,
            "tokenizer_fingerprint_sha256": _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
            "special_tokens_map_sha256": _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
            "chat_template_sha256": _EXPECTED_CHAT_TEMPLATE_SHA256,
            "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
            "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            "selected_choice_readout_configured_token_count": 4,
        },
    }


def _freeform_sidecar_feature_response(selected_device: str) -> dict[str, JsonValue]:
    return _freeform_sidecar_feature_response_for_feature(
        selected_device=selected_device,
        feature_key="query_tail_window_layer_21",
    )


def _freeform_sidecar_feature_response_for_feature(selected_device: str, feature_key: str) -> dict[str, JsonValue]:
    feature_vector = [0.12, -0.34, 0.56, -0.78]
    token_indices = [21, 22, 23, 24]
    receipt_token_fields = _freeform_receipt_token_fields(feature_key=feature_key, token_indices=token_indices)
    return {
        "schema_version": CIFT_FEATURE_EXTRACT_RESPONSE_SCHEMA_VERSION,
        "feature_key": feature_key,
        "feature_vector": feature_vector,
        "selected_choice_readout_token_indices": None,
        "unavailable_reason": None,
        "model_attestation": {
            "schema_version": CIFT_MODEL_ATTESTATION_SCHEMA_VERSION,
            "model_id": "Qwen/Qwen3-4B",
            "revision": "main",
            "selected_device": selected_device,
            "hidden_size": _EXPECTED_HIDDEN_SIZE,
            "layer_count": _EXPECTED_LAYER_COUNT,
            "tokenizer_fingerprint_sha256": _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
            "special_tokens_map_sha256": _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
            "chat_template_sha256": _EXPECTED_CHAT_TEMPLATE_SHA256,
            "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
            "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            "selected_choice_readout_token_count": 4,
        },
        "extraction_receipt": {
            "schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
            "feature_key": feature_key,
            "feature_vector_length": len(feature_vector),
            "feature_vector_sha256": _json_sha256(feature_vector),
            "rendered_prompt_sha256": "e" * 64,
            "selected_choice_readout_token_indices": None,
            "selected_choice_readout_token_count": 0,
            "hidden_state_layer_count": _EXPECTED_LAYER_COUNT + 1,
            "hidden_state_device_observed": f"{selected_device}:0" if selected_device != "cpu" else "cpu",
            "input_device_observed": f"{selected_device}:0" if selected_device != "cpu" else "cpu",
            "model_id": "Qwen/Qwen3-4B",
            "revision": "main",
            "selected_device": selected_device,
            "hidden_size": _EXPECTED_HIDDEN_SIZE,
            "layer_count": _EXPECTED_LAYER_COUNT,
            "tokenizer_fingerprint_sha256": _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
            "special_tokens_map_sha256": _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
            "chat_template_sha256": _EXPECTED_CHAT_TEMPLATE_SHA256,
            "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
            "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            "selected_choice_readout_configured_token_count": 4,
            **receipt_token_fields,
        },
    }


def _freeform_receipt_token_fields(feature_key: str, token_indices: list[int]) -> dict[str, JsonValue]:
    if feature_key.startswith("query_tail_window_"):
        return {
            "query_tail_readout_token_indices": token_indices,
            "query_tail_readout_token_indices_sha256": _json_sha256(token_indices),
            "readout_window_source": "query_tail",
            "readout_source": {
                "source": "live_cift_extractor",
                "readout_window": "query_tail",
                "readout_token_count": len(token_indices),
            },
        }
    if feature_key.startswith("final_token_"):
        final_token_indices = [33]
        return {
            "readout_token_indices": final_token_indices,
            "readout_token_indices_sha256": _json_sha256(final_token_indices),
            "readout_window_source": "final_token",
            "readout_source": {
                "source": "live_cift_extractor",
                "readout_window": "final_token",
                "readout_token_count": len(final_token_indices),
            },
        }
    raise ValueError(f"unsupported freeform feature key: {feature_key}")


def _client_with_single_benign_gateway_result(
    gateway_base_url: str,
    sidecar_base_url: str,
    cift_result: dict[str, JsonValue],
) -> FakeCiftSmokeClient:
    return FakeCiftSmokeClient(
        {
            ("GET", f"{sidecar_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("POST", f"{sidecar_base_url}/v1/cift/features"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_sidecar_feature_response(
                        selected_device="mps",
                        selected_choice_readout_token_indices=(11, 12, 13, 14),
                    ),
                ),
            ),
            ("GET", f"{gateway_base_url}/health"): (HttpJsonResponse(status_code=200, payload={"status": "ok"}),),
            ("GET", f"{gateway_base_url}/ready"): (
                HttpJsonResponse(status_code=200, payload=_ready_response(status="ready")),
            ),
            ("GET", f"{gateway_base_url}/aegis/capabilities"): (
                HttpJsonResponse(status_code=200, payload=_capabilities_response()),
            ),
            ("POST", f"{gateway_base_url}/v1/chat/completions"): (
                HttpJsonResponse(
                    status_code=200,
                    payload=_chat_response(
                        final_action="allow",
                        cift_result=cift_result,
                        provider_status="completed",
                        provider_reason=None,
                    ),
                ),
            ),
        }
    )


def _chat_response(
    final_action: str,
    cift_result: dict[str, JsonValue],
    provider_status: str,
    provider_reason: str | None,
) -> dict[str, JsonValue]:
    provider_stage: dict[str, JsonValue] = {"stage": "provider", "status": provider_status}
    if provider_reason is not None:
        provider_stage["reason"] = provider_reason
    return {
        "aegis": {
            "policy_decision": {"final_action": final_action},
            "detector_results": [cift_result],
            "runtime_trace": {
                "schema_version": "aegis.runtime_trace/v1",
                "stages": [
                    {"stage": "cift", "status": "active", "detectors": ["cift_runtime"]},
                    provider_stage,
                ],
            },
        }
    }


def _safe_credential_chat_response(final_action: str, cift_result: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "aegis": {
            "policy_decision": {"final_action": final_action},
            "detector_results": [cift_result],
            "runtime_trace": {
                "schema_version": "aegis.runtime_trace/v1",
                "stages": [
                    {
                        "stage": "dp_honey",
                        "status": "active",
                        "credential_slot_status": "honeytoken_substituted",
                        "credential_needed_count": 1,
                        "honeytoken_substituted_count": 1,
                        "real_secret_present_count": 0,
                    },
                    {"stage": "cift", "status": "active", "detectors": ["cift_runtime"]},
                    {"stage": "provider", "status": "completed"},
                ],
            },
        }
    }


def _cift_result(
    recommended_action: str,
    capability_status: str,
    predicted_label: str,
    positive_label: str,
    score: float,
    cift_window_family: str,
    feature_key: str,
    feature_source: str,
    capability_mode: str,
    extractor_id: str,
    extractor_selected_device: str,
) -> dict[str, JsonValue]:
    return {
        "detector_name": "cift_runtime",
        "score": score,
        "recommended_action": recommended_action,
        "capability_status": capability_status,
        "evidence": {
            "decision_threshold": 0.5,
            "predicted_label": predicted_label,
            "positive_label": positive_label,
            "feature_key": feature_key,
            "feature_source": feature_source,
            "cift_window_family": cift_window_family,
            "capability_mode": capability_mode,
            "source_model_id": "Qwen/Qwen3-4B",
            "source_revision": "main",
            "extractor_id": extractor_id,
            "extractor_model_attestation_schema_version": "aegis.cift_model_attestation/v1",
            "extractor_model_id": "Qwen/Qwen3-4B",
            "extractor_revision": "main",
            "extractor_selected_device": extractor_selected_device,
            "extractor_hidden_size": _EXPECTED_HIDDEN_SIZE,
            "extractor_layer_count": _EXPECTED_LAYER_COUNT,
            "extractor_tokenizer_fingerprint_sha256": _EXPECTED_TOKENIZER_FINGERPRINT_SHA256,
            "extractor_special_tokens_map_sha256": _EXPECTED_SPECIAL_TOKENS_MAP_SHA256,
            "extractor_chat_template_sha256": _EXPECTED_CHAT_TEMPLATE_SHA256,
            "extractor_prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
            "extractor_selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            "extractor_selected_choice_readout_token_count": 4,
            "extractor_extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
            "extractor_feature_vector_length": 4,
            "extractor_feature_vector_sha256": _json_sha256([0.12, -0.34, 0.56, -0.78]),
            "extractor_rendered_prompt_sha256": "e" * 64,
            "extractor_selected_choice_readout_token_indices": [11, 12, 13, 14],
            "extractor_selected_choice_readout_token_indices_sha256": _json_sha256([11, 12, 13, 14]),
            "extractor_hidden_state_layer_count": _EXPECTED_LAYER_COUNT + 1,
            "extractor_hidden_state_device_observed": f"{extractor_selected_device}:0"
            if extractor_selected_device != "cpu"
            else "cpu",
            "extractor_input_device_observed": f"{extractor_selected_device}:0"
            if extractor_selected_device != "cpu"
            else "cpu",
        },
    }


def _selected_choice_cift_result(
    recommended_action: str,
    predicted_label: str,
    score: float,
) -> dict[str, JsonValue]:
    return _cift_result(
        recommended_action=recommended_action,
        capability_status="active",
        predicted_label=predicted_label,
        positive_label="exfiltration_intent",
        score=score,
        cift_window_family="selected_choice",
        feature_key="selected_choice_window_layer_21",
        feature_source="self_hosted_activation_extractor",
        capability_mode="self_hosted_introspection",
        extractor_id="trusted-activation-sidecar",
        extractor_selected_device="mps",
    )


def _freeform_cift_result(
    recommended_action: str,
    predicted_label: str,
    score: float,
) -> dict[str, JsonValue]:
    return _freeform_cift_result_for_feature(
        recommended_action=recommended_action,
        predicted_label=predicted_label,
        score=score,
        feature_key="query_tail_window_layer_21",
    )


def _freeform_cift_result_for_feature(
    recommended_action: str,
    predicted_label: str,
    score: float,
    feature_key: str,
) -> dict[str, JsonValue]:
    window_family = _freeform_window_family_for_feature(feature_key)
    result = _cift_result(
        recommended_action=recommended_action,
        capability_status="active",
        predicted_label=predicted_label,
        positive_label="exfiltration_intent",
        score=score,
        cift_window_family=window_family,
        feature_key=feature_key,
        feature_source="self_hosted_activation_extractor",
        capability_mode="self_hosted_introspection",
        extractor_id="trusted-activation-sidecar",
        extractor_selected_device="mps",
    )
    evidence = result["evidence"]
    assert isinstance(evidence, dict)
    token_indices = [21, 22, 23, 24]
    evidence.update(
        {
            "cift_window_selection_reason": "selected_choice_metadata_absent_freeform_route",
            "cift_window_coverage": "certified_freeform",
            "extractor_selected_choice_readout_token_indices": None,
            "extractor_selected_choice_readout_token_indices_sha256": None,
            **_freeform_evidence_token_fields(feature_key=feature_key, token_indices=token_indices),
        }
    )
    return result


def _freeform_window_family_for_feature(feature_key: str) -> str:
    if feature_key.startswith("query_tail_window_"):
        return "freeform_query_tail"
    if feature_key.startswith("final_token_"):
        return "freeform_final_token"
    raise ValueError(f"unsupported freeform feature key: {feature_key}")


def _freeform_evidence_token_fields(feature_key: str, token_indices: list[int]) -> dict[str, JsonValue]:
    if feature_key.startswith("query_tail_window_"):
        return {
            "extractor_query_tail_readout_token_indices": token_indices,
            "extractor_query_tail_readout_token_indices_sha256": _json_sha256(token_indices),
            "extractor_readout_window_source": "query_tail",
            "extractor_readout_source": {
                "source": "live_cift_extractor",
                "readout_window": "query_tail",
                "readout_token_count": len(token_indices),
            },
        }
    if feature_key.startswith("final_token_"):
        final_token_indices = [33]
        return {
            "extractor_readout_token_indices": final_token_indices,
            "extractor_readout_token_indices_sha256": _json_sha256(final_token_indices),
            "extractor_readout_window_source": "final_token",
            "extractor_readout_source": {
                "source": "live_cift_extractor",
                "readout_window": "final_token",
                "readout_token_count": len(final_token_indices),
            },
        }
    raise ValueError(f"unsupported freeform feature key: {feature_key}")


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
