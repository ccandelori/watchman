from __future__ import annotations

import hashlib
import json

import pytest

from aegis.cift_contract import (
    CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)
from aegis.core.contracts import CapabilityMode, Message, ModelInfo, NormalizedTurn
from aegis.proxy.cift_extractor_client import (
    CiftExpectedModelAttestation,
    CiftHttpExtractorConfig,
    CiftHttpExtractorError,
    CiftHttpFeatureExtractor,
    urllib_cift_extractor_sender,
)


def test_cift_http_extractor_posts_turn_for_feature_and_selected_choice_metadata() -> None:
    calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        calls.append((url, payload, headers, timeout_seconds))
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [2.0, 3],
            "selected_choice_readout_token_indices": [7, 8],
            "model_attestation": _attestation_payload(),
            "extraction_receipt": _extraction_receipt_payload(feature_vector=(2.0, 3.0), token_indices=(7, 8)),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000",
            api_key="sidecar-token",
            timeout_seconds=2.5,
        ),
        sender=sender,
    )
    turn = _turn()

    feature_vector = extractor.extract_feature_vector(turn=turn, feature_key="selected_choice_window_layer_19")
    token_indices = extractor.extract_selected_choice_readout_token_indices(
        turn=turn,
        feature_key="selected_choice_window_layer_19",
    )

    assert feature_vector == (2.0, 3.0)
    assert token_indices == (7, 8)
    assert len(calls) == 2
    url, payload, headers, timeout_seconds = calls[0]
    assert url == "http://127.0.0.1:9000/v1/cift/features"
    assert payload["schema_version"] == "aegis.cift_feature_extract_request/v1"
    assert payload["feature_key"] == "selected_choice_window_layer_19"
    assert headers == {"Content-Type": "application/json", "Authorization": "Bearer sidecar-token"}
    assert timeout_seconds == 2.5


def test_cift_http_extractor_returns_validated_provenance_with_feature_extraction() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [2.0, 3],
            "selected_choice_readout_token_indices": [7, 8],
            "model_attestation": _attestation_payload(),
            "extraction_receipt": _extraction_receipt_payload(feature_vector=(2.0, 3.0), token_indices=(7, 8)),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000",
            api_key="sidecar-token",
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    extraction = extractor.extract_feature_extraction(turn=_turn(), feature_key="selected_choice_window_layer_19")

    assert extraction.feature_vector == (2.0, 3.0)
    assert extraction.selected_choice_readout_token_indices == (7, 8)
    assert extraction.provenance == {
        "extractor_id": "trusted-activation-sidecar",
        "model_attestation_schema_version": "aegis.cift_model_attestation/v1",
        "model_id": "Qwen/Qwen3-test",
        "revision": "main",
        "selected_device": "mps",
        "hidden_size": 2560,
        "layer_count": 36,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        "selected_choice_readout_token_count": 2,
        "extraction_receipt_schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
        "feature_vector_length": 2,
        "feature_vector_sha256": _json_sha256([2.0, 3.0]),
        "rendered_prompt_sha256": "e" * 64,
        "selected_choice_readout_token_indices": [7, 8],
        "selected_choice_readout_token_indices_sha256": _json_sha256([7, 8]),
        "hidden_state_layer_count": 37,
        "hidden_state_device_observed": "mps:0",
        "input_device_observed": "mps:0",
    }


def test_cift_http_extractor_does_not_replay_response_for_different_content_with_reused_ids() -> None:
    calls: list[dict[str, object]] = []

    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        calls.append(payload)
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [float(len(calls))],
            "selected_choice_readout_token_indices": [7, 8],
            "model_attestation": _attestation_payload(),
            "extraction_receipt": _extraction_receipt_payload(
                feature_vector=(float(len(calls)),),
                token_indices=(7, 8),
            ),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    first_feature_vector = extractor.extract_feature_vector(
        turn=_turn_with_content(content="benign request"),
        feature_key="selected_choice_window_layer_19",
    )
    second_feature_vector = extractor.extract_feature_vector(
        turn=_turn_with_content(content="malicious request with reused ids"),
        feature_key="selected_choice_window_layer_19",
    )

    assert first_feature_vector == (1.0,)
    assert second_feature_vector == (2.0,)
    assert len(calls) == 2
    first_turn = calls[0]["turn"]
    second_turn = calls[1]["turn"]
    assert isinstance(first_turn, dict)
    assert isinstance(second_turn, dict)
    assert first_turn["trace_id"] == second_turn["trace_id"]
    assert first_turn["session_id"] == second_turn["session_id"]
    assert first_turn["turn_index"] == second_turn["turn_index"]
    assert first_turn["messages"] != second_turn["messages"]


def test_cift_http_extractor_rejects_mismatched_response_feature_key() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "other_feature",
            "feature_vector": [2.0, 3.0],
            "model_attestation": _attestation_payload(),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="feature_key"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_cift_http_extractor_rejects_malformed_feature_values() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": ["bad"],
            "model_attestation": _attestation_payload(),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="feature_vector"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_cift_http_extractor_rejects_non_finite_feature_values() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [float("nan")],
            "model_attestation": _attestation_payload(),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="finite"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_cift_http_extractor_rejects_missing_model_attestation() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [2.0, 3.0],
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="model_attestation"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_cift_http_extractor_rejects_model_attestation_device_mismatch() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        attestation = _attestation_payload()
        attestation["selected_device"] = "cpu"
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [2.0, 3.0],
            "model_attestation": attestation,
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="selected_device"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_cift_http_extractor_preserves_unavailable_reason_for_missing_feature_vector() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": None,
            "selected_choice_readout_token_indices": None,
            "unavailable_reason": "selected_choice_geometry_missing",
            "model_attestation": _attestation_payload(),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="selected_choice_geometry_missing"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")
    with pytest.raises(CiftHttpExtractorError, match="selected_choice_geometry_missing"):
        extractor.extract_feature_extraction(turn=_turn(), feature_key="selected_choice_window_layer_19")


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("model_id", "Qwen/Qwen3-other"),
        ("revision", "other-revision"),
        ("hidden_size", 1280),
        ("layer_count", 24),
        ("tokenizer_fingerprint_sha256", "d" * 64),
        ("special_tokens_map_sha256", "e" * 64),
        ("chat_template_sha256", "f" * 64),
        ("prompt_renderer", "different_renderer"),
        ("selected_choice_geometry", "different_geometry"),
    ),
)
def test_cift_http_extractor_rejects_model_attestation_identity_mismatch(
    field_name: str,
    field_value: object,
) -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        attestation = _attestation_payload()
        attestation[field_name] = field_value
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [2.0, 3.0],
            "model_attestation": attestation,
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match=field_name):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_cift_http_extractor_rejects_model_attestation_readout_count_mismatch() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        attestation = _attestation_payload()
        attestation["selected_choice_readout_token_count"] = 4
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [2.0, 3.0],
            "selected_choice_readout_token_indices": [7, 8],
            "model_attestation": attestation,
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="selected_choice_readout_token_count"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_cift_http_extractor_rejects_selected_choice_index_count_mismatch() -> None:
    def sender(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "schema_version": "aegis.cift_feature_extract_response/v1",
            "feature_key": "selected_choice_window_layer_19",
            "feature_vector": [2.0, 3.0],
            "selected_choice_readout_token_indices": [7, 8, 9],
            "model_attestation": _attestation_payload(),
        }

    extractor = CiftHttpFeatureExtractor(
        config=_http_config(
            base_url="http://127.0.0.1:9000/v1/cift/features",
            api_key=None,
            timeout_seconds=2.5,
        ),
        sender=sender,
    )

    with pytest.raises(CiftHttpExtractorError, match="selected_choice_readout_token_indices length"):
        extractor.extract_feature_vector(turn=_turn(), feature_key="selected_choice_window_layer_19")


def test_urllib_cift_extractor_sender_rejects_non_finite_json_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"schema_version":"aegis.cift_feature_extract_response/v1",'
                b'"feature_key":"selected_choice_window_layer_19",'
                b'"feature_vector":[NaN]}'
            )

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("aegis.proxy.cift_extractor_client.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(CiftHttpExtractorError, match="invalid JSON"):
        urllib_cift_extractor_sender(
            url="http://127.0.0.1:9000/v1/cift/features",
            payload={
                "schema_version": "aegis.cift_feature_extract_request/v1",
                "feature_key": "selected_choice_window_layer_19",
                "turn": {},
            },
            headers={},
            timeout_seconds=2.5,
        )


def _turn() -> NormalizedTurn:
    return _turn_with_content(content="hello")


def _http_config(base_url: str, api_key: str | None, timeout_seconds: float) -> CiftHttpExtractorConfig:
    return CiftHttpExtractorConfig(
        extractor_id="trusted-activation-sidecar",
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        expected_attestation=CiftExpectedModelAttestation(
            model_id="Qwen/Qwen3-test",
            revision="main",
            selected_device="mps",
            hidden_size=2560,
            layer_count=36,
            tokenizer_fingerprint_sha256="a" * 64,
            special_tokens_map_sha256="b" * 64,
            chat_template_sha256="c" * 64,
            prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
            selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
            selected_choice_readout_token_count=2,
        ),
    )


def _attestation_payload() -> dict[str, object]:
    return {
        "schema_version": "aegis.cift_model_attestation/v1",
        "model_id": "Qwen/Qwen3-test",
        "revision": "main",
        "selected_device": "mps",
        "hidden_size": 2560,
        "layer_count": 36,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        "selected_choice_readout_token_count": 2,
    }


def _extraction_receipt_payload(
    feature_vector: tuple[float, ...],
    token_indices: tuple[int, ...] | None,
) -> dict[str, object]:
    return {
        "schema_version": CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
        "feature_key": "selected_choice_window_layer_19",
        "feature_vector_length": len(feature_vector),
        "feature_vector_sha256": _json_sha256([float(value) for value in feature_vector]),
        "rendered_prompt_sha256": "e" * 64,
        "selected_choice_readout_token_indices": None if token_indices is None else list(token_indices),
        "selected_choice_readout_token_count": 0 if token_indices is None else len(token_indices),
        "selected_choice_readout_token_indices_sha256": None
        if token_indices is None
        else _json_sha256([int(value) for value in token_indices]),
        "hidden_state_layer_count": 37,
        "hidden_state_device_observed": "mps:0",
        "input_device_observed": "mps:0",
        "model_id": "Qwen/Qwen3-test",
        "revision": "main",
        "selected_device": "mps",
        "hidden_size": 2560,
        "layer_count": 36,
        "tokenizer_fingerprint_sha256": "a" * 64,
        "special_tokens_map_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "prompt_renderer": CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        "selected_choice_geometry": CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        "selected_choice_readout_configured_token_count": 2,
    }


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _turn_with_content(content: str) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-cift-extractor-client",
        session_id="session-cift-extractor-client",
        turn_index=0,
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        model=ModelInfo(provider="mock", model_id="mock-model", revision=None, selected_device=None),
        messages=(Message(role="user", content=content),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={},
    )
