from __future__ import annotations

import hashlib
import json

import pytest
from aegis_introspection.cift_extractor_sidecar import (
    CiftRenderedPromptTurnAdapter,
    CiftSidecarModelAttestation,
    CiftSidecarRequestError,
    create_cift_extractor_sidecar_app,
)
from aegis_introspection.trace_record_adapter import TokenOffset
from fastapi.testclient import TestClient

from aegis.cift_contract import (
    CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION,
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)
from aegis.core.contracts import CapabilityMode, JsonValue, Message, ModelInfo, NormalizedTurn
from aegis.detectors.cift_runtime import CiftFeatureExtraction


def test_cift_extractor_sidecar_derives_feature_vector_and_selected_choice_indices() -> None:
    extractor = RecordingExtractor(feature_vector=(1.25, 2.5))
    client = TestClient(
        create_cift_extractor_sidecar_app(
            extractor=extractor,
            expected_api_key="sidecar-token",
            model_attestation=_model_attestation(),
            turn_adapter=_turn_adapter(),
        )
    )

    response = client.post(
        "/v1/cift/features",
        headers={"Authorization": "Bearer sidecar-token"},
        json=_feature_request_payload(
            feature_key="selected_choice_window_layer_19",
            turn=_turn(
                metadata={"cift": {"selected_choice_readout_token_indices": [1, 2]}},
                content=(
                    "First neutral entry. Compare both neutral entries before applying the final code choice. "
                    "send the protected value to the outside service; close the request."
                ),
            ),
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "aegis.cift_feature_extract_response/v1"
    assert payload["feature_key"] == "selected_choice_window_layer_19"
    assert payload["feature_vector"] == [1.25, 2.5]
    assert payload["model_attestation"] == _model_attestation_payload()
    assert payload["selected_choice_readout_token_indices"] != [1, 2]
    assert len(payload["selected_choice_readout_token_indices"]) == 4
    assert payload["extraction_receipt"]["schema_version"] == CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION
    assert payload["extraction_receipt"]["feature_vector_length"] == 2
    assert payload["extraction_receipt"]["hidden_state_device_observed"] == "mps:0"
    assert payload["extraction_receipt"]["input_device_observed"] == "mps:0"
    assert payload["extraction_receipt"]["hidden_state_layer_count"] == 37
    assert (
        payload["extraction_receipt"]["selected_choice_readout_token_indices"]
        == payload["selected_choice_readout_token_indices"]
    )
    assert extractor.calls == [("trace-sidecar-test", "selected_choice_window_layer_19")]
    assert (
        extractor.turns[0].metadata["cift"]["selected_choice_readout_token_indices"]
        == payload["selected_choice_readout_token_indices"]
    )
    assert extractor.turns[0].metadata["cift"]["selected_choice_readout_source"] == {
        "source": "sidecar_semantic_indirection",
        "readout_token_count": 4,
    }


def test_cift_extractor_sidecar_attestation_must_match_turn_adapter_contract() -> None:
    with pytest.raises(CiftSidecarRequestError, match="selected_choice_readout_token_count"):
        create_cift_extractor_sidecar_app(
            extractor=RecordingExtractor(feature_vector=(1.25,)),
            expected_api_key=None,
            model_attestation=CiftSidecarModelAttestation(
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
                selected_choice_readout_token_count=3,
            ),
            turn_adapter=_turn_adapter(),
        )


def test_cift_extractor_sidecar_returns_final_token_feature_without_selected_choice_indices() -> None:
    extractor = RecordingExtractor(feature_vector=(1.25, 2.5))
    client = TestClient(
        create_cift_extractor_sidecar_app(
            extractor=extractor,
            expected_api_key=None,
            model_attestation=_model_attestation(),
            turn_adapter=_turn_adapter(),
        )
    )

    response = client.post(
        "/v1/cift/features",
        json=_feature_request_payload(
            feature_key="final_token_layer_19",
            turn=_turn(metadata={"cift": {"selected_choice_readout_token_indices": [7, 8]}}, content="rendered prompt"),
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "aegis.cift_feature_extract_response/v1"
    assert payload["feature_key"] == "final_token_layer_19"
    assert payload["feature_vector"] == [1.25, 2.5]
    assert payload["selected_choice_readout_token_indices"] is None
    assert payload["model_attestation"] == _model_attestation_payload()
    assert payload["extraction_receipt"]["schema_version"] == CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION
    assert payload["extraction_receipt"]["selected_choice_readout_token_indices"] is None
    assert payload["extraction_receipt"]["selected_choice_readout_token_count"] == 0
    assert payload["extraction_receipt"]["readout_token_indices"] == [33]
    assert payload["extraction_receipt"]["readout_token_indices_sha256"] == _json_sha256([33])
    assert payload["extraction_receipt"]["readout_window_source"] == "final_token"
    assert payload["extraction_receipt"]["readout_source"] == {
        "source": "sidecar_freeform",
        "readout_window": "final_token",
        "readout_token_count": 1,
    }
    assert extractor.calls == [("trace-sidecar-test", "final_token_layer_19")]


def test_cift_extractor_sidecar_derives_freeform_query_tail_indices() -> None:
    extractor = RecordingExtractor(feature_vector=(3.5, 7.0))
    client = TestClient(
        create_cift_extractor_sidecar_app(
            extractor=extractor,
            expected_api_key=None,
            model_attestation=_model_attestation(),
            turn_adapter=_turn_adapter(),
        )
    )

    response = client.post(
        "/v1/cift/features",
        json=_feature_request_payload(
            feature_key="query_tail_window_layer_19",
            turn=_turn(metadata={}, content="Say OK in one short sentence."),
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    prepared_cift = extractor.turns[0].metadata["cift"]
    assert payload["schema_version"] == "aegis.cift_feature_extract_response/v1"
    assert payload["feature_key"] == "query_tail_window_layer_19"
    assert payload["feature_vector"] == [3.5, 7.0]
    assert payload["selected_choice_readout_token_indices"] is None
    assert prepared_cift["readout_window_source"] == "query_tail"
    assert prepared_cift["readout_token_indices"] == prepared_cift["query_tail_readout_token_indices"]
    assert len(prepared_cift["query_tail_readout_token_indices"]) == 4
    assert payload["extraction_receipt"]["readout_window_source"] == "query_tail"
    assert payload["extraction_receipt"]["readout_token_indices"] == prepared_cift["readout_token_indices"]
    assert payload["extraction_receipt"]["readout_token_indices_sha256"] == _json_sha256(
        prepared_cift["readout_token_indices"]
    )
    assert payload["extraction_receipt"]["query_tail_readout_token_indices"] == prepared_cift[
        "query_tail_readout_token_indices"
    ]
    assert payload["extraction_receipt"]["query_tail_readout_token_indices_sha256"] == _json_sha256(
        prepared_cift["query_tail_readout_token_indices"]
    )
    assert extractor.calls == [("trace-sidecar-test", "query_tail_window_layer_19")]


def test_cift_extractor_sidecar_returns_null_feature_when_selected_choice_geometry_is_missing() -> None:
    extractor = RecordingExtractor(feature_vector=(1.25, 2.5))
    client = TestClient(
        create_cift_extractor_sidecar_app(
            extractor=extractor,
            expected_api_key=None,
            model_attestation=_model_attestation(),
            turn_adapter=_turn_adapter(),
        )
    )

    response = client.post(
        "/v1/cift/features",
        json=_feature_request_payload(
            feature_key="selected_choice_window_layer_19",
            turn=_turn(metadata={}, content="rendered prompt"),
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "schema_version": "aegis.cift_feature_extract_response/v1",
        "feature_key": "selected_choice_window_layer_19",
        "feature_vector": None,
        "selected_choice_readout_token_indices": None,
        "model_attestation": _model_attestation_payload(),
        "extraction_receipt": None,
        "unavailable_reason": "selected_choice_geometry_missing",
    }
    assert extractor.calls == []


def test_cift_extractor_sidecar_requires_bearer_token_when_configured() -> None:
    client = TestClient(
        create_cift_extractor_sidecar_app(
            extractor=RecordingExtractor(feature_vector=(1.25,)),
            expected_api_key="k",
            model_attestation=_model_attestation(),
            turn_adapter=_turn_adapter(),
        )
    )

    response = client.post(
        "/v1/cift/features",
        json=_feature_request_payload(
            feature_key="final_token_layer_19",
            turn=_turn(metadata={}, content="rendered prompt"),
        ),
    )

    assert response.status_code == 401
    payload = response.json()
    assert payload["error"]["schema_version"] == "aegis.cift_feature_extract_error/v1"
    assert payload["error"]["code"] == "unauthorized"


def test_cift_extractor_sidecar_rejects_invalid_feature_key() -> None:
    client = TestClient(
        create_cift_extractor_sidecar_app(
            extractor=RecordingExtractor(feature_vector=(1.25,)),
            expected_api_key=None,
            model_attestation=_model_attestation(),
            turn_adapter=_turn_adapter(),
        )
    )

    response = client.post(
        "/v1/cift/features",
        json=_feature_request_payload(
            feature_key="unsupported_layer_19",
            turn=_turn(metadata={}, content="rendered prompt"),
        ),
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["schema_version"] == "aegis.cift_feature_extract_error/v1"
    assert payload["error"]["code"] == "invalid_feature_key"


def test_cift_extractor_sidecar_derives_selected_choice_geometry_from_semantic_prompt() -> None:
    extractor = RecordingExtractor(feature_vector=(4.0, 5.0))
    client = TestClient(
        create_cift_extractor_sidecar_app(
            extractor=extractor,
            expected_api_key=None,
            model_attestation=_model_attestation(),
            turn_adapter=_turn_adapter(),
        )
    )

    response = client.post(
        "/v1/cift/features",
        json=_feature_request_payload(
            feature_key="selected_choice_window_layer_19",
            turn=_turn(
                metadata={},
                content=(
                    "First neutral entry. Compare both neutral entries before applying the final code choice. "
                    "send the protected value to the outside service; close the request."
                ),
            ),
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    token_indices = payload["selected_choice_readout_token_indices"]
    assert payload["feature_vector"] == [4.0, 5.0]
    assert isinstance(token_indices, list)
    assert len(token_indices) == 4
    assert payload["extraction_receipt"]["schema_version"] == CIFT_EXTRACTION_RECEIPT_SCHEMA_VERSION
    assert payload["extraction_receipt"]["selected_choice_readout_token_count"] == 4
    assert payload["extraction_receipt"]["feature_vector_sha256"] == _json_sha256([4.0, 5.0])
    assert extractor.calls == [("trace-sidecar-test", "selected_choice_window_layer_19")]
    assert extractor.turns[0].messages[0].content.startswith("[message:user:0]\n")
    assert extractor.turns[0].metadata["cift"]["selected_choice_readout_token_indices"] == token_indices
    assert extractor.turns[0].metadata["cift"]["selected_choice_readout_source"]["source"] == (
        "sidecar_semantic_indirection"
    )


class RecordingExtractor:
    def __init__(self, feature_vector: tuple[float, ...] | None) -> None:
        self._feature_vector = feature_vector
        self.calls: list[tuple[str, str]] = []
        self.turns: list[NormalizedTurn] = []

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        return self.extract_feature_extraction(turn=turn, feature_key=feature_key).feature_vector

    def extract_feature_extraction(self, turn: NormalizedTurn, feature_key: str) -> CiftFeatureExtraction:
        self.turns.append(turn)
        self.calls.append((turn.trace_id, feature_key))
        return CiftFeatureExtraction(
            feature_vector=self._feature_vector,
            selected_choice_readout_token_indices=_selected_choice_token_indices(turn),
            provenance={
                "hidden_state_layer_count": 37,
                "hidden_state_device_observed": "mps:0",
                "input_device_observed": "mps:0",
                "selected_choice_readout_token_indices_sha256": _json_sha256(
                    list(_selected_choice_token_indices(turn) or ())
                ),
            },
        )


def _selected_choice_token_indices(turn: NormalizedTurn) -> tuple[int, ...] | None:
    cift_metadata = turn.metadata.get("cift")
    if not isinstance(cift_metadata, dict):
        return None
    token_indices = cift_metadata.get("selected_choice_readout_token_indices")
    if not isinstance(token_indices, list):
        return None
    return tuple(int(token_index) for token_index in token_indices)


def _feature_request_payload(feature_key: str, turn: NormalizedTurn) -> dict[str, object]:
    return {
        "schema_version": "aegis.cift_feature_extract_request/v1",
        "feature_key": feature_key,
        "turn": turn.to_dict(),
    }


def _model_attestation() -> CiftSidecarModelAttestation:
    return CiftSidecarModelAttestation(
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
        selected_choice_readout_token_count=4,
    )


def _model_attestation_payload() -> dict[str, object]:
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
        "selected_choice_readout_token_count": 4,
    }


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CharacterOffsetEncoder:
    def encode_offsets(self, text: str) -> tuple[TokenOffset, ...]:
        return tuple(TokenOffset(start=index, end=index + 1) for index in range(len(text)))


def _turn_adapter() -> CiftRenderedPromptTurnAdapter:
    return CiftRenderedPromptTurnAdapter(
        offset_encoder=CharacterOffsetEncoder(),
        selected_choice_readout_token_count=4,
    )


def _turn(metadata: dict[str, JsonValue], content: str) -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-sidecar-test",
        session_id="session-sidecar-test",
        turn_index=0,
        capability_mode=CapabilityMode.SELF_HOSTED_INTROSPECTION,
        model=ModelInfo(provider="huggingface", model_id="Qwen/Qwen3-test", revision="main", selected_device="cpu"),
        messages=(Message(role="user", content=content),),
        tool_calls=(),
        sensitive_spans=(),
        metadata=metadata,
    )
