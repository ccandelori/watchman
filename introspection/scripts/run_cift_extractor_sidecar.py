from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_extractor_sidecar import (  # noqa: E402
    CiftExtractorSidecarService,
    CiftRenderedPromptTurnAdapter,
    CiftSidecarHttpResponse,
    CiftSidecarModelAttestation,
    cift_sidecar_error_response,
)
from aegis_introspection.cift_live_extractor import (  # noqa: E402
    LiveCiftFeatureSetExtractor,
    LoadedModelHiddenStateRunner,
)
from aegis_introspection.cift_model_metadata import (  # noqa: E402
    CiftModelMetadataConfig,
    cift_model_metadata_report_from_loaded_objects,
)
from aegis_introspection.hf_offset_encoder import HuggingFaceOffsetEncoder  # noqa: E402
from aegis_introspection.model_loader import (  # noqa: E402
    LoadedCausalLM,
    ModelDTypeName,
    ModelLoadConfig,
    load_causal_lm,
    parse_model_dtype,
)

from aegis.cift_contract import (  # noqa: E402
    CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
    CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
)

_FEATURES_ROUTE = "/v1/cift/features"
_HEALTH_ROUTE = "/health"
_MAX_REQUEST_BODY_BYTES = 2_000_000


@dataclass(frozen=True)
class CiftExtractorSidecarCliConfig:
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    feature_keys: tuple[str, ...]
    selected_choice_readout_token_count: int
    expected_api_key: str | None
    host: str
    port: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trusted live CIFT activation extractor sidecar.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--feature-key", required=True, action="append")
    parser.add_argument("--selected-choice-readout-token-count", required=True, type=int)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env-var")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> CiftExtractorSidecarCliConfig:
    namespace = _build_parser().parse_args(argv)
    config = CiftExtractorSidecarCliConfig(
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        requested_device=str(namespace.device),
        local_files_only=not bool(namespace.allow_download),
        dtype_name=parse_model_dtype(str(namespace.dtype)),
        trust_remote_code=bool(namespace.trust_remote_code),
        feature_keys=_feature_keys(namespace.feature_key),
        selected_choice_readout_token_count=_positive_int(
            raw_value=namespace.selected_choice_readout_token_count,
            field_name="--selected-choice-readout-token-count",
        ),
        expected_api_key=_expected_api_key(
            raw_api_key=namespace.api_key,
            api_key_env_var=namespace.api_key_env_var,
        ),
        host=str(namespace.host),
        port=int(namespace.port),
    )
    _validate_model_device_policy(config)
    return config


def _feature_keys(raw_feature_keys: object) -> tuple[str, ...]:
    if not isinstance(raw_feature_keys, list):
        raise ValueError("--feature-key must be provided at least once.")
    feature_keys = tuple(str(item) for item in raw_feature_keys)
    if len(feature_keys) == 0:
        raise ValueError("--feature-key must be provided at least once.")
    if any(feature_key == "" for feature_key in feature_keys):
        raise ValueError("--feature-key values must not be empty.")
    if len(set(feature_keys)) != len(feature_keys):
        raise ValueError("--feature-key values must be unique.")
    return feature_keys


def _expected_api_key(raw_api_key: object, api_key_env_var: object) -> str | None:
    api_key = _optional_string(raw_api_key, "--api-key")
    env_var = _optional_string(api_key_env_var, "--api-key-env-var")
    if api_key is not None and env_var is not None:
        raise ValueError("--api-key and --api-key-env-var are mutually exclusive.")
    if env_var is None:
        return api_key
    value = os.environ.get(env_var)
    if value is None or value == "":
        raise ValueError(f"Environment variable {env_var} must contain a non-empty CIFT sidecar API key.")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if value == "":
        raise ValueError(f"{field_name} must not be empty when provided.")
    return value


def _positive_int(raw_value: object, field_name: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if raw_value < 1:
        raise ValueError(f"{field_name} must be positive.")
    return raw_value


def run_cli(config: CiftExtractorSidecarCliConfig) -> None:
    _validate_model_device_policy(config)
    loaded_model = load_causal_lm(
        ModelLoadConfig(
            model_id=config.model_id,
            revision=config.revision,
            requested_device=config.requested_device,
            local_files_only=config.local_files_only,
            dtype_name=config.dtype_name,
            trust_remote_code=config.trust_remote_code,
        )
    )
    extractor = LiveCiftFeatureSetExtractor(
        runner=LoadedModelHiddenStateRunner(loaded_model=loaded_model),
        feature_keys=config.feature_keys,
    )
    model_attestation = _model_attestation_from_loaded_model(
        config=config,
        loaded_model=loaded_model,
    )
    service = CiftExtractorSidecarService(
        extractor=extractor,
        expected_api_key=config.expected_api_key,
        model_attestation=model_attestation,
        turn_adapter=CiftRenderedPromptTurnAdapter(
            offset_encoder=HuggingFaceOffsetEncoder(loaded_model.tokenizer),
            selected_choice_readout_token_count=config.selected_choice_readout_token_count,
        ),
    )
    run_http_server(service=service, host=config.host, port=config.port)


def _model_attestation_from_loaded_model(
    config: CiftExtractorSidecarCliConfig,
    loaded_model: LoadedCausalLM,
) -> CiftSidecarModelAttestation:
    metadata = cift_model_metadata_report_from_loaded_objects(
        config=CiftModelMetadataConfig(
            model_id=config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        ),
        model_config=loaded_model.model.config,
        tokenizer=loaded_model.tokenizer,
    )
    return CiftSidecarModelAttestation(
        model_id=loaded_model.model_id,
        revision=loaded_model.revision,
        selected_device=loaded_model.device.name,
        hidden_size=metadata.hidden_size,
        layer_count=metadata.layer_count,
        tokenizer_fingerprint_sha256=metadata.tokenizer_fingerprint_sha256,
        special_tokens_map_sha256=metadata.special_tokens_map_sha256,
        chat_template_sha256=metadata.chat_template_sha256,
        prompt_renderer=CIFT_PROMPT_RENDERER_TRACE_BRIDGE_V1,
        selected_choice_geometry=CIFT_SELECTED_CHOICE_GEOMETRY_SEMANTIC_INDIRECTION_V1,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
    )


def _validate_model_device_policy(config: CiftExtractorSidecarCliConfig) -> None:
    if config.model_id == "Qwen/Qwen3-4B" and config.requested_device != "mps":
        raise ValueError("Qwen/Qwen3-4B CIFT extractor sidecar requires --device mps.")


def run_http_server(service: CiftExtractorSidecarService, host: str, port: int) -> None:
    handler_class = _request_handler_class(service)
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"CIFT extractor sidecar listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _request_handler_class(service: CiftExtractorSidecarService) -> type[BaseHTTPRequestHandler]:
    class CiftExtractorRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == _HEALTH_ROUTE:
                _write_json_response(
                    handler=self,
                    response=CiftSidecarHttpResponse(status_code=200, payload={"status": "ok"}),
                )
                return
            _write_json_response(handler=self, response=_route_not_found_response(method=self.command, path=route))

        def do_POST(self) -> None:
            route = urlsplit(self.path).path
            if route != _FEATURES_ROUTE:
                _write_json_response(handler=self, response=_route_not_found_response(method=self.command, path=route))
                return
            request_body = _read_json_request_body(self)
            if isinstance(request_body, CiftSidecarHttpResponse):
                _write_json_response(handler=self, response=request_body)
                return
            response = service.extract_features(
                raw_payload=request_body,
                authorization_header=self.headers.get("Authorization"),
            )
            _write_json_response(handler=self, response=response)

    return CiftExtractorRequestHandler


def _read_json_request_body(handler: BaseHTTPRequestHandler) -> object | CiftSidecarHttpResponse:
    raw_content_length = handler.headers.get("Content-Length")
    if raw_content_length is None:
        return cift_sidecar_error_response(
            status_code=400,
            code="invalid_request",
            message="Content-Length header is required.",
            details={},
        )
    try:
        content_length = int(raw_content_length)
    except ValueError:
        return cift_sidecar_error_response(
            status_code=400,
            code="invalid_request",
            message="Content-Length header must be an integer.",
            details={},
        )
    if content_length < 0:
        return cift_sidecar_error_response(
            status_code=400,
            code="invalid_request",
            message="Content-Length header must be non-negative.",
            details={},
        )
    if content_length > _MAX_REQUEST_BODY_BYTES:
        return cift_sidecar_error_response(
            status_code=413,
            code="request_too_large",
            message="CIFT feature extraction request body exceeds the configured limit.",
            details={"max_request_body_bytes": _MAX_REQUEST_BODY_BYTES},
        )
    raw_body = handler.rfile.read(content_length)
    try:
        decoded_body = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        return cift_sidecar_error_response(
            status_code=400,
            code="invalid_json",
            message="Request body must be UTF-8 encoded JSON.",
            details={},
        )
    try:
        return json.loads(decoded_body)
    except json.JSONDecodeError:
        return cift_sidecar_error_response(
            status_code=400,
            code="invalid_json",
            message="Request body must be valid JSON.",
            details={},
        )


def _write_json_response(handler: BaseHTTPRequestHandler, response: CiftSidecarHttpResponse) -> None:
    encoded_body = json.dumps(response.payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(response.status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded_body)))
    handler.end_headers()
    handler.wfile.write(encoded_body)


def _route_not_found_response(method: str, path: str) -> CiftSidecarHttpResponse:
    return cift_sidecar_error_response(
        status_code=404,
        code="route_not_found",
        message=f"No route for {method} {path}.",
        details={"method": method, "path": path},
    )


def main(argv: Sequence[str]) -> None:
    run_cli(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
