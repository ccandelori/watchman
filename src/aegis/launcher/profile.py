from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aegis.core.contracts import JsonValue

LAUNCHER_PROFILE_SCHEMA_VERSION = "aegis.launcher_profile/v1"
DEFAULT_PROFILE_NAME = "ollama-qwen3-4b-hermes"
DEFAULT_CIFT_ENV_PATH = (
    "introspection/data/reports/"
    "qwen3_4b_watchman_semantic_v9_480_selected_choice_immutable_l21_raw_linear_promoted_runtime_mps_receipt_recheck_strict_deployment_env.sh"
)
DEFAULT_CIFT_FREEFORM_ENV_PATH = (
    "introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/"
    "qwen3_4b_watchman_v14_freeform_final_token_l12_strict_deployment_env.sh"
)


class LauncherProfileError(ValueError):
    """Raised when a launcher profile cannot be loaded or saved."""


@dataclass(frozen=True)
class LockedCiftBinding:
    model_id: str
    revision: str
    device: str
    dtype: str
    feature_key: str
    freeform_feature_key: str
    selected_choice_readout_token_count: int
    hidden_size: int
    layer_count: int
    tokenizer_fingerprint_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
    strict_deployment_env_path: str
    freeform_strict_deployment_env_path: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "feature_key": self.feature_key,
            "freeform_feature_key": self.freeform_feature_key,
            "selected_choice_readout_token_count": self.selected_choice_readout_token_count,
            "hidden_size": self.hidden_size,
            "layer_count": self.layer_count,
            "tokenizer_fingerprint_sha256": self.tokenizer_fingerprint_sha256,
            "special_tokens_map_sha256": self.special_tokens_map_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "strict_deployment_env_path": self.strict_deployment_env_path,
            "freeform_strict_deployment_env_path": self.freeform_strict_deployment_env_path,
        }


@dataclass(frozen=True)
class LauncherProfile:
    schema_version: str
    name: str
    agent_kind: str
    provider_kind: str
    provider_base_url: str
    provider_model: str
    provider_api_key: str
    gateway_host: str
    gateway_port: int
    sidecar_host: str
    sidecar_port: int
    console_host: str
    console_port: int
    audit_jsonl_path: str
    cift_api_key: str
    mps_python_path: str
    cift_binding: LockedCiftBinding

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "agent_kind": self.agent_kind,
            "provider_kind": self.provider_kind,
            "provider_base_url": self.provider_base_url,
            "provider_model": self.provider_model,
            "provider_api_key": self.provider_api_key,
            "gateway_host": self.gateway_host,
            "gateway_port": self.gateway_port,
            "sidecar_host": self.sidecar_host,
            "sidecar_port": self.sidecar_port,
            "console_host": self.console_host,
            "console_port": self.console_port,
            "audit_jsonl_path": self.audit_jsonl_path,
            "cift_api_key": self.cift_api_key,
            "mps_python_path": self.mps_python_path,
            "cift_binding": self.cift_binding.to_dict(),
        }


def default_profile() -> LauncherProfile:
    return LauncherProfile(
        schema_version=LAUNCHER_PROFILE_SCHEMA_VERSION,
        name=DEFAULT_PROFILE_NAME,
        agent_kind="Hermes Agent",
        provider_kind="ollama",
        provider_base_url="http://127.0.0.1:11434/v1",
        provider_model="qwen3:4b",
        provider_api_key="ollama",
        gateway_host="127.0.0.1",
        gateway_port=8000,
        sidecar_host="127.0.0.1",
        sidecar_port=9000,
        console_host="127.0.0.1",
        console_port=8780,
        audit_jsonl_path="/tmp/aegis-local-agent-audit.jsonl",
        cift_api_key="set-a-deployment-secret",
        mps_python_path=".venv-mps313/bin/python",
        cift_binding=LockedCiftBinding(
            model_id="Qwen/Qwen3-4B",
            revision="1cfa9a7208912126459214e8b04321603b3df60c",
            device="mps",
            dtype="device",
            feature_key="selected_choice_window_layer_21",
            freeform_feature_key="final_token_layer_12",
            selected_choice_readout_token_count=4,
            hidden_size=2560,
            layer_count=36,
            tokenizer_fingerprint_sha256="41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04",
            special_tokens_map_sha256="edcb2fc2acbbe77f858a9c4fe51295ffdb84711efba5703ec5906b3d67282569",
            chat_template_sha256="a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
            strict_deployment_env_path=DEFAULT_CIFT_ENV_PATH,
            freeform_strict_deployment_env_path=DEFAULT_CIFT_FREEFORM_ENV_PATH,
        ),
    )


def load_profile(path: Path) -> LauncherProfile:
    if not path.exists():
        profile = default_profile()
        save_profile(path, profile)
        return profile
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LauncherProfileError(f"launcher profile '{path}' must contain a JSON object.")
    return profile_from_payload(payload)


def save_profile(path: Path, profile: LauncherProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def profile_from_payload(payload: dict[str, JsonValue]) -> LauncherProfile:
    binding_payload = _dict_field(payload, "cift_binding")
    default_binding = default_profile().cift_binding
    return LauncherProfile(
        schema_version=_string_field(payload, "schema_version", LAUNCHER_PROFILE_SCHEMA_VERSION),
        name=_string_field(payload, "name", DEFAULT_PROFILE_NAME),
        agent_kind=_string_field(payload, "agent_kind", "Hermes Agent"),
        provider_kind=_string_field(payload, "provider_kind", "ollama"),
        provider_base_url=_string_field(payload, "provider_base_url", "http://127.0.0.1:11434/v1"),
        provider_model=_string_field(payload, "provider_model", "qwen3:4b"),
        provider_api_key=_string_field(payload, "provider_api_key", "ollama"),
        gateway_host=_string_field(payload, "gateway_host", "127.0.0.1"),
        gateway_port=_int_field(payload, "gateway_port", 8000),
        sidecar_host=_string_field(payload, "sidecar_host", "127.0.0.1"),
        sidecar_port=_int_field(payload, "sidecar_port", 9000),
        console_host=_string_field(payload, "console_host", "127.0.0.1"),
        console_port=_int_field(payload, "console_port", 8780),
        audit_jsonl_path=_string_field(payload, "audit_jsonl_path", "/tmp/aegis-local-agent-audit.jsonl"),
        cift_api_key=_string_field(payload, "cift_api_key", "set-a-deployment-secret"),
        mps_python_path=_string_field(payload, "mps_python_path", ".venv-mps313/bin/python"),
        cift_binding=LockedCiftBinding(
            model_id=_string_field(binding_payload, "model_id", default_binding.model_id),
            revision=_string_field(binding_payload, "revision", default_binding.revision),
            device=_string_field(binding_payload, "device", default_binding.device),
            dtype=_string_field(binding_payload, "dtype", default_binding.dtype),
            feature_key=_string_field(binding_payload, "feature_key", default_binding.feature_key),
            freeform_feature_key=_string_field(
                binding_payload,
                "freeform_feature_key",
                default_binding.freeform_feature_key,
            ),
            selected_choice_readout_token_count=_int_field(
                binding_payload,
                "selected_choice_readout_token_count",
                default_binding.selected_choice_readout_token_count,
            ),
            hidden_size=_int_field(binding_payload, "hidden_size", default_binding.hidden_size),
            layer_count=_int_field(binding_payload, "layer_count", default_binding.layer_count),
            tokenizer_fingerprint_sha256=_string_field(
                binding_payload,
                "tokenizer_fingerprint_sha256",
                default_binding.tokenizer_fingerprint_sha256,
            ),
            special_tokens_map_sha256=_string_field(
                binding_payload,
                "special_tokens_map_sha256",
                default_binding.special_tokens_map_sha256,
            ),
            chat_template_sha256=_string_field(
                binding_payload,
                "chat_template_sha256",
                default_binding.chat_template_sha256,
            ),
            strict_deployment_env_path=_string_field(
                binding_payload,
                "strict_deployment_env_path",
                default_binding.strict_deployment_env_path,
            ),
            freeform_strict_deployment_env_path=_string_field(
                binding_payload,
                "freeform_strict_deployment_env_path",
                default_binding.freeform_strict_deployment_env_path,
            ),
        ),
    )


def editable_profile_update(profile: LauncherProfile, payload: dict[str, JsonValue]) -> LauncherProfile:
    return LauncherProfile(
        schema_version=profile.schema_version,
        name=_string_field(payload, "name", profile.name),
        agent_kind=_string_field(payload, "agent_kind", profile.agent_kind),
        provider_kind=_string_field(payload, "provider_kind", profile.provider_kind),
        provider_base_url=_string_field(payload, "provider_base_url", profile.provider_base_url),
        provider_model=_string_field(payload, "provider_model", profile.provider_model),
        provider_api_key=_string_field(payload, "provider_api_key", profile.provider_api_key),
        gateway_host=_string_field(payload, "gateway_host", profile.gateway_host),
        gateway_port=_int_field(payload, "gateway_port", profile.gateway_port),
        sidecar_host=_string_field(payload, "sidecar_host", profile.sidecar_host),
        sidecar_port=_int_field(payload, "sidecar_port", profile.sidecar_port),
        console_host=_string_field(payload, "console_host", profile.console_host),
        console_port=_int_field(payload, "console_port", profile.console_port),
        audit_jsonl_path=_string_field(payload, "audit_jsonl_path", profile.audit_jsonl_path),
        cift_api_key=_string_field(payload, "cift_api_key", profile.cift_api_key),
        mps_python_path=_string_field(payload, "mps_python_path", profile.mps_python_path),
        cift_binding=profile.cift_binding,
    )


def _string_field(payload: dict[str, JsonValue], key: str, fallback: str) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value != "":
        return value
    return fallback


def _int_field(payload: dict[str, JsonValue], key: str, fallback: int) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return fallback


def _dict_field(payload: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    return {}
