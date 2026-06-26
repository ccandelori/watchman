from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from ipaddress import ip_address
from math import isfinite
from pathlib import Path
from urllib.parse import urlparse

from aegis.audit.jsonl import JsonlAuditSink
from aegis.audit.memory import InMemoryAuditSink
from aegis.core.orchestrator import ModelProvider
from aegis.providers.mock import MockModelProvider
from aegis.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleProviderConfig,
    urllib_openai_sender,
)
from aegis.proxy.nimbus_profile import (
    ProxyConfigError as ProxyConfigError,
)
from aegis.proxy.nimbus_profile import (
    ProxyNimbusConfig as ProxyNimbusConfig,
)
from aegis.proxy.nimbus_profile import (
    nimbus_config_from_env as nimbus_config_from_env,
)


class ProviderKind(StrEnum):
    MOCK = "mock"
    OPENAI_COMPATIBLE = "openai_compatible"


class CiftProfile(StrEnum):
    BLACK_BOX = "black_box"
    SELF_HOSTED_WINDOW_SELECTOR = "self_hosted_window_selector"


class CiftCertificationMode(StrEnum):
    STRICT = "strict"
    GATEWAY_SMOKE_BOOTSTRAP = "gateway_smoke_bootstrap"


_TRUSTED_SELF_HOSTED_CIFT_FEATURE_SOURCE = "self_hosted_activation_extractor"


@dataclass(frozen=True)
class ProxyProviderConfig:
    kind: ProviderKind
    provider_name: str
    provider_target_url: str | None
    model_provider: ModelProvider
    mock_controls_enabled: bool


@dataclass(frozen=True)
class ProxyCiftConfig:
    profile: CiftProfile
    certification_mode: CiftCertificationMode
    detector_name: str
    selected_choice_model_path: Path | None
    fallback_model_path: Path | None
    certification_manifest_path: Path | None
    certification_report_path: Path | None
    certification_artifact_root: Path | None
    certification_manifest_sha256: str | None
    certification_report_sha256: str | None
    release_gate_report_path: Path | None
    release_gate_report_sha256: str | None
    required_device: str | None
    selected_choice_readout_token_count: int | None
    extractor_id: str | None
    extractor_base_url: str | None
    extractor_api_key: str | None
    extractor_timeout_seconds: float | None
    feature_source: str


def audit_sink_from_env(env: Mapping[str, str] | None = None) -> InMemoryAuditSink:
    values: Mapping[str, str] = os.environ if env is None else env
    path = values.get("AEGIS_AUDIT_JSONL_PATH")
    if path is None:
        return InMemoryAuditSink()
    if path == "":
        raise ProxyConfigError("AEGIS_AUDIT_JSONL_PATH must not be empty when provided.")
    return JsonlAuditSink(Path(path))


def provider_config_from_env(env: Mapping[str, str] | None = None) -> ProxyProviderConfig:
    values: Mapping[str, str] = os.environ if env is None else env
    provider_value = values.get("AEGIS_PROVIDER", ProviderKind.MOCK.value)
    try:
        provider_kind = ProviderKind(provider_value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in ProviderKind)
        raise ProxyConfigError(
            f"Unsupported AEGIS_PROVIDER '{provider_value}'. Supported values: {supported}."
        ) from exc

    if provider_kind == ProviderKind.MOCK:
        return ProxyProviderConfig(
            kind=provider_kind,
            provider_name="mock",
            provider_target_url=None,
            model_provider=MockModelProvider(
                default_content=values.get("AEGIS_MOCK_DEFAULT_CONTENT", "Aegis mock response.")
            ),
            mock_controls_enabled=True,
        )

    if provider_kind == ProviderKind.OPENAI_COMPATIBLE:
        base_url = values.get("AEGIS_OPENAI_BASE_URL", "")
        api_key = values.get("AEGIS_OPENAI_API_KEY", "")
        if base_url == "":
            raise ProxyConfigError("AEGIS_OPENAI_BASE_URL must be set when AEGIS_PROVIDER=openai_compatible.")
        if api_key == "":
            raise ProxyConfigError("AEGIS_OPENAI_API_KEY must be set when AEGIS_PROVIDER=openai_compatible.")
        timeout_seconds = _float_env(values, "AEGIS_OPENAI_TIMEOUT_SECONDS", 30.0)
        validated_base_url = _validated_http_base_url(base_url, "AEGIS_OPENAI_BASE_URL")
        return ProxyProviderConfig(
            kind=provider_kind,
            provider_name="openai_compatible",
            provider_target_url=validated_base_url,
            model_provider=OpenAICompatibleProvider(
                config=OpenAICompatibleProviderConfig(
                    base_url=validated_base_url,
                    api_key=api_key,
                    default_model=_optional_non_empty(values.get("AEGIS_OPENAI_MODEL")),
                    timeout_seconds=timeout_seconds,
                ),
                sender=urllib_openai_sender,
            ),
            mock_controls_enabled=False,
        )

    raise ProxyConfigError(f"Unhandled provider kind '{provider_kind.value}'.")


def cift_config_from_env(env: Mapping[str, str] | None = None) -> ProxyCiftConfig:
    values: Mapping[str, str] = os.environ if env is None else env
    profile_value = values.get("AEGIS_CIFT_PROFILE", CiftProfile.BLACK_BOX.value)
    try:
        profile = CiftProfile(profile_value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in CiftProfile)
        raise ProxyConfigError(
            f"Unsupported AEGIS_CIFT_PROFILE '{profile_value}'. Supported values: {supported}."
        ) from exc

    certification_mode = _cift_certification_mode_from_env(values)
    detector_name = _non_empty_env(values, "AEGIS_CIFT_DETECTOR_NAME", "cift_runtime")
    if profile == CiftProfile.BLACK_BOX:
        if certification_mode != CiftCertificationMode.STRICT:
            raise ProxyConfigError("AEGIS_CIFT_CERTIFICATION_MODE is only valid for self_hosted_window_selector.")
        return ProxyCiftConfig(
            profile=profile,
            certification_mode=certification_mode,
            detector_name=detector_name,
            selected_choice_model_path=None,
            fallback_model_path=None,
            certification_manifest_path=None,
            certification_report_path=None,
            certification_artifact_root=None,
            certification_manifest_sha256=None,
            certification_report_sha256=None,
            release_gate_report_path=None,
            release_gate_report_sha256=None,
            required_device=None,
            selected_choice_readout_token_count=None,
            extractor_id=None,
            extractor_base_url=None,
            extractor_api_key=None,
            extractor_timeout_seconds=None,
            feature_source="",
        )
    if profile == CiftProfile.SELF_HOSTED_WINDOW_SELECTOR:
        if certification_mode == CiftCertificationMode.STRICT:
            return _strict_self_hosted_cift_config_from_env(values=values, profile=profile, detector_name=detector_name)
        if certification_mode == CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP:
            return _gateway_smoke_bootstrap_cift_config_from_env(
                values=values,
                profile=profile,
                detector_name=detector_name,
            )
    raise ProxyConfigError(f"Unhandled CIFT profile '{profile.value}'.")


def _strict_self_hosted_cift_config_from_env(
    values: Mapping[str, str],
    profile: CiftProfile,
    detector_name: str,
) -> ProxyCiftConfig:
    _reject_strict_fallback_model_path(values)
    return ProxyCiftConfig(
        profile=profile,
        certification_mode=CiftCertificationMode.STRICT,
        detector_name=detector_name,
        selected_choice_model_path=Path(_required_non_empty_env(values, "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH")),
        fallback_model_path=None,
        extractor_id=_required_non_empty_env(values, "AEGIS_CIFT_EXTRACTOR_ID"),
        certification_manifest_path=Path(_required_non_empty_env(values, "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH")),
        certification_report_path=Path(_required_non_empty_env(values, "AEGIS_CIFT_CERTIFICATION_REPORT_PATH")),
        certification_artifact_root=Path(_required_non_empty_env(values, "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT")),
        certification_manifest_sha256=_required_sha256_env(values, "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256"),
        certification_report_sha256=_required_sha256_env(values, "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256"),
        release_gate_report_path=Path(_required_non_empty_env(values, "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH")),
        release_gate_report_sha256=_required_sha256_env(values, "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256"),
        required_device=_required_non_empty_env(values, "AEGIS_CIFT_REQUIRED_DEVICE"),
        selected_choice_readout_token_count=_required_positive_int_env(
            values,
            "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT",
        ),
        extractor_base_url=_optional_http_base_url_env(values, "AEGIS_CIFT_EXTRACTOR_BASE_URL"),
        extractor_api_key=_optional_string_env(values, "AEGIS_CIFT_EXTRACTOR_API_KEY"),
        extractor_timeout_seconds=_optional_positive_float_env(values, "AEGIS_CIFT_EXTRACTOR_TIMEOUT_SECONDS"),
        feature_source=_trusted_self_hosted_cift_feature_source_env(values, "AEGIS_CIFT_FEATURE_SOURCE"),
    )


def _reject_strict_fallback_model_path(values: Mapping[str, str]) -> None:
    if "AEGIS_CIFT_FALLBACK_MODEL_PATH" in values:
        raise ProxyConfigError(
            "AEGIS_CIFT_FALLBACK_MODEL_PATH is not supported in strict CIFT mode; "
            "strict selected-choice CIFT fails closed when selected-choice metadata is unavailable."
        )


def _gateway_smoke_bootstrap_cift_config_from_env(
    values: Mapping[str, str],
    profile: CiftProfile,
    detector_name: str,
) -> ProxyCiftConfig:
    _validate_gateway_smoke_bootstrap_env(values)
    return ProxyCiftConfig(
        profile=profile,
        certification_mode=CiftCertificationMode.GATEWAY_SMOKE_BOOTSTRAP,
        detector_name=detector_name,
        selected_choice_model_path=Path(_required_non_empty_env(values, "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH")),
        fallback_model_path=_optional_path_env(values, "AEGIS_CIFT_FALLBACK_MODEL_PATH"),
        extractor_id=_required_non_empty_env(values, "AEGIS_CIFT_EXTRACTOR_ID"),
        certification_manifest_path=None,
        certification_report_path=None,
        certification_artifact_root=None,
        certification_manifest_sha256=None,
        certification_report_sha256=None,
        release_gate_report_path=None,
        release_gate_report_sha256=None,
        required_device=_required_non_empty_env(values, "AEGIS_CIFT_REQUIRED_DEVICE"),
        selected_choice_readout_token_count=_required_positive_int_env(
            values,
            "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT",
        ),
        extractor_base_url=_required_http_base_url_env(values, "AEGIS_CIFT_EXTRACTOR_BASE_URL"),
        extractor_api_key=_optional_string_env(values, "AEGIS_CIFT_EXTRACTOR_API_KEY"),
        extractor_timeout_seconds=_optional_positive_float_env(values, "AEGIS_CIFT_EXTRACTOR_TIMEOUT_SECONDS"),
        feature_source=_non_empty_env(values, "AEGIS_CIFT_FEATURE_SOURCE", "self_hosted_activation_extractor"),
    )


def _cift_certification_mode_from_env(values: Mapping[str, str]) -> CiftCertificationMode:
    mode_value = values.get("AEGIS_CIFT_CERTIFICATION_MODE", CiftCertificationMode.STRICT.value)
    try:
        return CiftCertificationMode(mode_value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in CiftCertificationMode)
        raise ProxyConfigError(
            f"Unsupported AEGIS_CIFT_CERTIFICATION_MODE '{mode_value}'. Supported values: {supported}."
        ) from exc


def _validate_gateway_smoke_bootstrap_env(values: Mapping[str, str]) -> None:
    provider_value = values.get("AEGIS_PROVIDER", ProviderKind.MOCK.value)
    if provider_value != ProviderKind.MOCK.value:
        raise ProxyConfigError("gateway_smoke_bootstrap requires AEGIS_PROVIDER=mock.")
    forbidden_keys = (
        "AEGIS_CIFT_CERTIFICATION_MANIFEST_PATH",
        "AEGIS_CIFT_CERTIFICATION_REPORT_PATH",
        "AEGIS_CIFT_CERTIFICATION_ARTIFACT_ROOT",
        "AEGIS_CIFT_CERTIFICATION_MANIFEST_SHA256",
        "AEGIS_CIFT_CERTIFICATION_REPORT_SHA256",
        "AEGIS_CIFT_RELEASE_GATE_REPORT_PATH",
        "AEGIS_CIFT_RELEASE_GATE_REPORT_SHA256",
    )
    for key in forbidden_keys:
        if key in values:
            raise ProxyConfigError(f"{key} must not be set when AEGIS_CIFT_CERTIFICATION_MODE=gateway_smoke_bootstrap.")


def _float_env(values: Mapping[str, str], key: str, default: float) -> float:
    return _positive_float_env(values, key, default)


def _positive_float_env(values: Mapping[str, str], key: str, default: float) -> float:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    return _positive_float(value=raw_value, key=key)


def _optional_positive_float_env(values: Mapping[str, str], key: str) -> float | None:
    raw_value = values.get(key)
    if raw_value is None:
        return None
    return _positive_float(value=raw_value, key=key)


def _required_positive_int_env(values: Mapping[str, str], key: str) -> int:
    raw_value = _required_non_empty_env(values=values, key=key)
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ProxyConfigError(f"{key} must be an integer.") from exc
    if parsed < 1:
        raise ProxyConfigError(f"{key} must be positive.")
    return parsed


def _positive_float(value: str, key: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ProxyConfigError(f"{key} must be a number.") from exc
    if parsed <= 0:
        raise ProxyConfigError(f"{key} must be positive.")
    if not isfinite(parsed):
        raise ProxyConfigError(f"{key} must be finite.")
    return parsed


def _optional_non_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _required_non_empty_env(values: Mapping[str, str], key: str) -> str:
    value = values.get(key)
    if value is None or value == "":
        raise ProxyConfigError(f"{key} must be set.")
    return value


def _required_sha256_env(values: Mapping[str, str], key: str) -> str:
    value = _required_non_empty_env(values=values, key=key)
    if len(value) != 64:
        raise ProxyConfigError(f"{key} must be a 64-character SHA-256 hex digest.")
    for character in value:
        if character not in "0123456789abcdef":
            raise ProxyConfigError(f"{key} must be lowercase SHA-256 hex.")
    return value


def _optional_path_env(values: Mapping[str, str], key: str) -> Path | None:
    value = values.get(key)
    if value is None:
        return None
    if value == "":
        raise ProxyConfigError(f"{key} must not be empty.")
    return Path(value)


def _optional_string_env(values: Mapping[str, str], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if value == "":
        raise ProxyConfigError(f"{key} must not be empty.")
    return value


def _required_http_base_url_env(values: Mapping[str, str], key: str) -> str:
    return _validated_http_base_url(_required_non_empty_env(values=values, key=key), key)


def _optional_http_base_url_env(values: Mapping[str, str], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if value == "":
        raise ProxyConfigError(f"{key} must not be empty.")
    return _validated_http_base_url(value=value, key=key)


def _validated_http_base_url(value: str, key: str) -> str:
    if value.strip() != value:
        raise ProxyConfigError(f"{key} must not include leading or trailing whitespace.")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ProxyConfigError(f"{key} must use http or https.")
    if parsed.hostname is None or parsed.hostname == "":
        raise ProxyConfigError(f"{key} must include a host.")
    if parsed.username is not None or parsed.password is not None:
        raise ProxyConfigError(f"{key} must not include credentials in the URL.")
    if parsed.params != "" or parsed.query != "" or parsed.fragment != "":
        raise ProxyConfigError(f"{key} must not include params, query, or fragment components.")
    if parsed.scheme == "http" and not _is_loopback_hostname(parsed.hostname):
        raise ProxyConfigError(f"{key} may use http only for loopback hosts.")
    return value


def _is_loopback_hostname(hostname: str) -> bool:
    normalized = hostname.lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _non_empty_env(values: Mapping[str, str], key: str, default: str) -> str:
    value = values.get(key)
    if value is None:
        return default
    if value == "":
        raise ProxyConfigError(f"{key} must not be empty.")
    return value


def _trusted_self_hosted_cift_feature_source_env(values: Mapping[str, str], key: str) -> str:
    value = values.get(key)
    if value is None:
        return _TRUSTED_SELF_HOSTED_CIFT_FEATURE_SOURCE
    if value == "":
        raise ProxyConfigError(f"{key} must not be empty.")
    if value != _TRUSTED_SELF_HOSTED_CIFT_FEATURE_SOURCE:
        raise ProxyConfigError(f"{key} must be {_TRUSTED_SELF_HOSTED_CIFT_FEATURE_SOURCE} in strict CIFT mode.")
    return value
