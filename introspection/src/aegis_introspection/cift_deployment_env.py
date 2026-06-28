from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from aegis_introspection.cift_release_gate import (
    CiftReleaseGateConfig,
    evaluate_cift_release_gate,
    materialize_cift_release_gate_report,
)

_SHELL_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class CiftDeploymentEnvError(ValueError):
    """Raised when a CIFT deployment environment cannot be materialized."""


@dataclass(frozen=True)
class CiftDeploymentEnvConfig:
    runtime_model_path: Path
    repository_root: Path
    certification_manifest_path: Path
    certification_report_path: Path
    certification_artifact_root: Path
    required_device: str
    expected_detector_name: str
    expected_extractor_id: str
    expected_feature_source: str
    expected_selected_choice_readout_token_count: int
    extractor_base_url: str
    extractor_timeout_seconds: float
    extractor_api_key_env_var: str
    release_gate_report_path: Path


@dataclass(frozen=True)
class CiftDeploymentEnvCliConfig:
    deployment_config: CiftDeploymentEnvConfig
    output_path: Path | None


def materialize_cift_deployment_env(config: CiftDeploymentEnvConfig) -> str:
    _validate_deployment_env_config(config)
    window_family = _window_family_from_runtime_model_path(config.runtime_model_path)
    manifest_sha256 = _sha256_file(config.certification_manifest_path)
    report_sha256 = _sha256_file(config.certification_report_path)
    gate_config = CiftReleaseGateConfig(
        runtime_model_path=config.runtime_model_path,
        repository_root=config.repository_root,
        required_runtime_prevention_device=config.required_device,
        certification_manifest_path=config.certification_manifest_path,
        certification_report_path=config.certification_report_path,
        certification_artifact_root=config.certification_artifact_root,
        certification_manifest_sha256=manifest_sha256,
        certification_report_sha256=report_sha256,
        expected_detector_name=config.expected_detector_name,
        expected_extractor_id=config.expected_extractor_id,
        expected_feature_source=config.expected_feature_source,
        expected_selected_choice_readout_token_count=config.expected_selected_choice_readout_token_count,
        allow_embedded_artifact_only=False,
    )
    gate_report = evaluate_cift_release_gate(gate_config)
    materialize_cift_release_gate_report(
        config=gate_config,
        report=gate_report,
        output_path=config.release_gate_report_path,
    )
    if not gate_report.eligible:
        failures = "\n".join(f"- {failure}" for failure in gate_report.failed_requirements)
        raise CiftDeploymentEnvError(f"CIFT release gate failed; refusing to emit deployment env.\n{failures}")
    env_values = {
        "AEGIS_CIFT_PROFILE": "self_hosted_window_selector",
        "AEGIS_CIFT_CERTIFICATION_MODE": "strict",
        "AEGIS_CIFT_DETECTOR_NAME": config.expected_detector_name,
        "AEGIS_CIFT_REQUIRED_DEVICE": config.required_device,
        "AEGIS_CIFT_SELECTED_CHOICE_READOUT_TOKEN_COUNT": str(config.expected_selected_choice_readout_token_count),
        "AEGIS_CIFT_EXTRACTOR_ID": config.expected_extractor_id,
        "AEGIS_CIFT_EXTRACTOR_BASE_URL": config.extractor_base_url,
        "AEGIS_CIFT_EXTRACTOR_TIMEOUT_SECONDS": _format_float(config.extractor_timeout_seconds),
        "AEGIS_CIFT_FEATURE_SOURCE": config.expected_feature_source,
    }
    env_values.update(
        _route_binding_env_values(
            repository_root=config.repository_root,
            runtime_model_path=config.runtime_model_path,
            certification_manifest_path=config.certification_manifest_path,
            certification_report_path=config.certification_report_path,
            certification_artifact_root=config.certification_artifact_root,
            release_gate_report_path=config.release_gate_report_path,
            manifest_sha256=manifest_sha256,
            report_sha256=report_sha256,
            release_gate_report_sha256=_sha256_file(config.release_gate_report_path),
            window_family=window_family,
        )
    )
    if window_family.startswith("freeform_"):
        env_values["AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH"] = _selected_choice_model_path_env_value(
            repository_root=config.repository_root,
            certification_manifest_path=config.certification_manifest_path,
        )
    return _shell_exports(
        env_values=env_values,
        extractor_api_key_env_var=config.extractor_api_key_env_var,
        window_family=window_family,
    )


def run_deployment_env_cli(argv: Sequence[str]) -> int:
    config = _parse_args(argv)
    try:
        env_text = materialize_cift_deployment_env(config.deployment_config)
    except CiftDeploymentEnvError as exc:
        print(str(exc))
        return 1
    if config.output_path is None:
        print(env_text)
    else:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(env_text, encoding="utf-8")
        print(f"Wrote strict CIFT deployment env to {config.output_path}")
    return 0


def _parse_args(argv: Sequence[str]) -> CiftDeploymentEnvCliConfig:
    namespace = _build_parser().parse_args(argv)
    return CiftDeploymentEnvCliConfig(
        deployment_config=CiftDeploymentEnvConfig(
            runtime_model_path=Path(str(namespace.runtime_model)),
            repository_root=Path(str(namespace.repository_root)),
            certification_manifest_path=Path(str(namespace.certification_manifest)),
            certification_report_path=Path(str(namespace.certification_report)),
            certification_artifact_root=Path(str(namespace.certification_artifact_root)),
            required_device=str(namespace.required_device),
            expected_detector_name=str(namespace.expected_detector_name),
            expected_extractor_id=str(namespace.expected_extractor_id),
            expected_feature_source=str(namespace.expected_feature_source),
            expected_selected_choice_readout_token_count=int(namespace.expected_selected_choice_readout_token_count),
            extractor_base_url=str(namespace.extractor_base_url),
            extractor_timeout_seconds=float(namespace.extractor_timeout_seconds),
            extractor_api_key_env_var=str(namespace.extractor_api_key_env_var),
            release_gate_report_path=Path(str(namespace.release_gate_report_output)),
        ),
        output_path=_optional_path(namespace.output),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a strict CIFT deployment shell env after release-gate validation."
    )
    parser.add_argument("runtime_model")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--certification-manifest", required=True)
    parser.add_argument("--certification-report", required=True)
    parser.add_argument("--certification-artifact-root", required=True)
    parser.add_argument("--required-device", required=True)
    parser.add_argument("--expected-detector-name", default="cift_runtime")
    parser.add_argument("--expected-extractor-id", default="trusted-activation-sidecar")
    parser.add_argument("--expected-feature-source", default="self_hosted_activation_extractor")
    parser.add_argument("--expected-selected-choice-readout-token-count", required=True, type=int)
    parser.add_argument("--extractor-base-url", required=True)
    parser.add_argument("--extractor-timeout-seconds", required=True, type=float)
    parser.add_argument("--extractor-api-key-env-var", default="AEGIS_CIFT_EXTRACTOR_API_KEY")
    parser.add_argument("--release-gate-report-output", required=True)
    parser.add_argument("--output")
    return parser


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _validate_deployment_env_config(config: CiftDeploymentEnvConfig) -> None:
    _validate_existing_file(config.runtime_model_path, "runtime_model_path")
    _validate_existing_file(config.certification_manifest_path, "certification_manifest_path")
    _validate_existing_file(config.certification_report_path, "certification_report_path")
    if not config.certification_artifact_root.exists():
        raise CiftDeploymentEnvError("certification_artifact_root must exist.")
    _validate_non_empty(config.required_device, "required_device")
    _validate_non_empty(config.expected_detector_name, "expected_detector_name")
    _validate_non_empty(config.expected_extractor_id, "expected_extractor_id")
    _validate_non_empty(config.expected_feature_source, "expected_feature_source")
    _validate_non_empty(config.extractor_base_url, "extractor_base_url")
    _validate_non_empty(config.extractor_api_key_env_var, "extractor_api_key_env_var")
    if _SHELL_IDENTIFIER_PATTERN.fullmatch(config.extractor_api_key_env_var) is None:
        raise CiftDeploymentEnvError("extractor_api_key_env_var must be a shell-safe environment variable name.")
    if config.expected_selected_choice_readout_token_count < 1:
        raise CiftDeploymentEnvError("expected_selected_choice_readout_token_count must be positive.")
    if not math.isfinite(config.extractor_timeout_seconds) or config.extractor_timeout_seconds <= 0.0:
        raise CiftDeploymentEnvError("extractor_timeout_seconds must be a finite positive number.")


def _validate_existing_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise CiftDeploymentEnvError(f"{field_name} must point to an existing file.")


def _validate_non_empty(value: str, field_name: str) -> None:
    if value == "":
        raise CiftDeploymentEnvError(f"{field_name} must not be empty.")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window_family_from_runtime_model_path(path: Path) -> str:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftDeploymentEnvError(f"runtime_model_path must contain valid JSON: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise CiftDeploymentEnvError("runtime_model_path must contain a JSON object.")
    feature_key = decoded.get("feature_key")
    if not isinstance(feature_key, str) or feature_key == "":
        raise CiftDeploymentEnvError("runtime_model_path feature_key must be a non-empty string.")
    return _window_family_from_feature_key(feature_key)


def _window_family_from_feature_key(feature_key: str) -> str:
    if feature_key.startswith("selected_choice_window_"):
        return "selected_choice"
    if feature_key.startswith("query_tail_window_"):
        return "freeform_query_tail"
    if feature_key.startswith("readout_window_"):
        return "freeform_readout"
    if feature_key.startswith("final_token_"):
        return "freeform_final_token"
    if feature_key.startswith("mean_pool_"):
        return "freeform_mean_pool"
    return "unsupported"


def _route_binding_env_values(
    repository_root: Path,
    runtime_model_path: Path,
    certification_manifest_path: Path,
    certification_report_path: Path,
    certification_artifact_root: Path,
    release_gate_report_path: Path,
    manifest_sha256: str,
    report_sha256: str,
    release_gate_report_sha256: str,
    window_family: str,
) -> dict[str, str]:
    if window_family == "selected_choice":
        prefix = "AEGIS_CIFT_"
        model_path_key = "AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH"
    elif window_family.startswith("freeform_"):
        prefix = "AEGIS_CIFT_FREEFORM_"
        model_path_key = "AEGIS_CIFT_FREEFORM_MODEL_PATH"
    else:
        raise CiftDeploymentEnvError(f"runtime_model_path feature_key maps to unsupported route '{window_family}'.")
    return {
        model_path_key: _deployment_path(repository_root=repository_root, path=runtime_model_path),
        f"{prefix}CERTIFICATION_MANIFEST_PATH": _deployment_path(
            repository_root=repository_root,
            path=certification_manifest_path,
        ),
        f"{prefix}CERTIFICATION_REPORT_PATH": _deployment_path(
            repository_root=repository_root,
            path=certification_report_path,
        ),
        f"{prefix}CERTIFICATION_ARTIFACT_ROOT": _deployment_path(
            repository_root=repository_root,
            path=certification_artifact_root,
        ),
        f"{prefix}CERTIFICATION_MANIFEST_SHA256": manifest_sha256,
        f"{prefix}CERTIFICATION_REPORT_SHA256": report_sha256,
        f"{prefix}RELEASE_GATE_REPORT_PATH": _deployment_path(
            repository_root=repository_root,
            path=release_gate_report_path,
        ),
        f"{prefix}RELEASE_GATE_REPORT_SHA256": release_gate_report_sha256,
    }


def _deployment_path(repository_root: Path, path: Path) -> str:
    resolved_root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def _selected_choice_model_path_env_value(repository_root: Path, certification_manifest_path: Path) -> str:
    manifest = _json_object_from_path(path=certification_manifest_path, field_name="certification_manifest_path")
    training = manifest.get("training")
    if not isinstance(training, dict):
        raise CiftDeploymentEnvError("certification_manifest_path training must be an object for freeform env output.")
    raw_path = training.get("selected_choice_runtime_model_path")
    if not isinstance(raw_path, str) or raw_path == "":
        raise CiftDeploymentEnvError(
            "certification_manifest_path training.selected_choice_runtime_model_path must be set for freeform env output."
        )
    selected_choice_path = Path(raw_path)
    resolved_path = selected_choice_path if selected_choice_path.is_absolute() else repository_root / selected_choice_path
    _validate_existing_file(resolved_path, "certification_manifest_path training.selected_choice_runtime_model_path")
    return _deployment_path(repository_root=repository_root, path=resolved_path)


def _json_object_from_path(path: Path, field_name: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftDeploymentEnvError(f"{field_name} must contain valid JSON: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise CiftDeploymentEnvError(f"{field_name} must contain a JSON object.")
    return decoded


def _format_float(value: float) -> str:
    return str(value)


def _shell_exports(env_values: Mapping[str, str], extractor_api_key_env_var: str, window_family: str) -> str:
    route_label = "selected-choice" if window_family == "selected_choice" else "freeform"
    lines = [
        f"# Generated only after the hardened CIFT release gate passed for the {route_label} route.",
        "# Source or merge this file from the repository root before starting aegis-proxy.",
    ]
    for key in sorted(env_values):
        lines.append(f"export {key}={shlex.quote(env_values[key])}")
    lines.append(
        f'export AEGIS_CIFT_EXTRACTOR_API_KEY="${{{extractor_api_key_env_var}:?set {extractor_api_key_env_var}}}"'
    )
    return "\n".join(lines) + "\n"
