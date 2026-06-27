from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from aegis.console.service import (
    ConsoleSettings,
    GatewayFetcher,
    console_events,
    console_latency_summary,
    console_overview_from_events,
    console_trace,
    default_gateway_fetcher,
)
from aegis.core.contracts import JsonValue
from aegis.launcher.profile import (
    LauncherProfile,
    editable_profile_update,
    load_profile,
    save_profile,
)

LAUNCHER_STATE_SCHEMA_VERSION = "aegis.launcher_state/v1"
LAUNCHER_PREFLIGHT_SCHEMA_VERSION = "aegis.launcher_preflight/v1"
LAUNCHER_ACTION_SCHEMA_VERSION = "aegis.launcher_action/v1"
LAUNCHER_OBSERVABILITY_SCHEMA_VERSION = "aegis.launcher_observability/v1"
_LOG_BYTES = 12000
_MAX_OBSERVABILITY_LIMIT = 100
_LOCAL_PROVIDER_TIMEOUT_SECONDS = "90"
_OBSERVABILITY_REQUEST_TIMEOUT_SECONDS = 0.5
_CIFT_SMOKE_REPORT_PATH = "introspection/data/reports/qwen3_4b_launcher_cift_freeform_smoke_v1.json"


class LauncherServiceError(ValueError):
    """Raised when the launcher cannot satisfy a requested operation."""


@dataclass(frozen=True)
class LauncherSettings:
    repo_root: Path
    profile_path: Path
    log_dir: Path


@dataclass(frozen=True)
class ManagedCommand:
    action_id: str
    label: str
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    kind: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "kind": self.kind,
        }


@dataclass
class ManagedProcess:
    action_id: str
    label: str
    process: subprocess.Popen[bytes]
    started_at: float
    log_path: Path
    kind: str
    finished_at: float | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        exit_code = self.process.poll()
        if exit_code is not None and self.finished_at is None:
            self.finished_at = time.time()
        status = "running" if exit_code is None else "exited"
        observed_at = time.time()
        finished_at = self.finished_at if self.finished_at is not None else observed_at
        return {
            "action_id": self.action_id,
            "label": self.label,
            "pid": self.process.pid,
            "status": status,
            "exit_code": exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "observed_at": observed_at,
            "runtime_seconds": max(0.0, finished_at - self.started_at),
            "kind": self.kind,
            "log_path": str(self.log_path),
            "log_excerpt": _read_log_excerpt(self.log_path),
        }


class ProcessSupervisor:
    def __init__(self, settings: LauncherSettings) -> None:
        self._settings = settings
        self._processes: dict[str, ManagedProcess] = {}

    def statuses(self) -> dict[str, JsonValue]:
        return {action_id: process.to_dict() for action_id, process in sorted(self._processes.items())}

    def start(self, command: ManagedCommand) -> dict[str, JsonValue]:
        existing = self._processes.get(command.action_id)
        if existing is not None and existing.process.poll() is None:
            raise LauncherServiceError(f"{command.label} is already running with pid {existing.process.pid}.")
        command.cwd.mkdir(parents=True, exist_ok=True)
        self._settings.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._settings.log_dir / f"{command.action_id}.log"
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            command.argv,
            cwd=str(command.cwd),
            env=command.env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=False,
        )
        log_file.close()
        managed = ManagedProcess(
            action_id=command.action_id,
            label=command.label,
            process=process,
            started_at=time.time(),
            log_path=log_path,
            kind=command.kind,
        )
        self._processes[command.action_id] = managed
        return managed.to_dict()

    def stop(self, action_id: str) -> dict[str, JsonValue]:
        process = self._processes.get(action_id)
        if process is None:
            raise LauncherServiceError(f"no launched process is registered for '{action_id}'.")
        if process.process.poll() is None:
            process.process.terminate()
        return process.to_dict()

    def stop_all(self) -> dict[str, JsonValue]:
        stopped: dict[str, JsonValue] = {}
        for action_id in sorted(self._processes):
            stopped[action_id] = self.stop(action_id)
        return stopped

    def clear_statuses(self) -> dict[str, JsonValue]:
        running = sorted(action_id for action_id, process in self._processes.items() if process.process.poll() is None)
        if running:
            raise LauncherServiceError(f"Stop running services before clearing statuses: {', '.join(running)}.")
        cleared = sorted(self._processes)
        self._processes.clear()
        return {
            "status": "cleared",
            "cleared": list(cleared),
            "cleared_count": len(cleared),
        }


class LauncherService:
    def __init__(
        self,
        settings: LauncherSettings,
        supervisor: ProcessSupervisor | None,
        gateway_fetcher: GatewayFetcher | None = None,
    ) -> None:
        self._settings = settings
        self._supervisor = supervisor if supervisor is not None else ProcessSupervisor(settings=settings)
        self._gateway_fetcher = gateway_fetcher if gateway_fetcher is not None else default_gateway_fetcher

    def profile(self) -> LauncherProfile:
        return load_profile(self._settings.profile_path)

    def update_profile(self, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        profile = editable_profile_update(self.profile(), payload)
        save_profile(self._settings.profile_path, profile)
        return self.state()

    def state(self) -> dict[str, JsonValue]:
        profile = self.profile()
        preflight = run_preflight(settings=self._settings, profile=profile)
        commands = build_commands(settings=self._settings, profile=profile)
        return {
            "schema_version": LAUNCHER_STATE_SCHEMA_VERSION,
            "repo_root": str(self._settings.repo_root),
            "profile_path": str(self._settings.profile_path),
            "profile": profile.to_dict(),
            "agent_settings": agent_settings(profile),
            "provider_models": provider_models(profile),
            "preflight": preflight,
            "actions": [commands[action_id].to_dict() for action_id in sorted(commands)],
            "processes": self._supervisor.statuses(),
        }

    def preflight(self) -> dict[str, JsonValue]:
        return run_preflight(settings=self._settings, profile=self.profile())

    def start_action(self, action_id: str) -> dict[str, JsonValue]:
        commands = build_commands(settings=self._settings, profile=self.profile())
        command = commands.get(action_id)
        if command is None:
            raise LauncherServiceError(f"unknown launcher action '{action_id}'.")
        return self._supervisor.start(command)

    def stop_action(self, action_id: str) -> dict[str, JsonValue]:
        return self._supervisor.stop(action_id)

    def stop_all(self) -> dict[str, JsonValue]:
        return self._supervisor.stop_all()

    def clear_statuses(self) -> dict[str, JsonValue]:
        return self._supervisor.clear_statuses()

    def observability(self, limit: int) -> dict[str, JsonValue]:
        if limit <= 0:
            raise LauncherServiceError("observability limit must be positive.")
        if limit > _MAX_OBSERVABILITY_LIMIT:
            raise LauncherServiceError(f"observability limit must be at most {_MAX_OBSERVABILITY_LIMIT}.")
        settings = _console_settings(settings=self._settings, profile=self.profile())
        events = console_events(settings=settings, fetcher=self._gateway_fetcher, limit=limit, session_id=None)
        overview = console_overview_from_events(settings=settings, fetcher=self._gateway_fetcher, events_payload=events)
        return {
            "schema_version": LAUNCHER_OBSERVABILITY_SCHEMA_VERSION,
            "gateway_base_url": settings.gateway_base_url,
            "overview": overview,
            "events": events,
            "latency": console_latency_summary(events),
        }

    def observability_trace(self, trace_id: str) -> dict[str, JsonValue]:
        if trace_id == "":
            raise LauncherServiceError("trace_id must not be empty.")
        settings = _console_settings(settings=self._settings, profile=self.profile())
        try:
            trace = console_trace(settings=settings, fetcher=self._gateway_fetcher, trace_id=trace_id)
        except ValueError as exc:
            return {
                "status": "unavailable",
                "trace_id": trace_id,
                "error": str(exc),
                "trace": None,
            }
        return {
            "status": "ok",
            "trace_id": trace_id,
            "error": None,
            "trace": trace,
        }


def default_settings(repo_root: Path) -> LauncherSettings:
    return LauncherSettings(
        repo_root=repo_root,
        profile_path=repo_root / ".aegis" / "launcher-profiles" / "local-qwen3-4b.json",
        log_dir=repo_root / ".aegis" / "launcher-logs",
    )


def _console_settings(settings: LauncherSettings, profile: LauncherProfile) -> ConsoleSettings:
    return ConsoleSettings(
        gateway_base_url=f"http://{profile.gateway_host}:{profile.gateway_port}",
        request_timeout_seconds=_OBSERVABILITY_REQUEST_TIMEOUT_SECONDS,
        smoke_report_path=settings.repo_root / _CIFT_SMOKE_REPORT_PATH,
        sample_audit_jsonl_path=Path(profile.audit_jsonl_path),
        operator_profile="balanced",
    )


def agent_settings(profile: LauncherProfile) -> dict[str, JsonValue]:
    base_url = f"http://{profile.gateway_host}:{profile.gateway_port}/v1"
    text = (
        "Provider: Custom OpenAI-compatible\n"
        f"Base URL: {base_url}\n"
        "API key: aegis-local-dev-key\n"
        f"Model: {profile.provider_model}"
    )
    return {
        "provider": "Custom OpenAI-compatible",
        "base_url": base_url,
        "api_key": "aegis-local-dev-key",
        "model": profile.provider_model,
        "text": text,
    }


def provider_models(profile: LauncherProfile) -> dict[str, JsonValue]:
    installed = _ollama_model_names(profile) or []
    return {
        "installed": cast(JsonValue, installed),
        "selected": profile.provider_model,
        "selected_installed": profile.provider_model in installed,
    }


def run_preflight(settings: LauncherSettings, profile: LauncherProfile) -> dict[str, JsonValue]:
    checks = [
        _binary_check("uv", "uv package runner"),
        _binary_check("ollama", "Ollama CLI"),
        _path_check(settings.repo_root / profile.mps_python_path, "CIFT Python environment"),
        _path_check(
            settings.repo_root / "introspection/scripts/check_cift_device_preflight.py",
            "GPU preflight script",
        ),
        _path_check(settings.repo_root / "introspection/scripts/run_cift_extractor_sidecar.py", "CIFT sidecar script"),
        _path_check(
            settings.repo_root / profile.cift_binding.strict_deployment_env_path,
            "Selected-choice CIFT deployment env",
        ),
        _path_check(
            settings.repo_root / profile.cift_binding.freeform_strict_deployment_env_path,
            "Freeform CIFT deployment env",
        ),
        _port_check(profile.gateway_host, profile.gateway_port, "Aegis gateway port"),
        _port_check(profile.sidecar_host, profile.sidecar_port, "CIFT sidecar port"),
        _url_check(_ollama_tags_url(profile.provider_base_url), "Ollama server"),
        _ollama_model_check(profile),
    ]
    failed = [check for check in checks if check["status"] == "failed"]
    warned = [check for check in checks if check["status"] == "warn"]
    if failed:
        overall = "failed"
    elif warned:
        overall = "warn"
    else:
        overall = "passed"
    return {
        "schema_version": LAUNCHER_PREFLIGHT_SCHEMA_VERSION,
        "overall_status": overall,
        "checks": cast(list[JsonValue], checks),
    }


def build_commands(settings: LauncherSettings, profile: LauncherProfile) -> dict[str, ManagedCommand]:
    env = _base_env(profile)
    strict_env = {
        **_strict_cift_env(settings.repo_root / profile.cift_binding.strict_deployment_env_path),
        **_strict_cift_env(settings.repo_root / profile.cift_binding.freeform_strict_deployment_env_path),
    }
    gateway_env = {**env, **strict_env, **_gateway_env(profile)}
    sidecar_env = {**env, "PYTHONPATH": "src:introspection/src"}
    smoke_env = {**gateway_env, "PYTHONPATH": "src:introspection/src"}
    return {
        "open_ollama": ManagedCommand(
            action_id="open_ollama",
            label="Open Ollama",
            argv=("open", "-a", "Ollama"),
            env=env,
            cwd=settings.repo_root,
            kind="one-shot",
        ),
        "pull_model": ManagedCommand(
            action_id="pull_model",
            label="Pull provider model",
            argv=("ollama", "pull", profile.provider_model),
            env=env,
            cwd=settings.repo_root,
            kind="one-shot",
        ),
        "mps_preflight": ManagedCommand(
            action_id="mps_preflight",
            label="Run GPU preflight",
            argv=(
                str(settings.repo_root / profile.mps_python_path),
                "introspection/scripts/check_cift_device_preflight.py",
                "--device",
                profile.cift_binding.device,
            ),
            env=sidecar_env,
            cwd=settings.repo_root,
            kind="one-shot",
        ),
        "cift_sidecar": ManagedCommand(
            action_id="cift_sidecar",
            label="Start CIFT sidecar",
            argv=_sidecar_argv(settings=settings, profile=profile),
            env=sidecar_env,
            cwd=settings.repo_root,
            kind="long-running",
        ),
        "gateway": ManagedCommand(
            action_id="gateway",
            label="Start Aegis gateway",
            argv=("uv", "run", "aegis-proxy", "--host", profile.gateway_host, "--port", str(profile.gateway_port)),
            env=gateway_env,
            cwd=settings.repo_root,
            kind="long-running",
        ),
        "console": ManagedCommand(
            action_id="console",
            label="Start operator console",
            argv=(
                "uv",
                "run",
                "aegis-console",
                "--gateway-url",
                f"http://{profile.gateway_host}:{profile.gateway_port}",
                "--host",
                profile.console_host,
                "--port",
                str(profile.console_port),
            ),
            env=gateway_env,
            cwd=settings.repo_root,
            kind="long-running",
        ),
        "cift_smoke": ManagedCommand(
            action_id="cift_smoke",
            label="Run CIFT smoke",
            argv=_cift_smoke_argv(profile),
            env=smoke_env,
            cwd=settings.repo_root,
            kind="one-shot",
        ),
        "real_provider_smoke": ManagedCommand(
            action_id="real_provider_smoke",
            label="Run real-provider smoke",
            argv=_real_provider_smoke_argv(profile),
            env=smoke_env,
            cwd=settings.repo_root,
            kind="one-shot",
        ),
    }


def _sidecar_argv(settings: LauncherSettings, profile: LauncherProfile) -> tuple[str, ...]:
    binding = profile.cift_binding
    feature_key_args = _sidecar_feature_key_args(
        selected_choice_feature_key=binding.feature_key,
        freeform_feature_key=binding.freeform_feature_key,
    )
    return (
        str(settings.repo_root / profile.mps_python_path),
        "introspection/scripts/run_cift_extractor_sidecar.py",
        "--model-id",
        binding.model_id,
        "--revision",
        binding.revision,
        "--device",
        binding.device,
        "--dtype",
        binding.dtype,
        *feature_key_args,
        "--selected-choice-readout-token-count",
        str(binding.selected_choice_readout_token_count),
        "--host",
        profile.sidecar_host,
        "--port",
        str(profile.sidecar_port),
        "--api-key-env-var",
        "AEGIS_CIFT_EXTRACTOR_API_KEY",
    )


def _sidecar_feature_key_args(selected_choice_feature_key: str, freeform_feature_key: str) -> tuple[str, ...]:
    feature_keys = (selected_choice_feature_key, freeform_feature_key)
    argv: list[str] = []
    for feature_key in dict.fromkeys(feature_keys):
        argv.extend(("--feature-key", feature_key))
    return tuple(argv)


def _cift_smoke_argv(profile: LauncherProfile) -> tuple[str, ...]:
    binding = profile.cift_binding
    return (
        "uv",
        "run",
        "aegis-proxy-cift-smoke",
        "--url",
        f"http://{profile.gateway_host}:{profile.gateway_port}",
        "--sidecar-url",
        f"http://{profile.sidecar_host}:{profile.sidecar_port}",
        "--gateway-model",
        profile.provider_model,
        "--report-id",
        "qwen3_4b_launcher_cift_freeform_smoke_v1",
        "--timeout",
        "120",
        "--detector-name",
        "cift_runtime",
        "--sidecar-feature-key",
        binding.freeform_feature_key,
        "--expected-gateway-feature-source",
        "self_hosted_activation_extractor",
        "--expected-extractor-id",
        "trusted-activation-sidecar",
        "--expected-sidecar-model-id",
        binding.model_id,
        "--expected-sidecar-revision",
        binding.revision,
        "--expected-sidecar-device",
        binding.device,
        "--expected-sidecar-hidden-size",
        str(binding.hidden_size),
        "--expected-sidecar-layer-count",
        str(binding.layer_count),
        "--expected-sidecar-tokenizer-fingerprint-sha256",
        binding.tokenizer_fingerprint_sha256,
        "--expected-sidecar-special-tokens-map-sha256",
        binding.special_tokens_map_sha256,
        "--expected-sidecar-chat-template-sha256",
        binding.chat_template_sha256,
        "--selected-choice-readout-token-count",
        str(binding.selected_choice_readout_token_count),
        "--sidecar-api-key-env-var",
        "AEGIS_CIFT_EXTRACTOR_API_KEY",
        "--output",
        "introspection/data/reports/qwen3_4b_launcher_cift_freeform_smoke_v1.json",
    )


def _real_provider_smoke_argv(profile: LauncherProfile) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "aegis-proxy-smoke",
        "--url",
        f"http://{profile.gateway_host}:{profile.gateway_port}",
        "--timeout",
        "120",
        "--provider-mode",
        "real-provider",
        "--require-cift-pre-generation-block",
        "--output",
        "introspection/data/reports/aegis_launcher_real_provider_smoke_v1.json",
    )


def _base_env(profile: LauncherProfile) -> dict[str, str]:
    env = dict(os.environ)
    env["AEGIS_CIFT_EXTRACTOR_API_KEY"] = profile.cift_api_key
    return env


def _gateway_env(profile: LauncherProfile) -> dict[str, str]:
    return {
        "AEGIS_PROVIDER": "openai_compatible",
        "AEGIS_OPENAI_BASE_URL": profile.provider_base_url,
        "AEGIS_OPENAI_API_KEY": profile.provider_api_key,
        "AEGIS_OPENAI_MODEL": profile.provider_model,
        "AEGIS_OPENAI_TIMEOUT_SECONDS": _LOCAL_PROVIDER_TIMEOUT_SECONDS,
        "AEGIS_AUDIT_JSONL_PATH": profile.audit_jsonl_path,
        "AEGIS_CIFT_EXTRACTOR_BASE_URL": f"http://{profile.sidecar_host}:{profile.sidecar_port}",
    }


def _strict_cift_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("export "):
            continue
        key_value = line.removeprefix("export ").strip()
        if "=" not in key_value:
            continue
        key, raw_value = key_value.split("=", 1)
        if "${" in raw_value:
            continue
        env[key] = shlex.split(raw_value)[0] if raw_value else ""
    return env


def _binary_check(binary: str, label: str) -> dict[str, JsonValue]:
    path = shutil.which(binary)
    if path is None:
        return _check(label, "failed", f"{binary} was not found on PATH.")
    return _check(label, "passed", path)


def _path_check(path: Path, label: str) -> dict[str, JsonValue]:
    if path.exists():
        return _check(label, "passed", str(path))
    return _check(label, "failed", f"Missing {path}.")


def _port_check(host: str, port: int, label: str) -> dict[str, JsonValue]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        result = sock.connect_ex((host, port))
    if result == 0:
        return _check(label, "warn", f"{host}:{port} already accepts connections.")
    return _check(label, "passed", f"{host}:{port} is available.")


def _url_check(url: str, label: str) -> dict[str, JsonValue]:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return _check(label, "passed", f"HTTP {response.status} from {url}.")
    except urllib.error.URLError as error:
        return _check(label, "warn", f"{url} is not reachable: {error.reason}.")


def _ollama_model_check(profile: LauncherProfile) -> dict[str, JsonValue]:
    installed = _ollama_model_names(profile)
    if installed is None:
        return _check("Provider model", "warn", "Cannot inspect Ollama models because Ollama is not reachable.")
    if profile.provider_model in installed:
        return _check("Provider model", "passed", f"{profile.provider_model} is present in Ollama tags.")
    return _check("Provider model", "warn", f"{profile.provider_model} was not found. Use Pull provider model.")


def _ollama_model_names(profile: LauncherProfile) -> list[str] | None:
    url = _ollama_tags_url(profile.provider_base_url)
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.URLError:
        return None
    return _parse_ollama_model_names(payload)


def _parse_ollama_model_names(raw_payload: str) -> list[str]:
    try:
        payload: object = json.loads(raw_payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name)
    return sorted(dict.fromkeys(names))


def _ollama_tags_url(provider_base_url: str) -> str:
    parsed = urllib.parse.urlparse(provider_base_url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/api/tags", "", "", ""))


def _check(label: str, status: str, detail: str) -> dict[str, JsonValue]:
    return {"label": label, "status": status, "detail": detail}


def _read_log_excerpt(path: Path) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    offset = max(0, size - _LOG_BYTES)
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read(_LOG_BYTES)
    return data[-_LOG_BYTES:].decode("utf-8", errors="replace")
