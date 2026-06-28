from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import aegis.launcher.service as launcher_service
from aegis.console.service import ConsoleGatewayError, ConsoleSettings
from aegis.core.contracts import JsonValue
from aegis.launcher.app import create_app
from aegis.launcher.profile import default_profile, editable_profile_update, save_profile
from aegis.launcher.service import (
    LauncherService,
    LauncherSettings,
    ManagedCommand,
    ManagedProcess,
    ProcessSupervisor,
    _parse_ollama_model_names,
    build_commands,
)


class RecordingSupervisor:
    def __init__(self) -> None:
        self.started: list[ManagedCommand] = []
        self.stopped: list[str] = []

    def statuses(self) -> dict[str, JsonValue]:
        if len(self.started) == 0:
            return {}
        last = self.started[-1]
        return {
            last.action_id: {
                "action_id": last.action_id,
                "label": last.label,
                "pid": 12345,
                "status": "running",
                "exit_code": None,
                "started_at": 1.0,
                "kind": last.kind,
                "log_path": "/tmp/aegis-test.log",
                "log_excerpt": "started",
            }
        }

    def start(self, command: ManagedCommand) -> dict[str, JsonValue]:
        self.started.append(command)
        return self.statuses()[command.action_id]  # type: ignore[index]

    def stop(self, action_id: str) -> dict[str, JsonValue]:
        self.stopped.append(action_id)
        return {"action_id": action_id, "status": "stopping"}

    def stop_all(self) -> dict[str, JsonValue]:
        self.stopped.append("all")
        return {"all": "stopping"}

    def clear_statuses(self) -> dict[str, JsonValue]:
        cleared = [command.action_id for command in self.started]
        self.started.clear()
        return {"status": "cleared", "cleared": cleared, "cleared_count": len(cleared)}


def test_editable_profile_update_keeps_cift_binding_locked() -> None:
    profile = default_profile()
    updated = editable_profile_update(
        profile,
        {
            "provider_base_url": "http://127.0.0.1:1234/v1",
            "provider_model": "custom-model",
            "cift_binding": {"model_id": "Not/Allowed"},
        },
    )

    assert updated.provider_base_url == "http://127.0.0.1:1234/v1"
    assert updated.provider_model == "custom-model"
    assert updated.cift_binding.model_id == "Qwen/Qwen3-4B"
    assert updated.cift_binding.revision == "1cfa9a7208912126459214e8b04321603b3df60c"


def test_build_commands_uses_whitelisted_argv_not_shell(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = default_profile()
    _write_strict_env(tmp_path / profile.cift_binding.strict_deployment_env_path)
    _write_freeform_strict_env(tmp_path / profile.cift_binding.freeform_strict_deployment_env_path)

    commands = build_commands(settings=settings, profile=profile)

    assert commands["gateway"].argv == ("uv", "run", "aegis-proxy", "--host", "127.0.0.1", "--port", "8000")
    assert commands["gateway"].env["AEGIS_OPENAI_BASE_URL"] == "http://127.0.0.1:11434/v1"
    assert commands["gateway"].env["AEGIS_OPENAI_TIMEOUT_SECONDS"] == "90"
    assert commands["gateway"].env["AEGIS_CIFT_CERTIFICATION_MODE"] == "strict"
    assert commands["gateway"].env["AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH"] == "selected.json"
    assert commands["gateway"].env["AEGIS_CIFT_FREEFORM_MODEL_PATH"] == "freeform.json"
    assert commands["gateway"].env["AEGIS_CIFT_FREEFORM_CERTIFICATION_MANIFEST_SHA256"] == "b" * 64
    assert commands["gateway"].env["AEGIS_CIFT_EXTRACTOR_API_KEY"] == "set-a-deployment-secret"
    assert commands["cift_sidecar"].argv[0] == str(tmp_path / ".venv-mps313/bin/python")
    assert "Qwen/Qwen3-4B" in commands["cift_sidecar"].argv
    assert _feature_keys_from_argv(commands["cift_sidecar"].argv) == (
        "selected_choice_window_layer_21",
        "final_token_layer_12",
    )
    assert _argument_value(commands["cift_smoke"].argv, "--sidecar-feature-key") == "final_token_layer_12"
    assert _argument_value(commands["cift_smoke"].argv, "--report-id") == "qwen3_4b_launcher_cift_freeform_smoke_v1"
    assert "${AEGIS_CIFT_EXTRACTOR_API_KEY" not in commands["gateway"].env.values()


def test_parse_ollama_model_names_deduplicates_and_sorts() -> None:
    payload = """
    {
      "models": [
        {"name": "qwen3:4b"},
        {"name": "llama3.2:latest"},
        {"name": "qwen3:4b"},
        {"digest": "missing-name"},
        {"name": ""}
      ]
    }
    """

    assert _parse_ollama_model_names(payload) == ["llama3.2:latest", "qwen3:4b"]
    assert _parse_ollama_model_names("{not json") == []


def test_managed_process_reports_exit_duration_and_log_excerpt(tmp_path: Path) -> None:
    log_path = tmp_path / "process.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            (sys.executable, "-c", "print('failure output'); raise SystemExit(7)"),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=False,
        )
    started_at = time.time()
    process.wait(timeout=5)
    managed = ManagedProcess(
        action_id="cift_smoke",
        label="Run CIFT smoke",
        process=process,
        started_at=started_at,
        log_path=log_path,
        kind="one-shot",
    )

    payload = managed.to_dict()
    second_payload = managed.to_dict()

    assert payload["status"] == "exited"
    assert payload["exit_code"] == 7
    assert isinstance(payload["finished_at"], float)
    assert isinstance(payload["runtime_seconds"], float)
    assert payload["runtime_seconds"] >= 0.0
    assert payload["finished_at"] == second_payload["finished_at"]
    assert "failure output" in str(payload["log_excerpt"])


def test_process_supervisor_clears_exited_statuses_and_starts_fresh_log(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    supervisor = ProcessSupervisor(settings=settings)

    supervisor.start(_print_command(tmp_path, "cift_smoke", "old output"))
    old_status = _wait_for_status(supervisor, "cift_smoke", "exited", 5.0, 0.02)
    cleared = supervisor.clear_statuses()
    supervisor.start(_print_command(tmp_path, "cift_smoke", "new output"))
    new_status = _wait_for_status(supervisor, "cift_smoke", "exited", 5.0, 0.02)

    assert "old output" in str(old_status["log_excerpt"])
    assert cleared["cleared"] == ["cift_smoke"]
    assert supervisor.statuses() != {}
    assert "new output" in str(new_status["log_excerpt"])
    assert "old output" not in str(new_status["log_excerpt"])


def test_launcher_app_serves_state_updates_profile_and_starts_action(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    profile = default_profile()
    save_profile(settings.profile_path, profile)
    _write_strict_env(tmp_path / profile.cift_binding.strict_deployment_env_path)
    _write_freeform_strict_env(tmp_path / profile.cift_binding.freeform_strict_deployment_env_path)
    supervisor = RecordingSupervisor()
    service = LauncherService(settings=settings, supervisor=supervisor)  # type: ignore[arg-type]
    monkeypatch.setattr(launcher_service, "_sidecar_healthy", lambda running_profile: False)
    monkeypatch.setattr(launcher_service, "_gateway_ready", lambda running_profile: False)
    client = TestClient(create_app(service))

    page = client.get("/")
    state = client.get("/api/state")
    capabilities = client.get("/api/launcher/capabilities")
    update = client.put("/api/profile", json={"provider_model": "qwen3:4b-instruct"})
    start = client.post("/api/actions/gateway/start")

    assert page.status_code == 200
    assert "Aegis Local Launcher" in page.text
    assert state.status_code == 200
    assert state.json()["agent_settings"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert capabilities.status_code == 200
    assert capabilities.json()["schema_version"] == "aegis.launcher_capabilities/v1"
    assert capabilities.json()["features"]["observability"] is True
    assert "/api/observability" in capabilities.json()["routes"]
    assert update.status_code == 200
    assert update.json()["profile"]["provider_model"] == "qwen3:4b-instruct"
    assert update.json()["profile"]["cift_binding"]["model_id"] == "Qwen/Qwen3-4B"
    assert update.json()["profile"]["cift_binding"]["freeform_feature_key"] == "final_token_layer_12"
    assert (
        update.json()["profile"]["cift_binding"]["freeform_strict_deployment_env_path"]
        == "introspection/data/certifications/qwen3_4b_watchman_v14_freeform_final_token_l12/reports/"
        "qwen3_4b_watchman_v14_freeform_final_token_l12_strict_deployment_env.sh"
    )
    assert start.status_code == 200
    assert supervisor.started[-1].action_id == "gateway"
    assert supervisor.started[-1].env["AEGIS_OPENAI_MODEL"] == "qwen3:4b-instruct"
    assert supervisor.started[-1].env["AEGIS_CIFT_FREEFORM_MODEL_PATH"] == "freeform.json"
    clear = client.post("/api/actions/clear-statuses")

    assert clear.status_code == 200
    assert clear.json()["cleared"] == ["gateway"]
    assert supervisor.started == []


def test_launcher_adopts_already_running_cift_and_gateway(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    profile = default_profile()
    save_profile(settings.profile_path, profile)
    _write_strict_env(tmp_path / profile.cift_binding.strict_deployment_env_path)
    _write_freeform_strict_env(tmp_path / profile.cift_binding.freeform_strict_deployment_env_path)
    supervisor = RecordingSupervisor()
    service = LauncherService(settings=settings, supervisor=supervisor)  # type: ignore[arg-type]
    monkeypatch.setattr(launcher_service, "_sidecar_healthy", lambda running_profile: True)
    monkeypatch.setattr(launcher_service, "_gateway_ready", lambda running_profile: True)

    gateway_start = service.start_action("gateway")
    state = service.state()

    assert gateway_start["status"] == "running"
    assert gateway_start["source"] == "external"
    assert supervisor.started == []
    processes = state["processes"]
    assert isinstance(processes, dict)
    assert processes["cift_sidecar"]["status"] == "running"
    assert processes["cift_sidecar"]["source"] == "external"
    assert processes["gateway"]["status"] == "running"
    assert processes["gateway"]["source"] == "external"


def test_launcher_observability_summarizes_recent_events_and_latency(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = default_profile()
    save_profile(settings.profile_path, profile)
    calls: list[str] = []
    fetcher = _observability_fetcher(
        events=(
            _audit_record("trace-one", latency_ms=10.0, cift_latency_ms=4.0, nimbus_latency_ms=1.0),
            _audit_record("trace-two", latency_ms=30.0, cift_latency_ms=8.0, nimbus_latency_ms=3.0),
        )
    )

    def recording_fetcher(
        console_settings: ConsoleSettings,
        path: str,
        query: tuple[tuple[str, str], ...],
    ) -> dict[str, JsonValue]:
        calls.append(path)
        return fetcher(console_settings, path, query)

    service = LauncherService(
        settings=settings,
        supervisor=RecordingSupervisor(),  # type: ignore[arg-type]
        gateway_fetcher=recording_fetcher,
    )
    client = TestClient(create_app(service))

    response = client.get("/api/observability")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "aegis.launcher_observability/v1"
    assert payload["gateway_base_url"] == "http://127.0.0.1:8000"
    assert payload["overview"]["protection"]["gateway_online"] is True
    assert [event["trace_id"] for event in payload["events"]["events"]] == ["trace-one", "trace-two"]
    assert payload["events"]["events"][0]["latency_ms"] == 10.0
    assert payload["events"]["events"][0]["request"] == {
        "preview": "[REDACTED_SENSITIVE]",
        "preview_role": "user",
        "message_count": 1,
        "tool_call_count": 0,
        "sensitive_span_count": 0,
    }
    assert payload["latency"]["request_latency_ms"] == {
        "count": 2,
        "p50": 20.0,
        "p95": 30.0,
        "min": 10.0,
        "max": 30.0,
    }
    assert payload["latency"]["detector_latency_by_name_ms"]["cift_runtime"] == {
        "count": 2,
        "p50": 6.0,
        "p95": 8.0,
        "min": 4.0,
        "max": 8.0,
    }
    assert payload["latency"]["direct_provider_baseline"]["status"] == "not_measured"
    assert calls.count("/audit/recent") == 1


def test_launcher_observability_rejects_unbounded_limit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    save_profile(settings.profile_path, default_profile())
    service = LauncherService(
        settings=settings,
        supervisor=RecordingSupervisor(),  # type: ignore[arg-type]
        gateway_fetcher=_observability_fetcher(events=()),
    )
    client = TestClient(create_app(service))

    response = client.get("/api/observability?limit=101")

    assert response.status_code == 400
    assert response.json()["error"] == "observability limit must be at most 100."


def test_launcher_observability_rejects_non_integer_limit_with_error_shape(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    save_profile(settings.profile_path, default_profile())
    service = LauncherService(
        settings=settings,
        supervisor=RecordingSupervisor(),  # type: ignore[arg-type]
        gateway_fetcher=_observability_fetcher(events=()),
    )
    client = TestClient(create_app(service))

    response = client.get("/api/observability?limit=abc")

    assert response.status_code == 400
    assert response.json() == {"error": "invalid launcher request parameters."}


def test_launcher_observability_trace_returns_unavailable_when_gateway_has_no_trace(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    save_profile(settings.profile_path, default_profile())
    service = LauncherService(
        settings=settings,
        supervisor=RecordingSupervisor(),  # type: ignore[arg-type]
        gateway_fetcher=_observability_fetcher(events=(), include_trace=False),
    )
    client = TestClient(create_app(service))

    response = client.get("/api/observability/traces/missing-trace")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["trace_id"] == "missing-trace"
    assert payload["trace"] is None


def test_launcher_observability_trace_accepts_path_sensitive_trace_ids(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    save_profile(settings.profile_path, default_profile())
    service = LauncherService(
        settings=settings,
        supervisor=RecordingSupervisor(),  # type: ignore[arg-type]
        gateway_fetcher=_observability_fetcher(events=(), trace_id="session/trace"),
    )
    client = TestClient(create_app(service))

    response = client.get("/api/observability/trace?trace_id=session%2Ftrace")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["trace_id"] == "session/trace"
    assert response.json()["trace"]["trace_id"] == "session/trace"


def _print_command(repo_root: Path, action_id: str, output: str) -> ManagedCommand:
    return ManagedCommand(
        action_id=action_id,
        label="Print test output",
        argv=(sys.executable, "-c", f"print({output!r})"),
        env={},
        cwd=repo_root,
        kind="one-shot",
    )


def _feature_keys_from_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    feature_keys: list[str] = []
    for index, value in enumerate(argv):
        if value == "--feature-key":
            feature_keys.append(argv[index + 1])
    return tuple(feature_keys)


def _argument_value(argv: tuple[str, ...], argument_name: str) -> str:
    try:
        index = argv.index(argument_name)
    except ValueError as exc:
        raise AssertionError(f"{argument_name} not found in argv.") from exc
    return argv[index + 1]


def _wait_for_status(
    supervisor: ProcessSupervisor,
    action_id: str,
    expected_status: str,
    timeout_seconds: float,
    interval_seconds: float,
) -> dict[str, JsonValue]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = supervisor.statuses()[action_id]
        if status["status"] == expected_status:
            return status
        time.sleep(interval_seconds)
    raise AssertionError(f"{action_id} did not reach {expected_status}.")


def _observability_fetcher(
    events: tuple[dict[str, JsonValue], ...],
    include_trace: bool = True,
    trace_id: str = "trace-one",
) -> Callable[[ConsoleSettings, str, tuple[tuple[str, str], ...]], dict[str, JsonValue]]:
    payloads: dict[str, dict[str, JsonValue]] = {
        "/health": {"status": "ok"},
        "/ready": _ready_payload(),
        "/aegis/capabilities": _capabilities_payload(),
        "/audit/recent": {"events": [event for event in events]},
    }
    if include_trace:
        payloads["/audit/explain"] = {
            "trace_id": trace_id,
            "stage_timeline": [{"stage": "cift", "status": "blocked"}],
        }

    def fetcher(
        settings: ConsoleSettings,
        path: str,
        query: tuple[tuple[str, str], ...],
    ) -> dict[str, JsonValue]:
        _ = settings
        _ = query
        if path not in payloads:
            raise ConsoleGatewayError(f"{path} is unavailable.")
        return payloads[path]

    return fetcher


def _ready_payload() -> dict[str, JsonValue]:
    return {
        "ready": True,
        "status": "ready",
        "strict_protected_mode": {"enabled": True},
        "provider": {"status": "ready", "name": "openai_compatible"},
        "dp_honey": {"status": "ready"},
        "provider_egress_guard": {"status": "ready"},
        "nimbus": {"status": "deterministic_beta", "critic_version": "canary-v0"},
        "cift": {
            "status": "ready",
            "capability_mode": "self_hosted_introspection",
            "support_tier": "runtime-enforceable",
            "support_scope": "model-specific CIFT enforcement for Qwen/Qwen3-4B on mps",
            "support_reason": "strict certification binding and live extractor readiness are satisfied.",
            "extractor": {"selected_device": "mps"},
        },
    }


def _capabilities_payload() -> dict[str, JsonValue]:
    return {
        "audit": {"durable_jsonl_enabled": True, "durable_jsonl_path": "/tmp/aegis-local-agent-audit.jsonl"},
        "nimbus": {"status": "deterministic_beta", "critic_kind": "deterministic_canary"},
        "cift": {
            "capability_mode": "self_hosted_introspection",
            "runtime_binding": {
                "certification_id": "cert-qwen3-4b",
                "certification_mode": "strict",
                "runtime_model_sha256": "a" * 64,
                "release_gate_report_sha256": "b" * 64,
                "model_bundle_id": "freeform-linear",
                "source_model_id": "Qwen/Qwen3-4B",
                "source_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "source_selected_device": "mps",
                "source_hidden_size": 2560,
                "source_layer_count": 36,
                "feature_key": "final_token_layer_12",
                "selected_choice_readout_token_count": 4,
                "tokenizer_fingerprint_sha256": "c" * 64,
                "special_tokens_map_sha256": "d" * 64,
                "chat_template_sha256": "e" * 64,
            },
        },
    }


def _audit_record(
    trace_id: str,
    latency_ms: float,
    cift_latency_ms: float,
    nimbus_latency_ms: float,
) -> dict[str, JsonValue]:
    return {
        "trace_id": trace_id,
        "session_id": "session-observe",
        "turn_index": 0,
        "created_at": "2026-06-27T00:00:00+00:00",
        "latency_ms": latency_ms,
        "normalized_turn": {
            "trace_id": trace_id,
            "session_id": "session-observe",
            "turn_index": 0,
            "capability_mode": "self_hosted_introspection",
            "model": {"provider": "ollama", "model_id": "Qwen/Qwen3-4B", "selected_device": "mps"},
            "messages": [{"role": "user", "content": "[REDACTED_SENSITIVE]"}],
            "tool_calls": [],
            "sensitive_spans": [],
            "metadata": {
                "aegis_credential_slot_detection": {
                    "status": "honeytoken_substituted",
                    "credential_needed_count": 1,
                    "honeytoken_substituted_count": 1,
                    "real_secret_present_count": 0,
                },
                "dp_honey_canary_count": 1,
            },
        },
        "detector_results": [
            {
                "detector_name": "cift_runtime",
                "component": "cift",
                "score": 0.91,
                "confidence": 0.95,
                "recommended_action": "block",
                "capability_required": "self_hosted_introspection",
                "capability_status": "active",
                "latency_ms": cift_latency_ms,
            },
            {
                "detector_name": "nimbus",
                "component": "nimbus",
                "score": 0.2,
                "confidence": 0.8,
                "recommended_action": "allow",
                "capability_required": None,
                "capability_status": "active",
                "latency_ms": nimbus_latency_ms,
            },
        ],
        "policy_decision": {
            "final_action": "block",
            "reason": "cift_pre_generation_policy_block",
            "triggered_detectors": ["cift_runtime"],
            "risk_score": 0.91,
            "sanitized_output": "",
        },
        "model_response_metadata": {"provider": "skipped", "reason": "pre_generation_policy_block"},
        "runtime_evidence": {
            "schema_version": "aegis.audit_runtime_evidence/v1",
            "policy_mode": "severity",
            "provider_state": {"status": "skipped", "reason": "pre_generation_policy_block"},
            "credential_slot_status": "honeytoken_substituted",
            "latency_ms": latency_ms,
            "detector_latency_ms": {"cift_runtime": cift_latency_ms, "nimbus": nimbus_latency_ms},
            "fail_closed_events": [],
        },
    }


def _settings(repo_root: Path) -> LauncherSettings:
    return LauncherSettings(
        repo_root=repo_root,
        profile_path=repo_root / ".aegis" / "launcher-profiles" / "test.json",
        log_dir=repo_root / ".aegis" / "launcher-logs",
    )


def _write_strict_env(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "export AEGIS_CIFT_CERTIFICATION_MODE=strict",
                "export AEGIS_CIFT_EXTRACTOR_BASE_URL=http://127.0.0.1:9000",
                "export AEGIS_CIFT_SELECTED_CHOICE_MODEL_PATH=selected.json",
                'export AEGIS_CIFT_EXTRACTOR_API_KEY="${AEGIS_CIFT_EXTRACTOR_API_KEY:?set '
                'AEGIS_CIFT_EXTRACTOR_API_KEY}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_freeform_strict_env(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "export AEGIS_CIFT_CERTIFICATION_MODE=strict",
                "export AEGIS_CIFT_FREEFORM_MODEL_PATH=freeform.json",
                f"export AEGIS_CIFT_FREEFORM_CERTIFICATION_MANIFEST_SHA256={'b' * 64}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
