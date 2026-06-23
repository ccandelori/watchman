from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import JsonValue, NormalizedTurn
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.cift_runtime import (
    CiftFeatureExtractor,
    CiftFeatureVectorAnnotator,
    CiftRuntimeDetector,
    CiftRuntimeLinearModel,
    load_cift_runtime_model,
)
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider
from aegis_introspection.activations import HiddenStateForwardPass
from aegis_introspection.cift_live_extractor import (
    HiddenStateRunner,
    LiveCiftFeatureExtractor,
    LoadedModelHiddenStateRunner,
)
from aegis_introspection.cift_runtime_eval import load_runtime_requests_jsonl
from aegis_introspection.model_loader import ModelDTypeName, ModelLoadConfig, load_causal_lm
from aegis_introspection.sealed_holdout import assert_unsealed_jsonl_tags, assert_unsealed_paths


class CiftLiveBenchmarkError(ValueError):
    """Raised when live CIFT benchmark execution cannot be completed."""


@dataclass(frozen=True)
class CiftLiveBenchmarkConfig:
    runtime_turns_path: Path
    runtime_model_path: Path
    output_json_path: Path
    output_markdown_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftLiveBenchmarkRow:
    trace_id: str
    example_id: str | None
    turn_index: int
    detector_action: str
    policy_action: str
    capability_status: str
    score: float
    model_forward_ms: float
    feature_extraction_ms: float
    detector_ms: float
    total_runtime_ms: float

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "trace_id": self.trace_id,
            "example_id": self.example_id,
            "turn_index": self.turn_index,
            "detector_action": self.detector_action,
            "policy_action": self.policy_action,
            "capability_status": self.capability_status,
            "score": self.score,
            "model_forward_ms": self.model_forward_ms,
            "feature_extraction_ms": self.feature_extraction_ms,
            "detector_ms": self.detector_ms,
            "total_runtime_ms": self.total_runtime_ms,
        }


@dataclass(frozen=True)
class CiftLiveBenchmarkReport:
    schema_version: str
    model_id: str
    revision: str
    selected_device: str
    runtime_model_path: str
    runtime_turns_path: str
    request_count: int
    model_load_ms: float
    action_counts: dict[str, int]
    policy_action_counts: dict[str, int]
    capability_status_counts: dict[str, int]
    model_forward_ms: dict[str, float]
    feature_extraction_ms: dict[str, float]
    detector_ms: dict[str, float]
    total_runtime_ms: dict[str, float]
    rows: tuple[CiftLiveBenchmarkRow, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "selected_device": self.selected_device,
            "runtime_model_path": self.runtime_model_path,
            "runtime_turns_path": self.runtime_turns_path,
            "request_count": self.request_count,
            "model_load_ms": self.model_load_ms,
            "action_counts": self.action_counts,
            "policy_action_counts": self.policy_action_counts,
            "capability_status_counts": self.capability_status_counts,
            "model_forward_ms": self.model_forward_ms,
            "feature_extraction_ms": self.feature_extraction_ms,
            "detector_ms": self.detector_ms,
            "total_runtime_ms": self.total_runtime_ms,
            "rows": [row.to_dict() for row in self.rows],
        }


class TimingHiddenStateRunner:
    def __init__(self, wrapped: HiddenStateRunner) -> None:
        self.forward_latencies_ms: list[float] = []
        self._wrapped = wrapped

    def run(self, prompt: str) -> HiddenStateForwardPass:
        started_at = time.perf_counter()
        try:
            return self._wrapped.run(prompt)
        finally:
            self.forward_latencies_ms.append(_elapsed_ms(started_at))


class TimingFeatureExtractor:
    def __init__(self, wrapped: CiftFeatureExtractor) -> None:
        self.extraction_latencies_ms: list[float] = []
        self._wrapped = wrapped

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        started_at = time.perf_counter()
        try:
            return self._wrapped.extract_feature_vector(turn=turn, feature_key=feature_key)
        finally:
            self.extraction_latencies_ms.append(_elapsed_ms(started_at))


def run_cift_live_benchmark(config: CiftLiveBenchmarkConfig) -> CiftLiveBenchmarkReport:
    _validate_config(config)
    paths = (
        config.runtime_turns_path,
        config.runtime_model_path,
        config.output_json_path,
        config.output_markdown_path,
    )
    assert_unsealed_paths(
        paths=paths,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT benchmark",
    )
    assert_unsealed_jsonl_tags(
        path=config.runtime_turns_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT benchmark",
    )
    requests = load_runtime_requests_jsonl(config.runtime_turns_path)
    runtime_model = _runtime_model(config.runtime_model_path)
    started_load = time.perf_counter()
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
    model_load_ms = _elapsed_ms(started_load)
    timing_runner = TimingHiddenStateRunner(LoadedModelHiddenStateRunner(loaded_model=loaded_model))
    timing_extractor = TimingFeatureExtractor(
        LiveCiftFeatureExtractor(runner=timing_runner, feature_key=runtime_model.feature_key)
    )
    runtime = AegisRuntime(
        turn_annotators=(
            CiftFeatureVectorAnnotator(
                feature_key=runtime_model.feature_key,
                extractor=timing_extractor,
                source=config.feature_source,
            ),
        ),
        pre_generation_detectors=(CiftRuntimeDetector(detector_name=config.detector_name, model=runtime_model),),
        post_generation_detectors=(),
        session_detectors=(),
        policy_engine=SeverityPolicyEngine(),
        audit_sink=InMemoryAuditSink(),
        model_provider=MockModelProvider(default_content=config.mock_response),
    )
    rows = _benchmark_rows(
        runtime=runtime,
        requests=requests,
        timing_runner=timing_runner,
        timing_extractor=timing_extractor,
    )
    report = _report(
        config=config,
        selected_device=loaded_model.device.name,
        model_load_ms=model_load_ms,
        rows=rows,
    )
    write_cift_live_benchmark_json(config.output_json_path, report)
    write_cift_live_benchmark_markdown(config.output_markdown_path, report)
    return report


def write_cift_live_benchmark_json(path: Path, report: CiftLiveBenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_cift_live_benchmark_markdown(path: Path, report: CiftLiveBenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_benchmark_markdown(report), encoding="utf-8")


def _benchmark_rows(
    runtime: AegisRuntime,
    requests: tuple[RuntimeRequest, ...],
    timing_runner: TimingHiddenStateRunner,
    timing_extractor: TimingFeatureExtractor,
) -> tuple[CiftLiveBenchmarkRow, ...]:
    rows: list[CiftLiveBenchmarkRow] = []
    for request in requests:
        forward_count = len(timing_runner.forward_latencies_ms)
        extraction_count = len(timing_extractor.extraction_latencies_ms)
        started_at = time.perf_counter()
        response = runtime.evaluate_turn(request)
        total_runtime_ms = _elapsed_ms(started_at)
        if len(response.detector_results) != 1:
            raise CiftLiveBenchmarkError("Live CIFT benchmark expects exactly one detector result per request.")
        detector_result = response.detector_results[0]
        model_forward_ms = _latest_latency(
            latencies=timing_runner.forward_latencies_ms,
            previous_count=forward_count,
            metric_name="model_forward_ms",
        )
        feature_extraction_ms = _latest_latency(
            latencies=timing_extractor.extraction_latencies_ms,
            previous_count=extraction_count,
            metric_name="feature_extraction_ms",
        )
        rows.append(
            CiftLiveBenchmarkRow(
                trace_id=request.trace_id,
                example_id=_example_id(request.metadata),
                turn_index=request.turn_index,
                detector_action=detector_result.recommended_action.value,
                policy_action=response.policy_decision.final_action.value,
                capability_status=detector_result.capability_status.value,
                score=detector_result.score,
                model_forward_ms=model_forward_ms,
                feature_extraction_ms=feature_extraction_ms,
                detector_ms=detector_result.latency_ms,
                total_runtime_ms=total_runtime_ms,
            )
        )
    return tuple(rows)


def _report(
    config: CiftLiveBenchmarkConfig,
    selected_device: str,
    model_load_ms: float,
    rows: tuple[CiftLiveBenchmarkRow, ...],
) -> CiftLiveBenchmarkReport:
    return CiftLiveBenchmarkReport(
        schema_version="aegis_introspection.cift_live_benchmark/v1",
        model_id=config.model_id,
        revision=config.revision,
        selected_device=selected_device,
        runtime_model_path=str(config.runtime_model_path),
        runtime_turns_path=str(config.runtime_turns_path),
        request_count=len(rows),
        model_load_ms=model_load_ms,
        action_counts=_counts(tuple(row.detector_action for row in rows)),
        policy_action_counts=_counts(tuple(row.policy_action for row in rows)),
        capability_status_counts=_counts(tuple(row.capability_status for row in rows)),
        model_forward_ms=_summary(tuple(row.model_forward_ms for row in rows)),
        feature_extraction_ms=_summary(tuple(row.feature_extraction_ms for row in rows)),
        detector_ms=_summary(tuple(row.detector_ms for row in rows)),
        total_runtime_ms=_summary(tuple(row.total_runtime_ms for row in rows)),
        rows=rows,
    )


def _runtime_model(path: Path) -> CiftRuntimeLinearModel:
    return load_cift_runtime_model(path)


def _benchmark_markdown(report: CiftLiveBenchmarkReport) -> str:
    lines = [
        "# Live CIFT Benchmark",
        "",
        "## Source",
        "",
        f"- Model: `{report.model_id}`",
        f"- Revision: `{report.revision}`",
        f"- Selected device: `{report.selected_device}`",
        f"- Runtime model: `{report.runtime_model_path}`",
        f"- Runtime turns: `{report.runtime_turns_path}`",
        f"- Requests: `{report.request_count}`",
        f"- Model load: `{report.model_load_ms:.4f} ms`",
        "",
        "## Latency",
        "",
        "| Metric | Mean ms | Median ms | P95 ms | Min ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|",
        _summary_row("Model forward", report.model_forward_ms),
        _summary_row("Feature extraction", report.feature_extraction_ms),
        _summary_row("Detector", report.detector_ms),
        _summary_row("Total runtime", report.total_runtime_ms),
        "",
        "## Actions",
        "",
        f"- Detector actions: `{report.action_counts}`",
        f"- Policy actions: `{report.policy_action_counts}`",
        f"- Capability statuses: `{report.capability_status_counts}`",
        "",
        "## Rows",
        "",
        "| Example | Score | Detector Action | Policy Action | Forward ms | Feature ms | Total ms |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"`{row.example_id}` | "
            f"{row.score:.6f} | "
            f"`{row.detector_action}` | "
            f"`{row.policy_action}` | "
            f"{row.model_forward_ms:.4f} | "
            f"{row.feature_extraction_ms:.4f} | "
            f"{row.total_runtime_ms:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _summary_row(name: str, values: dict[str, float]) -> str:
    return (
        f"| {name} | {values['mean']:.4f} | {values['median']:.4f} | "
        f"{values['p95']:.4f} | {values['min']:.4f} | {values['max']:.4f} |"
    )


def _summary(values: tuple[float, ...]) -> dict[str, float]:
    if len(values) == 0:
        raise CiftLiveBenchmarkError("Cannot summarize an empty latency set.")
    ordered = tuple(sorted(values))
    return {
        "mean": sum(values) / len(values),
        "median": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(ordered_values: tuple[float, ...], quantile: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    index = quantile * (len(ordered_values) - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    weight = index - lower_index
    return ordered_values[lower_index] * (1.0 - weight) + ordered_values[upper_index] * weight


def _counts(values: tuple[str, ...]) -> dict[str, int]:
    return dict(Counter(values))


def _latest_latency(latencies: list[float], previous_count: int, metric_name: str) -> float:
    if len(latencies) != previous_count + 1:
        raise CiftLiveBenchmarkError(f"Expected one new {metric_name} value.")
    return latencies[-1]


def _example_id(metadata: dict[str, JsonValue]) -> str | None:
    value = metadata.get("example_id")
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise CiftLiveBenchmarkError("metadata.example_id must be a non-empty string when present.")
    return value


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _validate_config(config: CiftLiveBenchmarkConfig) -> None:
    if config.detector_name == "":
        raise CiftLiveBenchmarkError("detector_name must not be empty.")
    if config.feature_source == "":
        raise CiftLiveBenchmarkError("feature_source must not be empty.")
    if config.mock_response == "":
        raise CiftLiveBenchmarkError("mock_response must not be empty.")
    if config.model_id == "":
        raise CiftLiveBenchmarkError("model_id must not be empty.")
    if config.revision == "":
        raise CiftLiveBenchmarkError("revision must not be empty.")
