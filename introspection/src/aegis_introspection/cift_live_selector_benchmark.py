from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from aegis.audit.memory import InMemoryAuditSink
from aegis.core.contracts import (
    JsonValue,
    NormalizedTurn,
)
from aegis.core.orchestrator import AegisRuntime, RuntimeRequest
from aegis.detectors.cift_runtime import (
    CiftFeatureExtractor,
    CiftFeatureVectorAnnotator,
    CiftRuntimeLinearModel,
    CiftRuntimeWindowSelector,
    load_cift_runtime_model,
)
from aegis.policy.engine import SeverityPolicyEngine
from aegis.providers.mock import MockModelProvider
from aegis_introspection.runtime_requests import RuntimeRequestJsonlError
from aegis_introspection.runtime_requests import load_runtime_requests_jsonl as _load_shared_runtime_requests_jsonl
from aegis_introspection.sealed_holdout_policy import (
    SealedHoldoutPolicyError,
)
from aegis_introspection.sealed_holdout_policy import (
    assert_unsealed_jsonl_tags as _assert_shared_unsealed_jsonl_tags,
)
from aegis_introspection.sealed_holdout_policy import (
    assert_unsealed_paths as _assert_shared_unsealed_paths,
)

ModelDTypeName: TypeAlias = Literal["auto", "device", "float32", "float16", "bfloat16"]


class CiftLiveWindowSelectorBenchmarkError(ValueError):
    """Raised when live CIFT window-selector benchmarking cannot be completed."""


class HiddenStateRunner(Protocol):
    def run(self, prompt: str) -> object:
        """Return hidden states for a rendered prompt."""


class TimingHiddenStateRunner:
    def __init__(self, wrapped: HiddenStateRunner) -> None:
        self.forward_latencies_ms: list[float] = []
        self._wrapped = wrapped

    def run(self, prompt: str) -> object:
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


@dataclass(frozen=True)
class CiftLiveWindowSelectorBenchmarkConfig:
    runtime_turns_path: Path
    selected_choice_runtime_model_path: Path
    fallback_runtime_model_path: Path
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
class CiftLiveWindowSelectorBenchmarkRequestConfig:
    runtime_turns_path: Path
    selected_choice_runtime_model_path: Path
    fallback_runtime_model_path: Path
    output_json_path: Path
    output_markdown_path: Path
    detector_name: str
    feature_source: str
    mock_response: str
    model_id: str
    revision: str
    selected_device: str
    model_load_ms: float
    allow_sealed_holdout: bool


@dataclass(frozen=True)
class CiftLiveWindowSelectorBenchmarkRow:
    trace_id: str
    example_id: str | None
    turn_index: int
    expected_label: str | None
    expected_window_family: str | None
    window_family: str
    window_selection_reason: str
    model_bundle_id: str
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
            "expected_label": self.expected_label,
            "expected_window_family": self.expected_window_family,
            "window_family": self.window_family,
            "window_selection_reason": self.window_selection_reason,
            "model_bundle_id": self.model_bundle_id,
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
class CiftLiveWindowSelectorBenchmarkReport:
    schema_version: str
    model_id: str
    revision: str
    selected_device: str
    selected_choice_runtime_model_path: str
    fallback_runtime_model_path: str
    runtime_turns_path: str
    request_count: int
    model_load_ms: float
    expected_label_counts: dict[str, int]
    expected_window_family_counts: dict[str, int]
    window_family_counts: dict[str, int]
    window_family_mismatch_count: int
    action_counts: dict[str, int]
    policy_action_counts: dict[str, int]
    capability_status_counts: dict[str, int]
    model_forward_ms: dict[str, float]
    feature_extraction_ms: dict[str, float]
    detector_ms: dict[str, float]
    total_runtime_ms: dict[str, float]
    rows: tuple[CiftLiveWindowSelectorBenchmarkRow, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "selected_device": self.selected_device,
            "selected_choice_runtime_model_path": self.selected_choice_runtime_model_path,
            "fallback_runtime_model_path": self.fallback_runtime_model_path,
            "runtime_turns_path": self.runtime_turns_path,
            "request_count": self.request_count,
            "model_load_ms": self.model_load_ms,
            "expected_label_counts": self.expected_label_counts,
            "expected_window_family_counts": self.expected_window_family_counts,
            "window_family_counts": self.window_family_counts,
            "window_family_mismatch_count": self.window_family_mismatch_count,
            "action_counts": self.action_counts,
            "policy_action_counts": self.policy_action_counts,
            "capability_status_counts": self.capability_status_counts,
            "model_forward_ms": self.model_forward_ms,
            "feature_extraction_ms": self.feature_extraction_ms,
            "detector_ms": self.detector_ms,
            "total_runtime_ms": self.total_runtime_ms,
            "rows": [row.to_dict() for row in self.rows],
        }


def run_cift_live_window_selector_benchmark(
    config: CiftLiveWindowSelectorBenchmarkConfig,
) -> CiftLiveWindowSelectorBenchmarkReport:
    from aegis_introspection.cift_live_extractor import LoadedModelHiddenStateRunner
    from aegis_introspection.model_loader import ModelLoadConfig, load_causal_lm

    _validate_full_config(config)
    _assert_unsealed_paths(
        paths=(
            config.runtime_turns_path,
            config.selected_choice_runtime_model_path,
            config.fallback_runtime_model_path,
            config.output_json_path,
            config.output_markdown_path,
        ),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT window-selector benchmark",
    )
    _assert_unsealed_jsonl_tags(
        path=config.runtime_turns_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT window-selector benchmark",
    )
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
    return run_cift_live_window_selector_benchmark_with_runner(
        config=CiftLiveWindowSelectorBenchmarkRequestConfig(
            runtime_turns_path=config.runtime_turns_path,
            selected_choice_runtime_model_path=config.selected_choice_runtime_model_path,
            fallback_runtime_model_path=config.fallback_runtime_model_path,
            output_json_path=config.output_json_path,
            output_markdown_path=config.output_markdown_path,
            detector_name=config.detector_name,
            feature_source=config.feature_source,
            mock_response=config.mock_response,
            model_id=config.model_id,
            revision=config.revision,
            selected_device=loaded_model.device.name,
            model_load_ms=model_load_ms,
            allow_sealed_holdout=config.allow_sealed_holdout,
        ),
        runner=LoadedModelHiddenStateRunner(loaded_model=loaded_model),
    )


def run_cift_live_window_selector_benchmark_with_runner(
    config: CiftLiveWindowSelectorBenchmarkRequestConfig,
    runner: HiddenStateRunner,
) -> CiftLiveWindowSelectorBenchmarkReport:
    from aegis_introspection.cift_live_extractor import LiveCiftFeatureSetExtractor

    selected_choice_model = load_cift_runtime_model(config.selected_choice_runtime_model_path)
    fallback_model = load_cift_runtime_model(config.fallback_runtime_model_path)
    timing_runner = TimingHiddenStateRunner(runner)
    extractor = LiveCiftFeatureSetExtractor(
        runner=timing_runner,
        feature_keys=(selected_choice_model.feature_key, fallback_model.feature_key),
    )
    return _run_cift_live_window_selector_benchmark(
        config=config,
        selected_choice_model=selected_choice_model,
        fallback_model=fallback_model,
        extractor=extractor,
        timing_runner=timing_runner,
    )


def run_cift_live_window_selector_benchmark_with_extractor(
    config: CiftLiveWindowSelectorBenchmarkRequestConfig,
    extractor: CiftFeatureExtractor,
) -> CiftLiveWindowSelectorBenchmarkReport:
    selected_choice_model = load_cift_runtime_model(config.selected_choice_runtime_model_path)
    fallback_model = load_cift_runtime_model(config.fallback_runtime_model_path)
    return _run_cift_live_window_selector_benchmark(
        config=config,
        selected_choice_model=selected_choice_model,
        fallback_model=fallback_model,
        extractor=extractor,
        timing_runner=None,
    )


def write_cift_live_window_selector_benchmark_json(
    path: Path,
    report: CiftLiveWindowSelectorBenchmarkReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_cift_live_window_selector_benchmark_markdown(
    path: Path,
    report: CiftLiveWindowSelectorBenchmarkReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_benchmark_markdown(report), encoding="utf-8")


def _run_cift_live_window_selector_benchmark(
    config: CiftLiveWindowSelectorBenchmarkRequestConfig,
    selected_choice_model: CiftRuntimeLinearModel,
    fallback_model: CiftRuntimeLinearModel,
    extractor: CiftFeatureExtractor,
    timing_runner: TimingHiddenStateRunner | None,
) -> CiftLiveWindowSelectorBenchmarkReport:
    _validate_request_config(config)
    _assert_unsealed_paths(
        paths=(
            config.runtime_turns_path,
            config.selected_choice_runtime_model_path,
            config.fallback_runtime_model_path,
            config.output_json_path,
            config.output_markdown_path,
        ),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT window-selector benchmark",
    )
    _assert_unsealed_jsonl_tags(
        path=config.runtime_turns_path,
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="live CIFT window-selector benchmark",
    )
    requests = _load_runtime_requests_jsonl(config.runtime_turns_path)
    timing_extractor = TimingFeatureExtractor(extractor)
    runtime = AegisRuntime(
        turn_annotators=(
            CiftFeatureVectorAnnotator(
                feature_key=selected_choice_model.feature_key,
                extractor=timing_extractor,
                source=config.feature_source,
            ),
            CiftFeatureVectorAnnotator(
                feature_key=fallback_model.feature_key,
                extractor=timing_extractor,
                source=config.feature_source,
            ),
        ),
        pre_generation_detectors=(
            CiftRuntimeWindowSelector(
                detector_name=config.detector_name,
                selected_choice_model=selected_choice_model,
                fallback_model=fallback_model,
            ),
        ),
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
    report = _report(config=config, rows=rows)
    write_cift_live_window_selector_benchmark_json(config.output_json_path, report)
    write_cift_live_window_selector_benchmark_markdown(config.output_markdown_path, report)
    return report


def _benchmark_rows(
    runtime: AegisRuntime,
    requests: tuple[RuntimeRequest, ...],
    timing_runner: TimingHiddenStateRunner | None,
    timing_extractor: TimingFeatureExtractor,
) -> tuple[CiftLiveWindowSelectorBenchmarkRow, ...]:
    rows: list[CiftLiveWindowSelectorBenchmarkRow] = []
    for request in requests:
        forward_count = _forward_count(timing_runner)
        extraction_count = len(timing_extractor.extraction_latencies_ms)
        started_at = time.perf_counter()
        response = runtime.evaluate_turn(request)
        total_runtime_ms = _elapsed_ms(started_at)
        if len(response.detector_results) != 1:
            raise CiftLiveWindowSelectorBenchmarkError(
                "Live CIFT window-selector benchmark expects exactly one detector result per request."
            )
        detector_result = response.detector_results[0]
        rows.append(
            CiftLiveWindowSelectorBenchmarkRow(
                trace_id=request.trace_id,
                example_id=_example_id(request.metadata),
                turn_index=request.turn_index,
                expected_label=_eval_metadata_string(request.metadata, "label"),
                expected_window_family=_eval_metadata_string(request.metadata, "expected_cift_window_family"),
                window_family=_evidence_string(detector_result.evidence, "cift_window_family"),
                window_selection_reason=_evidence_string(detector_result.evidence, "cift_window_selection_reason"),
                model_bundle_id=_evidence_string(detector_result.evidence, "model_bundle_id"),
                detector_action=detector_result.recommended_action.value,
                policy_action=response.policy_decision.final_action.value,
                capability_status=detector_result.capability_status.value,
                score=detector_result.score,
                model_forward_ms=_request_forward_latency(timing_runner=timing_runner, previous_count=forward_count),
                feature_extraction_ms=_new_latency_sum(
                    latencies=timing_extractor.extraction_latencies_ms,
                    previous_count=extraction_count,
                    metric_name="feature_extraction_ms",
                ),
                detector_ms=detector_result.latency_ms,
                total_runtime_ms=total_runtime_ms,
            )
        )
    return tuple(rows)


def _report(
    config: CiftLiveWindowSelectorBenchmarkRequestConfig,
    rows: tuple[CiftLiveWindowSelectorBenchmarkRow, ...],
) -> CiftLiveWindowSelectorBenchmarkReport:
    return CiftLiveWindowSelectorBenchmarkReport(
        schema_version="aegis_introspection.cift_live_window_selector_benchmark/v1",
        model_id=config.model_id,
        revision=config.revision,
        selected_device=config.selected_device,
        selected_choice_runtime_model_path=str(config.selected_choice_runtime_model_path),
        fallback_runtime_model_path=str(config.fallback_runtime_model_path),
        runtime_turns_path=str(config.runtime_turns_path),
        request_count=len(rows),
        model_load_ms=config.model_load_ms,
        expected_label_counts=_optional_counts(tuple(row.expected_label for row in rows)),
        expected_window_family_counts=_optional_counts(tuple(row.expected_window_family for row in rows)),
        window_family_counts=_counts(tuple(row.window_family for row in rows)),
        window_family_mismatch_count=_window_family_mismatch_count(rows),
        action_counts=_counts(tuple(row.detector_action for row in rows)),
        policy_action_counts=_counts(tuple(row.policy_action for row in rows)),
        capability_status_counts=_counts(tuple(row.capability_status for row in rows)),
        model_forward_ms=_summary(tuple(row.model_forward_ms for row in rows)),
        feature_extraction_ms=_summary(tuple(row.feature_extraction_ms for row in rows)),
        detector_ms=_summary(tuple(row.detector_ms for row in rows)),
        total_runtime_ms=_summary(tuple(row.total_runtime_ms for row in rows)),
        rows=rows,
    )


def _benchmark_markdown(report: CiftLiveWindowSelectorBenchmarkReport) -> str:
    lines = [
        "# Live CIFT Window Selector Benchmark",
        "",
        "## Source",
        "",
        f"- Model: `{report.model_id}`",
        f"- Revision: `{report.revision}`",
        f"- Selected device: `{report.selected_device}`",
        f"- Selected-choice runtime model: `{report.selected_choice_runtime_model_path}`",
        f"- Fallback runtime model: `{report.fallback_runtime_model_path}`",
        f"- Runtime turns: `{report.runtime_turns_path}`",
        f"- Requests: `{report.request_count}`",
        f"- Model load: `{report.model_load_ms:.4f} ms`",
        f"- Window route mismatches: `{report.window_family_mismatch_count}`",
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
        f"- Window families: `{report.window_family_counts}`",
        f"- Expected window families: `{report.expected_window_family_counts}`",
        f"- Expected labels: `{report.expected_label_counts}`",
        f"- Detector actions: `{report.action_counts}`",
        f"- Policy actions: `{report.policy_action_counts}`",
        f"- Capability statuses: `{report.capability_status_counts}`",
        "",
        "## Rows",
        "",
        "| Example | Label | Window | Expected Window | Score | Detector Action | "
        "Policy Action | Forward ms | Feature ms | Total ms |",
        "|---|---|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"`{row.example_id}` | "
            f"`{row.expected_label}` | "
            f"`{row.window_family}` | "
            f"`{row.expected_window_family}` | "
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
        raise CiftLiveWindowSelectorBenchmarkError("Cannot summarize an empty latency set.")
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


def _optional_counts(values: tuple[str | None, ...]) -> dict[str, int]:
    return dict(Counter(value for value in values if value is not None))


def _window_family_mismatch_count(rows: tuple[CiftLiveWindowSelectorBenchmarkRow, ...]) -> int:
    return sum(
        1 for row in rows if row.expected_window_family is not None and row.expected_window_family != row.window_family
    )


def _forward_count(timing_runner: TimingHiddenStateRunner | None) -> int:
    if timing_runner is None:
        return 0
    return len(timing_runner.forward_latencies_ms)


def _request_forward_latency(timing_runner: TimingHiddenStateRunner | None, previous_count: int) -> float:
    if timing_runner is None:
        return 0.0
    if len(timing_runner.forward_latencies_ms) != previous_count + 1:
        raise CiftLiveWindowSelectorBenchmarkError("Expected one new model_forward_ms value.")
    return timing_runner.forward_latencies_ms[-1]


def _new_latency_sum(latencies: list[float], previous_count: int, metric_name: str) -> float:
    if len(latencies) <= previous_count:
        raise CiftLiveWindowSelectorBenchmarkError(f"Expected at least one new {metric_name} value.")
    return sum(latencies[previous_count:])


def _evidence_string(evidence: dict[str, JsonValue], field_name: str) -> str:
    value = evidence.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CiftLiveWindowSelectorBenchmarkError(f"detector evidence.{field_name} must be a non-empty string.")
    return value


def _example_id(metadata: dict[str, JsonValue]) -> str | None:
    value = metadata.get("example_id")
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise CiftLiveWindowSelectorBenchmarkError("metadata.example_id must be a non-empty string when present.")
    return value


def _eval_metadata_string(metadata: dict[str, JsonValue], field_name: str) -> str | None:
    eval_metadata = metadata.get("eval")
    if eval_metadata is None:
        return None
    if not isinstance(eval_metadata, dict):
        raise CiftLiveWindowSelectorBenchmarkError("metadata.eval must be an object when present.")
    value = eval_metadata.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise CiftLiveWindowSelectorBenchmarkError(f"metadata.eval.{field_name} must be a non-empty string.")
    return value


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _validate_full_config(config: CiftLiveWindowSelectorBenchmarkConfig) -> None:
    if config.model_id == "":
        raise CiftLiveWindowSelectorBenchmarkError("model_id must not be empty.")
    if config.revision == "":
        raise CiftLiveWindowSelectorBenchmarkError("revision must not be empty.")
    if config.requested_device == "":
        raise CiftLiveWindowSelectorBenchmarkError("requested_device must not be empty.")
    _validate_shared_config(
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
    )


def _validate_request_config(config: CiftLiveWindowSelectorBenchmarkRequestConfig) -> None:
    if config.model_id == "":
        raise CiftLiveWindowSelectorBenchmarkError("model_id must not be empty.")
    if config.revision == "":
        raise CiftLiveWindowSelectorBenchmarkError("revision must not be empty.")
    if config.selected_device == "":
        raise CiftLiveWindowSelectorBenchmarkError("selected_device must not be empty.")
    if config.model_load_ms < 0.0:
        raise CiftLiveWindowSelectorBenchmarkError("model_load_ms must not be negative.")
    _validate_shared_config(
        detector_name=config.detector_name,
        feature_source=config.feature_source,
        mock_response=config.mock_response,
    )


def _validate_shared_config(detector_name: str, feature_source: str, mock_response: str) -> None:
    if detector_name == "":
        raise CiftLiveWindowSelectorBenchmarkError("detector_name must not be empty.")
    if feature_source == "":
        raise CiftLiveWindowSelectorBenchmarkError("feature_source must not be empty.")
    if mock_response == "":
        raise CiftLiveWindowSelectorBenchmarkError("mock_response must not be empty.")


def _load_runtime_requests_jsonl(path: Path) -> tuple[RuntimeRequest, ...]:
    try:
        return _load_shared_runtime_requests_jsonl(path)
    except RuntimeRequestJsonlError as exc:
        raise CiftLiveWindowSelectorBenchmarkError(str(exc)) from exc


def _assert_unsealed_paths(paths: tuple[Path, ...], allow_sealed_holdout: bool, context: str) -> None:
    try:
        _assert_shared_unsealed_paths(paths=paths, allow_sealed_holdout=allow_sealed_holdout, context=context)
    except SealedHoldoutPolicyError as exc:
        raise CiftLiveWindowSelectorBenchmarkError(str(exc)) from exc


def _assert_unsealed_jsonl_tags(path: Path, allow_sealed_holdout: bool, context: str) -> None:
    try:
        _assert_shared_unsealed_jsonl_tags(path=path, allow_sealed_holdout=allow_sealed_holdout, context=context)
    except SealedHoldoutPolicyError as exc:
        raise CiftLiveWindowSelectorBenchmarkError(str(exc)) from exc
