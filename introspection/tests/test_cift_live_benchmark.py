from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from aegis_introspection.activations import HiddenStateForwardPass
from aegis_introspection.cift_live_benchmark import (
    CiftLiveBenchmarkReport,
    CiftLiveBenchmarkRow,
    TimingFeatureExtractor,
    TimingHiddenStateRunner,
    write_cift_live_benchmark_json,
    write_cift_live_benchmark_markdown,
)

from aegis.core.contracts import CapabilityMode, Message, ModelInfo, NormalizedTurn


class CiftLiveBenchmarkTest(unittest.TestCase):
    def test_timing_hidden_state_runner_records_forward_latency(self) -> None:
        runner = TimingHiddenStateRunner(FakeHiddenStateRunner())

        forward_pass = runner.run("prompt")

        self.assertEqual("prompt", forward_pass.prompt)
        self.assertEqual(1, len(runner.forward_latencies_ms))
        self.assertGreaterEqual(runner.forward_latencies_ms[0], 0.0)

    def test_timing_feature_extractor_records_extraction_latency(self) -> None:
        extractor = TimingFeatureExtractor(FakeFeatureExtractor(vector=(1.0, 2.0)))

        vector = extractor.extract_feature_vector(_turn(), "feature")

        self.assertEqual((1.0, 2.0), vector)
        self.assertEqual(1, len(extractor.extraction_latencies_ms))
        self.assertGreaterEqual(extractor.extraction_latencies_ms[0], 0.0)

    def test_benchmark_report_writes_json_and_markdown(self) -> None:
        report = _report()
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "benchmark.json"
            markdown_path = Path(directory) / "benchmark.md"

            write_cift_live_benchmark_json(json_path, report)
            write_cift_live_benchmark_markdown(markdown_path, report)

            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual("aegis_introspection.cift_live_benchmark/v1", decoded["schema_version"])
        self.assertEqual(2, decoded["request_count"])
        self.assertIn("Live CIFT Benchmark", markdown)
        self.assertIn("Model forward", markdown)


class FakeHiddenStateRunner:
    def run(self, prompt: str) -> HiddenStateForwardPass:
        return HiddenStateForwardPass(
            prompt=prompt,
            input_ids=torch.tensor(((1,),), dtype=torch.int64),
            attention_mask=torch.tensor(((1,),), dtype=torch.int64),
            hidden_states=(torch.tensor((((1.0, 2.0),),), dtype=torch.float32),),
        )


class FakeFeatureExtractor:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self._vector = vector

    def extract_feature_vector(self, turn: NormalizedTurn, feature_key: str) -> tuple[float, ...] | None:
        return self._vector


def _turn() -> NormalizedTurn:
    return NormalizedTurn(
        trace_id="trace-benchmark",
        session_id="session-benchmark",
        turn_index=1,
        capability_mode=CapabilityMode.OFFLINE_EVAL,
        model=ModelInfo(provider="mock", model_id="mock", revision=None, selected_device=None),
        messages=(Message(role="user", content="prompt"),),
        tool_calls=(),
        sensitive_spans=(),
        metadata={"example_id": "example-1"},
    )


def _report() -> CiftLiveBenchmarkReport:
    rows = (
        CiftLiveBenchmarkRow(
            trace_id="trace-1",
            example_id="safe-1",
            turn_index=1,
            detector_action="allow",
            policy_action="allow",
            capability_status="active",
            score=0.1,
            model_forward_ms=10.0,
            feature_extraction_ms=11.0,
            detector_ms=1.0,
            total_runtime_ms=12.0,
        ),
        CiftLiveBenchmarkRow(
            trace_id="trace-2",
            example_id="exfil-1",
            turn_index=2,
            detector_action="warn",
            policy_action="warn",
            capability_status="active",
            score=0.9,
            model_forward_ms=20.0,
            feature_extraction_ms=21.0,
            detector_ms=2.0,
            total_runtime_ms=22.0,
        ),
    )
    return CiftLiveBenchmarkReport(
        schema_version="aegis_introspection.cift_live_benchmark/v1",
        model_id="Qwen/Qwen3-test",
        revision="main",
        selected_device="cpu",
        runtime_model_path="model.json",
        runtime_turns_path="turns.jsonl",
        request_count=2,
        model_load_ms=100.0,
        action_counts={"allow": 1, "warn": 1},
        policy_action_counts={"allow": 1, "warn": 1},
        capability_status_counts={"active": 2},
        model_forward_ms={"mean": 15.0, "median": 15.0, "p95": 19.5, "min": 10.0, "max": 20.0},
        feature_extraction_ms={"mean": 16.0, "median": 16.0, "p95": 20.5, "min": 11.0, "max": 21.0},
        detector_ms={"mean": 1.5, "median": 1.5, "p95": 1.95, "min": 1.0, "max": 2.0},
        total_runtime_ms={"mean": 17.0, "median": 17.0, "p95": 21.5, "min": 12.0, "max": 22.0},
        rows=rows,
    )


if __name__ == "__main__":
    unittest.main()
