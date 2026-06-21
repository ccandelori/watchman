from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.adjudication import load_binary_error_analysis_report_json
from aegis_introspection.binary_tasks import BinaryMethodName
from aegis_introspection.detector_result_bridge import (
    CiftDetectorBridgeConfig,
    RecommendedAction,
    cift_prediction_to_detector_result,
)
from aegis_introspection.probe import JsonValue


@dataclass(frozen=True)
class ExportCiftDetectorResultsConfig:
    runtime_turns_path: Path
    error_report_path: Path
    output_path: Path
    task_name: str
    method_name: BinaryMethodName
    detector_name: str
    feature_key: str
    probe_version: str
    capability_required: str
    positive_label: str
    positive_score: float
    negative_score: float
    positive_action: str
    negative_action: str
    confidence: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CIFT predictions as Aegis DetectorResult-shaped JSONL.")
    parser.add_argument(
        "--runtime-turns",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "runtime_turns_dp_honey_lite_v3_selector_windows.jsonl"),
    )
    parser.add_argument(
        "--error-report",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_error_analysis_readout_window_layer_15_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "detector_results_cift_v3_selector_window_layer_15_v1.jsonl"
        ),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument(
        "--method",
        required=False,
        choices=("activation_probe", "word_tfidf", "char_tfidf"),
        default="activation_probe",
    )
    parser.add_argument("--detector-name", required=False, default="cift_selector_probe")
    parser.add_argument("--feature-key", required=False, default="readout_window_layer_15")
    parser.add_argument("--probe-version", required=False, default="dp_honey_lite_v3_selector_window_layer_15_v1")
    parser.add_argument("--capability-required", required=False, default="self_hosted_introspection")
    parser.add_argument("--positive-label", required=False, default="exfiltration_intent")
    parser.add_argument("--positive-score", required=False, type=float, default=1.0)
    parser.add_argument("--negative-score", required=False, type=float, default=0.0)
    parser.add_argument("--positive-action", required=False, choices=("allow", "warn", "sanitize", "block", "escalate"), default="warn")
    parser.add_argument("--negative-action", required=False, choices=("allow", "warn", "sanitize", "block", "escalate"), default="allow")
    parser.add_argument("--confidence", required=False, type=float, default=0.7736)
    return parser


def _parse_args(argv: Sequence[str]) -> ExportCiftDetectorResultsConfig:
    namespace = _build_parser().parse_args(argv)
    return ExportCiftDetectorResultsConfig(
        runtime_turns_path=Path(namespace.runtime_turns),
        error_report_path=Path(namespace.error_report),
        output_path=Path(namespace.output),
        task_name=str(namespace.task),
        method_name=cast(BinaryMethodName, namespace.method),
        detector_name=str(namespace.detector_name),
        feature_key=str(namespace.feature_key),
        probe_version=str(namespace.probe_version),
        capability_required=str(namespace.capability_required),
        positive_label=str(namespace.positive_label),
        positive_score=float(namespace.positive_score),
        negative_score=float(namespace.negative_score),
        positive_action=str(namespace.positive_action),
        negative_action=str(namespace.negative_action),
        confidence=float(namespace.confidence),
    )


def _load_runtime_turns_by_example_id(path: Path) -> dict[str, Mapping[str, object]]:
    turns_by_example_id: dict[str, Mapping[str, object]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise TypeError(f"Line {line_number}: expected a JSON object.")
            metadata = decoded.get("metadata")
            if not isinstance(metadata, dict):
                raise TypeError(f"Line {line_number}: field 'metadata' must be an object.")
            example_id = metadata.get("example_id")
            if not isinstance(example_id, str) or example_id == "":
                raise TypeError(f"Line {line_number}: metadata.example_id must be a non-empty string.")
            if example_id in turns_by_example_id:
                raise ValueError(f"Line {line_number}: duplicate runtime turn example_id '{example_id}'.")
            turns_by_example_id[example_id] = cast(Mapping[str, object], decoded)
    if len(turns_by_example_id) == 0:
        raise ValueError(f"No runtime turns found in {path}.")
    return turns_by_example_id


def _detector_config(config: ExportCiftDetectorResultsConfig) -> CiftDetectorBridgeConfig:
    return CiftDetectorBridgeConfig(
        detector_name=config.detector_name,
        feature_key=config.feature_key,
        task_name=config.task_name,
        probe_version=config.probe_version,
        capability_required=config.capability_required,
        positive_label=config.positive_label,
        positive_score=config.positive_score,
        negative_score=config.negative_score,
        positive_action=cast(RecommendedAction, config.positive_action),
        negative_action=cast(RecommendedAction, config.negative_action),
        confidence=config.confidence,
    )


def _method_predictions(config: ExportCiftDetectorResultsConfig):
    report = load_binary_error_analysis_report_json(config.error_report_path)
    for task in report.tasks:
        if task.task_name != config.task_name:
            continue
        for method in task.methods:
            if method.method_name == config.method_name:
                return method.predictions
        raise ValueError(f"Task '{config.task_name}' does not contain method '{config.method_name}'.")
    raise ValueError(f"Report does not contain task '{config.task_name}'.")


def run_export(config: ExportCiftDetectorResultsConfig) -> None:
    turns_by_example_id = _load_runtime_turns_by_example_id(config.runtime_turns_path)
    detector_config = _detector_config(config)
    predictions = _method_predictions(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            turn = turns_by_example_id.get(prediction.example_id)
            if turn is None:
                raise ValueError(f"Missing runtime turn for prediction example '{prediction.example_id}'.")
            result = cift_prediction_to_detector_result(prediction=prediction, config=detector_config)
            row: dict[str, JsonValue] = {
                "trace_id": cast(str, turn["trace_id"]),
                "session_id": cast(str, turn["session_id"]),
                "turn_index": cast(int, turn["turn_index"]),
                "example_id": prediction.example_id,
                "detector_result": result,
            }
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    print(f"Wrote {len(predictions)} CIFT detector-result rows to {config.output_path}")


def main(argv: Sequence[str]) -> None:
    run_export(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
