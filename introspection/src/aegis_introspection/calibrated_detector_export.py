from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from aegis_introspection.cift_calibration import CiftCalibrationReport, load_cift_calibration_report_json
from aegis_introspection.detector_result_bridge import (
    CalibratedCiftDetectorBridgeConfig,
    RecommendedAction,
    calibrated_cift_prediction_to_detector_result,
)
from aegis_introspection.probe import JsonValue


class CalibratedDetectorExportError(ValueError):
    """Raised when calibrated detector results cannot be exported."""


@dataclass(frozen=True)
class CalibratedDetectorExportConfig:
    runtime_turns_path: Path
    calibration_report_path: Path
    output_path: Path
    detector_name: str
    probe_version: str
    capability_required: str
    positive_action: RecommendedAction
    negative_action: RecommendedAction
    confidence: float


def export_calibrated_cift_detector_results(config: CalibratedDetectorExportConfig) -> int:
    turns_by_example_id = load_runtime_turns_by_example_id(config.runtime_turns_path)
    report = load_cift_calibration_report_json(config.calibration_report_path)
    detector_config = calibrated_detector_bridge_config(report=report, config=config)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        for prediction in report.predictions:
            turn = turns_by_example_id.get(prediction.example_id)
            if turn is None:
                raise CalibratedDetectorExportError(
                    f"Missing runtime turn for calibrated prediction example '{prediction.example_id}'."
                )
            result = calibrated_cift_prediction_to_detector_result(prediction=prediction, config=detector_config)
            row: dict[str, JsonValue] = {
                "trace_id": _required_string(turn, "trace_id", prediction.example_id),
                "session_id": _required_string(turn, "session_id", prediction.example_id),
                "turn_index": _required_int(turn, "turn_index", prediction.example_id),
                "example_id": prediction.example_id,
                "detector_result": result,
            }
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    return len(report.predictions)


def calibrated_detector_bridge_config(
    report: CiftCalibrationReport,
    config: CalibratedDetectorExportConfig,
) -> CalibratedCiftDetectorBridgeConfig:
    return CalibratedCiftDetectorBridgeConfig(
        detector_name=config.detector_name,
        feature_key=report.activation_feature_key,
        task_name=report.task_name,
        probe_version=config.probe_version,
        capability_required=config.capability_required,
        decision_threshold=report.decision_threshold,
        positive_action=config.positive_action,
        negative_action=config.negative_action,
        confidence=config.confidence,
    )


def load_runtime_turns_by_example_id(path: Path) -> dict[str, Mapping[str, object]]:
    turns_by_example_id: dict[str, Mapping[str, object]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            decoded = json.loads(line)
            record = _as_mapping(decoded, line_number)
            metadata = _as_mapping(record.get("metadata"), line_number)
            example_id = metadata.get("example_id")
            if not isinstance(example_id, str) or example_id == "":
                raise CalibratedDetectorExportError(
                    f"Line {line_number}: metadata.example_id must be a non-empty string."
                )
            if example_id in turns_by_example_id:
                raise CalibratedDetectorExportError(f"Line {line_number}: duplicate example_id '{example_id}'.")
            turns_by_example_id[example_id] = record

    if len(turns_by_example_id) == 0:
        raise CalibratedDetectorExportError(f"No runtime turns found in {path}.")
    return turns_by_example_id


def _as_mapping(value: object, line_number: int) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CalibratedDetectorExportError(f"Line {line_number}: expected a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(record: Mapping[str, object], field_name: str, example_id: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CalibratedDetectorExportError(
            f"Runtime turn for example '{example_id}' field '{field_name}' must be a non-empty string."
        )
    return value


def _required_int(record: Mapping[str, object], field_name: str, example_id: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int):
        raise CalibratedDetectorExportError(
            f"Runtime turn for example '{example_id}' field '{field_name}' must be an integer."
        )
    return value
