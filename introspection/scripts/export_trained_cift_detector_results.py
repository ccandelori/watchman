from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.detector_result_bridge import RecommendedAction
from aegis_introspection.trained_detector_export import (
    TrainedDetectorExportConfig,
    export_trained_cift_detector_results,
)


@dataclass(frozen=True)
class ExportTrainedCiftDetectorResultsCliConfig:
    runtime_turns_path: Path
    artifact_path: Path
    model_bundle_path: Path
    output_path: Path
    detector_name: str
    model_bundle_id: str
    capability_required: str
    positive_action: RecommendedAction
    negative_action: RecommendedAction
    confidence: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export frozen CIFT model bundle predictions as DetectorResult JSONL.")
    parser.add_argument(
        "--runtime-turns",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "runtime_turns_dp_honey_lite_v4_1_selector_windows.jsonl"),
    )
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "activations"
            / "qwen3_0_6b_dp_honey_lite_v4_1_selector_windows.pt"
        ),
    )
    parser.add_argument(
        "--model-bundle",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "models"
            / "cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1.pkl"
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "detector_results_cift_v4_1_selector_window_layer_15_trained_bundle_v1.jsonl"
        ),
    )
    parser.add_argument("--detector-name", required=False, default="cift_selector_probe")
    parser.add_argument(
        "--model-bundle-id",
        required=False,
        default="cift_qwen3_0_6b_dp_honey_lite_v4_1_selector_window_layer_15_v1",
    )
    parser.add_argument("--capability-required", required=False, default="self_hosted_introspection")
    parser.add_argument("--positive-action", required=False, choices=_recommended_actions(), default="warn")
    parser.add_argument("--negative-action", required=False, choices=_recommended_actions(), default="allow")
    parser.add_argument("--confidence", required=False, type=float, default=0.7704)
    return parser


def _recommended_actions() -> tuple[RecommendedAction, ...]:
    return ("allow", "warn", "sanitize", "block", "escalate")


def _parse_args(argv: Sequence[str]) -> ExportTrainedCiftDetectorResultsCliConfig:
    namespace = _build_parser().parse_args(argv)
    return ExportTrainedCiftDetectorResultsCliConfig(
        runtime_turns_path=Path(namespace.runtime_turns),
        artifact_path=Path(namespace.artifact),
        model_bundle_path=Path(namespace.model_bundle),
        output_path=Path(namespace.output),
        detector_name=str(namespace.detector_name),
        model_bundle_id=str(namespace.model_bundle_id),
        capability_required=str(namespace.capability_required),
        positive_action=_recommended_action(str(namespace.positive_action)),
        negative_action=_recommended_action(str(namespace.negative_action)),
        confidence=float(namespace.confidence),
    )


def _recommended_action(value: str) -> RecommendedAction:
    if value == "allow":
        return "allow"
    if value == "warn":
        return "warn"
    if value == "sanitize":
        return "sanitize"
    if value == "block":
        return "block"
    if value == "escalate":
        return "escalate"
    raise ValueError(f"Unsupported action '{value}'.")


def _export_config(config: ExportTrainedCiftDetectorResultsCliConfig) -> TrainedDetectorExportConfig:
    return TrainedDetectorExportConfig(
        runtime_turns_path=config.runtime_turns_path,
        artifact_path=config.artifact_path,
        model_bundle_path=config.model_bundle_path,
        output_path=config.output_path,
        detector_name=config.detector_name,
        model_bundle_id=config.model_bundle_id,
        capability_required=config.capability_required,
        positive_action=config.positive_action,
        negative_action=config.negative_action,
        confidence=config.confidence,
    )


def run_export(config: ExportTrainedCiftDetectorResultsCliConfig) -> None:
    row_count = export_trained_cift_detector_results(_export_config(config))
    print(f"Wrote {row_count} trained CIFT DetectorResult rows to {config.output_path}")


def main(argv: Sequence[str]) -> None:
    run_export(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
