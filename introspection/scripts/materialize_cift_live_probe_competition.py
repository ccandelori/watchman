from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
INTROSPECTION_SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (INTROSPECTION_SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_live_probe_competition import (  # noqa: E402
    CiftLiveProbeCompetitionConfig,
    CiftLiveProbeCompetitionError,
    CiftLiveProbeRun,
    materialize_cift_live_probe_competition,
)


@dataclass(frozen=True)
class MaterializeCiftLiveProbeCompetitionCliConfig:
    paper_metric_report_path: Path
    candidate_metric_report_path: Path
    evaluation_split_manifest_path: Path
    output_path: Path
    report_id: str
    feature_representation: str
    activation_feature_key: str
    candidate_probe_architecture: str
    candidate_training_loss: str
    paper_operating_threshold: float
    candidate_operating_threshold: float
    created_at: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize live sealed CIFT probe competition evidence from sealed metric reports."
    )
    parser.add_argument("--paper-metric-report", required=True)
    parser.add_argument("--candidate-metric-report", required=True)
    parser.add_argument("--evaluation-split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--feature-representation", required=True)
    parser.add_argument("--activation-feature-key", required=True)
    parser.add_argument("--candidate-probe-architecture", required=True)
    parser.add_argument("--candidate-training-loss", required=True)
    parser.add_argument("--paper-operating-threshold", required=True, type=float)
    parser.add_argument("--candidate-operating-threshold", required=True, type=float)
    parser.add_argument("--created-at", required=True)
    return parser


def _parse_args(argv: Sequence[str]) -> MaterializeCiftLiveProbeCompetitionCliConfig:
    namespace = _build_parser().parse_args(argv)
    return MaterializeCiftLiveProbeCompetitionCliConfig(
        paper_metric_report_path=Path(str(namespace.paper_metric_report)),
        candidate_metric_report_path=Path(str(namespace.candidate_metric_report)),
        evaluation_split_manifest_path=Path(str(namespace.evaluation_split_manifest)),
        output_path=Path(str(namespace.output)),
        report_id=str(namespace.report_id),
        feature_representation=str(namespace.feature_representation),
        activation_feature_key=str(namespace.activation_feature_key),
        candidate_probe_architecture=str(namespace.candidate_probe_architecture),
        candidate_training_loss=str(namespace.candidate_training_loss),
        paper_operating_threshold=float(namespace.paper_operating_threshold),
        candidate_operating_threshold=float(namespace.candidate_operating_threshold),
        created_at=str(namespace.created_at),
    )


def run_materializer(config: MaterializeCiftLiveProbeCompetitionCliConfig) -> None:
    paper_metric = _load_json_object(path=config.paper_metric_report_path, label="paper metric report")
    candidate_metric = _load_json_object(path=config.candidate_metric_report_path, label="candidate metric report")
    report = materialize_cift_live_probe_competition(
        config=CiftLiveProbeCompetitionConfig(
            report_id=config.report_id,
            training_dataset_id=_required_matching_string(
                left=candidate_metric,
                right=paper_metric,
                field_name="training_dataset_id",
            ),
            task_name=_required_matching_string(left=candidate_metric, right=paper_metric, field_name="task_name"),
            evaluation_split_id=_required_matching_string(
                left=candidate_metric,
                right=paper_metric,
                field_name="evaluation_split_id",
            ),
            evaluation_split_manifest_id=str(config.evaluation_split_manifest_path),
            evaluation_split_sha256=_sha256_file(config.evaluation_split_manifest_path),
            feature_representation=config.feature_representation,
            activation_feature_key=config.activation_feature_key,
            metric_name=_required_matching_string(left=candidate_metric, right=paper_metric, field_name="metric_name"),
            paper_probe=_probe_run_from_metric(
                metric=paper_metric,
                probe_architecture="mlp_128_64_1",
                training_loss="bce_with_l1_softplus_weight_sparsity",
                operating_threshold=config.paper_operating_threshold,
            ),
            candidate_probe=_probe_run_from_metric(
                metric=candidate_metric,
                probe_architecture=config.candidate_probe_architecture,
                training_loss=config.candidate_training_loss,
                operating_threshold=config.candidate_operating_threshold,
            ),
            higher_is_better=True,
            created_at=config.created_at,
        ),
        output_path=config.output_path,
    )
    print(f"Wrote CIFT live probe competition report to {config.output_path}")
    print(f"Report ID: {report.report_id}")
    print(f"paper={report.paper_probe_metric_value:.6f}")
    print(f"candidate={report.candidate_probe_metric_value:.6f}")
    print(f"delta={report.candidate_delta:.6f}")


def _probe_run_from_metric(
    metric: Mapping[str, object],
    probe_architecture: str,
    training_loss: str,
    operating_threshold: float,
) -> CiftLiveProbeRun:
    return CiftLiveProbeRun(
        source_report_id=_required_string(record=metric, field_name="report_id"),
        probe_architecture=probe_architecture,
        training_loss=training_loss,
        model_bundle_id=_runtime_model_bundle_id(metric),
        metric_value=_required_float(record=metric, field_name="metric_value"),
        false_negative_count=_required_int(record=metric, field_name="false_negative_count"),
        false_positive_count=_required_int(record=metric, field_name="false_positive_count"),
        false_negative_rate=_required_float(record=metric, field_name="false_negative_rate"),
        false_positive_rate=_required_float(record=metric, field_name="false_positive_rate"),
        operating_threshold=operating_threshold,
    )


def _runtime_model_bundle_id(metric: Mapping[str, object]) -> str:
    fallback_model_bundle_id = metric.get("fallback_model_bundle_id")
    if fallback_model_bundle_id is not None:
        if not isinstance(fallback_model_bundle_id, str) or fallback_model_bundle_id == "":
            raise CiftLiveProbeCompetitionError("fallback_model_bundle_id must be a non-empty string when present.")
        return fallback_model_bundle_id
    return _required_string(record=metric, field_name="selected_choice_model_bundle_id")


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiftLiveProbeCompetitionError(f"Invalid {label} JSON in {path}: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise CiftLiveProbeCompetitionError(f"{label} must contain a JSON object: {path}.")
    return cast(Mapping[str, object], decoded)


def _required_matching_string(
    left: Mapping[str, object],
    right: Mapping[str, object],
    field_name: str,
) -> str:
    left_value = _required_string(record=left, field_name=field_name)
    right_value = _required_string(record=right, field_name=field_name)
    if left_value != right_value:
        raise CiftLiveProbeCompetitionError(f"{field_name} must match between metric reports.")
    return left_value


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or value == "":
        raise CiftLiveProbeCompetitionError(f"{field_name} must be a non-empty string.")
    return value


def _required_float(record: Mapping[str, object], field_name: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be a number.")
    return float(value)


def _required_int(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CiftLiveProbeCompetitionError(f"{field_name} must be an integer.")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str]) -> int:
    try:
        run_materializer(_parse_args(argv))
    except CiftLiveProbeCompetitionError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
