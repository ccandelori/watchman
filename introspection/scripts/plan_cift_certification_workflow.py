from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = INTROSPECTION_ROOT.parent
SRC_PATH = INTROSPECTION_ROOT / "src"
RUNTIME_SRC_PATH = WORKSPACE_ROOT / "src"
for source_path in (SRC_PATH, RUNTIME_SRC_PATH):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from aegis_introspection.cift_certification_workflow import (  # noqa: E402
    CiftCertificationWorkflowConfig,
    build_cift_certification_workflow_manifest,
)
from aegis_introspection.cift_model_metadata import CiftModelMetadataConfig, discover_cift_model_metadata  # noqa: E402


@dataclass(frozen=True)
class PlanCiftCertificationWorkflowCliConfig:
    certification_id: str
    model_id: str
    revision: str
    corpus_path: Path
    runtime_turns_path: Path
    fallback_runtime_model_path: Path
    output_dir: Path
    output_json_path: Path
    training_dataset_id: str
    task_name: str
    positive_label: str
    behavior_id: str
    behavior_description: str
    layer_indices: tuple[int, ...]
    pooling_methods: tuple[str, ...]
    candidate_feature_key: str
    requested_device: str
    prompt_renderer: str
    selected_choice_geometry: str
    selected_choice_readout_token_count: int
    dtype_name: str
    metric_threshold: float
    ablation_delta_threshold: float
    created_at: str
    local_files_only: bool
    trust_remote_code: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan a model-specific CIFT calibration/certification workflow.")
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--runtime-turns", required=True)
    parser.add_argument("--fallback-runtime-model", required=True)
    parser.add_argument("--output-dir", required=False, default=str(INTROSPECTION_ROOT / "data"))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--training-dataset-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--behavior-id", required=True)
    parser.add_argument("--behavior-description", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--candidate-feature", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--prompt-renderer", required=True)
    parser.add_argument("--selected-choice-geometry", required=True)
    parser.add_argument("--selected-choice-readout-token-count", required=True, type=int)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--metric-threshold", required=True, type=float)
    parser.add_argument("--ablation-delta-threshold", required=True, type=float)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> PlanCiftCertificationWorkflowCliConfig:
    namespace = _build_parser().parse_args(argv)
    return PlanCiftCertificationWorkflowCliConfig(
        certification_id=str(namespace.certification_id),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        corpus_path=Path(str(namespace.corpus)),
        runtime_turns_path=Path(str(namespace.runtime_turns)),
        fallback_runtime_model_path=Path(str(namespace.fallback_runtime_model)),
        output_dir=Path(str(namespace.output_dir)),
        output_json_path=Path(str(namespace.output_json)),
        training_dataset_id=str(namespace.training_dataset_id),
        task_name=str(namespace.task),
        positive_label=str(namespace.positive_label),
        behavior_id=str(namespace.behavior_id),
        behavior_description=str(namespace.behavior_description),
        layer_indices=_parse_layer_indices(str(namespace.layers)),
        pooling_methods=_parse_pooling_methods(str(namespace.pooling)),
        candidate_feature_key=str(namespace.candidate_feature),
        requested_device=str(namespace.device),
        prompt_renderer=str(namespace.prompt_renderer),
        selected_choice_geometry=str(namespace.selected_choice_geometry),
        selected_choice_readout_token_count=_positive_int(
            raw_value=namespace.selected_choice_readout_token_count,
            field_name="--selected-choice-readout-token-count",
        ),
        dtype_name=str(namespace.dtype),
        metric_threshold=float(namespace.metric_threshold),
        ablation_delta_threshold=float(namespace.ablation_delta_threshold),
        created_at=str(namespace.created_at),
        local_files_only=not bool(namespace.allow_download),
        trust_remote_code=bool(namespace.trust_remote_code),
    )


def _parse_layer_indices(value: str) -> tuple[int, ...]:
    layer_indices = tuple(int(item.strip()) for item in value.split(",") if item.strip() != "")
    if len(layer_indices) == 0:
        raise ValueError("layers must contain at least one integer.")
    return layer_indices


def _parse_pooling_methods(value: str) -> tuple[str, ...]:
    pooling_methods = tuple(item.strip() for item in value.split(",") if item.strip() != "")
    if len(pooling_methods) == 0:
        raise ValueError("pooling must contain at least one method.")
    return pooling_methods


def _positive_int(raw_value: object, field_name: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if raw_value < 1:
        raise ValueError(f"{field_name} must be positive.")
    return raw_value


def _workflow_config(config: PlanCiftCertificationWorkflowCliConfig) -> CiftCertificationWorkflowConfig:
    return CiftCertificationWorkflowConfig(
        certification_id=config.certification_id,
        repository_root=WORKSPACE_ROOT,
        model_id=config.model_id,
        revision=config.revision,
        corpus_path=config.corpus_path,
        runtime_turns_path=config.runtime_turns_path,
        fallback_runtime_model_path=config.fallback_runtime_model_path,
        output_dir=config.output_dir,
        training_dataset_id=config.training_dataset_id,
        task_name=config.task_name,
        positive_label=config.positive_label,
        behavior_id=config.behavior_id,
        behavior_description=config.behavior_description,
        layer_indices=config.layer_indices,
        pooling_methods=config.pooling_methods,
        candidate_feature_key=config.candidate_feature_key,
        requested_device=config.requested_device,
        prompt_renderer=config.prompt_renderer,
        selected_choice_geometry=config.selected_choice_geometry,
        selected_choice_readout_token_count=config.selected_choice_readout_token_count,
        dtype_name=config.dtype_name,
        metric_threshold=config.metric_threshold,
        ablation_delta_threshold=config.ablation_delta_threshold,
        allow_download=not config.local_files_only,
        trust_remote_code=config.trust_remote_code,
        created_at=config.created_at,
    )


def run_cli(config: PlanCiftCertificationWorkflowCliConfig) -> None:
    model_metadata = discover_cift_model_metadata(
        CiftModelMetadataConfig(
            model_id=config.model_id,
            revision=config.revision,
            requested_device=config.requested_device,
            dtype_name=config.dtype_name,
            selected_readout_candidates=(config.candidate_feature_key,),
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
    )
    manifest = build_cift_certification_workflow_manifest(
        config=_workflow_config(config),
        model_metadata=model_metadata,
    )
    config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote CIFT certification workflow manifest to {config.output_json_path}")


def main(argv: Sequence[str]) -> None:
    run_cli(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
