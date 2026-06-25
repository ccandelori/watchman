from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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

from aegis_introspection.cift_selected_choice_calibration import (  # noqa: E402
    SelectedChoiceCalibrationConfig,
    SelectedChoiceCalibrationError,
    SelectedChoiceCalibrationSourceProfile,
    materialize_selected_choice_calibration,
    supported_selected_choice_calibration_source_profiles,
)
from aegis_introspection.hf_offset_encoder import (  # noqa: E402
    HuggingFaceOffsetEncoder,
    load_huggingface_tokenizer,
)

from aegis.core.contracts import CapabilityMode  # noqa: E402


@dataclass(frozen=True)
class MaterializeSelectedChoiceCalibrationCliConfig:
    trace_records_path: Path
    structured_prompts_path: Path
    runtime_turns_path: Path
    manifest_path: Path
    corpus_id: str
    calibration_split_id: str
    source_profile: SelectedChoiceCalibrationSourceProfile
    participant_ids: tuple[str, ...]
    variants_per_label: int
    model_provider: str
    model_id: str
    revision: str
    selected_device: str
    session_id: str
    sensitive_source: str
    readout_token_count: int
    capability_mode: CapabilityMode
    created_at: str
    local_files_only: bool
    overwrite: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a paired-semantic selected-choice CIFT calibration corpus."
    )
    parser.add_argument("--trace-records-output", required=True)
    parser.add_argument("--structured-prompts-output", required=True)
    parser.add_argument("--runtime-turns-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--calibration-split-id", required=True)
    parser.add_argument(
        "--source-profile", required=True, choices=supported_selected_choice_calibration_source_profiles()
    )
    parser.add_argument("--participant-id", required=True, action="append")
    parser.add_argument("--variants-per-label", required=True, type=int)
    parser.add_argument("--model-provider", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--selected-device", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sensitive-source", required=True)
    parser.add_argument("--readout-token-count", required=True, type=int)
    parser.add_argument("--capability-mode", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> MaterializeSelectedChoiceCalibrationCliConfig:
    namespace = _build_parser().parse_args(argv)
    participant_ids = tuple(str(value) for value in namespace.participant_id)
    return MaterializeSelectedChoiceCalibrationCliConfig(
        trace_records_path=Path(str(namespace.trace_records_output)),
        structured_prompts_path=Path(str(namespace.structured_prompts_output)),
        runtime_turns_path=Path(str(namespace.runtime_turns_output)),
        manifest_path=Path(str(namespace.manifest_output)),
        corpus_id=str(namespace.corpus_id),
        calibration_split_id=str(namespace.calibration_split_id),
        source_profile=cast(SelectedChoiceCalibrationSourceProfile, str(namespace.source_profile)),
        participant_ids=participant_ids,
        variants_per_label=int(namespace.variants_per_label),
        model_provider=str(namespace.model_provider),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        selected_device=str(namespace.selected_device),
        session_id=str(namespace.session_id),
        sensitive_source=str(namespace.sensitive_source),
        readout_token_count=int(namespace.readout_token_count),
        capability_mode=CapabilityMode(str(namespace.capability_mode)),
        created_at=str(namespace.created_at),
        local_files_only=not bool(namespace.allow_download),
        overwrite=bool(namespace.overwrite),
    )


def _materializer_config(
    config: MaterializeSelectedChoiceCalibrationCliConfig,
) -> SelectedChoiceCalibrationConfig:
    return SelectedChoiceCalibrationConfig(
        trace_records_path=config.trace_records_path,
        structured_prompts_path=config.structured_prompts_path,
        runtime_turns_path=config.runtime_turns_path,
        manifest_path=config.manifest_path,
        corpus_id=config.corpus_id,
        calibration_split_id=config.calibration_split_id,
        source_profile=config.source_profile,
        participant_ids=config.participant_ids,
        variants_per_label=config.variants_per_label,
        model_provider=config.model_provider,
        model_id=config.model_id,
        revision=config.revision,
        selected_device=config.selected_device,
        session_id=config.session_id,
        sensitive_source=config.sensitive_source,
        readout_token_count=config.readout_token_count,
        capability_mode=config.capability_mode,
        created_at=config.created_at,
        overwrite=config.overwrite,
    )


def run_materializer(config: MaterializeSelectedChoiceCalibrationCliConfig) -> None:
    tokenizer = load_huggingface_tokenizer(
        model_id=config.model_id,
        revision=config.revision,
        local_files_only=config.local_files_only,
    )
    result = materialize_selected_choice_calibration(
        config=_materializer_config(config),
        encoder=HuggingFaceOffsetEncoder(tokenizer),
    )
    print(f"Wrote trace records to {config.trace_records_path}")
    print(f"Wrote structured prompts to {config.structured_prompts_path}")
    print(f"Wrote runtime turns to {config.runtime_turns_path}")
    print(f"Wrote manifest to {config.manifest_path}")
    print(f"Selected-choice rows: {result.selected_choice_record_count}")


def main(argv: Sequence[str]) -> int:
    try:
        run_materializer(_parse_args(argv))
    except SelectedChoiceCalibrationError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
