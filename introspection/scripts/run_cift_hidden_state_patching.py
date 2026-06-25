from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.cift_hidden_state_patching import (  # noqa: E402
    HiddenStatePatchReport,
    HiddenStatePatchReportConfig,
    PatchObservableMode,
    SelectedChoiceHiddenStatePatchReportConfig,
    evaluate_hidden_state_patch_report,
    evaluate_selected_choice_hidden_state_patch_report,
    write_hidden_state_patch_report_json,
)
from aegis_introspection.model_loader import (  # noqa: E402
    LoadedCausalLM,
    ModelDTypeName,
    ModelLoadConfig,
    load_causal_lm,
    parse_model_dtype,
)
from aegis_introspection.prompts import StructuredPromptExample, load_structured_prompt_examples  # noqa: E402
from aegis_introspection.sealed_holdout import add_unseal_flag, assert_unsealed_paths  # noqa: E402


@dataclass(frozen=True)
class HiddenStatePatchingCliConfig:
    prompts_path: Path
    output_path: Path
    report_id: str
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    patch_layer_index: int
    observable_mode: PatchObservableMode
    positive_completion: str | None
    negative_completion: str | None
    minimum_margin_shift: float
    max_pairs: int | None
    created_at: str
    allow_sealed_holdout: bool


def run_hidden_state_patching_cli(argv: Sequence[str]) -> int:
    config = _parse_args(argv)
    assert_unsealed_paths(
        paths=(config.prompts_path, config.output_path),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="CIFT hidden-state patching",
    )
    loaded_model = load_causal_lm(_model_config(config))
    examples = load_structured_prompt_examples(config.prompts_path)
    report = _evaluate_report(
        loaded_model=loaded_model,
        examples=examples,
        config=config,
    )
    write_hidden_state_patch_report_json(path=config.output_path, report=report)
    print(f"Wrote CIFT hidden-state patching report to {config.output_path}")
    print(f"Report ID: {report.report_id}")
    print(f"Candidate pairs: {report.candidate_pair_count}")
    print(f"Eligible pairs: {report.eligible_pair_count}")
    print(f"Pairs: {report.pair_count}")
    print(f"Skipped pairs: {report.skipped_pair_count}")
    print(f"Truncated pairs: {report.truncated_pair_count}")
    print(f"safe_to_exfil_success_rate={report.safe_to_exfil_success_rate:.6f}")
    print(f"exfil_to_safe_success_rate={report.exfil_to_safe_success_rate:.6f}")
    print(f"directional_intervention_passed={report.directional_intervention_passed}")
    print(f"coverage_complete={report.coverage_complete}")
    print(f"passed={report.passed}")
    return 0 if report.passed else 1


def _parse_args(argv: Sequence[str]) -> HiddenStatePatchingCliConfig:
    namespace = _build_parser().parse_args(argv)
    return HiddenStatePatchingCliConfig(
        prompts_path=Path(str(namespace.prompts)),
        output_path=Path(str(namespace.output)),
        report_id=str(namespace.report_id),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        requested_device=str(namespace.device),
        local_files_only=not bool(namespace.allow_download),
        dtype_name=parse_model_dtype(str(namespace.dtype)),
        trust_remote_code=bool(namespace.trust_remote_code),
        patch_layer_index=int(namespace.patch_layer_index),
        observable_mode=_parse_observable_mode(str(namespace.observable_mode)),
        positive_completion=None if namespace.positive_completion is None else str(namespace.positive_completion),
        negative_completion=None if namespace.negative_completion is None else str(namespace.negative_completion),
        minimum_margin_shift=float(namespace.minimum_margin_shift),
        max_pairs=None if namespace.max_pairs is None else int(namespace.max_pairs),
        created_at=str(namespace.created_at),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CIFT transformer hidden-state patching evidence.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=False, default="main")
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=False, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--patch-layer-index", required=True, type=int)
    parser.add_argument("--observable-mode", required=True, choices=("fixed_completion", "paired_selected_choice"))
    parser.add_argument("--positive-completion", required=False)
    parser.add_argument("--negative-completion", required=False)
    parser.add_argument("--minimum-margin-shift", required=True, type=float)
    parser.add_argument("--max-pairs", required=False, type=int)
    parser.add_argument("--created-at", required=True)
    add_unseal_flag(parser)
    return parser


def _parse_observable_mode(raw_value: str) -> PatchObservableMode:
    if raw_value == "fixed_completion" or raw_value == "paired_selected_choice":
        return raw_value
    raise ValueError(f"Unsupported observable mode '{raw_value}'.")


def _report_config(
    config: HiddenStatePatchingCliConfig,
) -> HiddenStatePatchReportConfig | SelectedChoiceHiddenStatePatchReportConfig:
    if config.observable_mode == "fixed_completion":
        if config.positive_completion is None or config.negative_completion is None:
            raise ValueError("fixed_completion mode requires --positive-completion and --negative-completion.")
        return HiddenStatePatchReportConfig(
            report_id=config.report_id,
            patch_layer_index=config.patch_layer_index,
            positive_completion=config.positive_completion,
            negative_completion=config.negative_completion,
            minimum_margin_shift=config.minimum_margin_shift,
            max_pairs=config.max_pairs,
            created_at=config.created_at,
        )
    if config.observable_mode == "paired_selected_choice":
        if config.positive_completion is not None or config.negative_completion is not None:
            raise ValueError("paired_selected_choice mode does not accept fixed completion arguments.")
        return SelectedChoiceHiddenStatePatchReportConfig(
            report_id=config.report_id,
            patch_layer_index=config.patch_layer_index,
            minimum_margin_shift=config.minimum_margin_shift,
            max_pairs=config.max_pairs,
            created_at=config.created_at,
        )
    raise ValueError(f"Unsupported observable mode '{config.observable_mode}'.")


def _evaluate_report(
    loaded_model: LoadedCausalLM,
    examples: tuple[StructuredPromptExample, ...],
    config: HiddenStatePatchingCliConfig,
) -> HiddenStatePatchReport:
    report_config = _report_config(config)
    if isinstance(report_config, HiddenStatePatchReportConfig):
        return evaluate_hidden_state_patch_report(
            loaded_model=loaded_model,
            examples=examples,
            config=report_config,
        )
    return evaluate_selected_choice_hidden_state_patch_report(
        loaded_model=loaded_model,
        examples=examples,
        config=report_config,
    )


def _model_config(config: HiddenStatePatchingCliConfig) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_id=config.model_id,
        revision=config.revision,
        requested_device=config.requested_device,
        local_files_only=config.local_files_only,
        dtype_name=config.dtype_name,
        trust_remote_code=config.trust_remote_code,
    )


if __name__ == "__main__":
    raise SystemExit(run_hidden_state_patching_cli(tuple(sys.argv[1:])))
