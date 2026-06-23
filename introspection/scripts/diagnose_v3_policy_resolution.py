from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from transformers import BatchEncoding

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.artifacts import load_activation_artifact  # noqa: E402
from aegis_introspection.model_loader import (  # noqa: E402
    LoadedCausalLM,
    ModelDTypeName,
    ModelLoadConfig,
    load_causal_lm,
    parse_model_dtype,
)
from aegis_introspection.v3_policy_resolution import (  # noqa: E402
    V3PolicyResolver,
    evaluate_v3_policy_resolution,
    write_v3_policy_resolution_json,
    write_v3_policy_resolution_markdown,
)


class V3PolicyResolutionScriptError(ValueError):
    """Raised when the V3 policy-resolution script receives malformed model data."""


@dataclass(frozen=True)
class DiagnoseV3PolicyResolutionScriptConfig:
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    max_new_tokens: int
    max_examples: int | None


class CausalLmV3PolicyResolver:
    def __init__(self, loaded_model: LoadedCausalLM, max_new_tokens: int) -> None:
        self._loaded_model = loaded_model
        self._max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        encoded = self._loaded_model.tokenizer(prompt, return_tensors="pt")
        if not isinstance(encoded, BatchEncoding):
            raise V3PolicyResolutionScriptError("Expected tokenizer output to be a transformers.BatchEncoding.")

        encoded_tensors = _encoded_tensors(encoded)
        input_ids = encoded_tensors["input_ids"].to(self._loaded_model.device.torch_device)
        attention_mask = encoded_tensors["attention_mask"].to(self._loaded_model.device.torch_device)
        with torch.inference_mode():
            generated = self._loaded_model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._loaded_model.tokenizer.eos_token_id,
            )
        continuation = generated[0, input_ids.shape[1] :]
        return str(self._loaded_model.tokenizer.decode(continuation, skip_special_tokens=True)).strip()


def _encoded_tensors(encoded: BatchEncoding) -> Mapping[str, torch.Tensor]:
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor):
        raise V3PolicyResolutionScriptError("Expected tokenizer field 'input_ids' to be a torch.Tensor.")
    if not isinstance(attention_mask, torch.Tensor):
        raise V3PolicyResolutionScriptError("Expected tokenizer field 'attention_mask' to be a torch.Tensor.")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a local causal LM to resolve hard V3 policy actions.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "activations"
            / "qwen3_0_6b_dp_honey_lite_v3_all_pooling.pt"
        ),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "dp_honey_lite_v3_policy_resolution_v1.json"),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "reports" / "dp_honey_lite_v3_policy_resolution_v1_summary.md"),
    )
    parser.add_argument("--model-id", required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=False, default="main")
    parser.add_argument("--device", required=False, default="auto")
    parser.add_argument("--dtype", required=False, default="device")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-new-tokens", required=False, type=int, default=32)
    parser.add_argument("--max-examples", required=False, type=int, default=None)
    return parser


def _parse_args(argv: Sequence[str]) -> DiagnoseV3PolicyResolutionScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return DiagnoseV3PolicyResolutionScriptConfig(
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        requested_device=str(namespace.device),
        local_files_only=not bool(namespace.allow_download),
        dtype_name=parse_model_dtype(str(namespace.dtype)),
        trust_remote_code=bool(namespace.trust_remote_code),
        max_new_tokens=int(namespace.max_new_tokens),
        max_examples=cast(int | None, namespace.max_examples),
    )


def _model_config(config: DiagnoseV3PolicyResolutionScriptConfig) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_id=config.model_id,
        revision=config.revision,
        requested_device=config.requested_device,
        local_files_only=config.local_files_only,
        dtype_name=config.dtype_name,
        trust_remote_code=config.trust_remote_code,
    )


def run_resolution(config: DiagnoseV3PolicyResolutionScriptConfig) -> None:
    if config.max_new_tokens <= 0:
        raise V3PolicyResolutionScriptError("max_new_tokens must be positive.")

    artifact = load_activation_artifact(config.artifact_path)
    loaded_model = load_causal_lm(_model_config(config))
    resolver: V3PolicyResolver = CausalLmV3PolicyResolver(
        loaded_model=loaded_model,
        max_new_tokens=config.max_new_tokens,
    )
    report = evaluate_v3_policy_resolution(
        artifact=artifact,
        resolver=resolver,
        resolver_model_id=f"{loaded_model.model_id}@{loaded_model.revision}",
        max_examples=config.max_examples,
    )
    write_v3_policy_resolution_json(config.output_json_path, report)
    write_v3_policy_resolution_markdown(config.output_markdown_path, report)

    print(f"Wrote V3 policy resolution report to {config.output_json_path}")
    print(f"Wrote V3 policy resolution summary to {config.output_markdown_path}")
    for slice_report in report.slices:
        print(
            f"{slice_report.slice_name}: model_macro_f1={slice_report.model.macro_f1:.4f} "
            f"model_accuracy={slice_report.model.accuracy:.4f} invalid={slice_report.model.invalid_count}"
        )


def main(argv: Sequence[str]) -> None:
    run_resolution(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
