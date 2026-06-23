from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.activations import run_hidden_state_forward  # noqa: E402
from aegis_introspection.artifacts import ActivationArtifact  # noqa: E402
from aegis_introspection.features import (  # noqa: E402
    ActivationFeature,
    PoolingMethod,
    extract_activation_features,
    parse_layer_indices,
    parse_pooling_methods,
    stack_feature_rows,
)
from aegis_introspection.model_loader import (  # noqa: E402
    ModelDTypeName,
    ModelLoadConfig,
    load_causal_lm,
    parse_model_dtype,
)
from aegis_introspection.prompts import (  # noqa: E402
    PromptExample,
    StructuredPromptExample,
    load_prompt_examples,
    load_structured_prompt_examples,
)
from aegis_introspection.sealed_holdout import (  # noqa: E402
    add_unseal_flag,
    assert_unsealed_paths,
    assert_unsealed_tag_rows,
)

PromptExtractionExample: TypeAlias = PromptExample | StructuredPromptExample


@dataclass(frozen=True)
class ExtractionScriptConfig:
    prompts_path: Path
    output_path: Path
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    layer_indices: tuple[int, ...]
    pooling_methods: tuple[PoolingMethod, ...]
    allow_sealed_holdout: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract hidden-state features for labeled Aegis prompts.")
    parser.add_argument("--prompts", required=False, default=str(INTROSPECTION_ROOT / "data" / "prompts.jsonl"))
    parser.add_argument(
        "--output",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_features.pt"),
    )
    parser.add_argument("--model-id", required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=False, default="main")
    parser.add_argument("--device", required=False, default="auto")
    parser.add_argument("--dtype", required=False, default="device")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--layers", required=False, default="0,7,14,21,28")
    parser.add_argument("--pooling", required=False, default="final_token,mean_pool")
    parser.add_argument("--allow-download", action="store_true")
    add_unseal_flag(parser)
    return parser


def _parse_args(argv: Sequence[str]) -> ExtractionScriptConfig:
    namespace = _build_parser().parse_args(argv)
    return ExtractionScriptConfig(
        prompts_path=Path(namespace.prompts),
        output_path=Path(namespace.output),
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        requested_device=str(namespace.device),
        local_files_only=not bool(namespace.allow_download),
        dtype_name=parse_model_dtype(str(namespace.dtype)),
        trust_remote_code=bool(namespace.trust_remote_code),
        layer_indices=parse_layer_indices(str(namespace.layers)),
        pooling_methods=parse_pooling_methods(str(namespace.pooling)),
        allow_sealed_holdout=bool(namespace.allow_sealed_holdout),
    )


def _build_model_config(config: ExtractionScriptConfig) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_id=config.model_id,
        revision=config.revision,
        requested_device=config.requested_device,
        local_files_only=config.local_files_only,
        dtype_name=config.dtype_name,
        trust_remote_code=config.trust_remote_code,
    )


def _save_artifact(
    config: ExtractionScriptConfig,
    examples: tuple[PromptExtractionExample, ...],
    selected_device: str,
    feature_tensors: dict[str, torch.Tensor],
) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact: ActivationArtifact = {
        "metadata": {
            "model_id": config.model_id,
            "revision": config.revision,
            "selected_device": selected_device,
            "dtype": config.dtype_name,
            "trust_remote_code": config.trust_remote_code,
            "layer_indices": config.layer_indices,
            "pooling_methods": config.pooling_methods,
        },
        "example_ids": tuple(example.id for example in examples),
        "labels": tuple(example.label for example in examples),
        "families": tuple(example.family for example in examples),
        "texts": tuple(example.text for example in examples),
        "tags": tuple(example.tags for example in examples),
        "features": feature_tensors,
    }
    torch.save(artifact, config.output_path)


def _load_examples(config: ExtractionScriptConfig) -> tuple[PromptExtractionExample, ...]:
    if _requires_structured_prompts(config.pooling_methods):
        return load_structured_prompt_examples(config.prompts_path)
    return load_prompt_examples(config.prompts_path)


def _requires_structured_prompts(pooling_methods: tuple[PoolingMethod, ...]) -> bool:
    structured_methods = frozenset(
        ("readout_window", "query_tail_window", "selected_choice_window", "combined_readout_window")
    )
    return any(method in structured_methods for method in pooling_methods)


def _readout_token_indices(
    example: PromptExtractionExample,
    pooling_methods: tuple[PoolingMethod, ...],
) -> tuple[int, ...] | None:
    if "readout_window" not in pooling_methods and "combined_readout_window" not in pooling_methods:
        return None
    if not isinstance(example, StructuredPromptExample):
        raise TypeError("readout_window pooling requires structured prompt examples.")
    return example.readout_token_indices


def _selected_choice_readout_token_indices(
    example: PromptExtractionExample,
    pooling_methods: tuple[PoolingMethod, ...],
) -> tuple[int, ...] | None:
    if "selected_choice_window" not in pooling_methods and "combined_readout_window" not in pooling_methods:
        return None
    if not isinstance(example, StructuredPromptExample):
        raise TypeError("selected_choice_window pooling requires structured prompt examples.")
    return example.selected_choice_readout_token_indices


def _query_tail_readout_token_indices(
    example: PromptExtractionExample,
    pooling_methods: tuple[PoolingMethod, ...],
) -> tuple[int, ...] | None:
    if "query_tail_window" not in pooling_methods:
        return None
    if not isinstance(example, StructuredPromptExample):
        raise TypeError("query_tail_window pooling requires structured prompt examples.")
    return example.query_tail_readout_token_indices


def run_extraction(config: ExtractionScriptConfig) -> None:
    assert_unsealed_paths(
        paths=(config.prompts_path, config.output_path),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="activation extraction",
    )
    examples = _load_examples(config)
    assert_unsealed_tag_rows(
        tag_rows=(example.tags for example in examples),
        allow_sealed_holdout=config.allow_sealed_holdout,
        context="activation extraction",
    )
    loaded_model = load_causal_lm(_build_model_config(config))

    feature_rows: list[tuple[ActivationFeature, ...]] = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example.id} {example.label}")
        forward_pass = run_hidden_state_forward(loaded_model, example.text)
        feature_rows.append(
            extract_activation_features(
                forward_pass=forward_pass,
                layer_indices=config.layer_indices,
                pooling_methods=config.pooling_methods,
                readout_token_indices=_readout_token_indices(example=example, pooling_methods=config.pooling_methods),
                query_tail_readout_token_indices=_query_tail_readout_token_indices(
                    example=example,
                    pooling_methods=config.pooling_methods,
                ),
                selected_choice_readout_token_indices=_selected_choice_readout_token_indices(
                    example=example,
                    pooling_methods=config.pooling_methods,
                ),
            )
        )

    feature_tensors = stack_feature_rows(tuple(feature_rows))
    _save_artifact(
        config=config,
        examples=examples,
        selected_device=loaded_model.device.name,
        feature_tensors=feature_tensors,
    )
    print(f"Wrote activation features to {config.output_path}")


def main(argv: Sequence[str]) -> None:
    run_extraction(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
