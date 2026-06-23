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

from aegis_introspection.activations import run_hidden_state_forward, summarize_hidden_states  # noqa: E402
from aegis_introspection.features import normalize_layer_index, parse_layer_indices  # noqa: E402
from aegis_introspection.model_loader import (  # noqa: E402
    ModelDTypeName,
    ModelLoadConfig,
    load_causal_lm,
    parse_model_dtype,
)


@dataclass(frozen=True)
class HiddenStateSmokeConfig:
    model_id: str
    revision: str
    requested_device: str
    local_files_only: bool
    dtype_name: ModelDTypeName
    trust_remote_code: bool
    layer_indices: tuple[int, ...]
    prompt: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify that a Transformers model exposes hidden states.")
    parser.add_argument("--model-id", required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=False, default="main")
    parser.add_argument("--device", required=False, default="auto")
    parser.add_argument("--dtype", required=False, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--layers", required=False, default="0,-1")
    parser.add_argument("--prompt", required=False, default="Aegis hidden-state smoke test.")
    return parser


def _parse_args(argv: Sequence[str]) -> HiddenStateSmokeConfig:
    namespace = _build_parser().parse_args(argv)
    return HiddenStateSmokeConfig(
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        requested_device=str(namespace.device),
        local_files_only=not bool(namespace.allow_download),
        dtype_name=parse_model_dtype(str(namespace.dtype)),
        trust_remote_code=bool(namespace.trust_remote_code),
        layer_indices=parse_layer_indices(str(namespace.layers)),
        prompt=str(namespace.prompt),
    )


def _build_model_config(config: HiddenStateSmokeConfig) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_id=config.model_id,
        revision=config.revision,
        requested_device=config.requested_device,
        local_files_only=config.local_files_only,
        dtype_name=config.dtype_name,
        trust_remote_code=config.trust_remote_code,
    )


def run_smoke(config: HiddenStateSmokeConfig) -> None:
    loaded_model = load_causal_lm(_build_model_config(config))
    forward_pass = run_hidden_state_forward(loaded_model=loaded_model, prompt=config.prompt)
    summaries = summarize_hidden_states(forward_pass.hidden_states)
    layer_count = len(summaries)
    selected_layer_indices = tuple(
        normalize_layer_index(layer_index=layer_index, layer_count=layer_count)
        for layer_index in config.layer_indices
    )

    print(f"model_id: {loaded_model.model_id}")
    print(f"revision: {loaded_model.revision}")
    print(f"selected_device: {loaded_model.device.name}")
    print(f"dtype: {config.dtype_name}")
    print(f"input_tokens: {int(forward_pass.input_ids.shape[1])}")
    print(f"hidden_state_count: {layer_count}")
    for layer_index in selected_layer_indices:
        summary = summaries[layer_index]
        print(
            "layer "
            f"{summary.layer_index}: shape={summary.shape} dtype={summary.dtype} device={summary.device}"
        )


def main(argv: Sequence[str]) -> None:
    run_smoke(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
