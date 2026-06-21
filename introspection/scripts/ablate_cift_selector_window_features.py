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

from aegis_introspection.artifacts import load_activation_artifact
from aegis_introspection.binary_tasks import BinaryTaskConfig, BinaryTaskError
from aegis_introspection.cift_feature_ablation import (
    CiftFeatureAblationVariant,
    evaluate_grouped_cift_feature_ablation,
    write_cift_feature_ablation_json,
    write_cift_feature_ablation_markdown,
)


@dataclass(frozen=True)
class AblateCiftSelectorWindowFeaturesCliConfig:
    artifact_path: Path
    output_json_path: Path
    output_markdown_path: Path
    task_name: str
    baseline_variant_id: str
    variants: tuple[CiftFeatureAblationVariant, ...]
    fold_count: int
    random_seed: int
    max_iter: int
    regularization_c: float
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]


def _feature_key(layer_index: int) -> str:
    return f"readout_window_layer_{layer_index:02d}"


def _concat_feature_key(layer_indices: tuple[int, ...]) -> str:
    return "concat(" + ",".join(_feature_key(layer_index) for layer_index in layer_indices) + ")"


def _layer_range(start: int, end: int) -> tuple[int, ...]:
    if start > end:
        raise BinaryTaskError("Layer range start must be less than or equal to end.")
    return tuple(range(start, end + 1))


def _default_selector_window_variants() -> tuple[CiftFeatureAblationVariant, ...]:
    return (
        CiftFeatureAblationVariant(
            variant_id="baseline_layer_15",
            feature_key="readout_window_layer_15",
        ),
        CiftFeatureAblationVariant(
            variant_id="single_layer_21",
            feature_key="readout_window_layer_21",
        ),
        CiftFeatureAblationVariant(
            variant_id="local_concat_14_15_16",
            feature_key=_concat_feature_key((14, 15, 16)),
        ),
        CiftFeatureAblationVariant(
            variant_id="selector_band_12_18",
            feature_key=_concat_feature_key(_layer_range(12, 18)),
        ),
        CiftFeatureAblationVariant(
            variant_id="late_band_15_21",
            feature_key=_concat_feature_key(_layer_range(15, 21)),
        ),
        CiftFeatureAblationVariant(
            variant_id="last_quarter_22_28",
            feature_key=_concat_feature_key(_layer_range(22, 28)),
        ),
        CiftFeatureAblationVariant(
            variant_id="all_readout_layers_00_28",
            feature_key=_concat_feature_key(_layer_range(0, 28)),
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grouped CIFT selector-window feature ablations.")
    parser.add_argument(
        "--artifact",
        required=False,
        default=str(INTROSPECTION_ROOT / "data" / "activations" / "qwen3_0_6b_dp_honey_lite_v3_selector_windows.pt"),
    )
    parser.add_argument(
        "--output-json",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_feature_ablation_v1.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        required=False,
        default=str(
            INTROSPECTION_ROOT
            / "data"
            / "reports"
            / "dp_honey_lite_v3_selector_window_feature_ablation_v1_summary.md"
        ),
    )
    parser.add_argument("--task", required=False, default="safe_secret_vs_exfiltration")
    parser.add_argument("--baseline-variant", required=False, default="baseline_layer_15")
    parser.add_argument(
        "--variant",
        required=False,
        action="append",
        help="Named feature variant in variant_id=feature_key form. May be repeated.",
    )
    parser.add_argument("--folds", required=False, type=int, default=5)
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--max-iter", required=False, type=int, default=1000)
    parser.add_argument("--regularization-c", required=False, type=float, default=1.0)
    parser.add_argument("--word-ngram-min", required=False, type=int, default=1)
    parser.add_argument("--word-ngram-max", required=False, type=int, default=2)
    parser.add_argument("--char-ngram-min", required=False, type=int, default=3)
    parser.add_argument("--char-ngram-max", required=False, type=int, default=5)
    return parser


def _parse_args(argv: Sequence[str]) -> AblateCiftSelectorWindowFeaturesCliConfig:
    namespace = _build_parser().parse_args(argv)
    return AblateCiftSelectorWindowFeaturesCliConfig(
        artifact_path=Path(namespace.artifact),
        output_json_path=Path(namespace.output_json),
        output_markdown_path=Path(namespace.output_md),
        task_name=str(namespace.task),
        baseline_variant_id=str(namespace.baseline_variant),
        variants=_parse_variants(namespace.variant),
        fold_count=int(namespace.folds),
        random_seed=int(namespace.seed),
        max_iter=int(namespace.max_iter),
        regularization_c=float(namespace.regularization_c),
        word_ngram_range=(int(namespace.word_ngram_min), int(namespace.word_ngram_max)),
        char_ngram_range=(int(namespace.char_ngram_min), int(namespace.char_ngram_max)),
    )


def _parse_variants(raw_variants: list[str] | None) -> tuple[CiftFeatureAblationVariant, ...]:
    if raw_variants is None:
        return _default_selector_window_variants()
    return tuple(_parse_variant(raw_variant) for raw_variant in raw_variants)


def _parse_variant(raw_variant: str) -> CiftFeatureAblationVariant:
    parts = raw_variant.split("=", maxsplit=1)
    if len(parts) != 2:
        raise BinaryTaskError("Variant must use variant_id=feature_key form.")
    variant_id = parts[0].strip()
    feature_key = parts[1].strip()
    if variant_id == "" or feature_key == "":
        raise BinaryTaskError("Variant id and feature key must be non-empty.")
    return CiftFeatureAblationVariant(variant_id=variant_id, feature_key=feature_key)


def _binary_task_config(config: AblateCiftSelectorWindowFeaturesCliConfig) -> BinaryTaskConfig:
    return BinaryTaskConfig(
        fold_count=config.fold_count,
        random_seed=config.random_seed,
        max_iter=config.max_iter,
        regularization_c=config.regularization_c,
        activation_feature_key="readout_window_layer_15",
        word_ngram_range=config.word_ngram_range,
        char_ngram_range=config.char_ngram_range,
    )


def run_ablation(config: AblateCiftSelectorWindowFeaturesCliConfig) -> None:
    artifact = load_activation_artifact(config.artifact_path)
    report = evaluate_grouped_cift_feature_ablation(
        artifact=artifact,
        task_name=config.task_name,
        variants=config.variants,
        baseline_variant_id=config.baseline_variant_id,
        config=_binary_task_config(config),
    )
    write_cift_feature_ablation_json(config.output_json_path, report)
    write_cift_feature_ablation_markdown(config.output_markdown_path, report)
    baseline = next(variant for variant in report.variants if variant.is_baseline)
    best = report.variants[0]
    print(f"Wrote CIFT feature ablation JSON to {config.output_json_path}")
    print(f"Wrote CIFT feature ablation summary to {config.output_markdown_path}")
    print(f"best={best.variant_id} macro_f1={best.macro_f1_mean:.4f} accuracy={best.accuracy_mean:.4f}")
    print(
        f"baseline={baseline.variant_id} "
        f"macro_f1={baseline.macro_f1_mean:.4f} accuracy={baseline.accuracy_mean:.4f}"
    )


def main(argv: Sequence[str]) -> None:
    run_ablation(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
