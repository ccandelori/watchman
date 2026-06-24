"""Command-line interface for DP-HONEY (``python -m detect.dp_honey``).

Subcommands: ``list-formats``, ``preview-corpus``, ``train``, ``generate``,
``inspect-model``, ``validate``, ``report``.

Every :class:`DPHoneyError` is mapped to a concise stderr message and exit code 1;
argparse handles usage errors with exit code 2. Commands that emit token-like
material print a synthetic/non-functional safety banner to stderr so output copied
into a demo is never mistaken for real credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from . import scanner
from .artifact_status import inspect_artifact, validate_artifact
from .bigram import (
    DEFAULT_CLIP,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_EPSILON,
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TRAIN_SEED,
)
from .errors import DPHoneyError
from .formats import get_format, list_formats
from .operations import (
    GENERATE_MAX,
    FormatModelSource,
    GenerateRequest,
    ModelArtifactSource,
    ReportRequest,
    TrainRequest,
    generate_tokens,
    run_report_request,
    train_to_artifact,
)
from .realism import REPORT_MAX, enforce_count_limit

DESCRIPTION = (
    "DP-HONEY: generate synthetic, shape-only honeytokens for credential-leak "
    "detection research. Every output is a non-functional decoy -- never a real, "
    "valid, signed, or usable credential."
)

SAFETY_BANNER = "# DP-HONEY: synthetic, shape-only honeytokens -- NOT real, valid, or usable credentials."


def main(argv: Sequence[str] | None = None) -> int:
    """Parse *argv* and dispatch. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler = cast(Callable[[argparse.Namespace], int], args.func)
        return handler(args)
    except DPHoneyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dp-honey", description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-formats", help="list registered shape-only formats")
    p_list.add_argument("--json", action="store_true", help="emit format specs as JSON")
    p_list.set_defaults(func=cmd_list_formats)

    p_prev = sub.add_parser("preview-corpus", help="print synthetic training examples")
    p_prev.add_argument("--format", required=True, help="format slug")
    p_prev.add_argument("--count", type=int, default=10, help="number of examples")
    p_prev.add_argument("--seed", type=int, default=0, help="corpus seed")
    p_prev.set_defaults(func=cmd_preview_corpus)

    p_train = sub.add_parser("train", help="train and save a model artifact")
    p_train.add_argument("--format", required=True, help="format slug")
    p_train.add_argument("--out", required=True, help="output artifact path")
    p_train.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    p_train.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    p_train.add_argument("--corpus-size", type=int, default=DEFAULT_CORPUS_SIZE, dest="corpus_size")
    p_train.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED, help="training seed")
    p_train.add_argument("--force", action="store_true", help="overwrite an existing artifact")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="generate synthetic honeytokens")
    _add_model_source(p_gen)
    p_gen.add_argument("--count", type=int, required=True, help=f"number to generate (<= {GENERATE_MAX})")
    p_gen.add_argument("--json", action="store_true", help="emit a JSON array instead of lines")
    p_gen.set_defaults(func=cmd_generate)

    p_inspect = sub.add_parser("inspect-model", help="show artifact metadata (lenient)")
    p_inspect.add_argument("--model", required=True, help="path to a saved model artifact")
    p_inspect.set_defaults(func=cmd_inspect_model)

    p_validate = sub.add_parser("validate", help="strictly validate a model artifact")
    p_validate.add_argument("--model", required=True, help="path to a saved model artifact")
    p_validate.set_defaults(func=cmd_validate)

    p_report = sub.add_parser("report", help="generate a batch and compute realism metrics")
    _add_model_source(p_report)
    p_report.add_argument("--count", type=int, required=True, help=f"batch size (<= {REPORT_MAX})")
    p_report.set_defaults(func=cmd_report)

    p_scan = sub.add_parser("scan", help="detect secret-shaped substrings in text")
    p_scan.add_argument("--file", help="path to scan (default: stdin)")
    p_scan.add_argument(
        "--show-matches",
        action="store_true",
        help="also include matched values (OFF by default; handle with care)",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_auto = sub.add_parser("auto-decoy", help="scan text and emit a matching decoy per finding")
    p_auto.add_argument("--file", help="path to scan (default: stdin)")
    p_auto.add_argument("--seed", type=int, default=0)
    p_auto.set_defaults(func=cmd_auto_decoy)

    return parser


def _add_model_source(parser: argparse.ArgumentParser) -> None:
    """Add the mutually-exclusive --format/--model source plus training/sample args."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--format", help="format slug to train on the fly")
    source.add_argument("--model", help="path to a saved model artifact")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    parser.add_argument("--corpus-size", type=int, default=DEFAULT_CORPUS_SIZE, dest="corpus_size")
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED, dest="train_seed")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED, help="sample seed")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_REPAIR_ATTEMPTS,
        dest="max_attempts",
        help="bounded repair attempts per token",
    )


def _source_from_args(args: argparse.Namespace) -> FormatModelSource | ModelArtifactSource:
    """Resolve a typed model source from --model or --format arguments."""
    if args.model:
        return ModelArtifactSource(path=Path(args.model))
    return FormatModelSource(
        format_slug=args.format,
        epsilon=args.epsilon,
        clip=args.clip,
        corpus_size=args.corpus_size,
        train_seed=args.train_seed,
    )


def _emit_safety_banner() -> None:
    print(SAFETY_BANNER, file=sys.stderr)


def _read_input(path: str | None) -> str:
    if path:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise DPHoneyError(f"could not read input {path}: {exc}") from exc
    return sys.stdin.read()


# --- command handlers ----------------------------------------------------------


def cmd_list_formats(args: argparse.Namespace) -> int:
    specs = list_formats()
    if args.json:
        print(json.dumps([spec.to_snapshot() for spec in specs], indent=2))
        return 0
    print("# DP-HONEY formats -- all outputs are synthetic, shape-only, non-functional decoys.")
    for spec in specs:
        print(f"{spec.slug}\t{spec.name}\t[{spec.category}]")
    return 0


def cmd_preview_corpus(args: argparse.Namespace) -> int:
    enforce_count_limit(args.count, maximum=GENERATE_MAX, label="--count")
    spec = get_format(args.format)
    _emit_safety_banner()
    rng = np.random.default_rng(args.seed)
    for _ in range(args.count):
        print(spec.random_example(rng))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    result = train_to_artifact(
        TrainRequest(
            format_slug=args.format,
            output_path=Path(args.out),
            epsilon=args.epsilon,
            clip=args.clip,
            corpus_size=args.corpus_size,
            train_seed=args.seed,
            force=args.force,
        )
    )
    print(f"trained {args.format} -> {result.path}")
    print(
        f"  epsilon={result.epsilon} clip={result.clip} corpus_size={result.corpus_size} train_seed={result.train_seed}"
    )
    print("  NOTE: synthetic, shape-only model; outputs are not real credentials.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    result = generate_tokens(
        GenerateRequest(
            source=_source_from_args(args),
            count=args.count,
            sample_seed=args.seed,
            max_repair_attempts=args.max_attempts,
        )
    )
    _emit_safety_banner()
    if args.json:
        print(json.dumps(list(result.tokens), indent=2))
    else:
        for token in result.tokens:
            print(token)
    return 0


def cmd_inspect_model(args: argparse.Namespace) -> int:
    inspection = inspect_artifact(args.model)
    safety = inspection.safety if isinstance(inspection.safety, dict) else {}
    print(f"artifact: {args.model}")
    print(f"  schema_version: {inspection.schema_version}")
    print(f"  format: {inspection.format_slug} (registry_version={inspection.registry_version})")
    print(
        f"  epsilon={inspection.epsilon} clip={inspection.clip} "
        f"corpus_size={inspection.corpus_size} train_seed={inspection.train_seed}"
    )
    print(f"  alphabet_size: {inspection.alphabet_size}")
    print(f"  snapshot_status: {inspection.snapshot_status.value}")
    print(f"  safety: synthetic_only={safety.get('synthetic_only')} provider_valid={safety.get('provider_valid')}")
    if safety.get("note"):
        print(f"  note: {safety['note']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_artifact(args.model)
    if not result.valid:
        raise DPHoneyError(result.error or f"invalid artifact: {args.model}")
    print(f"valid: {args.model}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = run_report_request(
        ReportRequest(
            source=_source_from_args(args),
            count=args.count,
            sample_seed=args.seed,
            max_repair_attempts=args.max_attempts,
        )
    )
    _emit_safety_banner()
    print(json.dumps(report, indent=2))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    findings = scanner.scan(text)
    output: dict[str, object] = {"findings": findings}
    if args.show_matches:
        print("# DP-HONEY warning: --show-matches echoes matched input values.", file=sys.stderr)
        output["matches"] = [text[int(finding["start"]) : int(finding["end"])] for finding in findings]
    print(json.dumps(output, indent=2))
    return 0


def cmd_auto_decoy(args: argparse.Namespace) -> int:
    _emit_safety_banner()
    result = scanner.auto_decoy(_read_input(args.file), seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
