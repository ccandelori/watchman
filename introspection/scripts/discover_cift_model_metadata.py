from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
INTROSPECTION_ROOT = SCRIPT_PATH.parents[1]
SRC_PATH = INTROSPECTION_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aegis_introspection.cift_model_metadata import (  # noqa: E402
    CiftModelMetadataConfig,
    cift_model_metadata_report_to_json,
    discover_cift_model_metadata,
)


@dataclass(frozen=True)
class DiscoverCiftModelMetadataCliConfig:
    model_id: str
    revision: str
    output_path: Path
    local_files_only: bool
    trust_remote_code: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover model/tokenizer metadata for CIFT certification.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> DiscoverCiftModelMetadataCliConfig:
    namespace = _build_parser().parse_args(argv)
    return DiscoverCiftModelMetadataCliConfig(
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        output_path=Path(namespace.output),
        local_files_only=not bool(namespace.allow_download),
        trust_remote_code=bool(namespace.trust_remote_code),
    )


def run_cli(config: DiscoverCiftModelMetadataCliConfig) -> None:
    report = discover_cift_model_metadata(
        CiftModelMetadataConfig(
            model_id=config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(cift_model_metadata_report_to_json(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote CIFT model metadata to {config.output_path}")
    print(f"Model: {report.model_id}@{report.revision}")
    print(f"Layers: {report.layer_count}")
    print(f"Hidden size: {report.hidden_size}")
    print(f"Tokenizer fingerprint: {report.tokenizer_fingerprint_sha256}")
    print(f"Chat template SHA-256: {report.chat_template_sha256}")


def main(argv: Sequence[str]) -> None:
    run_cli(_parse_args(argv))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
