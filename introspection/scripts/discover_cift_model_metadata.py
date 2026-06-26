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

from aegis.cift_contract import CIFT_SUPPORT_STATE_FAILED_CERTIFICATION  # noqa: E402


@dataclass(frozen=True)
class DiscoverCiftModelMetadataCliConfig:
    model_id: str
    revision: str
    output_path: Path
    requested_device: str
    dtype_name: str
    selected_readout_candidates: tuple[str, ...]
    local_files_only: bool
    trust_remote_code: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover model/tokenizer metadata for CIFT certification.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--selected-readout-candidate", action="append", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _parse_args(argv: Sequence[str]) -> DiscoverCiftModelMetadataCliConfig:
    namespace = _build_parser().parse_args(argv)
    return DiscoverCiftModelMetadataCliConfig(
        model_id=str(namespace.model_id),
        revision=str(namespace.revision),
        output_path=Path(namespace.output),
        requested_device=str(namespace.device),
        dtype_name=str(namespace.dtype),
        selected_readout_candidates=tuple(str(value) for value in namespace.selected_readout_candidate),
        local_files_only=not bool(namespace.allow_download),
        trust_remote_code=bool(namespace.trust_remote_code),
    )


def run_cli(config: DiscoverCiftModelMetadataCliConfig) -> int:
    try:
        report = discover_cift_model_metadata(
            CiftModelMetadataConfig(
                model_id=config.model_id,
                revision=config.revision,
                requested_device=config.requested_device,
                dtype_name=config.dtype_name,
                selected_readout_candidates=config.selected_readout_candidates,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
            )
        )
    except (ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _write_failure_report(config=config, failure_reason=str(exc))
        print(f"CIFT model metadata discovery failed: {config.model_id}@{config.revision}")
        print(f"Metadata report: {config.output_path}")
        print(f"- {exc}")
        return 1
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(cift_model_metadata_report_to_json(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote CIFT model metadata to {config.output_path}")
    print(f"Model: {report.model_id}@{report.revision}")
    print(f"Support state: {report.support_state}")
    print(f"Device/dtype: {report.selected_device}/{report.resolved_torch_dtype}")
    print(f"Hidden-state support: {report.hidden_state_support}")
    print(f"Layers: {report.layer_count}")
    print(f"Hidden size: {report.hidden_size}")
    print(f"Selected readout candidates: {', '.join(report.selected_readout_candidates)}")
    print(f"Tokenizer fingerprint: {report.tokenizer_fingerprint_sha256}")
    print(f"Chat template SHA-256: {report.chat_template_sha256}")
    return 0


def _write_failure_report(config: DiscoverCiftModelMetadataCliConfig, failure_reason: str) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(
            {
                "schema_version": "aegis_introspection.cift_model_metadata/v1",
                "support_state": CIFT_SUPPORT_STATE_FAILED_CERTIFICATION,
                "model_id": config.model_id,
                "revision": config.revision,
                "resolved_revision": config.revision,
                "requested_device": config.requested_device,
                "dtype_name": config.dtype_name,
                "selected_readout_candidates": list(config.selected_readout_candidates),
                "failure_reason": failure_reason,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str]) -> int:
    return run_cli(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
