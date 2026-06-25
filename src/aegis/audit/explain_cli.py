from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aegis.audit.explain import explain_audit_record
from aegis.audit.jsonl import JsonlAuditSinkError, find_jsonl_audit_record
from aegis.core.contracts import JsonValue


class AuditExplainCliError(ValueError):
    """Raised when a durable audit explanation cannot be produced."""


@dataclass(frozen=True)
class AuditExplainCliConfig:
    input_path: Path
    output_path: Path | None
    trace_id: str | None
    session_id: str | None


def parse_args(argv: Sequence[str]) -> AuditExplainCliConfig:
    parser = argparse.ArgumentParser(description="Explain one redacted Aegis audit JSONL trace or session.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a redacted Aegis audit JSONL file.")
    parser.add_argument("--trace-id", required=False, help="Trace id to explain.")
    parser.add_argument("--session-id", required=False, help="Session id to explain. Uses the newest matching event.")
    parser.add_argument("--output", required=False, type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    trace_id = _optional_non_empty_string(args.trace_id, "--trace-id")
    session_id = _optional_non_empty_string(args.session_id, "--session-id")
    if trace_id is None and session_id is None:
        raise AuditExplainCliError("--trace-id or --session-id must be provided.")
    return AuditExplainCliConfig(
        input_path=Path(str(args.input)),
        output_path=None if args.output is None else Path(str(args.output)),
        trace_id=trace_id,
        session_id=session_id,
    )


def explain_audit_jsonl(config: AuditExplainCliConfig) -> dict[str, JsonValue]:
    record = find_jsonl_audit_record(path=config.input_path, trace_id=config.trace_id, session_id=config.session_id)
    if record is None:
        raise AuditExplainCliError("No audit record matched the requested trace_id/session_id.")
    try:
        return explain_audit_record(record)
    except ValueError as exc:
        raise AuditExplainCliError(str(exc)) from exc


def render_audit_explanation_json(explanation: dict[str, JsonValue]) -> str:
    return json.dumps(explanation, allow_nan=False, indent=2, sort_keys=True) + "\n"


def run_cli(config: AuditExplainCliConfig) -> int:
    explanation = explain_audit_jsonl(config)
    rendered = render_audit_explanation_json(explanation)
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


def main() -> None:
    try:
        run_cli(parse_args(tuple(sys.argv[1:])))
    except (AuditExplainCliError, JsonlAuditSinkError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc


def _optional_non_empty_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise AuditExplainCliError(f"{field_name} must be a non-empty string when provided.")
    return value
