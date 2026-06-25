"""Launch the local Aegis/Watchman Console."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from aegis.console.app import create_app
from aegis.console.service import ConsoleSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis-console",
        description="Run the local Aegis/Watchman Console.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8780, help="port")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000", help="Aegis gateway base URL")
    parser.add_argument("--gateway-timeout", type=float, default=2.0, help="gateway request timeout in seconds")
    parser.add_argument("--smoke-report", type=Path, help="optional latest smoke JSON report path")
    parser.add_argument("--sample-audit-jsonl", type=Path, help="optional audit JSONL used when live audit is empty")
    parser.add_argument(
        "--profile",
        choices=("observe", "balanced", "strict"),
        default="balanced",
        help="operator profile label shown when gateway strict mode is not active",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = ConsoleSettings(
        gateway_base_url=str(args.gateway_url),
        request_timeout_seconds=float(args.gateway_timeout),
        smoke_report_path=args.smoke_report,
        sample_audit_jsonl_path=args.sample_audit_jsonl,
        operator_profile=str(args.profile),
    )
    app = create_app(settings=settings, gateway_fetcher=None)
    print(f"Aegis Watchman Console on http://{args.host}:{args.port} -> {settings.gateway_base_url}")
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
