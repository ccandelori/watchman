"""Launch the DP-HONEY web UI with uvicorn (localhost only by default)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-honey-ui",
        description="Run the DP-HONEY web UI. Binds to localhost by default; every output is synthetic and shape-only.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import uvicorn

    from .app import app

    print(f"DP-HONEY UI on http://{args.host}:{args.port}  (synthetic, shape-only — not real credentials)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
