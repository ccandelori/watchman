"""Launch the local Aegis setup launcher."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis-launcher",
        description="Run the local Aegis setup launcher. Binds to localhost by default.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8790, help="port")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Aegis repository root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import uvicorn

    from aegis.launcher.app import create_app
    from aegis.launcher.service import LauncherService, default_settings

    repo_root = Path(args.repo_root).resolve()
    app = create_app(LauncherService(settings=default_settings(repo_root), supervisor=None))
    print(f"Aegis Local Launcher on http://{args.host}:{args.port} -> {repo_root}")
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
