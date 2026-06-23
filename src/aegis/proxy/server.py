"""Command-line entrypoint for the Aegis development proxy."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import uvicorn


@dataclass(frozen=True)
class ProxyServerConfig:
    host: str
    port: int


def parse_args(argv: Sequence[str]) -> ProxyServerConfig:
    parser = argparse.ArgumentParser(description="Run the Aegis development proxy.")
    parser.add_argument("--host", required=True, help="Host interface to bind, for example 127.0.0.1.")
    parser.add_argument("--port", required=True, type=int, help="TCP port to bind, for example 8000.")
    args = parser.parse_args(argv)
    return ProxyServerConfig(host=args.host, port=args.port)


def run_server(config: ProxyServerConfig) -> None:
    uvicorn.run(
        "aegis.proxy.http_app:create_default_http_app",
        factory=True,
        host=config.host,
        port=config.port,
    )


def main() -> None:
    run_server(parse_args(tuple(sys.argv[1:])))
