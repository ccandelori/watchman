"""FastAPI app for the local Aegis/Watchman Console."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aegis.console.service import (
    ConsoleGatewayError,
    ConsoleServiceError,
    ConsoleSettings,
    GatewayFetcher,
    console_events,
    console_overview,
    console_setup,
    console_trace,
    default_gateway_fetcher,
    settings_from_process_env,
)
from aegis.core.contracts import JsonValue

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(settings: ConsoleSettings, gateway_fetcher: GatewayFetcher | None) -> FastAPI:
    fetcher = default_gateway_fetcher if gateway_fetcher is None else gateway_fetcher
    app = FastAPI(
        title="Aegis Watchman Console",
        description="Local operator console for the Aegis/Watchman sentinel.",
    )

    @app.exception_handler(ConsoleServiceError)
    async def _on_console_error(request: Request, exc: ConsoleServiceError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(ConsoleGatewayError)
    async def _on_gateway_error(request: Request, exc: ConsoleGatewayError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": str(exc)})

    @app.get("/api/overview", response_model=None)
    def api_overview() -> dict[str, JsonValue]:
        return console_overview(settings=settings, fetcher=fetcher)

    @app.get("/api/events", response_model=None)
    def api_events(limit: int = 20, session_id: str | None = None) -> dict[str, JsonValue]:
        return console_events(settings=settings, fetcher=fetcher, limit=limit, session_id=session_id)

    @app.get("/api/trace", response_model=None)
    def api_trace(trace_id: str) -> dict[str, JsonValue]:
        return console_trace(settings=settings, fetcher=fetcher, trace_id=trace_id)

    @app.get("/api/setup", response_model=None)
    def api_setup() -> dict[str, JsonValue]:
        return console_setup(settings)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


app = create_app(settings_from_process_env(), None)
