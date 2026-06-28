"""FastAPI app for the local Aegis launcher."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aegis.core.contracts import JsonValue
from aegis.launcher.service import LauncherService, LauncherServiceError, default_settings

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(service: LauncherService) -> FastAPI:
    app = FastAPI(
        title="Aegis Local Launcher",
        description="Local setup launcher for Aegis Watchman.",
    )

    @app.exception_handler(LauncherServiceError)
    async def _on_launcher_error(request: Request, exc: LauncherServiceError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "invalid launcher request parameters."})

    @app.get("/api/state", response_model=None)
    def api_state() -> dict[str, JsonValue]:
        return service.state()

    @app.get("/api/launcher/capabilities", response_model=None)
    def api_launcher_capabilities() -> dict[str, JsonValue]:
        return service.capabilities()

    @app.put("/api/profile", response_model=None)
    def api_update_profile(body: dict[str, Any]) -> dict[str, JsonValue]:
        return service.update_profile(body)

    @app.post("/api/preflight", response_model=None)
    def api_preflight() -> dict[str, JsonValue]:
        return service.preflight()

    @app.post("/api/actions/{action_id}/start", response_model=None)
    def api_start_action(action_id: str) -> dict[str, JsonValue]:
        return service.start_action(action_id)

    @app.post("/api/actions/{action_id}/stop", response_model=None)
    def api_stop_action(action_id: str) -> dict[str, JsonValue]:
        return service.stop_action(action_id)

    @app.post("/api/actions/stop-all", response_model=None)
    def api_stop_all() -> dict[str, JsonValue]:
        return service.stop_all()

    @app.post("/api/actions/clear-statuses", response_model=None)
    def api_clear_statuses() -> dict[str, JsonValue]:
        return service.clear_statuses()

    @app.get("/api/observability", response_model=None)
    def api_observability(limit: int = 20) -> dict[str, JsonValue]:
        return service.observability(limit=limit)

    @app.get("/api/observability/traces/{trace_id}", response_model=None)
    def api_observability_trace(trace_id: str) -> dict[str, JsonValue]:
        return service.observability_trace(trace_id=trace_id)

    @app.get("/api/observability/trace", response_model=None)
    def api_observability_trace_query(trace_id: str) -> dict[str, JsonValue]:
        return service.observability_trace(trace_id=trace_id)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


app = create_app(LauncherService(settings=default_settings(Path.cwd()), supervisor=None))
