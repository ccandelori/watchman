"""FastAPI app for the DP-HONEY web UI: thin routes over the service layer."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..errors import DPHoneyError
from . import service

_STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="DP-HONEY UI",
        description="Synthetic, shape-only honeytoken generator. Outputs are never real credentials.",
    )

    @app.exception_handler(DPHoneyError)
    async def _on_dphoney_error(request: Request, exc: DPHoneyError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.get("/api/formats")
    def api_formats() -> list:
        return service.list_formats_payload()

    @app.post("/api/preview-corpus")
    def api_preview(body: dict) -> dict:
        examples = service.preview_corpus(body.get("format"), int(body.get("count", 10)), int(body.get("seed", 0)))
        return {"examples": examples}

    @app.post("/api/generate")
    def api_generate(body: dict) -> dict:
        return service.run_generate(body)

    @app.post("/api/report")
    def api_report(body: dict) -> dict:
        return service.run_report(body)

    @app.post("/api/scan")
    def api_scan(body: dict) -> dict:
        return service.run_scan(body.get("text", ""))

    @app.post("/api/auto-decoy")
    def api_auto_decoy(body: dict) -> dict:
        return service.run_auto_decoy(body.get("text", ""), seed=int(body.get("seed", 0)))

    @app.post("/api/train")
    def api_train(body: dict) -> dict:
        return service.run_train(body)

    @app.get("/api/models")
    def api_models() -> list:
        return service.list_models()

    @app.post("/api/inspect")
    def api_inspect(body: dict) -> dict:
        return service.run_inspect(body.get("model"))

    @app.post("/api/validate")
    def api_validate(body: dict) -> dict:
        return service.run_validate(body.get("model"))

    @app.get("/api/models/{name}/download")
    def api_download(name: str) -> FileResponse:
        ref = service.resolve_model_ref(name)
        return FileResponse(ref, media_type="application/json", filename=ref.name)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


app = create_app()
