"""FastAPI wrapper for the Aegis mock proxy surface."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

from json import JSONDecodeError
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aegis.core.contracts import JsonValue
from aegis.proxy.mock_app import MockProxyApp, create_default_proxy


def create_http_app(proxy: MockProxyApp) -> FastAPI:
    app = FastAPI(
        title="Aegis Proxy",
        description="Development HTTP surface for the Aegis runtime spine.",
    )

    @app.get("/health")
    def health() -> JSONResponse:
        return _proxy_response(proxy, method="GET", path="/health", body={})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await _request_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        return _proxy_response(proxy, method="POST", path="/v1/chat/completions", body=body)

    @app.get("/audit/recent")
    def audit_recent() -> JSONResponse:
        return _proxy_response(proxy, method="GET", path="/audit/recent", body={})

    @app.post("/test/reset")
    async def test_reset(request: Request) -> JSONResponse:
        body = await _request_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        return _proxy_response(proxy, method="POST", path="/test/reset", body=body)

    return app


def create_default_http_app() -> FastAPI:
    return create_http_app(create_default_proxy())


async def _request_json_object(request: Request) -> dict[str, JsonValue] | JSONResponse:
    try:
        raw_body = await request.json()
    except JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Request body must be valid JSON."})
    if not isinstance(raw_body, dict):
        return JSONResponse(status_code=400, content={"error": "Request body must be a JSON object."})
    return cast(dict[str, JsonValue], raw_body)


def _proxy_response(proxy: MockProxyApp, method: str, path: str, body: dict[str, JsonValue]) -> JSONResponse:
    status_code, payload = proxy.handle(method=method, path=path, body=body)
    return JSONResponse(status_code=status_code, content=payload)
