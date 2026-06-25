"""FastAPI wrapper for the Aegis mock proxy surface."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from aegis.core.contracts import JsonValue
from aegis.detectors.cift_runtime import CiftFeatureExtractor
from aegis.proxy.mock_app import (
    MockProxyApp,
    create_default_proxy,
    create_default_proxy_with_cift_extractors,
    proxy_error_payload,
)


def create_http_app(proxy: MockProxyApp) -> FastAPI:
    app = FastAPI(
        title="Aegis Proxy",
        description="Development HTTP surface for the Aegis runtime spine.",
    )

    @app.get("/health")
    def health() -> JSONResponse:
        return _proxy_response(proxy, method="GET", path="/health", body={})

    @app.get("/ready")
    def ready() -> JSONResponse:
        return _proxy_response(proxy, method="GET", path="/ready", body={})

    @app.get("/aegis/capabilities")
    def capabilities() -> JSONResponse:
        return _proxy_response(proxy, method="GET", path="/aegis/capabilities", body={})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await _request_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        return _proxy_response(proxy, method="POST", path="/v1/chat/completions", body=body)

    @app.get("/audit/recent")
    def audit_recent(limit: int = 20, session_id: str | None = None) -> JSONResponse:
        body: dict[str, JsonValue] = {"limit": limit}
        if session_id is not None:
            body["session_id"] = session_id
        return _proxy_response(proxy, method="GET", path="/audit/recent", body=body)

    @app.get("/audit/explain")
    def audit_explain(trace_id: str | None = None, session_id: str | None = None) -> JSONResponse:
        body: dict[str, JsonValue] = {}
        if trace_id is not None:
            body["trace_id"] = trace_id
        if session_id is not None:
            body["session_id"] = session_id
        return _proxy_response(proxy, method="GET", path="/audit/explain", body=body)

    @app.post("/test/reset")
    async def test_reset(request: Request) -> JSONResponse:
        body = await _request_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        return _proxy_response(proxy, method="POST", path="/test/reset", body=body)

    @app.post("/test/seed-canary")
    async def test_seed_canary(request: Request) -> JSONResponse:
        body = await _request_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        return _proxy_response(proxy, method="POST", path="/test/seed-canary", body=body)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _http_exception_response(request=request, exc=exc)

    return app


def create_default_http_app() -> FastAPI:
    return create_http_app(create_default_proxy())


def create_default_http_app_with_cift_extractors(cift_extractors: Mapping[str, CiftFeatureExtractor]) -> FastAPI:
    return create_http_app(create_default_proxy_with_cift_extractors(cift_extractors=cift_extractors))


async def _request_json_object(request: Request) -> dict[str, JsonValue] | JSONResponse:
    try:
        raw_body = await request.json()
    except JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content=proxy_error_payload(code="invalid_json", message="Request body must be valid JSON.", details={}),
        )
    if not isinstance(raw_body, dict):
        return JSONResponse(
            status_code=400,
            content=proxy_error_payload(
                code="invalid_request",
                message="Request body must be a JSON object.",
                details={},
            ),
        )
    return cast(dict[str, JsonValue], raw_body)


def _proxy_response(proxy: MockProxyApp, method: str, path: str, body: dict[str, JsonValue]) -> JSONResponse:
    status_code, payload = proxy.handle(method=method, path=path, body=body)
    return JSONResponse(status_code=status_code, content=payload)


def _http_exception_response(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code == 404:
        code = "route_not_found"
        message = f"No route for {request.method} {request.url.path}."
    elif exc.status_code == 405:
        code = "method_not_allowed"
        message = f"Method {request.method} is not allowed for {request.url.path}."
    else:
        code = "http_error"
        message = str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=proxy_error_payload(
            code=code,
            message=message,
            details={"method": request.method, "path": request.url.path},
        ),
    )
