from __future__ import annotations

from aegis.core.orchestrator import AegisRuntime, AegisRuntimeResponse, RuntimeRequest


def evaluate_turn(runtime: AegisRuntime, request: RuntimeRequest) -> AegisRuntimeResponse:
    return runtime.evaluate_turn(request)
