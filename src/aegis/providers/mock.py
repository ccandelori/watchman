from __future__ import annotations

from aegis.core.contracts import NormalizedTurn
from aegis.core.orchestrator import ModelResponse


class MockModelProvider:
    def __init__(self, default_content: str) -> None:
        self._default_content = default_content

    def generate(self, turn: NormalizedTurn) -> ModelResponse:
        content = turn.metadata.get("mock_response")
        if isinstance(content, str) and content != "":
            output_text = content
        else:
            output_text = self._default_content
        return ModelResponse(output_text=output_text, metadata={"provider": "mock"})
