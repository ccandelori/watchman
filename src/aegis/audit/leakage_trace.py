from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Literal, TextIO, cast

from aegis.core.leakage import LeakageTrace


class LeakageTraceWriter:
    """Minimal JSONL writer for LeakageTrace records."""

    def __init__(self, path: str | Path, mode: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = cast(TextIO, self.path.open(mode=mode, encoding="utf-8"))

    def write(self, trace: LeakageTrace) -> None:
        self._file.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> LeakageTraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False
