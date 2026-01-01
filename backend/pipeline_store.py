from __future__ import annotations

from typing import Any


class PipelineStateStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def get(self, session_id: str) -> dict[str, Any]:
        return dict(self._data.get(session_id, {}))

    def update(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        record = dict(self._data.get(session_id, {}))
        record.update(kwargs)
        self._data[session_id] = record
        return dict(record)

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def set_reference(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, reference=payload, stage="reference")

    def set_grid(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, grid=payload, stage="grid")

    def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, overrides=overrides)


pipeline_store = PipelineStateStore()
