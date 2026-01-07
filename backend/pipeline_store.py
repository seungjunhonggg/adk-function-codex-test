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

    def add_status(self, session_id: str, stage: str, message: str, done: bool = False) -> dict[str, Any]:
        record = dict(self._data.get(session_id, {}))
        history = list(record.get("status_history", []))
        history.append(
            {"stage": stage, "message": message, "done": bool(done)}
        )
        record["status_history"] = history[-60:]
        status = dict(record.get("status", {}))
        status[stage] = {"message": message, "done": bool(done)}
        record["status"] = status
        self._data[session_id] = record
        return dict(record)

    def get_status_history(self, session_id: str) -> list[dict[str, Any]]:
        record = self._data.get(session_id, {})
        history = record.get("status_history", [])
        return list(history) if isinstance(history, list) else []

    def get_status(self, session_id: str) -> dict[str, Any]:
        record = self._data.get(session_id, {})
        status = record.get("status", {})
        return dict(status) if isinstance(status, dict) else {}

    def get_events(self, session_id: str) -> dict[str, Any]:
        record = self._data.get(session_id, {})
        events = record.get("events", {})
        return dict(events) if isinstance(events, dict) else {}

    def get_event(self, session_id: str, event_type: str) -> Any | None:
        return self.get_events(session_id).get(event_type)

    def set_event(self, session_id: str, event_type: str, payload: Any) -> dict[str, Any]:
        record = dict(self._data.get(session_id, {}))
        events = dict(record.get("events", {}))
        events[event_type] = payload
        record["events"] = events
        self._data[session_id] = record
        return dict(record)

    def set_pending_memory_summary(
        self, session_id: str, summary: str, label: str | None = None
    ) -> dict[str, Any]:
        return self.update(
            session_id,
            pending_memory_summary=summary,
            pending_memory_label=label or "",
        )

    def pop_pending_memory_summary(self, session_id: str) -> str | None:
        record = dict(self._data.get(session_id, {}))
        summary = record.pop("pending_memory_summary", None)
        record.pop("pending_memory_label", None)
        self._data[session_id] = record
        return summary

    def set_reference(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, reference=payload, stage="reference")

    def set_grid(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, grid=payload, stage="grid")

    def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, overrides=overrides)


pipeline_store = PipelineStateStore()
