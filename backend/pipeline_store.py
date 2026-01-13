from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from .config import PIPELINE_STATE_DB_PATH


class PipelineStateStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._db_path = (db_path or "").strip()
        self._db_ready = False
        if self._db_path:
            self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pipeline_state (
                        session_id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            self._db_ready = True
        except sqlite3.Error:
            self._db_ready = False

    def _load_from_db(self, session_id: str) -> dict[str, Any] | None:
        if not self._db_ready or not session_id:
            return None
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT payload FROM pipeline_state WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            if not row or not row[0]:
                return None
            payload = json.loads(row[0])
            return payload if isinstance(payload, dict) else None
        except (sqlite3.Error, json.JSONDecodeError):
            return None

    def _persist(self, session_id: str, record: dict[str, Any]) -> None:
        if not self._db_ready or not session_id:
            return
        try:
            payload = json.dumps(record, ensure_ascii=False, default=str)
            updated_at = datetime.utcnow().isoformat()
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_state (session_id, payload, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, payload, updated_at),
                )
                conn.commit()
        except sqlite3.Error:
            return

    def _delete_from_db(self, session_id: str) -> None:
        if not self._db_ready or not session_id:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "DELETE FROM pipeline_state WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
        except sqlite3.Error:
            return

    def _get_record(self, session_id: str) -> dict[str, Any]:
        if not session_id:
            return {}
        record = self._data.get(session_id)
        if record is None:
            record = self._load_from_db(session_id) or {}
            self._data[session_id] = record
        return record

    def get(self, session_id: str) -> dict[str, Any]:
        record = self._get_record(session_id)
        return dict(record)

    def update(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        record = dict(self._get_record(session_id))
        record.update(kwargs)
        self._data[session_id] = record
        self._persist(session_id, record)
        return dict(record)

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)
        self._delete_from_db(session_id)

    def add_status(self, session_id: str, stage: str, message: str, done: bool = False) -> dict[str, Any]:
        record = dict(self._get_record(session_id))
        history = list(record.get("status_history", []))
        history.append(
            {"stage": stage, "message": message, "done": bool(done)}
        )
        record["status_history"] = history[-60:]
        status = dict(record.get("status", {}))
        status[stage] = {"message": message, "done": bool(done)}
        record["status"] = status
        self._data[session_id] = record
        self._persist(session_id, record)
        return dict(record)

    def get_status_history(self, session_id: str) -> list[dict[str, Any]]:
        record = self._get_record(session_id)
        history = record.get("status_history", [])
        return list(history) if isinstance(history, list) else []

    def get_status(self, session_id: str) -> dict[str, Any]:
        record = self._get_record(session_id)
        status = record.get("status", {})
        return dict(status) if isinstance(status, dict) else {}

    def get_events(self, session_id: str) -> dict[str, Any]:
        record = self._get_record(session_id)
        events = record.get("events", {})
        return dict(events) if isinstance(events, dict) else {}

    def get_event(self, session_id: str, event_type: str) -> Any | None:
        return self.get_events(session_id).get(event_type)

    def set_event(self, session_id: str, event_type: str, payload: Any) -> dict[str, Any]:
        record = dict(self._get_record(session_id))
        events = dict(record.get("events", {}))
        events[event_type] = payload
        record["events"] = events
        self._data[session_id] = record
        self._persist(session_id, record)
        return dict(record)

    def get_stage_inputs(self, session_id: str) -> dict[str, Any]:
        record = self._get_record(session_id)
        stage_inputs = record.get("stage_inputs", {})
        return dict(stage_inputs) if isinstance(stage_inputs, dict) else {}

    def set_stage_inputs(
        self, session_id: str, stage: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not session_id or not stage:
            return self.get(session_id)
        record = dict(self._get_record(session_id))
        stage_inputs = record.get("stage_inputs", {})
        if not isinstance(stage_inputs, dict):
            stage_inputs = {}
        stage_inputs = dict(stage_inputs)
        stage_inputs[stage] = payload
        record["stage_inputs"] = stage_inputs
        self._data[session_id] = record
        self._persist(session_id, record)
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
        record = dict(self._get_record(session_id))
        summary = record.pop("pending_memory_summary", None)
        record.pop("pending_memory_label", None)
        self._data[session_id] = record
        self._persist(session_id, record)
        return summary

    def set_reference(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, reference=payload, stage="reference")

    def set_grid(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, grid=payload, stage="grid")

    def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        return self.update(session_id, overrides=overrides)


pipeline_store = PipelineStateStore(PIPELINE_STATE_DB_PATH)
