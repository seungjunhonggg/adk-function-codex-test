from __future__ import annotations

from typing import Dict


class LotStore:
    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}

    def set(self, session_id: str, lot_payload: dict) -> None:
        self._data[session_id] = dict(lot_payload)

    def get(self, session_id: str) -> dict | None:
        record = self._data.get(session_id)
        return dict(record) if record else None

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)


lot_store = LotStore()
