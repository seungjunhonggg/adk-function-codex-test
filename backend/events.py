from typing import Dict, Set

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder

from .context import current_session_id
from .pipeline_store import pipeline_store


class EventBus:
    def __init__(self) -> None:
        self._clients_by_session: Dict[str, Set[WebSocket]] = {}
        self._client_sessions: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, session_id: str | None = None) -> None:
        await websocket.accept()
        session = (session_id or "").strip() or current_session_id.get()
        self._client_sessions[websocket] = session
        self._clients_by_session.setdefault(session, set()).add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        session = self._client_sessions.pop(websocket, None)
        if not session:
            return
        clients = self._clients_by_session.get(session)
        if not clients:
            return
        clients.discard(websocket)
        if not clients:
            self._clients_by_session.pop(session, None)

    def has_clients(self, session_id: str | None = None) -> bool:
        session = (session_id or "").strip() or current_session_id.get()
        return bool(self._clients_by_session.get(session))

    async def broadcast(self, event: dict, session_id: str | None = None) -> None:
        session = (session_id or "").strip() or current_session_id.get()
        event_out = dict(event) if isinstance(event, dict) else {"payload": event}
        if "workflow_id" not in event_out:
            workflow_id = pipeline_store.get(session).get("workflow_id")
            if workflow_id:
                event_out["workflow_id"] = workflow_id
        safe_event = jsonable_encoder(event_out)
        stale = []
        for ws in list(self._clients_by_session.get(session, set())):
            try:
                await ws.send_json(safe_event)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


event_bus = EventBus()
