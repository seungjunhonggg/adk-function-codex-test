import asyncio
import copy
import json
import logging
import time
import uuid
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.genai import types

from backend.core.agents import root_agent
from backend.core.db_production import (
    PostgresSession,
    fetch_session_state,
    upsert_session_state,
    upsert_session_ip,
    fetch_all_sessions,
    delete_session_by_id,
    _ensure_agent_tables,
    _get_agent_connection,
    AGENT_SESSIONS_TABLE,
    AGENT_MESSAGES_TABLE,
)

logger = logging.getLogger(__name__)

APP_NAME = "mlcc_simulation"
USER_ID = "default_user"


# ---------------------------------------------------------------------------
# PostgresSessionService: ADK BaseSessionService backed by PostgresSession
# ---------------------------------------------------------------------------
class PostgresSessionService(BaseSessionService):
    """Google ADK SessionService implementation using PostgreSQL.

    Uses the existing PostgresSession (message storage) and
    fetch_session_state / upsert_session_state (state persistence)
    from db_production.py.
    """

    def __init__(self) -> None:
        # In-memory cache keyed by (app_name, user_id, session_id).
        # Keeps the authoritative Session objects for the running process
        # so the Runner can mutate them during a turn.
        self._cache: dict[tuple[str, str, str], Session] = {}

    # -- helpers -------------------------------------------------------------

    def _key(self, app_name: str, user_id: str, session_id: str):
        return (app_name, user_id, session_id)

    def _pg_session(self, session_id: str) -> PostgresSession:
        return PostgresSession(session_id=session_id)

    async def _persist_state(self, session: Session) -> None:
        """Write session state to Postgres in a background thread."""
        await asyncio.to_thread(
            upsert_session_state, session.id, session.state
        )

    async def _load_state(self, session_id: str) -> dict[str, Any]:
        """Read session state from Postgres."""
        state = await asyncio.to_thread(fetch_session_state, session_id)
        return state or {}

    async def _persist_event(self, session_id: str, event: Event) -> None:
        """Serialize an Event and append it to the Postgres message store."""
        pg = self._pg_session(session_id)
        data = event.model_dump(mode="json", exclude_none=True)
        await pg.add_items([data])

    async def _load_events(
        self, session_id: str, limit: int | None = None
    ) -> list[Event]:
        """Load Events from Postgres message store."""
        pg = self._pg_session(session_id)
        rows = await pg.get_items(limit=limit)
        events: list[Event] = []
        for row in rows:
            try:
                events.append(Event.model_validate(row))
            except Exception:
                continue
        return events

    # -- BaseSessionService overrides ----------------------------------------

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        session_id = (
            session_id.strip()
            if session_id and session_id.strip()
            else str(uuid.uuid4())
        )

        key = self._key(app_name, user_id, session_id)
        if key in self._cache:
            return copy.deepcopy(self._cache[key])

        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=state or {},
            last_update_time=time.time(),
        )
        self._cache[key] = session

        # Persist initial state
        if state:
            await self._persist_state(session)

        # Ensure Postgres session row exists
        pg = self._pg_session(session_id)
        await pg.add_items([])  # triggers INSERT … ON CONFLICT DO NOTHING

        return copy.deepcopy(session)

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        key = self._key(app_name, user_id, session_id)

        if key in self._cache:
            session = copy.deepcopy(self._cache[key])
        else:
            # Try to restore from DB
            state = await self._load_state(session_id)
            events = await self._load_events(session_id)
            if not state and not events:
                return None

            session = Session(
                app_name=app_name,
                user_id=user_id,
                id=session_id,
                state=state,
                events=events,
                last_update_time=time.time(),
            )
            self._cache[key] = session
            session = copy.deepcopy(session)

        # Apply config filters on the copy
        if config:
            if config.num_recent_events:
                session.events = session.events[-config.num_recent_events:]
            if config.after_timestamp:
                session.events = [
                    e for e in session.events if e.timestamp >= config.after_timestamp
                ]

        return session

    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        sessions = []
        for (a, u, _), s in self._cache.items():
            if a != app_name:
                continue
            if user_id is not None and u != user_id:
                continue
            copied = copy.deepcopy(s)
            copied.events = []
            sessions.append(copied)
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        key = self._key(app_name, user_id, session_id)
        self._cache.pop(key, None)
        pg = self._pg_session(session_id)
        await pg.clear_session()

    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event

        key = self._key(session.app_name, session.user_id, session.id)

        # Update in-memory session via parent class logic
        await super().append_event(session=session, event=event)
        session.last_update_time = event.timestamp

        # Update storage session
        storage = self._cache.get(key)
        if storage is not None:
            storage.events.append(event)
            storage.last_update_time = event.timestamp

            # Apply state delta
            if event.actions and event.actions.state_delta:
                storage.state.update(event.actions.state_delta)

        # Persist event and state to Postgres
        await self._persist_event(session.id, event)
        if storage is not None:
            await self._persist_state(storage)

        return event


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MLCC Simulation Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


session_service = PostgresSessionService()
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    session_id = req.session_id or str(uuid.uuid4())

    # Ensure session exists
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )

    # Save client IP
    client_ip = request.client.host if request.client else None
    await asyncio.to_thread(upsert_session_ip, session_id, client_ip)

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=req.message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=user_content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_response += part.text

    return ChatResponse(session_id=session_id, response=final_response)


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


# ---------------------------------------------------------------------------
# 테스트 엔드포인트: table_data SSE 파이프라인 검증용
# GET /test/table → table_data SSE + final 이벤트를 전송한다.
# ---------------------------------------------------------------------------
@app.get("/test/table")
async def test_table():
    async def gen():
        payload = json.dumps({"url": "/artifacts/test_sample.json"})
        yield _sse("table_data", payload)
        yield _sse("final", json.dumps({"session_id": "test", "response": ""}))
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    session_id = req.session_id or str(uuid.uuid4())

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )

    client_ip = request.client.host if request.client else None
    await asyncio.to_thread(upsert_session_ip, session_id, client_ip)

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=req.message)],
    )

    async def event_generator():
        final_text = ""
        logs: list[dict] = []

        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=user_content,
        ):
            # artifact_delta 감지 → table_data SSE 전송
            if event.actions and getattr(event.actions, "artifact_delta", None):
                for file_url in event.actions.artifact_delta.values():
                    payload = json.dumps({"url": file_url})
                    yield _sse("table_data", payload)

            if not event.content or not event.content.parts:
                continue

            for part in event.content.parts:
                # Tool 호출 시작 → progress
                if part.function_call:
                    tool_name = part.function_call.name
                    logs.append({"text": f"{tool_name} 실행 중", "status": "in_progress"})
                    yield _sse("progress", json.dumps({"logs": logs}))

                # Tool 응답 → progress 완료 + 트리거 감지
                if part.function_response:
                    tool_name = part.function_response.name
                    resp = part.function_response.response or {}
                    # 진행 로그 업데이트
                    for log in logs:
                        if log["status"] == "in_progress":
                            log["status"] = "done"
                    yield _sse("progress", json.dumps({"logs": logs}))

                    # _frontend_trigger가 있으면 프론트에 전송
                    if isinstance(resp, dict) and "_frontend_trigger" in resp:
                        trigger = resp["_frontend_trigger"]
                        yield _sse("trigger", json.dumps(trigger))

                # 텍스트 응답 누적
                if part.text:
                    final_text += part.text

        yield _sse(
            "final",
            json.dumps({"session_id": session_id, "response": final_text}),
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions")
async def list_sessions():
    """저장된 세션 목록을 최근 업데이트 순으로 반환한다."""
    sessions = await asyncio.to_thread(fetch_all_sessions)
    return {"sessions": sessions}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """세션의 저장된 메시지(이벤트)를 반환한다."""
    pg = PostgresSession(session_id=session_id)
    messages = await pg.get_items()
    return {"messages": messages}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션과 관련 메시지를 모두 삭제한다."""
    # 인메모리 캐시에서도 제거한다.
    key = (APP_NAME, USER_ID, session_id)
    session_service._cache.pop(key, None)
    # DB에서 삭제한다.
    await asyncio.to_thread(delete_session_by_id, session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
