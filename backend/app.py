from pathlib import Path

from agents import Runner, SQLiteSession
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import triage_agent
from .config import OPENAI_API_KEY, SESSION_DB_PATH
from .context import current_session_id
from .db import init_db
from .events import event_bus
from .tools import collect_simulation_params, emit_process_data, execute_simulation


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="공정 모니터링 데모")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class TestRequest(BaseModel):
    session_id: str
    query: str = ""
    params: dict | None = None


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/builder")
async def builder() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "builder.html")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    if not OPENAI_API_KEY:
        return {
            "assistant_message": (
                "OPENAI_API_KEY가 설정되지 않았습니다. 환경 변수로 설정한 뒤 재시작해주세요."
            )
        }

    session = SQLiteSession(request.session_id, SESSION_DB_PATH)
    token = current_session_id.set(request.session_id)
    try:
        result = await Runner.run(
            triage_agent,
            input=request.message,
            session=session,
        )
    finally:
        current_session_id.reset(token)

    return {"assistant_message": result.final_output}


@app.post("/api/test/trigger")
async def trigger_test(request: TestRequest) -> dict:
    token = current_session_id.set(request.session_id)
    try:
        await emit_process_data(request.query or "", limit=8)
        params = request.params or {
            "temperature": 120,
            "voltage": 3.7,
            "size": 12,
            "capacity": 6,
        }
        safe_params = {
            key: value
            for key, value in params.items()
            if key in {"temperature", "voltage", "size", "capacity"}
        }
        collect_simulation_params(**safe_params)
        result = await execute_simulation()
    finally:
        current_session_id.reset(token)

    return {"status": "ok", "simulation": result}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await event_bus.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
    except Exception:
        event_bus.disconnect(websocket)


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
