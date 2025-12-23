from pathlib import Path

from agents import Runner, SQLiteSession
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import triage_agent
from .config import OPENAI_API_KEY, SESSION_DB_PATH
from .context import current_session_id
from .db_connections import connect_and_save, get_schema, list_connections
from .db import init_db
from .events import event_bus
from .simulation import extract_simulation_params, simulation_store
from .tools import (
    collect_simulation_params,
    emit_lot_info,
    emit_simulation_form,
    execute_simulation,
    run_lot_simulation,
)
from .workflow import (
    ensure_workflow,
    execute_workflow,
    load_workflow,
    normalize_workflow,
    preview_workflow,
    save_workflow,
    validate_workflow,
)


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="공정 모니터링 데모")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class TestRequest(BaseModel):
    session_id: str
    query: str = ""
    params: dict | None = None


class WorkflowRequest(BaseModel):
    workflow: dict


class WorkflowPreviewRequest(BaseModel):
    workflow: dict
    message: str


class DBConnectRequest(BaseModel):
    name: str
    db_type: str
    host: str
    port: int = 5432
    database: str
    user: str
    password: str


class SimulationParamsRequest(BaseModel):
    session_id: str
    temperature: float | None = None
    voltage: float | None = None
    size: float | None = None
    capacity: float | None = None
    production_mode: str | None = None


class SimulationRunRequest(BaseModel):
    session_id: str


@app.on_event("startup")
async def startup() -> None:
    init_db()
    ensure_workflow()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/builder")
async def builder() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "builder.html")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    session = SQLiteSession(request.session_id, SESSION_DB_PATH)
    token = current_session_id.set(request.session_id)
    try:
        workflow = load_workflow()
        workflow_run = await execute_workflow(
            workflow,
            request.message,
            request.session_id,
            session,
            allow_llm=bool(OPENAI_API_KEY),
        )
        if workflow_run is not None:
            await _maybe_update_simulation_from_message(
                request.message, request.session_id
            )
            return {"assistant_message": workflow_run.assistant_message}

        if not OPENAI_API_KEY:
            await _maybe_update_simulation_from_message(
                request.message, request.session_id
            )
            return {
                "assistant_message": (
                    "OPENAI_API_KEY가 설정되지 않았습니다. "
                    "환경 변수로 설정한 뒤 재시작해주세요."
                )
            }

        result = await Runner.run(triage_agent, input=request.message, session=session)
        await _maybe_update_simulation_from_message(request.message, request.session_id)
        return {"assistant_message": result.final_output}
    finally:
        current_session_id.reset(token)


async def _maybe_update_simulation_from_message(message: str, session_id: str) -> None:
    if not simulation_store.is_active(session_id):
        return
    parsed = extract_simulation_params(message)
    if not parsed:
        return
    current = simulation_store.get(session_id)
    missing_only = {key: value for key, value in parsed.items() if key not in current}
    if not missing_only:
        return
    update_result = collect_simulation_params(**missing_only)
    await emit_simulation_form(
        update_result.get("params", {}),
        update_result.get("missing", []),
    )


@app.post("/api/test/trigger")
async def trigger_test(request: TestRequest) -> dict:
    token = current_session_id.set(request.session_id)
    try:
        if request.query:
            await emit_lot_info(request.query, limit=1)
            result = await run_lot_simulation(request.query)
        else:
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


@app.get("/api/workflow")
async def get_workflow() -> dict:
    return load_workflow()


@app.post("/api/workflow")
async def set_workflow(request: WorkflowRequest) -> dict:
    normalized = normalize_workflow(request.workflow)
    valid, errors = validate_workflow(normalized)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})
    return save_workflow(normalized)


@app.post("/api/workflow/validate")
async def validate_workflow_route(request: WorkflowRequest) -> dict:
    normalized = normalize_workflow(request.workflow)
    valid, errors = validate_workflow(normalized)
    return {"valid": valid, "errors": errors}


@app.post("/api/workflow/preview")
async def preview_workflow_route(request: WorkflowPreviewRequest) -> dict:
    normalized = normalize_workflow(request.workflow)
    valid, errors = validate_workflow(normalized)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})
    preview = preview_workflow(normalized, request.message)
    if preview is None:
        raise HTTPException(
            status_code=400, detail={"errors": ["경로를 찾지 못했습니다."]}
        )
    return preview


@app.post("/api/simulation/params")
async def update_simulation_params_route(request: SimulationParamsRequest) -> dict:
    token = current_session_id.set(request.session_id)
    try:
        update_result = collect_simulation_params(
            temperature=request.temperature,
            voltage=request.voltage,
            size=request.size,
            capacity=request.capacity,
            production_mode=request.production_mode,
        )
        await emit_simulation_form(
            update_result["params"], update_result["missing"]
        )
        if update_result["missing"]:
            return {
                "status": "missing",
                "missing": update_result["missing"],
                "params": update_result["params"],
            }
        result = await execute_simulation()
        return {
            "status": result.get("status"),
            "params": update_result["params"],
            "result": result.get("result"),
        }
    finally:
        current_session_id.reset(token)


@app.post("/api/simulation/run")
async def run_simulation_route(request: SimulationRunRequest) -> dict:
    token = current_session_id.set(request.session_id)
    try:
        params = simulation_store.get(request.session_id)
        missing = simulation_store.missing(request.session_id)
        await emit_simulation_form(params, missing)
        if missing:
            return {"status": "missing", "missing": missing, "params": params}
        result = await execute_simulation()
        return {
            "status": result.get("status"),
            "params": params,
            "result": result.get("result"),
        }
    finally:
        current_session_id.reset(token)


@app.get("/api/db/connections")
async def list_db_connections() -> dict:
    return {"connections": list_connections()}


@app.post("/api/db/connect")
async def connect_db(request: DBConnectRequest) -> dict:
    try:
        record = connect_and_save(request.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=400, detail={"error": str(error)}) from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail={"error": str(error)}) from error
    schema = record.get("schema") or {}
    return {
        "connection": {
            "id": record.get("id"),
            "name": record.get("name"),
            "db_type": record.get("db_type"),
            "host": record.get("host"),
            "port": record.get("port"),
            "database": record.get("database"),
            "user": record.get("user"),
            "updated_at": record.get("updated_at"),
        },
        "schema": schema,
    }


@app.get("/api/db/schema/{connection_id}")
async def get_db_schema(connection_id: str) -> dict:
    schema = get_schema(connection_id)
    if schema is None:
        raise HTTPException(status_code=404, detail={"error": "연결을 찾을 수 없습니다."})
    return {"schema": schema}


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
