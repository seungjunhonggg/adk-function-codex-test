import json
from pathlib import Path
import re

from agents import Runner, SQLiteSession
from agents.tracing import set_tracing_disabled
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import auto_message_agent, triage_agent
from .config import LOT_DB_CONNECTION_ID, OPENAI_API_KEY, SESSION_DB_PATH, TRACING_ENABLED
from .context import current_session_id
from .db_connections import connect_and_save, get_schema, list_connections, preload_schema
from .db import init_db
from .events import event_bus
from .observability import WorkflowRunHooks
from .pipeline_store import pipeline_store
from .reference_lot import (
    build_grid_values,
    load_reference_rules,
    normalize_reference_rules,
    select_reference_lot,
    select_reference_lot_by_id,
)
from .simulation import (
    call_grid_search_api,
    extract_simulation_params,
    recommendation_store,
    simulation_store,
)
from .tools import (
    collect_simulation_params,
    emit_lot_info,
    emit_simulation_form,
    execute_simulation,
    run_lot_simulation,
)
from .workflow import (
    apply_saved_workflow,
    delete_saved_workflow,
    ensure_workflow,
    ensure_workflow_store,
    execute_workflow,
    list_saved_workflows,
    load_workflow,
    normalize_workflow,
    preview_workflow,
    save_workflow_entry,
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
    temperature: str | None = None
    voltage: float | None = None
    size: float | None = None
    capacity: float | None = None
    production_mode: str | None = None
    model_name: str | None = None


class SimulationRunRequest(BaseModel):
    session_id: str


class RecommendationParamsRequest(BaseModel):
    session_id: str
    params: dict[str, float | str] | None = None


@app.on_event("startup")
async def startup() -> None:
    init_db()
    ensure_workflow()
    ensure_workflow_store()
    set_tracing_disabled(not TRACING_ENABLED)
    if LOT_DB_CONNECTION_ID:
        try:
            preload_schema(LOT_DB_CONNECTION_ID, force=True)
        except Exception:
            pass


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
        sim_response = await _maybe_handle_simulation_message(
            request.message, request.session_id
        )
        if sim_response is not None:
            await _append_user_message(request.session_id, request.message)
            await _append_assistant_message(request.session_id, sim_response)
            return {"assistant_message": sim_response}

        result = await Runner.run(
            triage_agent,
            input=request.message,
            session=session,
            hooks=WorkflowRunHooks(),
        )
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


def _format_simulation_params(params: dict) -> str:
    if not params:
        return "입력 없음"
    mapping = [
        ("model_name", "기종명"),
        ("temperature", "온도"),
        ("voltage", "전압"),
        ("size", "크기"),
        ("capacity", "용량"),
        ("production_mode", "양산/개발"),
    ]
    parts = []
    for key, label in mapping:
        value = params.get(key)
        if value is None or value == "":
            continue
        if key == "production_mode":
            if value == "mass":
                value = "양산"
            elif value == "dev":
                value = "개발"
        parts.append(f"{label}={value}")
    return ", ".join(parts) if parts else "입력 없음"


def _format_simulation_result(result: dict | None) -> str:
    if not result:
        return "결과 없음"
    parts = []
    model_name = result.get("recommended_model")
    if model_name:
        parts.append(f"추천 기종={model_name}")
    rep_lot = result.get("representative_lot")
    if rep_lot:
        parts.append(f"대표 LOT={rep_lot}")
    params = result.get("params")
    if isinstance(params, dict):
        parts.append(f"파라미터={len(params)}개")
    if not parts:
        return "결과 수신"
    return ", ".join(parts)


def _format_missing_fields(missing: list[str]) -> str:
    if not missing:
        return "없음"
    mapping = {
        "temperature": "온도",
        "voltage": "전압",
        "size": "크기",
        "capacity": "용량",
        "production_mode": "양산/개발",
    }
    labels = [mapping.get(item, item) for item in missing]
    return ", ".join(labels)


async def _append_system_note(session_id: str, content: str) -> None:
    session = SQLiteSession(session_id, SESSION_DB_PATH)
    await session.add_items([{"role": "system", "content": content}])


async def _append_user_message(session_id: str, content: str) -> None:
    session = SQLiteSession(session_id, SESSION_DB_PATH)
    await session.add_items([{"role": "user", "content": content}])


async def _append_assistant_message(session_id: str, content: str) -> None:
    session = SQLiteSession(session_id, SESSION_DB_PATH)
    await session.add_items([{"role": "assistant", "content": content}])


async def _emit_chat_message(session_id: str, content: str) -> None:
    if not content:
        return
    await _append_assistant_message(session_id, content)
    await event_bus.broadcast(
        {"type": "chat_message", "payload": {"role": "assistant", "content": content}}
    )


async def _generate_simulation_auto_message(
    params: dict, missing: list[str], result: dict | None
) -> str:
    if not OPENAI_API_KEY:
        if missing:
            missing_text = _format_missing_fields(missing)
            return f"추천을 실행하려면 {missing_text} 값을 알려주세요."
        if result:
            return (
                f"인접기종 추천 결과가 나왔습니다. {_format_simulation_result(result)}. "
                "예측 시뮬레이션도 진행할까요?"
            )
        return "추천 입력을 확인해 주세요."

    payload = {
        "params": params,
        "missing_fields": missing,
        "result": result or {},
    }
    prompt = (
        "Compose a short Korean assistant message based on the JSON below. "
        "Rules: 1-2 sentences, no mention of tools/routing. "
        "If missing_fields is not empty, ask for those fields in one question. "
        "If result is present, summarize recommended_model, representative_lot, "
        "and the count of params, then ask whether to run a prediction simulation. "
        f"JSON: {json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        run = await Runner.run(auto_message_agent, input=prompt)
        message = (run.final_output or "").strip()
    except Exception:
        message = ""
    if message:
        return message
    if missing:
        missing_text = _format_missing_fields(missing)
        return f"추천을 실행하려면 {missing_text} 값을 알려주세요."
    if result:
        return (
            f"인접기종 추천 결과가 나왔습니다. {_format_simulation_result(result)}. "
            "예측 시뮬레이션도 진행할까요?"
        )
    return "추천 입력을 확인해 주세요."


def _format_prediction_result(result: dict | None) -> str:
    if not result:
        return "결과 없음"
    parts = []
    capacity = result.get("predicted_capacity")
    dc_capacity = result.get("predicted_dc_capacity")
    reliability = result.get("reliability_pass_prob")
    if capacity is not None:
        parts.append(f"예측 용량={capacity}")
    if dc_capacity is not None:
        parts.append(f"예측 DC 용량={dc_capacity}")
    if reliability is not None:
        parts.append(f"신뢰성 통과확률={reliability}")
    return ", ".join(parts) if parts else "결과 수신"


async def _generate_prediction_auto_message(result: dict | None) -> str:
    summary = _format_prediction_result(result)
    return f"예측 결과가 나왔습니다. {summary}."


def _contains_any(message: str, tokens: tuple[str, ...]) -> bool:
    lowered = (message or "").lower()
    return any(token in lowered for token in tokens)


def _extract_lot_id(message: str) -> str | None:
    match = re.search(r"(lot[-_ ]?[a-z0-9-]+)", message or "", re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1)
    return raw.upper().replace("_", "-").replace(" ", "")


def _is_affirmative(message: str) -> bool:
    lowered = (message or "").strip().lower()
    if not lowered:
        return False
    affirmative = {
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "yep",
        "네",
        "예",
        "응",
        "그래",
        "좋아",
        "가능",
        "진행",
        "실행",
        "해줘",
    }
    if lowered in affirmative:
        return True
    return any(token in lowered for token in affirmative)


def _is_recommendation_intent(message: str) -> bool:
    tokens = (
        "추천",
        "인접",
        "시뮬",
        "simulation",
        "simulate",
        "what-if",
        "예측",
    )
    return _contains_any(message, tokens)


def _extract_grid_overrides(message: str) -> dict[str, float]:
    overrides: dict[str, float] = {}
    mapping = {
        "sheet_t": [r"sheet\s*t", r"sheet_t", r"시트"],
        "laydown": [r"laydown", r"레이다운"],
        "active_layer": [r"active\s*layer", r"active_layer", r"액티브"],
    }
    for name, patterns in mapping.items():
        for pattern in patterns:
            match = re.search(
                rf"{pattern}\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
                message,
                re.IGNORECASE,
            )
            if match:
                try:
                    overrides[name] = float(match.group(1))
                except (TypeError, ValueError):
                    continue
                break
    return overrides


def _should_rerun_grid(message: str) -> bool:
    tokens = ("다시", "재실행", "리런", "run", "grid", "그리드", "예측")
    return _contains_any(message, tokens)


async def _run_reference_pipeline(
    session_id: str,
    params: dict,
    model_override: str | None = None,
    reference_override: str | None = None,
    grid_overrides: dict | None = None,
) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    model_name = model_override
    simulation_result = None

    if not model_name:
        simulation_result = await execute_simulation()
        inner = simulation_result.get("result") if isinstance(simulation_result, dict) else {}
        model_name = inner.get("recommended_model")

    if not model_name:
        return {
            "message": "추천 기종을 확인하지 못했습니다. 입력 파라미터를 다시 확인해 주세요."
        }

    reference_result = select_reference_lot(model_name)
    if reference_override:
        reference_result = select_reference_lot_by_id(reference_override)

    if reference_result.get("status") != "ok":
        return {
            "message": "레퍼런스 LOT를 찾지 못했습니다. 조건 또는 DB 설정을 확인해 주세요."
        }

    pipeline_store.set_reference(session_id, reference_result)
    lot_payload = {
        "lot_id": reference_result.get("lot_id"),
        "columns": reference_result.get("columns", []),
        "rows": [reference_result.get("row")],
        "source": reference_result.get("source") or "postgresql",
    }
    await event_bus.broadcast({"type": "lot_result", "payload": lot_payload})

    if simulation_result is None:
        synthetic = {
            "recommended_model": model_name,
            "representative_lot": reference_result.get("lot_id"),
            "params": {},
        }
        await event_bus.broadcast(
            {"type": "simulation_result", "payload": {"params": params, "result": synthetic}}
        )
        simulation_result = {"status": "ok", "result": synthetic}

    defect_rates = reference_result.get("defect_rates", [])
    if defect_rates:
        stats = {
            "count": len(defect_rates),
            "avg": sum(item["defect_rate"] for item in defect_rates) / len(defect_rates),
        }
        await event_bus.broadcast(
            {
                "type": "defect_rate_chart",
                "payload": {
                    "lots": defect_rates[:50],
                    "filters": {"chip_prod_id": model_name},
                    "stats": stats,
                },
            }
        )

    grid_overrides = grid_overrides or {}
    grid_values = build_grid_values(reference_result.get("row", {}), rules, grid_overrides)
    grid_payload = {
        "chip_prod_id": model_name,
        "lot_id": reference_result.get("lot_id"),
        "grid": grid_values,
        "max_results": rules.get("grid_search", {}).get("max_results", 100),
    }
    grid_results = await call_grid_search_api(grid_payload)
    top_k = rules.get("grid_search", {}).get("top_k", 10)
    top_k = max(1, int(top_k))
    top_candidates = grid_results[:top_k]
    for idx, item in enumerate(top_candidates, start=1):
        item["rank"] = idx

    pipeline_store.set_grid(
        session_id,
        {
            "chip_prod_id": model_name,
            "lot_id": reference_result.get("lot_id"),
            "results": grid_results,
            "top": top_candidates,
            "overrides": grid_overrides,
        },
    )

    await event_bus.broadcast(
        {
            "type": "design_candidates",
            "payload": {
                "candidates": top_candidates,
                "total": len(grid_results),
                "offset": 0,
                "limit": len(top_candidates),
                "target": model_name,
            },
        }
    )

    if simulation_result and simulation_result.get("result"):
        summary = _format_simulation_result(simulation_result.get("result"))
        return {
            "message": (
            f"추천 기종은 {model_name}입니다. 기준 LOT를 선정했고 그리드 서치를 완료했습니다. "
            f"추천 결과 요약: {summary}. 상위 후보를 확인해 주세요."
            ),
            "simulation_result": simulation_result,
        }
    return {
        "message": (
            f"기종 {model_name} 기준으로 레퍼런스 LOT를 선정하고 그리드 서치를 완료했습니다. "
            "상위 후보를 확인해 주세요."
        ),
        "simulation_result": simulation_result,
    }


async def _handle_pipeline_edit_message(
    message: str, session_id: str
) -> str | None:
    state = pipeline_store.get(session_id)
    if not state:
        return None

    grid_overrides = _extract_grid_overrides(message)
    reference_override = None
    lot_id = _extract_lot_id(message)
    if lot_id and _contains_any(message, ("기준", "레퍼런스", "reference")):
        reference_override = lot_id

    if not grid_overrides and not reference_override and not _should_rerun_grid(message):
        return None

    params = simulation_store.get(session_id)
    model_override = params.get("model_name")
    result = await _run_reference_pipeline(
        session_id,
        params,
        model_override=model_override,
        reference_override=reference_override,
        grid_overrides=grid_overrides or (state.get("grid") or {}).get("overrides"),
    )
    return result.get("message")


async def _handle_pipeline_run_message(
    message: str, session_id: str
) -> str | None:
    params = extract_simulation_params(message)
    has_params = bool(params)
    intent = _is_recommendation_intent(message)
    if simulation_store.is_active(session_id) and not (has_params or intent):
        if _is_affirmative(message):
            stored_params = simulation_store.get(session_id)
            model_override = stored_params.get("model_name")
            missing = simulation_store.missing(session_id)
            if missing and not model_override:
                return await _generate_simulation_auto_message(
                    stored_params, missing, None
                )
            result = await _run_reference_pipeline(
                session_id, stored_params, model_override=model_override
            )
            return result.get("message")
        return None

    if not (has_params or intent):
        return None

    if has_params:
        update_result = collect_simulation_params(**params)
    else:
        update_result = {
            "params": simulation_store.get(session_id),
            "missing": simulation_store.missing(session_id),
        }
    await emit_simulation_form(
        update_result.get("params", {}), update_result.get("missing", [])
    )

    model_override = update_result.get("params", {}).get("model_name")
    missing = update_result.get("missing", [])
    if missing and not model_override:
        return await _generate_simulation_auto_message(
            update_result.get("params", {}), missing, None
        )

    result = await _run_reference_pipeline(
        session_id,
        update_result.get("params", {}),
        model_override=model_override,
    )
    return result.get("message")


async def _maybe_handle_simulation_message(
    message: str, session_id: str
) -> str | None:
    edit_message = await _handle_pipeline_edit_message(message, session_id)
    if edit_message is not None:
        return edit_message
    return await _handle_pipeline_run_message(message, session_id)


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
                "production_mode": "mass",
            }
            safe_params = {
                key: value
                for key, value in params.items()
                if key in {"temperature", "voltage", "size", "capacity", "production_mode"}
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


@app.get("/api/workflows")
async def list_workflows() -> dict:
    return list_saved_workflows()


@app.post("/api/workflows")
async def save_workflow_catalog(request: WorkflowRequest) -> dict:
    normalized = normalize_workflow(request.workflow)
    valid, errors = validate_workflow(normalized)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})
    return save_workflow_entry(normalized)


@app.post("/api/workflows/{workflow_id}/apply")
async def apply_workflow(workflow_id: str) -> dict:
    result = apply_saved_workflow(workflow_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "워크플로우를 찾을 수 없습니다."})
    return {"status": "ok", "active_id": result.get("active_id"), "workflow": result.get("workflow")}


@app.delete("/api/workflows/{workflow_id}")
async def remove_workflow(workflow_id: str) -> dict:
    deleted = delete_saved_workflow(workflow_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"error": "워크플로우를 찾을 수 없습니다."})
    return {"status": "ok"}


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
            model_name=request.model_name,
        )
        await emit_simulation_form(
            update_result["params"], update_result["missing"]
        )
        params_summary = _format_simulation_params(update_result.get("params", {}))
        missing = update_result.get("missing", [])
        missing_text = ", ".join(missing) if missing else "없음"
        await _append_system_note(
            request.session_id,
            f"추천 입력 패널 업데이트: {params_summary}. 누락: {missing_text}.",
        )
        if update_result["missing"]:
            return {
                "status": "missing",
                "missing": update_result["missing"],
                "params": update_result["params"],
            }
        return {
            "status": "ready",
            "params": update_result["params"],
            "missing": [],
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
        params_summary = _format_simulation_params(params)
        missing_text = ", ".join(missing) if missing else "없음"
        await _append_system_note(
            request.session_id,
            f"추천 실행 요청(패널): {params_summary}. 누락: {missing_text}.",
        )
        model_override = params.get("model_name")
        if missing and not model_override:
            auto_message = await _generate_simulation_auto_message(params, missing, None)
            await _emit_chat_message(request.session_id, auto_message)
            return {"status": "missing", "missing": missing, "params": params}
        pipeline_result = await _run_reference_pipeline(
            request.session_id, params, model_override=model_override
        )
        await _append_system_note(
            request.session_id, "추천 파이프라인 실행 완료."
        )
        message = pipeline_result.get("message")
        if message:
            await _emit_chat_message(request.session_id, message)
        simulation_result = pipeline_result.get("simulation_result") or {}
        inner = simulation_result.get("result") if isinstance(simulation_result, dict) else {}
        return {
            "status": simulation_result.get("status") or "ok",
            "params": params,
            "result": inner,
        }
    finally:
        current_session_id.reset(token)


@app.post("/api/recommendation/params")
async def update_recommendation_params_route(
    request: RecommendationParamsRequest,
) -> dict:
    token = current_session_id.set(request.session_id)
    try:
        current = recommendation_store.get(request.session_id)
        if not current:
            await _append_system_note(
                request.session_id, "추천 파라미터 업데이트 실패: 추천 결과 없음."
            )
            return {"status": "missing", "missing": ["recommendation"]}
        params = request.params or {}
        updated = recommendation_store.update_params(request.session_id, params)
        if not updated or not isinstance(updated.get("params"), dict):
            await _append_system_note(
                request.session_id, "추천 파라미터 업데이트 실패: 파라미터 없음."
            )
            return {"status": "missing", "missing": ["params"]}
        result_payload = {
            "recommended_model": updated.get("recommended_model"),
            "representative_lot": updated.get("representative_lot"),
            "params": updated.get("params"),
        }
        payload = {
            "params": updated.get("input_params") or {},
            "result": result_payload,
        }
        await event_bus.broadcast({"type": "simulation_result", "payload": payload})
        await _append_system_note(
            request.session_id,
            f"추천 파라미터 업데이트: {len(result_payload.get('params', {}))}개.",
        )
        auto_message = await _generate_simulation_auto_message(
            payload.get("params", {}), [], payload.get("result")
        )
        await _emit_chat_message(request.session_id, auto_message)
        return {"status": "ok", **payload}
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
    session_id = websocket.query_params.get("session_id")
    await event_bus.connect(websocket, session_id=session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
    except Exception:
        event_bus.disconnect(websocket)


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
