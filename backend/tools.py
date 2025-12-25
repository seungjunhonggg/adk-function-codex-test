from agents import function_tool


KNOWLEDGE_BASE = [
    {
        "id": "kb-process",
        "title": "LOT 모니터링",
        "content": "이 워크플로우는 LOT 정보 조회와 예측 시뮬레이션을 지원합니다.",
    },
    {
        "id": "kb-db",
        "title": "LOT 조회",
        "content": "get_lot_info 도구는 LOT ID로 PostgreSQL에서 정보를 조회합니다.",
    },
    {
        "id": "kb-sim",
        "title": "시뮬레이션",
        "content": "run_lot_simulation 도구는 LOT 정보를 시뮬레이션 입력으로 변환합니다.",
    },
]

from .context import current_session_id
from .config import (
    LOT_DB_COLUMNS,
    LOT_DB_CONNECTION_ID,
    LOT_DB_FILTER_OPERATOR,
    LOT_DB_LOT_COLUMN,
    LOT_DB_SCHEMA,
    LOT_DB_TABLE,
    LOT_PARAM_CAPACITY_COLUMN,
    LOT_PARAM_PRODUCTION_COLUMN,
    LOT_PARAM_SIZE_COLUMN,
    LOT_PARAM_TEMPERATURE_COLUMN,
    LOT_PARAM_VOLTAGE_COLUMN,
)
from .db import query_process_data
from .db_connections import execute_table_query
from .events import event_bus
from .observability import emit_workflow_log
from .lot_store import lot_store
from .simulation import call_simulation_api, extract_simulation_params, simulation_store


def _parse_columns(value: str | None) -> list[str]:
    if not value:
        return []
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def _format_tool_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) > 120:
        return f"{text[:117]}..."
    return text


def _format_tool_args(**kwargs: object) -> str:
    parts = []
    for key, value in kwargs.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={_format_tool_value(value)}")
    return ", ".join(parts) if parts else "no_args"


async def _log_route(route: str, detail: str) -> None:
    await emit_workflow_log("ROUTE", f"{route}: {detail}")


async def _log_tool_call(name: str, **kwargs: object) -> None:
    await emit_workflow_log("TOOL_CALL", f"{name}({_format_tool_args(**kwargs)})")


async def _log_tool_result(name: str, summary: str) -> None:
    await emit_workflow_log("TOOL_RESULT", f"{name}: {summary}")


def _normalize_row(row: dict) -> dict:
    normalized = {}
    for key, value in row.items():
        normalized[key] = value
        normalized[str(key).lower()] = value
    return normalized


def _extract_simulation_params_from_lot(row: dict) -> dict:
    normalized = _normalize_row(row)
    mapping = {
        "temperature": LOT_PARAM_TEMPERATURE_COLUMN,
        "voltage": LOT_PARAM_VOLTAGE_COLUMN,
        "size": LOT_PARAM_SIZE_COLUMN,
        "capacity": LOT_PARAM_CAPACITY_COLUMN,
        "production_mode": LOT_PARAM_PRODUCTION_COLUMN,
    }
    params: dict[str, object] = {}
    for key, column in mapping.items():
        if not column:
            continue
        value = normalized.get(column) or normalized.get(column.lower())
        if value is not None:
            params[key] = value
    return params


def _query_lot_rows(lot_id: str, limit: int) -> dict:
    if LOT_DB_CONNECTION_ID and LOT_DB_TABLE:
        columns = _parse_columns(LOT_DB_COLUMNS)
        result = execute_table_query(
            connection_id=LOT_DB_CONNECTION_ID,
            schema_name=LOT_DB_SCHEMA or "public",
            table_name=LOT_DB_TABLE,
            columns=columns,
            filter_column=LOT_DB_LOT_COLUMN,
            filter_operator=LOT_DB_FILTER_OPERATOR or "=",
            filter_value=lot_id,
            limit=limit,
        )
        return {
            "lot_id": lot_id,
            "columns": result.get("columns", []),
            "rows": result.get("rows", []),
            "source": "postgresql",
        }
    if LOT_DB_CONNECTION_ID or LOT_DB_TABLE:
        raise RuntimeError("LOT DB 설정이 불완전합니다. 연결/테이블 정보를 확인하세요.")

    rows = query_process_data(lot_id, limit)
    if rows:
        for row in rows:
            row.setdefault("lot_id", lot_id)
    return {
        "lot_id": lot_id,
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
        "source": "sqlite",
    }


async def emit_lot_info(lot_id: str = "", limit: int = 12) -> dict:
    if not lot_id:
        await emit_frontend_trigger("LOT ID가 필요합니다.")
        return {"error": "LOT ID가 필요합니다.", "rows": []}
    try:
        payload = _query_lot_rows(lot_id, limit)
    except Exception as error:
        message = f"LOT 조회 실패: {error}"
        await event_bus.broadcast(
            {
                "type": "frontend_trigger",
                "payload": {"message": message, "data": {"lot_id": lot_id}},
            }
        )
        return {"error": str(error), "rows": [], "lot_id": lot_id}

    rows = payload.get("rows", [])
    if rows:
        lot_store.set(
            current_session_id.get(),
            {"lot_id": lot_id, "row": rows[0], "rows": rows},
        )
    await event_bus.broadcast({"type": "lot_result", "payload": payload})
    if rows:
        message = f"LOT {lot_id} 데이터를 우측 패널에 표시했습니다."
    else:
        message = f"LOT {lot_id} 결과가 없습니다."
    await emit_frontend_trigger(message, payload)
    return payload


async def emit_process_data(query: str = "", limit: int = 12) -> dict:
    return await emit_lot_info(query, limit)


def collect_simulation_params(
    temperature: float | None = None,
    voltage: float | None = None,
    size: float | None = None,
    capacity: float | None = None,
    production_mode: str | None = None,
) -> dict:
    session_id = current_session_id.get()
    params = simulation_store.update(
        session_id,
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
        production_mode=production_mode,
    )
    missing = simulation_store.missing(session_id)
    return {"params": params, "missing": missing}


async def execute_simulation() -> dict:
    session_id = current_session_id.get()
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    if missing:
        return {"status": "missing", "missing": missing}

    result = await call_simulation_api(params)
    await event_bus.broadcast(
        {
            "type": "simulation_result",
            "payload": {"params": params, "result": result},
        }
    )
    await emit_frontend_trigger("시뮬레이션 결과가 우측 패널에 표시되었습니다.", {"params": params, "result": result})
    return {"status": "ok", "result": result}


async def emit_frontend_trigger(message: str, payload: dict | None = None) -> dict:
    event = {
        "type": "frontend_trigger",
        "payload": {
            "message": message,
            "data": payload or {},
        },
    }
    await event_bus.broadcast(event)
    return event["payload"]


async def emit_simulation_form(params: dict, missing: list[str]) -> dict:
    session_id = current_session_id.get()
    simulation_store.activate(session_id)
    event = {"type": "simulation_form", "payload": {"params": params, "missing": missing}}
    await event_bus.broadcast(event)
    return event["payload"]


def _search_knowledge_base(query: str, top_k: int) -> list[dict]:
    if not query:
        return KNOWLEDGE_BASE[:top_k]
    lowered = query.lower()
    scored = []
    for item in KNOWLEDGE_BASE:
        content = f"{item['title']} {item['content']}".lower()
        score = content.count(lowered)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


@function_tool
async def get_lot_info(lot_id: str = "", limit: int = 12) -> dict:
    """Fetch LOT/process data by lot_id and emit it to the UI panel."""
    await _log_route("db_lookup", "get_lot_info")
    await _log_tool_call("get_lot_info", lot_id=lot_id, limit=limit)
    result = await emit_lot_info(lot_id, limit)
    if isinstance(result, dict) and result.get("error"):
        await _log_tool_result("get_lot_info", f"error={result.get('error')}")
    else:
        rows = result.get("rows", []) if isinstance(result, dict) else []
        await _log_tool_result("get_lot_info", f"rows={len(rows)}")
    return result


@function_tool
async def get_process_data(query: str = "", limit: int = 12) -> dict:
    """Search process records by query (line/status) and emit results to the UI."""
    await _log_route("db_lookup", "get_process_data")
    await _log_tool_call("get_process_data", query=query, limit=limit)
    result = await emit_process_data(query, limit)
    rows = result.get("rows", []) if isinstance(result, dict) else []
    await _log_tool_result("get_process_data", f"rows={len(rows)}")
    return result


@function_tool
async def update_simulation_params(
    message: str | None = None,
    temperature: float | None = None,
    voltage: float | None = None,
    size: float | None = None,
    capacity: float | None = None,
    production_mode: str | None = None,
) -> dict:
    """Update simulation params from user input and emit missing fields to the UI."""
    await _log_tool_call(
        "update_simulation_params",
        message=message,
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
        production_mode=production_mode,
    )
    parsed = extract_simulation_params(message) if message else {}
    if temperature is not None:
        parsed["temperature"] = temperature
    if voltage is not None:
        parsed["voltage"] = voltage
    if size is not None:
        parsed["size"] = size
    if capacity is not None:
        parsed["capacity"] = capacity
    if production_mode is not None:
        parsed["production_mode"] = production_mode

    result = collect_simulation_params(**parsed)
    await emit_simulation_form(result["params"], result["missing"])
    missing = result.get("missing", [])
    await _log_tool_result(
        "update_simulation_params",
        f"missing={len(missing)} params={len(result.get('params', {}))}",
    )
    return result


@function_tool
async def run_simulation() -> dict:
    """Open the simulation form, then run if params are complete."""
    await _log_route("simulation", "run_simulation")
    await _log_tool_call("run_simulation")
    session_id = current_session_id.get()
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    await emit_simulation_form(params, missing)
    if missing:
        await _log_tool_result("run_simulation", f"status=missing missing={len(missing)}")
        return {"status": "missing", "missing": missing, "params": params}
    result = await execute_simulation()
    summary = f"status={result.get('status')}"
    inner = result.get("result") if isinstance(result, dict) else None
    if isinstance(inner, dict) and inner.get("predicted_yield") is not None:
        summary = f"{summary}, yield={inner.get('predicted_yield')}"
    await _log_tool_result("run_simulation", summary)
    return result


@function_tool
async def open_simulation_form() -> dict:
    """Open the simulation input form with current params and missing fields."""
    await _log_tool_call("open_simulation_form")
    session_id = current_session_id.get()
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    await emit_simulation_form(params, missing)
    await _log_tool_result(
        "open_simulation_form",
        f"missing={len(missing)} params={len(params)}",
    )
    return {"params": params, "missing": missing}


@function_tool
async def run_lot_simulation(lot_id: str = "") -> dict:
    """Fetch LOT data, derive params, run simulation, and emit the result to the UI."""
    session_id = current_session_id.get()
    lot_payload = None
    await _log_route("simulation", "run_lot_simulation")
    await _log_tool_call("run_lot_simulation", lot_id=lot_id)
    if lot_id:
        lot_payload = await emit_lot_info(lot_id, limit=1)
        if lot_payload.get("error"):
            await _log_tool_result("run_lot_simulation", f"status=error error={lot_payload.get('error')}")
            return {"status": "error", "error": lot_payload.get("error")}

    stored = lot_store.get(session_id)
    row = stored.get("row") if stored else None
    if not row:
        await emit_frontend_trigger("LOT 정보가 없습니다. 먼저 LOT를 조회해주세요.")
        await _log_tool_result("run_lot_simulation", "status=missing missing=lot_id")
        return {"status": "missing", "missing": ["lot_id"]}

    params = _extract_simulation_params_from_lot(row)
    update_result = collect_simulation_params(**params)
    await emit_simulation_form(update_result["params"], update_result["missing"])
    result = await execute_simulation()
    payload = {"status": result.get("status"), "lot_id": lot_id or stored.get("lot_id")}
    if result.get("status") == "missing":
        missing = result.get("missing", [])
        payload["missing"] = missing
        if missing:
            missing_text = ", ".join(missing)
            await emit_frontend_trigger(
                f"시뮬레이션 입력이 부족합니다: {missing_text}"
            )
        await _log_tool_result("run_lot_simulation", f"status=missing missing={len(missing)}")
        return payload
    payload["result"] = result.get("result")
    payload["params"] = simulation_store.get(session_id)
    await _log_tool_result("run_lot_simulation", "status=ok")
    return payload


@function_tool(strict_mode=False)
async def frontend_trigger(message: str, data: dict | None = None) -> dict:
    """Send a freeform message to the frontend UI."""
    await _log_tool_call("frontend_trigger", message=message)
    result = await emit_frontend_trigger(message, data)
    await _log_tool_result("frontend_trigger", "sent")
    return result


@function_tool
def file_search(query: str = "", top_k: int = 3) -> dict:
    """Search the internal knowledge base for the query."""
    results = _search_knowledge_base(query, max(1, top_k))
    payload = {"results": results, "query": query, "count": len(results)}
    try:
        import asyncio

        loop = asyncio.get_running_loop()
        loop.create_task(_log_tool_call("file_search", query=query, top_k=top_k))
        loop.create_task(_log_tool_result("file_search", f"count={payload['count']}"))
    except RuntimeError:
        pass
    return payload
