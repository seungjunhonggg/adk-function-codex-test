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
    LOT_PARAM_SIZE_COLUMN,
    LOT_PARAM_TEMPERATURE_COLUMN,
    LOT_PARAM_VOLTAGE_COLUMN,
)
from .db import query_process_data
from .db_connections import execute_table_query
from .events import event_bus
from .lot_store import lot_store
from .simulation import call_simulation_api, simulation_store


def _parse_columns(value: str | None) -> list[str]:
    if not value:
        return []
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


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
    return payload


async def emit_process_data(query: str = "", limit: int = 12) -> dict:
    return await emit_lot_info(query, limit)


def collect_simulation_params(
    temperature: float | None = None,
    voltage: float | None = None,
    size: float | None = None,
    capacity: float | None = None,
) -> dict:
    session_id = current_session_id.get()
    params = simulation_store.update(
        session_id,
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
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
    return await emit_lot_info(lot_id, limit)


@function_tool
async def get_process_data(query: str = "", limit: int = 12) -> dict:
    return await emit_process_data(query, limit)


@function_tool
def update_simulation_params(
    temperature: float | None = None,
    voltage: float | None = None,
    size: float | None = None,
    capacity: float | None = None,
) -> dict:
    return collect_simulation_params(
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
    )


@function_tool
async def run_simulation() -> dict:
    return await execute_simulation()


@function_tool
async def run_lot_simulation(lot_id: str = "") -> dict:
    session_id = current_session_id.get()
    lot_payload = None
    if lot_id:
        lot_payload = await emit_lot_info(lot_id, limit=1)
        if lot_payload.get("error"):
            return {"status": "error", "error": lot_payload.get("error")}

    stored = lot_store.get(session_id)
    row = stored.get("row") if stored else None
    if not row:
        await emit_frontend_trigger("LOT 정보가 없습니다. 먼저 LOT를 조회해주세요.")
        return {"status": "missing", "missing": ["lot_id"]}

    params = _extract_simulation_params_from_lot(row)
    collect_simulation_params(**params)
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
        return payload
    payload["result"] = result.get("result")
    payload["params"] = simulation_store.get(session_id)
    return payload


@function_tool(strict_mode=False)
async def frontend_trigger(message: str, data: dict | None = None) -> dict:
    return await emit_frontend_trigger(message, data)


@function_tool
def file_search(query: str = "", top_k: int = 3) -> dict:
    results = _search_knowledge_base(query, max(1, top_k))
    return {"results": results, "query": query, "count": len(results)}
