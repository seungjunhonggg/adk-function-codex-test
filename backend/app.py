import json
import uuid
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import asyncio
import logging

import psycopg2
from psycopg2 import sql

from agents import Runner, SQLiteSession
from agents.tracing import set_tracing_disabled
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import (
    conversation_agent,
    orchestrator_agent,
    simulation_flow_agent,
)
from .config import (
    OPENAI_API_KEY,
    SESSION_DB_PATH,
    TRACING_ENABLED,
    DEMO_LATENCY_SECONDS,
    BRIEFING_STREAM_DELAY_SECONDS,
    DEBUG_PRINT,
)
from .column_labels import build_column_label_map, get_column_label_map
from .context import current_session_id
from .db_connections import (
    connect_and_save,
    execute_table_query,
    execute_table_query_multi,
    execute_table_query_aggregate,
    get_connection,
    get_schema,
    list_connections,
    preload_schema,
)
from .db import init_db
from .events import event_bus
from .observability import WorkflowRunHooks
from .pipeline_store import pipeline_store
from .reference_lot import (
    build_grid_values,
    load_reference_rules,
    normalize_reference_rules,
    select_reference_from_params,
    get_defect_rates_by_lot_id,
    get_lot_detail_by_id,
)
from .simulation import (
    call_grid_search_api,
    recommendation_store,
    simulation_store,
)
from .tools import (
    collect_simulation_params,
    emit_lot_info,
    emit_simulation_form,
    execute_simulation,
    run_lot_simulation_impl,
)


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="공정 모니터링 데모")
logger = logging.getLogger(__name__)

DEBUG_PRINT_MAX_LEN = 160


def _debug_short(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return f"[{len(value)}]"
    text = " ".join(str(value).split())
    if len(text) > DEBUG_PRINT_MAX_LEN:
        return text[: DEBUG_PRINT_MAX_LEN - 3] + "..."
    return text


def _debug_print(stage: str, **meta: object) -> None:
    if not DEBUG_PRINT:
        return
    parts = []
    for key, value in meta.items():
        if value is None or value == "" or value == []:
            continue
        parts.append(f"{key}={_debug_short(value)}")
    suffix = " ".join(parts)
    line = f"[debug] {stage}"
    if suffix:
        line = f"{line} {suffix}"
    print(line, flush=True)

SIMULATION_EVENT_TYPES = {
    "simulation_form",
    "simulation_result",
    "lot_result",
    "defect_rate_chart",
    "design_candidates",
    "final_briefing",
    "final_defect_chart",
}
DEFECT_CHART_TABLE = "mdh_base_view_total_4"
INSPECTION_CHART_PROMPT = "검사불량률도 보여드릴까요?"


@dataclass
class WorkflowOutcome:
    message: str
    ui_event: dict | None = None
    memory_summary: str | None = None
    streamed: bool = False


class ChatRequest(BaseModel):
    session_id: str
    message: str
    intent: str | None = None
    params: dict | None = None


class TestRequest(BaseModel):
    session_id: str
    query: str = ""
    params: dict | None = None


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
    size: str | None = None
    capacity: float | None = None
    production_mode: str | None = None
    chip_prod_id: str | None = None


@app.on_event("startup")
async def startup() -> None:
    init_db()
    set_tracing_disabled(not TRACING_ENABLED)
    for connection in list_connections():
        connection_id = connection.get("id")
        if not connection_id:
            logger.warning("DB 연결 ID가 누락되어 스키마 갱신을 건너뜁니다.")
            continue
        logger.info("DB 스키마 갱신 시작: %s", connection_id)
        try:
            preload_schema(connection_id, force=True)
            logger.info("DB 스키마 갱신 완료: %s", connection_id)
        except Exception:
            logger.exception("DB 스키마 갱신 실패: %s", connection_id)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    session = SQLiteSession(request.session_id, SESSION_DB_PATH)
    token = current_session_id.set(request.session_id)
    try:
        message = request.message or ""
        if request.params:
            params_line = f"PARAMS_JSON: {json.dumps(request.params, ensure_ascii=False)}"
            message = f"{params_line}\n{message}" if message else params_line

        if request.intent == "simulation_run" or simulation_store.is_active(request.session_id):
            entry_agent = simulation_flow_agent
            workflow_id = "simulation"
        else:
            entry_agent = orchestrator_agent
            workflow_id = "chat"

        pipeline_store.update(request.session_id, workflow_id=workflow_id)
        try:
            run = await Runner.run(
                entry_agent,
                input=message,
                session=session,
                hooks=WorkflowRunHooks(),
            )
        except Exception:
            return {"assistant_message": "응답 생성 중 오류가 발생했습니다."}

        response = (run.final_output or "").strip()
        return {"assistant_message": response or "응답을 생성하지 못했습니다."}
    finally:
        current_session_id.reset(token)


def _format_simulation_params(params: dict) -> str:
    if not params:
        return "입력 없음"
    mapping = [
        ("chip_prod_id", "기종명"),
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
    chip_prod_id = result.get("recommended_chip_prod_id")
    if chip_prod_id:
        parts.append(f"추천 기종={chip_prod_id}")
    rep_lot = result.get("representative_lot")
    if rep_lot:
        parts.append(f"대표 LOT={rep_lot}")
    params = result.get("params")
    if isinstance(params, dict):
        parts.append(f"파라미터={len(params)}개")
    if not parts:
        return "결과 수신"
    return ", ".join(parts)


MEMORY_SUMMARY_MAX_LEN = 600
MEMORY_SUMMARY_SKIP_KEYS = {
    "top_candidates",
    "grid_results",
    "defect_rates",
    "rows",
    "row",
    "columns",
    "defect_stats",
    "post_grid_defects",
    "candidate_matches",
    "design_blocks",
}
BRIEFING_STREAM_CHUNK_SIZE = 80


def _resolve_post_grid_payload(summary_payload: dict | None) -> dict | None:
    if not isinstance(summary_payload, dict):
        return None
    candidate_matches = summary_payload.get("candidate_matches")
    if isinstance(candidate_matches, dict):
        return candidate_matches
    post_grid_defects = summary_payload.get("post_grid_defects")
    if isinstance(post_grid_defects, dict):
        return post_grid_defects
    return None


def _resolve_post_grid_available_columns(post_grid_payload: dict | None) -> list[str]:
    if not isinstance(post_grid_payload, dict):
        return []
    available_columns = post_grid_payload.get("available_columns")
    if isinstance(available_columns, list) and available_columns:
        return [str(column) for column in available_columns if column]
    columns = post_grid_payload.get("columns")
    if isinstance(columns, list) and columns:
        return [str(column) for column in columns if column]
    items = post_grid_payload.get("items")
    if not isinstance(items, list):
        return []
    collected: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_columns = item.get("columns")
        if isinstance(item_columns, list) and item_columns:
            collected.extend(str(column) for column in item_columns if column)
    if collected:
        return list(dict.fromkeys(collected))
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = item.get("rows") or item.get("lot_rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            collected.extend(str(column) for column in row.keys() if column)
    return list(dict.fromkeys(collected))


def _summary_token(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(round(value, 6))
    if isinstance(value, str):
        return value.strip()
    return None


def _join_summary_tokens(values: list[object], limit: int = 6) -> str | None:
    tokens = [str(value) for value in values if value not in (None, "")]
    if not tokens:
        return None
    if len(tokens) > limit:
        extra = len(tokens) - limit
        tokens = tokens[:limit] + [f"+{extra}"]
    return ",".join(tokens)


def _build_memory_summary(payload: dict, label: str = "final_briefing") -> str:
    items: list[tuple[str, str]] = [("briefing", label)]
    ordered_keys = [
        "chip_prod_id",
        "reference_lot",
        "candidate_total",
        "defect_rate_overall",
        "source",
    ]

    for key in ordered_keys:
        token = _summary_token(payload.get(key))
        if token:
            items.append((key, token))

    top_candidates = payload.get("top_candidates")
    if isinstance(top_candidates, list):
        items.append(("top_candidates", str(len(top_candidates))))

    design_blocks = payload.get("design_blocks")
    if isinstance(design_blocks, list):
        items.append(("design_blocks", str(len(design_blocks))))

    defect_stats = payload.get("defect_stats")
    if isinstance(defect_stats, dict):
        for stat_key in ("avg", "min", "max", "count"):
            token = _summary_token(defect_stats.get(stat_key))
            if token:
                items.append((f"defect_{stat_key}", token))

    post_grid_payload = _resolve_post_grid_payload(payload)
    if isinstance(post_grid_payload, dict):
        columns = (
            post_grid_payload.get("columns")
            or post_grid_payload.get("defect_columns")
            or post_grid_payload.get("match_fields")
        )
        if isinstance(columns, list):
            token = _join_summary_tokens(columns)
            if token:
                items.append(("post_grid_cols", token))
        pg_items = post_grid_payload.get("items")
        if isinstance(pg_items, list):
            items.append(("post_grid_items", str(len(pg_items))))
        recent = _summary_token(post_grid_payload.get("recent_months"))
        if recent:
            items.append(("post_grid_recent_months", recent))

    for key, value in payload.items():
        if key in MEMORY_SUMMARY_SKIP_KEYS:
            continue
        if any(key == existing[0] for existing in items):
            continue
        token = _summary_token(value)
        if token:
            items.append((key, token))

    summary = "BRIEFING_SUMMARY " + "; ".join(f"{key}={value}" for key, value in items)
    if len(summary) > MEMORY_SUMMARY_MAX_LEN:
        summary = summary[: MEMORY_SUMMARY_MAX_LEN - 3].rstrip() + "..."
    return summary


def _format_missing_fields(missing: list[str]) -> str:
    if not missing:
        return "없음"
    mapping = {
        "temperature": "온도",
        "voltage": "전압",
        "size": "크기",
        "capacity": "용량",
        "production_mode": "양산/개발",
        "chip_prod_id": "기종명",
    }
    labels = [mapping.get(item, item) for item in missing]
    return ", ".join(labels)


def _row_value(row: dict, column: str | None) -> object:
    if not row or not column:
        return None
    if column in row:
        return row.get(column)
    lower = column.lower()
    return row.get(lower)


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_selection_overrides(raw: dict | None) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, int] = {}
    top_k_value = (
        raw.get("top_k")
        if raw.get("top_k") is not None
        else raw.get("top") or raw.get("topn")
    )
    top_k = _coerce_positive_int(top_k_value)
    if top_k is not None:
        normalized["top_k"] = top_k
    max_blocks_value = (
        raw.get("max_blocks")
        if raw.get("max_blocks") is not None
        else raw.get("blocks")
    )
    max_blocks = _coerce_positive_int(max_blocks_value)
    if max_blocks is not None:
        normalized["max_blocks"] = max_blocks
    return normalized


def _attach_run_id(payload: object, run_id: str) -> object:
    if not run_id or not isinstance(payload, dict):
        return payload
    if payload.get("run_id") == run_id:
        return payload
    updated = dict(payload)
    updated["run_id"] = run_id
    return updated


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _json_safe_dict(payload: dict) -> dict:
    return {str(key): _json_safe_value(value) for key, value in payload.items()}


def _json_safe_rows(rows: list[object]) -> list[object]:
    safe_rows: list[object] = []
    for row in rows:
        if isinstance(row, dict):
            safe_rows.append(_json_safe_dict(row))
        else:
            safe_rows.append(_json_safe_value(row))
    return safe_rows


def _resolve_grid_payload_columns_from_rules(rules: dict) -> list[str]:
    grid = rules.get("grid_search", {}) if isinstance(rules, dict) else {}
    payload = grid.get("payload_columns")
    columns: list[str] = []
    if isinstance(payload, list):
        columns = payload
    elif isinstance(payload, dict):
        for key in ("ref", "sim"):
            subset = payload.get(key)
            if isinstance(subset, list):
                columns.extend(subset)
    if not columns:
        return []
    return list(dict.fromkeys([str(column) for column in columns if column]))


def _resolve_grid_payload_options(rules: dict) -> tuple[object, bool]:
    grid = rules.get("grid_search", {}) if isinstance(rules, dict) else {}
    fill_value = grid.get("payload_fill_value", -1)
    fallback = bool(grid.get("payload_fallback_to_ref", True))
    return fill_value, fallback


def _build_grid_search_payload(
    ref_row: dict,
    sim_params: dict,
    grid_overrides: dict | None = None,
    payload_columns: list[str] | None = None,
    payload_missing_columns: list[str] | None = None,
    payload_fill_value: object = -1,
    payload_fallback_to_ref: bool = True,
    rules: dict | None = None,
) -> dict:
    ref_payload = dict(ref_row) if isinstance(ref_row, dict) else {}
    sim_payload = dict(sim_params) if isinstance(sim_params, dict) else {}
    if grid_overrides:
        sim_payload.update(
            {key: value for key, value in grid_overrides.items() if value is not None}
        )

    if payload_columns:
        missing_set = set(payload_missing_columns or [])

        def _build_payload(
            columns: list[str],
            primary: dict,
            fallback: dict | None,
        ) -> dict:
            payload: dict = {}
            for column in columns:
                if column in missing_set:
                    payload[column] = payload_fill_value
                    continue
                if column in primary:
                    payload[column] = primary.get(column)
                    continue
                if fallback is not None and column in fallback:
                    payload[column] = fallback.get(column)
                else:
                    payload[column] = None
            return payload

        ref_payload = _build_payload(payload_columns, ref_payload, None)
        sim_payload = _build_payload(
            payload_columns,
            sim_payload,
            ref_payload if payload_fallback_to_ref else None,
        )

    electrode = _coerce_float(_row_value(ref_payload, "electrode_c_avg"))
    grinding_l = _coerce_float(_row_value(ref_payload, "grinding_l_avg"))
    grinding_t = _coerce_float(_row_value(ref_payload, "grinding_t_avg"))
    grinding_w = _coerce_float(_row_value(ref_payload, "grinding_w_avg"))
    if grinding_w is None:
        grinding_w = grinding_t


    total_cover = _row_value(ref_payload, "total_cover_layer_num")
    if total_cover is None:
        top_cover = _coerce_float(_row_value(ref_payload, "top_cover_layer_num"))
        bot_cover = _coerce_float(_row_value(ref_payload, "bot_cover_layer_num"))
        if top_cover is not None and bot_cover is not None:
            total_cover = top_cover + bot_cover

    ldn_avr_value = _row_value(ref_payload, "ldn_avr_value")

    active_layer = (
        grid_overrides.get("active_layer")
        if isinstance(grid_overrides, dict) and "active_layer" in grid_overrides
        else _row_value(ref_payload, "active_layer")
    )

    def _list_value(value: object) -> list[object]:
        if value in (None, ""):
            return []
        return [value]

    params_payload = {
        "screen_chip_size_leng": _list_value(
            _row_value(ref_payload, "screen_chip_size_leng")
        ),
        "screen_mrgn_leng": _list_value(_row_value(ref_payload, "screen_mrgn_leng")),
        "screen_chip_size_widh": _list_value(
            _row_value(ref_payload, "screen_chip_size_widh")
        ),
        "screen_mrgn_widh": _list_value(_row_value(ref_payload, "screen_mrgn_widh")),
        "active_layer": _list_value(active_layer),
        "cover_sheet_thk": _list_value(_row_value(ref_payload, "cover_sheet_thk")),
        "total_cover_layer_num": _list_value(total_cover),
        "gap_sheet_thk": _list_value(_row_value(ref_payload, "gap_sheet_thk")),
        "ldn_avr_value": _list_value(ldn_avr_value),
        "cast_dsgn_thk": _list_value(_row_value(ref_payload, "cast_dsgn_thk")),
    }
    normalized_rules = rules or {}
    if normalized_rules:
        factor_map: dict[str, str] = {}
        grid = normalized_rules.get("grid_search", {})
        if isinstance(grid, dict):
            for factor in grid.get("factors", []):
                if not isinstance(factor, dict):
                    continue
                name = factor.get("name")
                column = factor.get("column") or name
                if name and column:
                    factor_map[str(name)] = str(column)
        base_row = ref_row if isinstance(ref_row, dict) else ref_payload
        factor_values = build_grid_values(base_row, normalized_rules, grid_overrides)
        factor_columns = {}
        for name, values in factor_values.items():
            column = factor_map.get(name) or name
            if not column:
                continue
            if not isinstance(values, list):
                values = [values]
            factor_columns[column] = values
        if factor_columns:
            params_payload.update(factor_columns)
    params_payload = _json_safe_dict(params_payload)

    targets = {
        "target_electrode_c_avg": electrode * 1.05 if electrode is not None else None,
        "target_grinding_l_avg": grinding_l,
        "target_grinding_w_avg": grinding_w,
        "target_grinding_t_avg": grinding_t,
        "target_dc_cap": -1,
    }
    targets = _json_safe_dict(targets)

    return {
        "sim_type": "ver1",
        "data": {
            "ref": _json_safe_dict(ref_payload),
            "sim": _json_safe_dict(sim_payload),
        },
        "targets": targets,
        "params": params_payload,
    }


async def _append_system_note(session_id: str, content: str) -> None:
    session = SQLiteSession(session_id, SESSION_DB_PATH)
    await session.add_items([{"role": "system", "content": content}])


async def _append_user_message(session_id: str, content: str) -> None:
    session = SQLiteSession(session_id, SESSION_DB_PATH)
    await session.add_items([{"role": "user", "content": content}])


async def _append_assistant_message(session_id: str, content: str) -> None:
    session = SQLiteSession(session_id, SESSION_DB_PATH)
    summary = pipeline_store.pop_pending_memory_summary(session_id)
    store_content = summary or content
    await session.add_items([{"role": "assistant", "content": store_content}])


async def _emit_chat_message(session_id: str, content: str) -> None:
    if not content:
        return
    await _append_assistant_message(session_id, content)
    await event_bus.broadcast(
        {"type": "chat_message", "payload": {"role": "assistant", "content": content}}
    )


def _split_stream_chunks(
    text: str, max_chars: int = BRIEFING_STREAM_CHUNK_SIZE
) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(\s+)", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if buffer and len(buffer) + len(part) > max_chars:
            chunks.append(buffer)
            buffer = part
        else:
            buffer += part
    if buffer:
        chunks.append(buffer)
    return chunks


async def _stream_briefing_blocks(
    session_id: str, blocks: list[dict]
) -> bool:
    if not blocks:
        return False
    if not event_bus.has_clients(session_id):
        return False
    message_id = f"briefing-{uuid.uuid4().hex}"
    await event_bus.broadcast(
        {
            "type": "chat_stream_start",
            "payload": {"message_id": message_id, "role": "assistant"},
        },
        session_id=session_id,
    )
    delay = BRIEFING_STREAM_DELAY_SECONDS
    block_index = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("value") or "")
            if not text:
                continue
            block_index += 1
            block_id = f"{message_id}-b{block_index}"
            await event_bus.broadcast(
                {
                    "type": "chat_stream_block_start",
                    "payload": {"message_id": message_id, "block_id": block_id},
                },
                session_id=session_id,
            )
            for chunk in _split_stream_chunks(text):
                await event_bus.broadcast(
                    {
                        "type": "chat_stream_delta",
                        "payload": {
                            "message_id": message_id,
                            "block_id": block_id,
                            "delta": chunk,
                        },
                    },
                    session_id=session_id,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            await event_bus.broadcast(
                {
                    "type": "chat_stream_block_end",
                    "payload": {"message_id": message_id, "block_id": block_id},
                },
                session_id=session_id,
            )
            continue
        if block_type == "table":
            markdown = str(block.get("markdown") or "")
            if not markdown:
                continue
            await event_bus.broadcast(
                {
                    "type": "briefing_table",
                    "payload": {
                        "message_id": message_id,
                        "markdown": markdown,
                        "animate": True,
                    },
                },
                session_id=session_id,
            )
            if delay > 0:
                await asyncio.sleep(delay)
    await event_bus.broadcast(
        {
            "type": "chat_stream_end",
            "payload": {"message_id": message_id},
        },
        session_id=session_id,
    )
    return True


async def _emit_pipeline_status(stage: str, message: str, done: bool = False) -> None:
    pipeline_store.add_status(current_session_id.get(), stage, message, done)
    await event_bus.broadcast(
        {"type": "pipeline_status", "payload": {"stage": stage, "message": message, "done": done}}
    )


async def _emit_pipeline_stage_tables(
    stage: str, tables: list[dict], notes: list[str] | None = None, session_id: str | None = None
) -> None:
    payload = {
        "stage": stage,
        "tables": tables,
        "notes": notes or [],
    }
    session = (session_id or "").strip() or current_session_id.get()
    stored = pipeline_store.get_event(session, "stage_tables")
    if not isinstance(stored, dict):
        stored = {}
    stored[stage] = payload
    pipeline_store.set_event(session, "stage_tables", stored)
    await event_bus.broadcast({"type": "pipeline_stage_tables", "payload": payload}, session_id=session)


POST_GRID_LABEL_FALLBACKS = {
    "rank": "순위",
    "predicted_target": "예측값",
    "lot_count": "LOT 수",
    "sample_lots": "샘플 LOT",
    "lot_id": "LOT",
    "defect_rate": "평균 불량률",
}


def _get_selection_overrides(session_id: str) -> dict[str, int]:
    stage_inputs = pipeline_store.get_stage_inputs(session_id)
    selection = stage_inputs.get("selection") if isinstance(stage_inputs, dict) else {}
    if not isinstance(selection, dict):
        return {}
    sim_run_id = simulation_store.get_run_id(session_id)
    if sim_run_id and selection.get("run_id") not in ("", None, sim_run_id):
        return {}
    overrides = selection.get("overrides")
    if isinstance(overrides, dict):
        return _normalize_selection_overrides(overrides)
    return _normalize_selection_overrides(selection)


def _get_reference_override(session_id: str) -> str | None:
    stage_inputs = pipeline_store.get_stage_inputs(session_id)
    reference = stage_inputs.get("reference") if isinstance(stage_inputs, dict) else {}
    if not isinstance(reference, dict):
        return None
    sim_run_id = simulation_store.get_run_id(session_id)
    if sim_run_id and reference.get("run_id") not in ("", None, sim_run_id):
        return None
    override = reference.get("reference_override")
    return str(override).strip() if override else None


def _get_grid_overrides(session_id: str) -> dict[str, float]:
    stage_inputs = pipeline_store.get_stage_inputs(session_id)
    grid = stage_inputs.get("grid") if isinstance(stage_inputs, dict) else {}
    if isinstance(grid, dict):
        sim_run_id = simulation_store.get_run_id(session_id)
        if sim_run_id and grid.get("run_id") not in ("", None, sim_run_id):
            grid = {}
        overrides = grid.get("overrides")
        if isinstance(overrides, dict):
            return overrides
    state = pipeline_store.get(session_id)
    grid_state = state.get("grid") if isinstance(state, dict) else {}
    if not isinstance(grid_state, dict):
        return {}
    stored_overrides = grid_state.get("overrides")
    return stored_overrides if isinstance(stored_overrides, dict) else {}


async def _handle_detailed_briefing(session_id: str) -> WorkflowOutcome:
    summary_payload = pipeline_store.get_event(session_id, "final_briefing")
    if not isinstance(summary_payload, dict) or not summary_payload:
        return WorkflowOutcome("브리핑할 데이터가 아직 없습니다.")
    state = pipeline_store.get(session_id)
    reference_payload = state.get("reference")
    if not isinstance(reference_payload, dict) or not reference_payload:
        return WorkflowOutcome("레퍼런스 정보가 없어 상세 브리핑을 생성할 수 없습니다.")
    grid_state = state.get("grid")
    if not isinstance(grid_state, dict):
        grid_state = {}
    post_grid_payload = _resolve_post_grid_payload(summary_payload)

    rules = normalize_reference_rules(load_reference_rules())
    db_config = rules.get("db", {})
    label_connection_id = db_config.get("connection_id") or ""
    label_schema = db_config.get("schema") or "public"
    column_label_map = get_column_label_map(label_connection_id, label_schema)
    defect_source = rules.get("defect_rate_source") or {}
    defect_connection_id = defect_source.get("connection_id") or label_connection_id
    defect_schema = defect_source.get("schema") or label_schema
    defect_label_map = (
        get_column_label_map(defect_connection_id, defect_schema)
        if defect_connection_id and defect_connection_id != label_connection_id
        else column_label_map
    )
    briefing_columns = _load_briefing_table_columns(rules)
    detail_override_columns = briefing_columns.get("ref_lot_selected") or []
    run_id = simulation_store.get_run_id(session_id)

    briefing_config = _resolve_final_briefing_config(rules)
    reference_table_limit = briefing_config.get("reference_table_max_rows", 10)
    detail_chunk_size = briefing_config.get("reference_detail_chunk_size", 6)

    reference_columns = reference_payload.get("reference_columns") or []
    reference_rows = reference_payload.get("reference_rows") or []
    if not isinstance(reference_columns, list):
        reference_columns = []
    if not isinstance(reference_rows, list):
        reference_rows = []
    reference_column_labels = reference_payload.get("reference_column_labels")
    if not isinstance(reference_column_labels, dict):
        reference_column_labels = build_column_label_map(
            reference_columns, column_label_map
        )
    selected_row = reference_payload.get("row")
    if not isinstance(selected_row, dict):
        selected_row = {}
    selected_lot_id = reference_payload.get("lot_id") or summary_payload.get("reference_lot") or ""
    if not selected_lot_id:
        selected_lot_id = str(_row_value(selected_row, db_config.get("lot_id_column") or "lot_id") or "")

    safe_reference_rows = _json_safe_rows(reference_rows)
    reference_table_columns = _resolve_briefing_columns(
        briefing_columns,
        "ref_lot_candidate",
        reference_columns,
        available=reference_columns,
    )
    reference_table_labels = build_column_label_map(
        reference_table_columns, column_label_map
    )
    reference_table = _build_markdown_table(
        reference_table_columns,
        safe_reference_rows,
        row_limit=reference_table_limit,
        column_labels=reference_table_labels,
    )
    reference_table_label = ""
    reference_table_note = ""
    if reference_table:
        display_count = min(len(reference_rows), reference_table_limit)
        reference_table_label = f"레퍼런스 LOT 후보 표 (상위 {display_count}개):"
        if len(reference_rows) > reference_table_limit:
            reference_table_note = f"※ 상위 {reference_table_limit}개만 표시"
    else:
        reference_table_label = "레퍼런스 LOT 후보 표를 생성하지 못했습니다."

    detail_result = (
        get_lot_detail_by_id(
            selected_lot_id,
            columns=detail_override_columns or None,
            use_available_columns=True,
        )
        if selected_lot_id
        else {}
    )
    detail_row = detail_result.get("row") if detail_result.get("status") == "ok" else None
    if not isinstance(detail_row, dict):
        detail_row = selected_row if isinstance(selected_row, dict) else {}
    detail_columns = detail_result.get("columns") if detail_result.get("status") == "ok" else None
    if not isinstance(detail_columns, list) or not detail_columns:
        detail_columns = list(detail_row.keys()) if isinstance(detail_row, dict) else []
    detail_columns = [str(column) for column in detail_columns if column]
    if not detail_columns and isinstance(detail_row, dict):
        detail_columns = [str(column) for column in detail_row.keys() if column]
    detail_columns = _resolve_briefing_columns(
        briefing_columns,
        "ref_lot_selected",
        detail_columns,
        available=detail_columns,
    )
    detail_column_labels = build_column_label_map(detail_columns, column_label_map)
    detail_row_safe = _json_safe_dict(detail_row) if isinstance(detail_row, dict) else {}
    detail_tables = _build_markdown_table_chunks(
        detail_columns,
        detail_row_safe,
        detail_chunk_size,
        column_labels=detail_column_labels,
    )
    detail_table_text = (
        "\n\n".join(detail_tables) if detail_tables else "상세 정보를 찾지 못했습니다."
    )
    detail_note = ""
    if len(detail_tables) > 1:
        detail_note = "컬럼이 많아 표를 여러 개로 나눴습니다."

    top_candidates = grid_state.get("top")
    if not isinstance(top_candidates, list):
        top_candidates = []
    grid_columns_default = [
        "rank",
        "electrode_c_avg",
        "grinding_t_avg",
        "active_layer",
        "cast_dsgn_thk",
        "ldn_avr_value",
    ]
    grid_columns = _resolve_briefing_columns(
        briefing_columns,
        "grid_search",
        grid_columns_default,
    )
    grid_rows = _build_grid_table_rows(top_candidates, grid_columns, selected_row)
    grid_column_labels = build_column_label_map(grid_columns, column_label_map)
    grid_table = _build_markdown_table(
        grid_columns,
        _json_safe_rows(grid_rows),
        column_labels=grid_column_labels,
    )

    post_grid_available = None
    post_grid_requested = briefing_columns.get("post_grid_lot_search") or []
    missing_post_grid_columns: list[str] = []
    available_columns = _resolve_post_grid_available_columns(post_grid_payload)
    if available_columns:
        post_grid_available = available_columns
    if post_grid_available and post_grid_requested:
        missing_post_grid_columns = [
            column
            for column in post_grid_requested
            if column not in post_grid_available
        ]
    post_grid_columns = _resolve_briefing_columns(
        briefing_columns,
        "post_grid_lot_search",
        [],
        available=post_grid_available,
    )
    logger.info(
        "Briefing columns resolved: session=%s run=%s ref_candidate=%s ref_selected=%s grid=%s post_grid=%s",
        session_id,
        run_id,
        reference_table_columns,
        detail_columns,
        grid_columns,
        post_grid_columns,
    )
    post_grid_rows = _build_post_grid_table_rows(post_grid_payload, post_grid_columns)
    post_grid_label_map = _build_post_grid_label_map(
        column_label_map, defect_label_map
    )
    post_grid_column_labels = build_column_label_map(
        post_grid_columns, post_grid_label_map
    )
    post_grid_table = _build_markdown_table(
        post_grid_columns,
        _json_safe_rows(post_grid_rows),
        column_labels=post_grid_column_labels,
    )
    post_grid_diagnostic = _format_post_grid_diagnostics(
        post_grid_payload, post_grid_label_map
    )

    reference_tables: list[dict] = []
    reference_notes: list[str] = []
    if reference_table:
        reference_tables.append(
            {
                "title": reference_table_label or "레퍼런스 LOT 후보 표",
                "markdown": reference_table,
            }
        )
        if reference_table_note:
            reference_notes.append(reference_table_note)
    else:
        if reference_table_label:
            reference_notes.append(reference_table_label)
    if detail_tables:
        total_tables = len(detail_tables)
        for idx, table in enumerate(detail_tables, start=1):
            title = "레퍼런스 LOT 상세"
            if total_tables > 1:
                title = f"{title} ({idx}/{total_tables})"
            reference_tables.append({"title": title, "markdown": table})
        if detail_note:
            reference_notes.append(detail_note)
    elif detail_table_text:
        reference_notes.append(detail_table_text)
    if reference_tables or reference_notes:
        await _emit_pipeline_stage_tables(
            "reference", reference_tables, reference_notes, session_id=session_id
        )

    grid_tables: list[dict] = []
    grid_notes: list[str] = []
    if grid_table:
        grid_tables.append({"title": "그리드 서치 결과", "markdown": grid_table})
    else:
        grid_notes.append("그리드 결과가 없습니다.")
    if post_grid_table:
        grid_tables.append(
            {"title": "최근 LOT 불량률 현황", "markdown": post_grid_table}
        )
    else:
        if not post_grid_columns:
            grid_notes.append("post_grid_lot_search 컬럼 설정이 없습니다.")
            if missing_post_grid_columns:
                grid_notes.append(
                    "요청 컬럼 누락: " + ", ".join(missing_post_grid_columns)
                )
        else:
            grid_notes.append("최근 LOT 불량률 데이터가 없습니다.")
        if post_grid_diagnostic:
            grid_notes.append(f"매칭 진단: {post_grid_diagnostic}")
    await _emit_pipeline_stage_tables(
        "grid", grid_tables, grid_notes, session_id=session_id
    )

    params = simulation_store.get(session_id)
    input_summary = _format_input_conditions(params)
    condition_summary = _format_defect_condition_summary(rules, defect_label_map)
    sort_summary = _format_sort_summary(rules, column_label_map)
    intro_parts = [input_summary]
    if condition_summary:
        intro_parts.append(f"해당 조건으로 {condition_summary} 조건을 필터링하였으며")
    if sort_summary:
        intro_parts.append(f"정렬 우선순위는 {sort_summary} 입니다.")
    reference_intro = " ".join(intro_parts).strip()
    total_lot_count = len(reference_rows)
    ref_label = selected_lot_id or "Ref LOT"
    detail_intro = f"선택된 Ref는 {ref_label} 으로 정보는 하기와 같습니다."
    if detail_note:
        detail_intro = f"{detail_intro}\n{detail_note}"

    electrode_ref_value = _row_value(selected_row, "electrode_c_avg")
    electrode_ref_text = _format_brief_value(electrode_ref_value)
    grid_intro = (
        "Ref LOT의 모재/첨가제가 동일하다는 조건 하에 S/T, L/D, 층수를 변경하여 "
        "최적 설계를 진행하였습니다."
    )
    if electrode_ref_value not in (None, ""):
        grid_intro += (
            f" 최적설계는 {ref_label}의 연마용량인 {electrode_ref_text}로부터 "
            "5% 증가시켜 설계를 진행하였습니다."
        )
    grid_intro += " Sheet T, LayDown, 층수를 가변하여 최적설계를 진행한 결과는 하기와 같습니다."
    simulation_result = pipeline_store.get_event(session_id, "simulation_result")
    if not isinstance(simulation_result, dict):
        simulation_result = {}
    summary_body = (
        f"최종 요약: {_format_simulation_result(simulation_result.get('result'))}. "
        "상위 후보를 확인해 주세요."
        if simulation_result and simulation_result.get("result")
        else "최종 요약: 추천 결과를 확인해 주세요."
    )
    rewrite_sections = {
        "1": f"{reference_intro} 나온 LOT는 총 {total_lot_count}개입니다. "
        f"검색된 LOT {total_lot_count}개 중 가장 적합한 LOT를 선정하였습니다.",
        "2": detail_intro,
        "3": grid_intro,
        "5": summary_body,
    }
    rewritten = await _maybe_rewrite_briefing_sections(rewrite_sections)

    reference_step_text = (
        "1) 레퍼 LOT 추천 결과\n"
        f"{rewritten.get('1', rewrite_sections['1'])}"
    )
    if reference_table_label:
        reference_step_text = f"{reference_step_text}\n\n{reference_table_label}"
    reference_step = reference_step_text
    if reference_table:
        reference_step = f"{reference_step}\n\n{reference_table}"
    if reference_table_note:
        reference_step = f"{reference_step}\n\n{reference_table_note}"

    reference_detail_text = (
        "2) 레퍼 정보 제공\n"
        f"{rewritten.get('2', rewrite_sections['2'])}"
    )
    reference_detail_step = reference_detail_text
    reference_detail_step = f"{reference_detail_step}\n\n{detail_table_text}"

    grid_text = (
        "3) 그리드 서치 결과 제공\n"
        f"{rewritten.get('3', rewrite_sections['3'])}"
    )
    if grid_table:
        grid_step = f"{grid_text}\n\n{grid_table}"
    else:
        grid_step = f"{grid_text}\n\n그리드 결과가 없습니다."
    post_grid_title = "4) 최근 LOT 불량률 현황"
    if post_grid_table:
        post_grid_step = f"{post_grid_title}\n\n{post_grid_table}"
    else:
        post_grid_note = "최근 LOT 불량률 데이터가 없습니다."
        if not post_grid_columns:
            post_grid_note = "post_grid_lot_search 컬럼 설정이 없습니다."
            if missing_post_grid_columns:
                post_grid_note = (
                    f"{post_grid_note}\n요청 컬럼 누락: "
                    f"{', '.join(missing_post_grid_columns)}"
                )
        if post_grid_diagnostic:
            post_grid_note = f"{post_grid_note}\n{post_grid_diagnostic}"
        post_grid_step = f"{post_grid_title}\n\n{post_grid_note}"
    summary_step = f"5) {rewritten.get('5', rewrite_sections['5'])}"
    sections = [
        "브리핑 시작",
        reference_step,
        reference_detail_step,
        grid_step,
        post_grid_step,
        summary_step,
    ]
    message = "\n\n".join(sections)
    stream_blocks: list[dict] = [{"type": "text", "value": "브리핑 시작"}]
    if reference_step_text:
        stream_blocks.append({"type": "text", "value": reference_step_text})
    if reference_table:
        stream_blocks.append({"type": "table", "markdown": reference_table})
        if reference_table_note:
            stream_blocks.append({"type": "text", "value": reference_table_note})
    if reference_detail_text:
        if detail_tables:
            stream_blocks.append({"type": "text", "value": reference_detail_text})
            for table in detail_tables:
                stream_blocks.append({"type": "table", "markdown": table})
        else:
            stream_blocks.append({"type": "text", "value": reference_detail_step})
    if grid_text:
        stream_blocks.append({"type": "text", "value": grid_text})
        if grid_table:
            stream_blocks.append({"type": "table", "markdown": grid_table})
        else:
            stream_blocks.append({"type": "text", "value": "그리드 결과가 없습니다."})
    if post_grid_table:
        stream_blocks.append({"type": "text", "value": post_grid_title})
        stream_blocks.append({"type": "table", "markdown": post_grid_table})
    elif post_grid_step:
        stream_blocks.append({"type": "text", "value": post_grid_step})
    if summary_step:
        stream_blocks.append({"type": "text", "value": summary_step})
    process_chart_payload = _build_defect_chart_payload(
        rules,
        summary_payload,
        "공정",
        DEFECT_CHART_TABLE,
    )
    if process_chart_payload:
        message = f"{message}\n\n{INSPECTION_CHART_PROMPT}"
        stream_blocks.append({"type": "text", "value": INSPECTION_CHART_PROMPT})
    streamed = await _stream_briefing_blocks(session_id, stream_blocks)
    if process_chart_payload:
        await _emit_defect_chart(session_id, process_chart_payload)
    memory_summary = _build_memory_summary(summary_payload)
    pipeline_store.update(
        session_id,
        briefing_text=message,
        briefing_text_mode="detail",
        briefing_text_run_id=simulation_store.get_run_id(session_id),
        briefing_summary=memory_summary,
    )
    pipeline_store.set_pending_memory_summary(
        session_id, memory_summary, label="final_briefing"
    )
    return WorkflowOutcome(message, memory_summary=memory_summary, streamed=streamed)


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


def _build_final_briefing_payload(
    chip_prod_id: str,
    reference_result: dict,
    grid_results: list[dict],
    top_candidates: list[dict],
    defect_stats: dict,
    rules: dict | None = None,
    candidate_matches: dict | None = None,
    design_blocks_override: list[dict] | None = None,
    defect_rates: list[dict] | None = None,
    defect_rate_overall: float | None = None,
    chip_prod_id_count: int | None = None,
    chip_prod_id_samples: list[str] | None = None,
    column_labels: dict | None = None,
) -> dict:
    rules = rules or {}
    config = _resolve_final_briefing_config(rules)
    preview = []
    for item in top_candidates[: config.get("max_blocks", 3)]:
        if not isinstance(item, dict):
            continue
        preview.append(
            {
                "rank": item.get("rank"),
                "predicted_target": item.get("predicted_target"),
                "design": item.get("design"),
            }
        )
    payload = {
        "chip_prod_id": chip_prod_id,
        "reference_lot": reference_result.get("lot_id"),
        "candidate_total": len(grid_results),
        "top_candidates": preview,
        "defect_stats": defect_stats,
        "source": reference_result.get("source") or "postgresql",
    }
    if defect_rates is not None:
        payload["defect_rates"] = defect_rates
    if defect_rate_overall is not None:
        payload["defect_rate_overall"] = defect_rate_overall
    if chip_prod_id_count is not None:
        payload["chip_prod_id_count"] = chip_prod_id_count
    if chip_prod_id_samples is not None:
        payload["chip_prod_id_samples"] = chip_prod_id_samples
    if candidate_matches is not None:
        payload["candidate_matches"] = candidate_matches
        if not design_blocks_override:
            payload["design_blocks"] = _build_design_blocks_from_matches(
                candidate_matches, rules, column_labels
            )
    if design_blocks_override:
        payload["design_blocks"] = design_blocks_override
    return payload


def _summarize_metric_values(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None, "median": None}
    sorted_values = sorted(values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        median = sorted_values[mid]
    return {
        "count": count,
        "min": min(sorted_values),
        "max": max(sorted_values),
        "avg": sum(sorted_values) / count,
        "median": median,
    }


def _compute_histogram(
    values: list[float],
    bins: int,
    range_min: float | None,
    range_max: float | None,
    normalize: str,
    value_unit: str | None,
) -> dict:
    if not values:
        return {"bins": [], "normalize": normalize, "value_unit": value_unit}

    values_min = min(values)
    values_max = max(values)
    lower = values_min if range_min is None else range_min
    upper = values_max if range_max is None else range_max
    if lower == upper:
        upper = lower + 1e-6
    bins = max(1, int(bins))

    width = (upper - lower) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        if value < lower or value > upper:
            continue
        idx = int((value - lower) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    total = max(1, sum(counts))
    bin_entries = []
    for idx, count in enumerate(counts):
        start = lower + idx * width
        end = start + width
        entry = {"start": start, "end": end, "count": count}
        if normalize == "percent":
            entry["value"] = round((count / total) * 100, 3)
        elif normalize == "ratio":
            entry["value"] = round(count / total, 6)
        bin_entries.append(entry)

    return {
        "bins": bin_entries,
        "normalize": normalize,
        "value_unit": value_unit,
    }


def _resolve_final_briefing_config(rules: dict) -> dict:
    config = rules.get("final_briefing")
    if not isinstance(config, dict):
        config = {}
    try:
        max_blocks = int(config.get("max_blocks", 3))
    except (TypeError, ValueError):
        max_blocks = 3
    if max_blocks <= 0:
        max_blocks = 3
    design_fields = config.get("design_fields")
    if not isinstance(design_fields, list):
        design_fields = []
    design_labels = config.get("design_labels")
    if not isinstance(design_labels, dict):
        design_labels = {}
    try:
        chart_bins = int(config.get("chart_bins", 8))
    except (TypeError, ValueError):
        chart_bins = 8
    chart_bins = max(1, chart_bins)
    try:
        reference_table_max_rows = int(config.get("reference_table_max_rows", 10))
    except (TypeError, ValueError):
        reference_table_max_rows = 10
    reference_table_max_rows = max(1, reference_table_max_rows)
    try:
        reference_detail_chunk_size = int(config.get("reference_detail_chunk_size", 6))
    except (TypeError, ValueError):
        reference_detail_chunk_size = 6
    reference_detail_chunk_size = max(1, reference_detail_chunk_size)
    reference_detail_columns = config.get("reference_detail_columns")
    if not isinstance(reference_detail_columns, list):
        reference_detail_columns = []
    return {
        "max_blocks": max_blocks,
        "design_fields": design_fields,
        "design_labels": design_labels,
        "chart_bins": chart_bins,
        "reference_table_max_rows": reference_table_max_rows,
        "reference_detail_chunk_size": reference_detail_chunk_size,
        "reference_detail_columns": reference_detail_columns,
    }


def _build_design_display(
    design: dict,
    design_fields: list[str],
    design_labels: dict,
    column_labels: dict | None = None,
) -> list[dict]:
    if not isinstance(design, dict) or not design:
        return []
    keys = design_fields or list(design.keys())
    display = []
    for key in keys:
        if key not in design:
            continue
        value = design.get(key)
        if value is None or value == "":
            continue
        label = design_labels.get(key) or (column_labels or {}).get(key) or key
        display.append({"key": key, "label": label, "value": value})
    return display


def _build_design_blocks(
    post_grid_defects: dict | None,
    rules: dict,
    column_labels: dict | None = None,
) -> list[dict]:
    if not isinstance(post_grid_defects, dict):
        return []
    items = post_grid_defects.get("items")
    if not isinstance(items, list) or not items:
        return []
    config = _resolve_final_briefing_config(rules)
    value_unit = str(post_grid_defects.get("value_unit") or "").strip().lower()
    blocks: list[dict] = []
    for index, item in enumerate(items[: config["max_blocks"]], start=1):
        if not isinstance(item, dict):
            continue
        design = item.get("design")
        if not isinstance(design, dict):
            design = {}
        display = _build_design_display(
            design,
            config.get("design_fields", []),
            config.get("design_labels", {}),
            column_labels,
        )
        lot_rates = item.get("lot_defect_rates")
        if not isinstance(lot_rates, list):
            lot_rates = []
        values: list[float] = []
        for entry in lot_rates:
            if not isinstance(entry, dict):
                continue
            rate = entry.get("defect_rate")
            try:
                values.append(float(rate))
            except (TypeError, ValueError):
                continue
        stats = _summarize_metric_values(values)
        if value_unit:
            stats["value_unit"] = value_unit
        metrics = []
        if stats.get("avg") is not None:
            metrics.append(
                {
                    "key": "defect_avg",
                    "label": "평균 불량률",
                    "value": stats.get("avg"),
                    "unit": value_unit,
                }
            )
        chart = None
        if values:
            histogram = _compute_histogram(
                values, config.get("chart_bins", 8), None, None, "count", value_unit
            )
            chart = {
                "histogram": histogram,
                "stats": stats,
                "chart_type": "histogram",
                "metric_label": "평균 불량률",
                "value_unit": value_unit,
            }
        rank = item.get("rank") or index
        blocks.append(
            {
                "rank": rank,
                "predicted_target": item.get("predicted_target"),
                "design_display": display,
                "design": design,
                "metrics": metrics,
                "lot_count": item.get("lot_count"),
                "lot_samples": item.get("sample_lots"),
                "chart": chart,
            }
        )
    return blocks


def _build_design_blocks_from_matches(
    match_payload: dict | None,
    rules: dict,
    column_labels: dict | None = None,
) -> list[dict]:
    if not isinstance(match_payload, dict):
        return []
    items = match_payload.get("items")
    if not isinstance(items, list) or not items:
        return []
    config = _resolve_final_briefing_config(rules)
    value_unit = str(match_payload.get("value_unit") or "").strip().lower()
    row_limit = match_payload.get("row_limit")
    blocks: list[dict] = []
    for index, item in enumerate(items[: config["max_blocks"]], start=1):
        if not isinstance(item, dict):
            continue
        design = item.get("design")
        if not isinstance(design, dict):
            design = {}
        display = _build_design_display(
            design,
            config.get("design_fields", []),
            config.get("design_labels", {}),
            column_labels,
        )
        row_count = item.get("row_count")
        metrics = []
        if isinstance(row_count, int):
            metrics.append(
                {
                    "key": "match_rows",
                    "label": "매칭 LOT",
                    "value": row_count,
                    "unit": "건",
                }
            )
        chart = None
        averages = item.get("defect_column_avgs")
        if isinstance(averages, list) and averages:
            values = []
            lots = []
            for entry in averages:
                if not isinstance(entry, dict):
                    continue
                avg_value = entry.get("avg")
                try:
                    avg_num = float(avg_value)
                except (TypeError, ValueError):
                    continue
                values.append(avg_num)
                lots.append(
                    {
                        "label": _resolve_column_label(entry.get("column"), column_labels)
                        or entry.get("column"),
                        "defect_rate": avg_num,
                    }
                )
            if values:
                stats = _summarize_metric_values(values)
                if value_unit:
                    stats["value_unit"] = value_unit
                chart = {
                    "lots": lots,
                    "stats": stats,
                    "chart_type": "bar",
                    "bar_orientation": "vertical",
                    "metric_label": "불량 인자 평균",
                    "value_unit": value_unit,
                }

        rank = item.get("rank") or index
        blocks.append(
            {
                "rank": rank,
                "predicted_target": item.get("predicted_target"),
                "design_display": display,
                "design": design,
                "metrics": metrics,
                "lot_count": row_count,
                "lot_samples": item.get("sample_lots"),
                "chart": chart,
                "match_rows": _json_safe_rows(item.get("rows") or []),
                "match_columns": item.get("columns") or [],
                "match_column_labels": build_column_label_map(
                    item.get("columns") or [], column_labels or {}
                ),
                "match_row_count": row_count,
                "match_row_limit": row_limit,
                "match_recent_months": match_payload.get("recent_months"),
            }
        )
    return blocks


def _build_post_grid_chart_payload(post_grid_defects: dict | None) -> dict | None:
    if not isinstance(post_grid_defects, dict):
        return None
    items = post_grid_defects.get("items")
    if not isinstance(items, list) or not items:
        return None
    charts = []
    value_unit = str(post_grid_defects.get("value_unit") or "").strip().lower()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        lot_rates = item.get("lot_defect_rates")
        if not isinstance(lot_rates, list) or not lot_rates:
            continue
        lots = []
        values: list[float] = []
        for entry in lot_rates:
            if not isinstance(entry, dict):
                continue
            rate = entry.get("defect_rate")
            try:
                rate_value = float(rate)
            except (TypeError, ValueError):
                continue
            lots.append(
                {
                    "lot_id": entry.get("lot_id"),
                    "defect_rate": rate_value,
                }
            )
            values.append(rate_value)
        if not values:
            continue
        stats = _summarize_metric_values(values)
        if value_unit:
            stats["value_unit"] = value_unit
        histogram = _compute_histogram(values, 8, None, None, "count", value_unit)
        rank = item.get("rank") or index
        charts.append(
            {
                "title": f"TOP{rank} 설계안",
                "rank": rank,
                "lot_count": item.get("lot_count"),
                "lots": lots,
                "stats": stats,
                "histogram": histogram,
                "chart_type": "histogram",
                "metric_label": "설계안 평균 불량률",
                "value_unit": value_unit,
            }
        )
    if not charts:
        return None
    return {
        "charts": charts,
        "metric_label": "설계안별 평균 불량률",
        "value_unit": value_unit,
    }


def _build_candidate_defect_chart_payload(
    match_payload: dict | None, column_labels: dict | None = None
) -> dict | None:
    if not isinstance(match_payload, dict):
        return None
    items = match_payload.get("items")
    if not isinstance(items, list) or not items:
        return None
    value_unit = str(match_payload.get("value_unit") or "").strip().lower()
    target = next(
        (
            item
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("defect_column_avgs"), list)
            and item.get("defect_column_avgs")
        ),
        None,
    )
    if not target:
        return None
    averages = target.get("defect_column_avgs") or []
    lots = []
    values: list[float] = []
    for entry in averages:
        if not isinstance(entry, dict):
            continue
        avg_value = entry.get("avg")
        try:
            avg_num = float(avg_value)
        except (TypeError, ValueError):
            continue
        values.append(avg_num)
        label = entry.get("column")
        label = _resolve_column_label(label, column_labels) or label
        lots.append(
            {
                "label": label,
                "defect_rate": avg_num,
            }
        )
    if not values:
        return None
    stats = _summarize_metric_values(values)
    if value_unit:
        stats["value_unit"] = value_unit
    rank = target.get("rank")
    label = "불량 인자 평균"
    if isinstance(rank, int):
        label = f"TOP{rank} 불량 인자 평균"
    return {
        "lots": lots,
        "stats": stats,
        "chart_type": "bar",
        "bar_orientation": "vertical",
        "metric_label": label,
        "value_unit": value_unit,
    }


def _format_defect_rate_display(value: object, value_unit: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if value_unit == "ratio":
        return f"{numeric * 100:.2f}%"
    if value_unit == "percent":
        return f"{numeric:.2f}%"
    return f"{numeric:.4f}"


def _escape_markdown_cell(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        value = round(value, 6)
    text = str(value)
    return (
        text.replace("\r", " ")
        .replace("\n", " ")
        .replace("|", "\\|")
    )


def _resolve_column_label(column: object, column_labels: dict | None) -> str:
    if column is None:
        return ""
    text = str(column)
    if not column_labels:
        return text
    return str(column_labels.get(text, text))


def _build_markdown_table(
    columns: list[str],
    rows: list[object],
    row_limit: int | None = None,
    column_labels: dict | None = None,
) -> str | None:
    if not columns or not rows:
        return None
    safe_columns = [str(column) for column in columns]
    header_labels = [
        _escape_markdown_cell(_resolve_column_label(column, column_labels))
        for column in safe_columns
    ]
    header = "| " + " | ".join(header_labels) + " |"
    separator = "| " + " | ".join(["---"] * len(safe_columns)) + " |"
    lines = [header, separator]
    if isinstance(row_limit, int) and row_limit > 0:
        rows = rows[:row_limit]
    for row in rows:
        if isinstance(row, dict):
            cells = [_escape_markdown_cell(row.get(column)) for column in safe_columns]
        elif isinstance(row, (list, tuple)):
            cells = [
                _escape_markdown_cell(row[index] if index < len(row) else None)
                for index in range(len(safe_columns))
            ]
        else:
            cells = [
                _escape_markdown_cell(row) if index == 0 else "-"
                for index in range(len(safe_columns))
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _extract_markdown_table_blocks(text: str) -> list[str]:
    if not text:
        return []
    matches = re.findall(
        r"(?:^|\n)(\|.+\|\n\|[ -:|]+\|\n(?:\|.*\|\n?)*)",
        text,
    )
    blocks = [match.strip() for match in matches if match.strip()]
    return blocks


def _format_brief_value(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_capacity_value(value: object) -> str:
    if value is None or value == "":
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_value = abs(numeric)
    if abs_value >= 1_000_000:
        return f"{numeric / 1_000_000:.3g}uF"
    if abs_value >= 1_000:
        return f"{numeric / 1_000:.3g}nF"
    return f"{numeric:.3g}pF"


def _format_input_conditions(params: dict) -> str:
    if not isinstance(params, dict):
        return "입력 조건이 없습니다."
    parts = []
    temperature = params.get("temperature")
    if temperature not in (None, ""):
        parts.append(_format_brief_value(temperature))
    voltage = params.get("voltage")
    if voltage not in (None, ""):
        parts.append(f"전압 {_format_brief_value(voltage)}V")
    size = params.get("size")
    if size not in (None, ""):
        parts.append(f"사이즈 {_format_brief_value(size)}")
    capacity = params.get("capacity")
    if capacity not in (None, ""):
        parts.append(f"용량 {_format_capacity_value(capacity)}")
    if not parts:
        return "입력 조건이 없습니다."
    return f"프로님이 입력하신 조건은 {', '.join(parts)} 입니다."


def _format_condition_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_brief_value(item) for item in value)
    return _format_brief_value(value)


def _format_defect_condition_summary(
    rules: dict, column_labels: dict | None = None
) -> str:
    conditions = rules.get("defect_conditions") or []
    if not isinstance(conditions, list):
        return ""
    parts: list[str] = []
    for item in conditions:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("key") or item.get("name")
        if not column:
            continue
        column = _resolve_column_label(column, column_labels)
        operator = str(item.get("operator") or "=").lower()
        value = item.get("value")
        if operator in {"=", "=="}:
            parts.append(f"{column}={_format_condition_value(value)}")
        elif operator == "in":
            parts.append(f"{column} in ({_format_condition_value(value)})")
        elif operator == "between":
            parts.append(f"{column} between {_format_condition_value(value)}")
        elif operator == "is_null":
            parts.append(f"{column} is null")
        elif operator == "is_not_null":
            parts.append(f"{column} is not null")
        else:
            parts.append(f"{column} {operator} {_format_condition_value(value)}")
    return ", ".join(parts)


def _format_sort_summary(rules: dict, column_labels: dict | None = None) -> str:
    selection = rules.get("selection") or {}
    sort_specs = selection.get("lot_search_sort") or []
    if not isinstance(sort_specs, list):
        return ""
    parts: list[str] = []
    for item in sort_specs:
        if not isinstance(item, dict):
            continue
        columns = item.get("columns")
        if not isinstance(columns, list):
            columns = []
        if not columns:
            column = item.get("column") or item.get("name")
            if column:
                columns = [column]
        if not columns:
            continue
        column_text = "+".join(
            _resolve_column_label(column, column_labels) for column in columns if column
        )
        if not column_text:
            continue
        priority = item.get("value_priority")
        if isinstance(priority, dict) and priority:
            ordered = sorted(
                ((str(key), value) for key, value in priority.items()),
                key=lambda entry: entry[1],
                reverse=True,
            )
            rank_text = " > ".join(key for key, _ in ordered if key)
            if rank_text:
                parts.append(f"{column_text} 등급({rank_text})")
                continue
        order = str(item.get("order") or item.get("direction") or "asc").lower()
        order_text = "높은순" if order == "desc" else "낮은순"
        parts.append(f"{column_text} {order_text}")
    return " → ".join(parts)


def _extract_preserve_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*(?:%|V|uF|nF|pF))?", text):
        candidate = match.strip()
        if candidate and candidate not in tokens:
            tokens.append(candidate)
    for match in re.findall(r"\b[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b", text):
        if match and match not in tokens:
            tokens.append(match)
    for match in re.findall(r"\b[A-Za-z0-9_.-]*[_/][A-Za-z0-9_.-]*\b", text):
        if match and match not in tokens:
            tokens.append(match)
    return tokens


def _token_present(token: str, text: str) -> bool:
    if token in text:
        return True
    if " " in token:
        compact_token = token.replace(" ", "")
        if compact_token and compact_token in text.replace(" ", ""):
            return True
    return False


async def _maybe_rewrite_briefing_sections(
    sections: dict[str, str],
) -> dict[str, str]:
    if not OPENAI_API_KEY or not sections:
        return sections
    cleaned: dict[str, str] = {}
    tokens_map: dict[str, list[str]] = {}
    for key, value in sections.items():
        text = str(value or "").strip()
        if not text:
            continue
        cleaned[key] = text
        tokens_map[key] = _extract_preserve_tokens(text)
    if not cleaned:
        return sections

    payload = {"sections": cleaned}
    prompt = (
        "다음 JSON의 sections 문장을 간결하고 자연스럽게 다듬어 주세요. "
        "규칙: 1) 의미 변경 금지 2) 숫자/ID/기술용어/약어는 그대로 유지 "
        "3) 각 섹션은 1~2문장 4) 목록/번호/표현 추가 금지 "
        "5) 출력은 동일한 JSON 구조로만 반환\n"
        f"JSON: {json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        run = await Runner.run(
            conversation_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
        raw = (run.final_output or "").strip()
        parsed = json.loads(raw)
    except Exception:
        return sections

    output_sections = parsed.get("sections") if isinstance(parsed, dict) else None
    if not isinstance(output_sections, dict):
        return sections

    merged = dict(sections)
    for key, original in cleaned.items():
        candidate = output_sections.get(key)
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        candidate = candidate.strip()
        tokens = tokens_map.get(key, [])
        if any(not _token_present(token, candidate) for token in tokens):
            continue
        merged[key] = candidate
    return merged


def _build_markdown_table_chunks(
    columns: list[str],
    row: dict,
    chunk_size: int,
    column_labels: dict | None = None,
) -> list[str]:
    if not columns or not isinstance(row, dict):
        return []
    try:
        size = int(chunk_size)
    except (TypeError, ValueError):
        size = 6
    size = max(1, size)
    tables = []
    for start in range(0, len(columns), size):
        chunk = columns[start : start + size]
        table = _build_markdown_table(chunk, [row], column_labels=column_labels)
        if table:
            tables.append(table)
    return tables


def _get_available_table_columns(
    connection_id: str, schema_name: str, table_name: str
) -> list[str]:
    if not connection_id or not schema_name or not table_name:
        return []
    schema_payload = get_schema(connection_id) or {}
    schemas = schema_payload.get("schemas") if isinstance(schema_payload, dict) else None
    if not isinstance(schemas, dict):
        return []
    tables = schemas.get(schema_name, {}).get("tables", {})
    if not isinstance(tables, dict):
        return []
    table = tables.get(table_name, {})
    if not isinstance(table, dict):
        return []
    columns = table.get("columns", [])
    if not isinstance(columns, list):
        return []
    return [
        col.get("name")
        for col in columns
        if isinstance(col, dict) and col.get("name")
    ]


def _load_defect_columns_from_label_map(
    rules: dict,
    defect_type: str,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    db_config = rules.get("db", {}) if isinstance(rules, dict) else {}
    connection_id = str(db_config.get("connection_id") or "").strip()
    schema_name = str(db_config.get("schema") or "public")
    if not connection_id:
        return [], {}, {}
    try:
        result = execute_table_query(
            connection_id=connection_id,
            schema_name=schema_name,
            table_name="column_label_map",
            columns=["column_name", "label_ko", "value_unit", "unit"],
            filter_column="defect",
            filter_operator="=",
            filter_value=defect_type,
            limit=5000,
        )
    except Exception:
        try:
            result = execute_table_query(
                connection_id=connection_id,
                schema_name=schema_name,
                table_name="column_label_map",
                columns=["column_name", "label_ko"],
                filter_column="defect",
                filter_operator="=",
                filter_value=defect_type,
                limit=5000,
            )
        except Exception:
            return [], {}, {}
    rows = result.get("rows") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return [], {}, {}
    columns: list[str] = []
    label_map: dict[str, str] = {}
    unit_map: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("column_name") or "").strip()
        if not name:
            continue
        if name not in columns:
            columns.append(name)
        label = str(row.get("label_ko") or "").strip()
        if label:
            label_map[name] = label
        unit = str(row.get("value_unit") or row.get("unit") or "").strip().lower()
        if unit and unit not in {"none", "null", "-"}:
            unit_map[name] = unit
    return columns, label_map, unit_map


def _collect_post_grid_lot_ids(summary_payload: dict) -> list[str]:
    post_grid = _resolve_post_grid_payload(summary_payload)
    if not isinstance(post_grid, dict):
        return []
    items = post_grid.get("items")
    if not isinstance(items, list):
        return []
    lot_ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = item.get("rows")
        if isinstance(rows, list) and rows:
            lot_id_column = None
            columns = item.get("columns")
            if isinstance(columns, list):
                for column in columns:
                    lowered = str(column).strip().lower()
                    if lowered in {"lot_id", "lotid"}:
                        lot_id_column = str(column)
                        break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                lot_id = None
                if lot_id_column:
                    lot_id = row.get(lot_id_column)
                if lot_id in (None, ""):
                    for key, value in row.items():
                        lowered = str(key).strip().lower()
                        if lowered in {"lot_id", "lotid"}:
                            lot_id = value
                            break
                if lot_id in (None, ""):
                    continue
                lot_ids.append(str(lot_id))
            continue
        lot_rates = item.get("lot_defect_rates")
        if isinstance(lot_rates, list) and lot_rates:
            for entry in lot_rates:
                if not isinstance(entry, dict):
                    continue
                lot_id = entry.get("lot_id")
                if lot_id in (None, ""):
                    continue
                lot_ids.append(str(lot_id))
            continue
        sample_lots = item.get("sample_lots")
        if isinstance(sample_lots, list) and sample_lots:
            for lot_id in sample_lots:
                if lot_id in (None, ""):
                    continue
                lot_ids.append(str(lot_id))
    return list(dict.fromkeys(lot_ids))


def _resolve_value_unit(unit_map: dict[str, str], columns: list[str]) -> str:
    units = {unit_map.get(column) for column in columns if unit_map.get(column)}
    if len(units) == 1:
        return units.pop() or ""
    return ""


def _build_defect_chart_payload(
    rules: dict,
    summary_payload: dict,
    defect_type: str,
    table_name: str,
) -> dict | None:
    lot_ids = _collect_post_grid_lot_ids(summary_payload)
    if not lot_ids:
        return None
    columns, label_map, unit_map = _load_defect_columns_from_label_map(
        rules, defect_type
    )
    if not columns:
        return None
    db_config = rules.get("db", {}) if isinstance(rules, dict) else {}
    connection_id = str(db_config.get("connection_id") or "").strip()
    schema_name = str(db_config.get("schema") or "public")
    lot_id_column = db_config.get("lot_id_column") or "lot_id"
    if not connection_id:
        return None
    available = _get_available_table_columns(connection_id, schema_name, table_name)
    if available:
        columns = [column for column in columns if column in available]
    if not columns:
        return None
    metrics = [
        {"column": column, "agg": "avg", "alias": column} for column in columns
    ]
    filters = [{"column": lot_id_column, "operator": "in", "value": lot_ids}]
    try:
        result = execute_table_query_aggregate(
            connection_id=connection_id,
            schema_name=schema_name,
            table_name=table_name,
            metrics=metrics,
            filters=filters,
        )
    except Exception:
        return None
    rows = result.get("rows") if isinstance(result, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0] if isinstance(rows[0], dict) else {}
    lots: list[dict] = []
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        label = label_map.get(column, column)
        lots.append({"label": label, "defect_rate": value})
    if not lots:
        return None
    metric_label = "공정불량률" if defect_type == "공정" else "검사불량률"
    value_unit = _resolve_value_unit(unit_map, columns)
    return {
        "lots": lots,
        "metric_label": metric_label,
        "chart_type": "bar",
        "bar_orientation": "vertical",
        "value_unit": value_unit,
        "source": table_name,
    }


async def _emit_defect_chart(
    session_id: str, payload: dict | None
) -> bool:
    if not payload:
        return False
    pipeline_store.set_event(session_id, "defect_rate_chart", payload)
    await event_bus.broadcast(
        {"type": "defect_rate_chart", "payload": payload}, session_id=session_id
    )
    return True


def _parse_briefing_columns(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    parts = re.split(r"[,\n]", text)
    return [part.strip() for part in parts if part.strip()]


def _load_briefing_table_columns(rules: dict) -> dict[str, list[str]]:
    steps = (
        "ref_lot_candidate",
        "ref_lot_selected",
        "grid_search",
        "post_grid_lot_search",
    )
    columns_map: dict[str, list[str]] = {step: [] for step in steps}
    db_config = rules.get("db", {}) if isinstance(rules, dict) else {}
    connection_id = str(db_config.get("connection_id") or "").strip()
    if not connection_id:
        logger.warning(
            "Briefing columns not loaded: missing db.connection_id."
        )
        return columns_map
    schema_name = str(db_config.get("schema") or "public")
    limit = 100
    for step in steps:
        try:
            result = execute_table_query(
                connection_id=connection_id,
                schema_name=schema_name,
                table_name="column_briefing_table",
                columns=["columns"],
                filter_column="step",
                filter_operator="=",
                filter_value=step,
                limit=limit,
            )
        except Exception:
            logger.warning(
                "Briefing columns load failed: step=%s", step, exc_info=True
            )
            continue
        rows = result.get("rows") if isinstance(result, dict) else None
        if not rows:
            logger.info("Briefing columns empty: step=%s", step)
            continue
        parsed_columns: list[str] = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            columns_value = row.get("columns")
            parsed = _parse_briefing_columns(columns_value)
            if not parsed:
                continue
            for column in parsed:
                if column in seen:
                    continue
                seen.add(column)
                parsed_columns.append(column)
        if parsed_columns:
            columns_map[step] = parsed_columns
            logger.info(
                "Briefing columns loaded: step=%s rows=%s columns=%s",
                step,
                len(rows),
                parsed_columns,
            )
        else:
            logger.info(
                "Briefing columns empty after parse: step=%s rows=%s",
                step,
                len(rows),
            )
    return columns_map


def _resolve_briefing_columns(
    columns_map: dict[str, list[str]],
    step: str,
    fallback: list[str],
    available: list[str] | None = None,
) -> list[str]:
    selected = list(columns_map.get(step) or [])
    if selected and available:
        available_set = set(available)
        filtered = [column for column in selected if column in available_set]
        if filtered:
            selected = filtered
        else:
            selected = []
    if selected:
        return selected
    return list(fallback)


def _build_grid_table_rows(
    candidates: list[dict],
    columns: list[str],
    selected_row: dict | None = None,
) -> list[dict]:
    rows: list[dict] = []
    alias_keys = {
        "electrode_c_avg": ("electrode_c_avg",),
        "grinding_t_avg": ("grinding_t_avg", "grinding_t_size"),
        "active_layer": ("active_layer",),
        "cast_dsgn_thk": ("cast_dsgn_thk",),
        "ldn_avr_value": ("ldn_avr_value",),
    }
    for index, candidate in enumerate(candidates[:3], start=1):
        if not isinstance(candidate, dict):
            continue
        design = candidate.get("design")
        if not isinstance(design, dict):
            design = {}
        row: dict[str, object] = {}
        for column in columns:
            if column in alias_keys:
                row[column] = _resolve_design_value(
                    design, selected_row or {}, alias_keys[column]
                )
                continue
            if column in candidate:
                row[column] = candidate.get(column)
                continue
            if column in design:
                row[column] = design.get(column)
                continue
            if selected_row and column in selected_row:
                row[column] = selected_row.get(column)
                continue
            row[column] = None
        if "rank" in columns and row.get("rank") in (None, ""):
            row["rank"] = candidate.get("rank") or index
        rows.append(row)
    return rows


def _build_post_grid_table_rows(
    post_grid_defects: dict | None,
    columns: list[str],
) -> list[dict]:
    if not isinstance(post_grid_defects, dict):
        return []
    items = post_grid_defects.get("items")
    if not isinstance(items, list) or not items:
        return []
    rows: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        design = item.get("design")
        if not isinstance(design, dict):
            design = {}
        lot_count = item.get("lot_count")
        if lot_count is None:
            lot_count = item.get("row_count")
        sample_lots = item.get("sample_lots") or []
        if not sample_lots:
            rows = item.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    lot_id = row.get("lot_id")
                    if lot_id in (None, ""):
                        continue
                    sample_lots.append(str(lot_id))
                    if len(sample_lots) >= 5:
                        break
        base = {
            "rank": item.get("rank"),
            "predicted_target": item.get("predicted_target"),
            "lot_count": lot_count,
            "sample_lots": ", ".join(
                str(value)
                for value in sample_lots
                if value not in (None, "")
            ),
        }
        base.update(design)
        display_rows = item.get("lot_rows") or item.get("rows")
        if isinstance(display_rows, list) and display_rows:
            for lot_row in display_rows:
                if not isinstance(lot_row, dict):
                    continue
                row = dict(base)
                row.update(lot_row)
                rows.append({column: row.get(column) for column in columns})
            continue
        lot_rates = item.get("lot_defect_rates")
        if not isinstance(lot_rates, list) or not lot_rates:
            row = {column: base.get(column) for column in columns}
            rows.append(row)
            continue
        for lot in lot_rates:
            if not isinstance(lot, dict):
                continue
            row = dict(base)
            row.update(lot)
            rows.append({column: row.get(column) for column in columns})
    return rows


def _build_post_grid_label_map(
    column_label_map: dict, defect_label_map: dict
) -> dict:
    label_map = dict(column_label_map or {})
    if isinstance(defect_label_map, dict):
        label_map.update(defect_label_map)
    label_map.update(POST_GRID_LABEL_FALLBACKS)
    return label_map


def _format_post_grid_diagnostics(
    post_grid_defects: dict | None,
    column_labels: dict | None = None,
    limit: int = 3,
) -> str:
    if not isinstance(post_grid_defects, dict):
        return ""
    diagnostics = post_grid_defects.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        items = post_grid_defects.get("items")
        if not isinstance(items, list) or not items:
            return ""
        messages: list[str] = []
        for index, item in enumerate(items[:limit], start=1):
            if not isinstance(item, dict):
                continue
            rank = item.get("rank") or index
            parts: list[str] = []
            missing_fields = item.get("missing_fields") or []
            matched_values = item.get("matched_values") or {}
            row_count = item.get("row_count")
            if missing_fields:
                labels = [
                    _resolve_column_label(column, column_labels)
                    for column in missing_fields
                ]
                parts.append(f"누락 컬럼: {', '.join(labels)}")
            if matched_values:
                labels = [
                    _resolve_column_label(column, column_labels)
                    for column in matched_values.keys()
                ]
                parts.append(f"필터: {', '.join(labels)}")
            if row_count is not None:
                parts.append(f"조회 {row_count}건")
                if row_count == 0 and matched_values:
                    parts.append("매칭 LOT 없음")
            if not matched_values:
                parts.append("설계값 필터 없음")
            if parts:
                messages.append(f"TOP{rank} - " + "; ".join(parts))
        return " / ".join(messages)
    reason_map = {
        "design_missing": "설계값 없음",
        "no_design_filters": "설계값 필터 없음",
        "no_rows": "매칭 LOT 없음",
        "no_recent_rows": "최근 기간 LOT 없음",
    }
    messages: list[str] = []
    for index, item in enumerate(diagnostics[:limit], start=1):
        if not isinstance(item, dict):
            continue
        rank = item.get("rank") or index
        parts: list[str] = []
        missing_columns = item.get("missing_columns") or []
        missing_values = item.get("missing_values") or []
        filter_columns = item.get("filter_columns") or []
        row_count = item.get("row_count")
        recent_count = item.get("recent_row_count")
        reason = item.get("reason")
        if missing_columns:
            labels = [
                _resolve_column_label(column, column_labels) for column in missing_columns
            ]
            parts.append(f"누락 컬럼: {', '.join(labels)}")
        if missing_values:
            labels = [
                _resolve_column_label(column, column_labels) for column in missing_values
            ]
            parts.append(f"값 없음: {', '.join(labels)}")
        if filter_columns:
            labels = [
                _resolve_column_label(column, column_labels) for column in filter_columns
            ]
            parts.append(f"필터: {', '.join(labels)}")
        if row_count is not None:
            if recent_count is not None:
                parts.append(f"조회 {row_count}건/최근 {recent_count}건")
            else:
                parts.append(f"조회 {row_count}건")
        if reason and reason != "ok":
            parts.append(reason_map.get(reason, str(reason)))
        if parts:
            messages.append(f"TOP{rank} - " + "; ".join(parts))
    return " / ".join(messages)


def _resolve_design_value(
    design: dict, ref_row: dict, keys: tuple[str, ...]
) -> object:
    if isinstance(design, dict):
        for key in keys:
            value = design.get(key)
            if value not in (None, ""):
                return value
    for key in keys:
        value = _row_value(ref_row, key)
        if value not in (None, ""):
            return value
    return None


def _build_defect_condition_table_payload(
    defect_items: list[dict] | None,
    chip_prod_id: str,
    value_unit: str = "ratio",
    column_labels: dict | None = None,
) -> dict | None:
    if not defect_items:
        return None
    rows = []
    has_counts = any(
        isinstance(item, dict)
        and (
            "value1_count" in item
            or "value2_count" in item
            or "defect_rate" in item
        )
        for item in defect_items
    )
    if has_counts:
        for entry in defect_items:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label") or entry.get("key") or entry.get("column") or "-"
            label = _resolve_column_label(label, column_labels) or label
            total = entry.get("value1_count")
            passed = entry.get("value2_count")
            fail_count = None
            if isinstance(total, (int, float)) and isinstance(passed, (int, float)):
                fail_count = int(total) - int(passed)
            rows.append(
                {
                    "조건": label,
                    "전체 LOT": total,
                    "조건통과 LOT": passed,
                    "미달 LOT": fail_count,
                    "불량률": _format_defect_rate_display(entry.get("defect_rate"), value_unit),
                }
            )
        if not rows:
            return None
        return {
            "chart_type": "table",
            "metric_label": "불량 조건 요약",
            "value_unit": value_unit,
            "table_columns": ["조건", "전체 LOT", "조건통과 LOT", "미달 LOT", "불량률"],
            "table_rows": rows,
            "filters": {"chip_prod_id": chip_prod_id},
        }

    for entry in defect_items:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or entry.get("key") or entry.get("column") or "-"
        label = _resolve_column_label(label, column_labels) or label
        operator = entry.get("operator") or "="
        value = entry.get("value")
        if isinstance(value, (list, tuple)):
            value_text = ", ".join(str(item) for item in value)
        elif value is None:
            value_text = "-"
        else:
            value_text = str(value)
        rows.append(
            {
                "조건": label,
                "연산": operator,
                "값": value_text,
            }
        )
    if not rows:
        return None
    return {
        "chart_type": "table",
        "metric_label": "불량 조건 요약",
        "value_unit": "",
        "table_columns": ["조건", "연산", "값"],
        "table_rows": rows,
        "filters": {"chip_prod_id": chip_prod_id},
    }


def _format_design_summary(design: dict, limit: int = 14) -> str:
    if not isinstance(design, dict) or not design:
        return "-"
    parts = []
    for key, value in design.items():
        if value is None or value == "":
            continue
        if isinstance(value, float):
            value = round(value, 6)
        parts.append(f"{key}={value}")
    if not parts:
        return "-"
    if len(parts) > limit:
        extra = len(parts) - limit
        parts = parts[:limit] + [f"+{extra}"]
    return ", ".join(parts)


def _build_post_grid_step(
    post_grid_defects: dict | None,
    include_prefix: bool = True,
    column_labels: dict | None = None,
) -> str:
    prefix = "3) " if include_prefix else ""
    if not isinstance(post_grid_defects, dict):
        return f"{prefix}TOP3 설계조건 기반 불량실적 조회를 진행했습니다."
    columns = post_grid_defects.get("columns")
    if not isinstance(columns, list):
        columns = []
    items = post_grid_defects.get("items")
    if not isinstance(items, list):
        items = []
    recent_months = post_grid_defects.get("recent_months")
    if not columns:
        return (
            f"{prefix}TOP3 설계조건 기반 불량실적 컬럼 설정이 없어 해당 단계는 생략했습니다."
        )
    if not items:
        return (
            f"{prefix}TOP3 설계조건으로 동일 조건 LOT 불량실적을 조회했지만 "
            "매칭 LOT가 없습니다."
        )
    display_columns = [
        _resolve_column_label(column, column_labels) for column in columns if column
    ]
    if not display_columns:
        display_columns = columns
    period = (
        f"최근 {recent_months}개월"
        if isinstance(recent_months, int) and recent_months > 0
        else "최근 기간"
    )
    details = []
    for index, item in enumerate(items[:3], start=1):
        if not isinstance(item, dict):
            continue
        rank = item.get("rank") or index
        design_text = _format_design_summary(item.get("design") or {})
        lot_count = item.get("lot_count") or 0
        sample_lots = [str(lot) for lot in (item.get("sample_lots") or []) if lot]
        lot_detail = f"{lot_count}개"
        if sample_lots:
            lot_detail = f"{lot_detail} (예: {', '.join(sample_lots)})"
        details.append(f"TOP{rank} 설계안[{design_text}] → LOT {lot_detail}")
    summary = " / ".join(details) if details else "매칭 LOT 없음"
    return (
        f"{prefix}TOP3 설계조건으로 {period} LOT 불량실적을 조회했습니다. "
        f"{summary}"
    )


async def _maybe_demo_sleep() -> None:
    if DEMO_LATENCY_SECONDS <= 0:
        return
    await asyncio.sleep(DEMO_LATENCY_SECONDS)


async def _run_reference_pipeline(
    session_id: str,
    params: dict,
    chip_prod_override: str | None = None,
    reference_override: str | None = None,
    grid_overrides: dict | None = None,
    selection_overrides: dict | None = None,
    emit_briefing: bool | None = None,
) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    selection_overrides = _normalize_selection_overrides(selection_overrides)
    if selection_overrides:
        grid_config = dict(rules.get("grid_search") or {})
        if "top_k" in selection_overrides:
            grid_config["top_k"] = selection_overrides["top_k"]
        rules["grid_search"] = grid_config
        final_config = dict(rules.get("final_briefing") or {})
        if "max_blocks" in selection_overrides:
            final_config["max_blocks"] = selection_overrides["max_blocks"]
        rules["final_briefing"] = final_config
    db_config = rules.get("db", {})
    chip_prod_column = db_config.get("chip_prod_id_column") or "chip_prod_id"
    lot_id_column = db_config.get("lot_id_column") or "lot_id"
    run_id = simulation_store.ensure_run_id(session_id)
    _debug_print(
        "pipeline.start",
        session=session_id,
        run_id=run_id,
        params=_format_simulation_params(params),
        chip_prod_override=chip_prod_override,
        reference_override=reference_override,
    )
    pipeline_store.set_stage_inputs(
        session_id,
        "recommendation",
        {
            "run_id": run_id,
            "params": params,
            "missing": simulation_store.missing(session_id),
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    label_connection_id = db_config.get("connection_id") or ""
    label_schema = db_config.get("schema") or "public"
    column_label_map = get_column_label_map(label_connection_id, label_schema)
    defect_source = rules.get("defect_rate_source") or {}
    defect_connection_id = defect_source.get("connection_id") or label_connection_id
    defect_schema = defect_source.get("schema") or label_schema
    defect_label_map = (
        get_column_label_map(defect_connection_id, defect_schema)
        if defect_connection_id and defect_connection_id != label_connection_id
        else column_label_map
    )
    briefing_columns = _load_briefing_table_columns(rules)
    logger.info(
        "Briefing columns map loaded: session=%s run=%s columns=%s",
        session_id,
        run_id,
        briefing_columns,
    )
    detail_override_columns = briefing_columns.get("ref_lot_selected") or []

    await _emit_pipeline_status("recommendation", "조건 매칭 중...")
    await _emit_pipeline_status("reference", "조건에 맞는 LOT 조회 중...")
    reference_result = select_reference_from_params(
        params,
        chip_prod_override=chip_prod_override,
        briefing_columns=briefing_columns,
    )

    if reference_override:
        override_detail = get_lot_detail_by_id(
            reference_override,
            columns=detail_override_columns or None,
            use_available_columns=True,
        )
        if override_detail.get("status") == "ok":
            override_row = override_detail.get("row") or {}
            override_chip = _row_value(override_row, chip_prod_column)
            override_chip_id = str(override_chip) if override_chip is not None else ""
            if override_chip_id:
                override_result = select_reference_from_params(
                    params,
                    chip_prod_override=override_chip_id,
                    briefing_columns=briefing_columns,
                )
                if override_result.get("status") == "ok":
                    reference_result = override_result
                else:
                    reference_result = {
                        "status": "ok",
                        "chip_prod_ids": [override_chip_id],
                        "selected_chip_prod_id": override_chip_id,
                        "selected_lot_id": reference_override,
                        "selected_row": override_row,
                        "defect_rates": [],
                        "columns": override_detail.get("columns", []),
                        "source": override_detail.get("source"),
                        "reference_columns": [],
                        "reference_rows": [],
                    }
                reference_result["selected_lot_id"] = reference_override
                reference_result["selected_row"] = override_row
                reference_result["selected_chip_prod_id"] = override_chip_id
        else:
            reference_result = {
                "status": "missing",
                "error": "레퍼런스 LOT 정보를 찾지 못했습니다.",
            }

    _debug_print(
        "reference.result",
        status=reference_result.get("status"),
        selected_lot=reference_result.get("selected_lot_id"),
        selected_chip=reference_result.get("selected_chip_prod_id"),
        error=reference_result.get("error"),
    )
    if reference_result.get("status") != "ok":
        await _emit_pipeline_status(
            "reference", "레퍼런스 LOT을 찾지 못했습니다.", done=True
        )
        return {
            "message": reference_result.get("error")
            or "레퍼런스 LOT를 찾지 못했습니다. 조건 또는 DB 설정을 확인해 주세요."
        }

    selected_row = reference_result.get("selected_row") or {}
    selected_chip_prod_id = reference_result.get("selected_chip_prod_id") or ""
    selected_lot_id = reference_result.get("selected_lot_id") or str(
        _row_value(selected_row, lot_id_column) or ""
    )
    chip_prod_id = selected_chip_prod_id or chip_prod_override or ""

    chip_prod_ids = reference_result.get("chip_prod_ids") or []
    if not isinstance(chip_prod_ids, list):
        chip_prod_ids = []
    chip_prod_id_count = len(chip_prod_ids)
    chip_prod_id_samples = [str(item) for item in chip_prod_ids[:10]]
    reference_columns = reference_result.get("reference_columns") or []
    reference_rows = reference_result.get("reference_rows") or []
    if not isinstance(reference_columns, list):
        reference_columns = []
    if not isinstance(reference_rows, list):
        reference_rows = []
    selected_columns = reference_result.get("columns", [])
    if not isinstance(selected_columns, list):
        selected_columns = []
    selected_column_labels = build_column_label_map(selected_columns, column_label_map)
    reference_column_labels = build_column_label_map(
        reference_columns, column_label_map
    )

    reference_payload = {
        "status": "ok",
        "lot_id": selected_lot_id,
        "chip_prod_id": chip_prod_id,
        "row": selected_row,
        "columns": selected_columns,
        "column_labels": selected_column_labels,
        "source": reference_result.get("source") or "postgresql",
        "reference_columns": reference_columns,
        "reference_rows": _json_safe_rows(reference_rows),
        "reference_column_labels": reference_column_labels,
    }
    reference_payload = _attach_run_id(reference_payload, run_id)
    pipeline_store.set_reference(session_id, reference_payload)
    pipeline_store.set_stage_inputs(
        session_id,
        "reference",
        {
            "run_id": run_id,
            "chip_prod_id": chip_prod_id,
            "reference_override": reference_override,
            "selected_lot_id": selected_lot_id,
            "chip_prod_id_count": chip_prod_id_count,
            "chip_prod_id_samples": chip_prod_id_samples,
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    lot_payload = {
        "lot_id": selected_lot_id,
        "columns": reference_payload.get("columns", []),
        "rows": [selected_row],
        "column_labels": reference_payload.get("column_labels", {}),
        "source": reference_payload.get("source") or "postgresql",
    }
    lot_payload = _attach_run_id(lot_payload, run_id)
    await event_bus.broadcast({"type": "lot_result", "payload": lot_payload})
    pipeline_store.set_event(session_id, "lot_result", lot_payload)
    await _maybe_demo_sleep()
    await _emit_pipeline_status("reference", "레퍼런스 LOT 조회 완료", done=True)

    synthetic = {
        "recommended_chip_prod_id": chip_prod_id,
        "representative_lot": selected_lot_id,
        "params": {},
    }
    simulation_payload = {"params": params, "result": synthetic}
    if run_id:
        simulation_payload["run_id"] = run_id
    await event_bus.broadcast(
        {"type": "simulation_result", "payload": simulation_payload}
    )
    pipeline_store.set_event(
        session_id,
        "simulation_result",
        simulation_payload,
    )
    simulation_result = {"status": "ok", "result": synthetic}
    await _emit_pipeline_status("recommendation", "조건 매칭 완료", done=True)

    defect_payload = (
        get_defect_rates_by_lot_id(selected_lot_id)
        if selected_lot_id
        else {"status": "missing", "defect_rates": [], "source": "none"}
    )
    condition_filters = rules.get("defect_conditions") or []
    condition_table = _build_defect_condition_table_payload(
        condition_filters, chip_prod_id, "ratio", column_labels=defect_label_map
    )
    defect_rates: list[dict] = []
    defect_stats = {}
    defect_rate_overall = None
    if condition_table:
        condition_table = _attach_run_id(condition_table, run_id)
        rate_values = [
            item.get("defect_rate")
            for item in defect_rates
            if item.get("defect_rate") is not None
        ]
        if rate_values:
            defect_stats = {
                "count": len(rate_values),
                "avg": sum(rate_values) / len(rate_values),
                "min": min(rate_values),
                "max": max(rate_values),
                "value_unit": "ratio",
            }
            defect_rate_overall = (
                rate_values[0] if len(rate_values) == 1 else defect_stats.get("avg")
            )
        await event_bus.broadcast(
            {"type": "defect_rate_chart", "payload": condition_table}
        )
        pipeline_store.set_event(session_id, "defect_rate_chart", condition_table)
    else:
        defect_rates = defect_payload.get("defect_rates", []) or []
        if defect_rates:
            rate_values = [
                item.get("defect_rate")
                for item in defect_rates
                if item.get("defect_rate") is not None
            ]
            if rate_values:
                defect_stats = {
                    "count": len(rate_values),
                    "avg": sum(rate_values) / len(rate_values),
                    "min": min(rate_values),
                    "max": max(rate_values),
                }
                defect_stats["value_unit"] = "ratio"
                defect_rate_overall = (
                    rate_values[0] if len(rate_values) == 1 else defect_stats.get("avg")
                )
            chart_lots = []
            for item in defect_rates:
                if item.get("defect_rate") is None:
                    continue
                label = item.get("label") or item.get("key") or item.get("column")
                label = _resolve_column_label(label, defect_label_map) or label
                chart_lots.append(
                    {"label": label, "defect_rate": item.get("defect_rate")}
                )
            defect_source = defect_payload.get("source") or "postgresql"
            chart_payload = {
                "lots": chart_lots,
                "filters": {"chip_prod_id": chip_prod_id},
                "stats": defect_stats,
                "source": defect_source,
                "metric_label": "불량률",
                "value_unit": "ratio",
            }
            chart_payload = _attach_run_id(chart_payload, run_id)
            await event_bus.broadcast(
                {"type": "defect_rate_chart", "payload": chart_payload}
            )
            pipeline_store.set_event(
                session_id,
                "defect_rate_chart",
                chart_payload,
            )

    grid_overrides = grid_overrides or {}
    grid_payload_columns = reference_result.get("grid_payload_columns") or []
    if not grid_payload_columns:
        grid_payload_columns = _resolve_grid_payload_columns_from_rules(rules)
    grid_payload_missing = reference_result.get("grid_payload_missing_columns") or []
    payload_fill_value, payload_fallback = _resolve_grid_payload_options(rules)
    grid_payload = _build_grid_search_payload(
        selected_row,
        params,
        grid_overrides,
        payload_columns=grid_payload_columns,
        payload_missing_columns=grid_payload_missing,
        payload_fill_value=payload_fill_value,
        payload_fallback_to_ref=payload_fallback,
        rules=rules,
    )
    pipeline_store.set_stage_inputs(
        session_id,
        "grid",
        {
            "run_id": run_id,
            "overrides": grid_overrides,
            "payload_columns": grid_payload_columns,
            "payload_missing_columns": grid_payload_missing,
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    await _emit_pipeline_status("grid", "그리드 탐색 중...")
    grid_results = await call_grid_search_api(grid_payload)
    await _maybe_demo_sleep()
    if not isinstance(grid_results, list):
        grid_results = []
    top_k = rules.get("grid_search", {}).get("top_k", 10)
    top_k = max(1, int(top_k))
    top_candidates = grid_results[:top_k]
    for idx, item in enumerate(top_candidates, start=1):
        item["rank"] = idx

    design_label_map: dict[str, str] = {}
    if column_label_map:
        design_keys: list[str] = []
        for candidate in top_candidates:
            design = candidate.get("design")
            if isinstance(design, dict):
                design_keys.extend(str(key) for key in design.keys() if key)
        if design_keys:
            design_label_map = build_column_label_map(
                list(dict.fromkeys(design_keys)), column_label_map
            )

    raw_rules = load_reference_rules()
    raw_db = raw_rules.get("db") if isinstance(raw_rules, dict) else {}
    grid_match_fields = raw_rules.get("grid_match_fields") if isinstance(raw_rules, dict) else []
    if not isinstance(grid_match_fields, list):
        grid_match_fields = []
    grid_match_fields = [
        str(field).strip() for field in grid_match_fields if str(field).strip()
    ]
    grid_extra_columns = list(
        dict.fromkeys(
            grid_match_fields
            + (briefing_columns.get("grid_search") or [])
            + (briefing_columns.get("post_grid_lot_search") or [])
        )
    )
    grid_result = sorted(top_candidates, key=lambda x: x.get("rank") or 0)[:top_k]
    filtered_grid = []
    for sim in grid_result:
        design = sim.get("design") if isinstance(sim, dict) else None
        if not isinstance(design, dict):
            design = {}
        filtered_grid.append({k: design.get(k) for k in grid_extra_columns if k in design})

    match_values = [
        tuple(sim.get(field) for field in grid_match_fields)
        for sim in filtered_grid
    ]
    match_recent_months = 6
    match_table = "mdh_base_view_total_4"
    connection_id = str(raw_db.get("connection_id") or db_config.get("connection_id") or "")
    schema_name = str(raw_db.get("schema") or db_config.get("schema") or "public")

    result_rows: list[dict] = []
    result_columns: list[str] = []
    source = "none"
    if connection_id and grid_match_fields and match_values:
        connection = get_connection(connection_id)
        if connection:
            try:
                conn = psycopg2.connect(
                    host=connection.get("host"),
                    port=connection.get("port"),
                    database=connection.get("database"),
                    user=connection.get("user"),
                    password=connection.get("password"),
                )
                cur = conn.cursor()
                cutoff = datetime.now() - timedelta(days=match_recent_months * 30)
                columns_sql = sql.SQL(", ").join(map(sql.Identifier, grid_match_fields))
                query = sql.SQL(
                    "SELECT * FROM {schema}.{table} WHERE ({cols}) IN %s AND design_input_date >= %s"
                ).format(
                    schema=sql.Identifier(schema_name),
                    table=sql.Identifier(match_table),
                    cols=columns_sql,
                )
                cur.execute(query, (tuple(match_values), cutoff.isoformat()))
                db_rows = cur.fetchall()
                result_columns = [desc[0] for desc in cur.description]
                result_rows = [dict(zip(result_columns, row)) for row in db_rows]
                source = "postgresql"
            except Exception as exc:
                _debug_print("grid.match.error", error=str(exc))
                result_rows = []
                result_columns = []
                source = "error"
            finally:
                try:
                    cur.close()
                    conn.close()
                except Exception:
                    pass

    match_rows: dict[tuple, list[dict]] = {}
    for row in result_rows:
        key = tuple(row.get(field) for field in grid_match_fields)
        match_rows.setdefault(key, []).append(row)

    candidate_items: list[dict] = []
    for index, candidate in enumerate(grid_result, start=1):
        if not isinstance(candidate, dict):
            continue
        design = (
            candidate.get("design")
            if isinstance(candidate.get("design"), dict)
            else {}
        )
        match_key = tuple(design.get(field) for field in grid_match_fields)
        rows = match_rows.get(match_key, [])
        candidate_items.append(
            {
                "rank": candidate.get("rank") or index,
                "predicted_target": candidate.get("predicted_target"),
                "design": design,
                "match_fields": grid_match_fields,
                "matched_values": {field: design.get(field) for field in grid_match_fields},
                "rows": rows,
                "columns": result_columns,
                "row_count": len(rows),
                "source": source,
            }
        )
    candidate_matches = {
        "items": candidate_items,
        "match_fields": grid_match_fields,
        "recent_months": match_recent_months,
        "columns": result_columns,
        "table": match_table,
    }
    design_blocks_override = _build_design_blocks_from_matches(
        candidate_matches, rules, column_label_map
    )
    pipeline_store.set_grid(
        session_id,
        {
            "run_id": run_id,
            "chip_prod_id": chip_prod_id,
            "lot_id": selected_lot_id,
            "results": grid_results,
            "top": top_candidates,
            "overrides": grid_overrides,
        },
    )
    pipeline_store.set_event(session_id, "final_defect_chart", None)

    candidate_chart = _build_candidate_defect_chart_payload(
        candidate_matches, defect_label_map
    )
    if candidate_chart:
        candidate_chart = _attach_run_id(candidate_chart, run_id)
        await event_bus.broadcast(
            {"type": "defect_rate_chart", "payload": candidate_chart}
        )
        pipeline_store.set_event(
            session_id,
            "defect_rate_chart",
            candidate_chart,
        )

    design_payload = {
        "candidates": top_candidates,
        "total": len(grid_results),
        "offset": 0,
        "limit": len(top_candidates),
        "target": chip_prod_id,
        "design_labels": design_label_map,
    }
    design_payload = _attach_run_id(design_payload, run_id)
    await event_bus.broadcast(
        {"type": "design_candidates", "payload": design_payload}
    )
    pipeline_store.set_event(
        session_id,
        "design_candidates",
        design_payload,
    )
    briefing_config = _resolve_final_briefing_config(rules)
    pipeline_store.set_stage_inputs(
        session_id,
        "selection",
        {
            "run_id": run_id,
            "overrides": selection_overrides,
            "top_k": top_k,
            "max_blocks": briefing_config.get("max_blocks"),
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    summary_payload = _build_final_briefing_payload(
        chip_prod_id,
        reference_payload,
        grid_results,
        top_candidates,
        defect_stats,
        rules=rules,
        candidate_matches=candidate_matches,
        design_blocks_override=design_blocks_override or None,
        defect_rates=defect_rates,
        defect_rate_overall=defect_rate_overall,
        chip_prod_id_count=chip_prod_id_count,
        chip_prod_id_samples=chip_prod_id_samples,
        column_labels=column_label_map,
    )
    summary_payload = _attach_run_id(summary_payload, run_id)
    memory_summary = _build_memory_summary(summary_payload)
    await event_bus.broadcast({"type": "final_briefing", "payload": summary_payload})
    pipeline_store.set_event(session_id, "final_briefing", summary_payload)
    await _emit_pipeline_status("grid", "그리드 탐색 완료", done=True)

    if emit_briefing is None:
        emit_briefing = True
    if not emit_briefing:
        return {
            "message": "추천 결과를 저장했습니다. 브리핑 단계에서 요약을 제공합니다.",
            "simulation_result": simulation_result,
        }

    reference_table_limit = briefing_config.get("reference_table_max_rows", 10)
    detail_chunk_size = briefing_config.get("reference_detail_chunk_size", 6)

    safe_reference_rows = _json_safe_rows(reference_rows)
    reference_table_columns = _resolve_briefing_columns(
        briefing_columns,
        "ref_lot_candidate",
        reference_columns,
        available=reference_columns,
    )
    reference_table_labels = build_column_label_map(
        reference_table_columns, column_label_map
    )
    reference_table = _build_markdown_table(
        reference_table_columns,
        safe_reference_rows,
        row_limit=reference_table_limit,
        column_labels=reference_table_labels,
    )
    reference_table_label = ""
    reference_table_note = ""
    if reference_table:
        display_count = min(len(reference_rows), reference_table_limit)
        reference_table_label = f"레퍼런스 LOT 후보 표 (상위 {display_count}개):"
        if len(reference_rows) > reference_table_limit:
            reference_table_note = f"※ 상위 {reference_table_limit}개만 표시"
    else:
        reference_table_label = "레퍼런스 LOT 후보 표를 생성하지 못했습니다."

    detail_override_columns = briefing_columns.get("ref_lot_selected") or []
    detail_result = (
        get_lot_detail_by_id(
            selected_lot_id,
            columns=detail_override_columns or None,
            use_available_columns=True,
        )
        if selected_lot_id
        else {}
    )
    detail_row = detail_result.get("row") if detail_result.get("status") == "ok" else None
    if not isinstance(detail_row, dict):
        detail_row = selected_row if isinstance(selected_row, dict) else {}
    detail_columns = detail_result.get("columns") if detail_result.get("status") == "ok" else None
    if not isinstance(detail_columns, list) or not detail_columns:
        detail_columns = list(detail_row.keys()) if isinstance(detail_row, dict) else []
    detail_columns = [str(column) for column in detail_columns if column]
    if not detail_columns and isinstance(detail_row, dict):
        detail_columns = [str(column) for column in detail_row.keys() if column]
    detail_columns = _resolve_briefing_columns(
        briefing_columns,
        "ref_lot_selected",
        detail_columns,
        available=detail_columns,
    )
    detail_column_labels = build_column_label_map(detail_columns, column_label_map)
    detail_row_safe = _json_safe_dict(detail_row) if isinstance(detail_row, dict) else {}
    detail_tables = _build_markdown_table_chunks(
        detail_columns,
        detail_row_safe,
        detail_chunk_size,
        column_labels=detail_column_labels,
    )
    detail_table_text = (
        "\n\n".join(detail_tables) if detail_tables else "상세 정보를 찾지 못했습니다."
    )
    detail_note = ""
    if len(detail_tables) > 1:
        detail_note = "컬럼이 많아 표를 여러 개로 나눴습니다."

    grid_columns_default = [
        "rank",
        "electrode_c_avg",
        "grinding_t_avg",
        "active_layer",
        "cast_dsgn_thk",
        "ldn_avr_value",
    ]
    grid_columns = _resolve_briefing_columns(
        briefing_columns,
        "grid_search",
        grid_columns_default,
    )
    grid_rows = _build_grid_table_rows(top_candidates, grid_columns, selected_row)
    grid_column_labels = build_column_label_map(grid_columns, column_label_map)
    grid_table = _build_markdown_table(
        grid_columns,
        _json_safe_rows(grid_rows),
        column_labels=grid_column_labels,
    )

    post_grid_payload = _resolve_post_grid_payload(summary_payload)
    post_grid_available = None
    post_grid_requested = briefing_columns.get("post_grid_lot_search") or []
    missing_post_grid_columns: list[str] = []
    available_columns = _resolve_post_grid_available_columns(post_grid_payload)
    if available_columns:
        post_grid_available = available_columns
    if post_grid_available and post_grid_requested:
        missing_post_grid_columns = [
            column
            for column in post_grid_requested
            if column not in post_grid_available
        ]
    post_grid_columns = _resolve_briefing_columns(
        briefing_columns,
        "post_grid_lot_search",
        [],
        available=post_grid_available,
    )
    logger.info(
        "Briefing columns resolved (detail): session=%s run=%s ref_candidate=%s ref_selected=%s grid=%s post_grid=%s",
        session_id,
        simulation_store.get_run_id(session_id),
        reference_table_columns,
        detail_columns,
        grid_columns,
        post_grid_columns,
    )
    post_grid_rows = _build_post_grid_table_rows(post_grid_payload, post_grid_columns)
    post_grid_label_map = _build_post_grid_label_map(
        column_label_map, defect_label_map
    )
    post_grid_column_labels = build_column_label_map(
        post_grid_columns, post_grid_label_map
    )
    post_grid_table = _build_markdown_table(
        post_grid_columns,
        _json_safe_rows(post_grid_rows),
        column_labels=post_grid_column_labels,
    )
    post_grid_diagnostic = _format_post_grid_diagnostics(
        post_grid_payload, post_grid_label_map
    )

    reference_tables: list[dict] = []
    reference_notes: list[str] = []
    if reference_table:
        reference_tables.append(
            {
                "title": reference_table_label or "레퍼런스 LOT 후보 표",
                "markdown": reference_table,
            }
        )
        if reference_table_note:
            reference_notes.append(reference_table_note)
    else:
        if reference_table_label:
            reference_notes.append(reference_table_label)
    if detail_tables:
        total_tables = len(detail_tables)
        for idx, table in enumerate(detail_tables, start=1):
            title = "레퍼런스 LOT 상세"
            if total_tables > 1:
                title = f"{title} ({idx}/{total_tables})"
            reference_tables.append({"title": title, "markdown": table})
        if detail_note:
            reference_notes.append(detail_note)
    elif detail_table_text:
        reference_notes.append(detail_table_text)
    if reference_tables or reference_notes:
        await _emit_pipeline_stage_tables(
            "reference", reference_tables, reference_notes, session_id=session_id
        )

    grid_tables: list[dict] = []
    grid_notes: list[str] = []
    if grid_table:
        grid_tables.append({"title": "그리드 서치 결과", "markdown": grid_table})
    else:
        grid_notes.append("그리드 결과가 없습니다.")
    if post_grid_table:
        grid_tables.append(
            {"title": "최근 LOT 불량률 현황", "markdown": post_grid_table}
        )
    else:
        if not post_grid_columns:
            grid_notes.append("post_grid_lot_search 컬럼 설정이 없습니다.")
            if missing_post_grid_columns:
                grid_notes.append(
                    "요청 컬럼 누락: " + ", ".join(missing_post_grid_columns)
                )
        else:
            grid_notes.append("최근 LOT 불량률 데이터가 없습니다.")
        if post_grid_diagnostic:
            grid_notes.append(f"매칭 진단: {post_grid_diagnostic}")
    await _emit_pipeline_stage_tables(
        "grid", grid_tables, grid_notes, session_id=session_id
    )

    input_summary = _format_input_conditions(params)
    condition_summary = _format_defect_condition_summary(rules, defect_label_map)
    sort_summary = _format_sort_summary(rules, column_label_map)
    intro_parts = [input_summary]
    if condition_summary:
        intro_parts.append(f"해당 조건으로 {condition_summary} 조건을 필터링하였으며")
    if sort_summary:
        intro_parts.append(f"정렬 우선순위는 {sort_summary} 입니다.")
    reference_intro = " ".join(intro_parts).strip()
    total_lot_count = len(reference_rows)
    ref_label = selected_lot_id or "Ref LOT"
    detail_intro = f"선택된 Ref는 {ref_label} 으로 정보는 하기와 같습니다."
    if detail_note:
        detail_intro = f"{detail_intro}\n{detail_note}"

    electrode_ref_value = _row_value(selected_row, "electrode_c_avg")
    electrode_ref_text = _format_brief_value(electrode_ref_value)
    grid_intro = (
        "Ref LOT의 모재/첨가제가 동일하다는 조건 하에 S/T, L/D, 층수를 변경하여 "
        "최적 설계를 진행하였습니다."
    )
    if electrode_ref_value not in (None, ""):
        grid_intro += (
            f" 최적설계는 {ref_label}의 연마용량인 {electrode_ref_text}로부터 "
            "5% 증가시켜 설계를 진행하였습니다."
        )
    grid_intro += " Sheet T, LayDown, 층수를 가변하여 최적설계를 진행한 결과는 하기와 같습니다."
    summary_body = (
        f"최종 요약: {_format_simulation_result(simulation_result.get('result'))}. "
        "상위 후보를 확인해 주세요."
        if simulation_result and simulation_result.get("result")
        else "최종 요약: 추천 결과를 확인해 주세요."
    )
    rewrite_sections = {
        "1": f"{reference_intro} 나온 LOT는 총 {total_lot_count}개입니다. "
        f"검색된 LOT {total_lot_count}개 중 가장 적합한 LOT를 선정하였습니다.",
        "2": detail_intro,
        "3": grid_intro,
        "5": summary_body,
    }
    rewritten = await _maybe_rewrite_briefing_sections(rewrite_sections)

    reference_step_text = (
        "1) 레퍼 LOT 추천 결과\n"
        f"{rewritten.get('1', rewrite_sections['1'])}"
    )
    if reference_table_label:
        reference_step_text = f"{reference_step_text}\n\n{reference_table_label}"
    reference_step = reference_step_text
    if reference_table:
        reference_step = f"{reference_step}\n\n{reference_table}"
    if reference_table_note:
        reference_step = f"{reference_step}\n\n{reference_table_note}"

    reference_detail_text = (
        "2) 레퍼 정보 제공\n"
        f"{rewritten.get('2', rewrite_sections['2'])}"
    )
    reference_detail_step = reference_detail_text
    reference_detail_step = f"{reference_detail_step}\n\n{detail_table_text}"

    grid_text = (
        "3) 그리드 서치 결과 제공\n"
        f"{rewritten.get('3', rewrite_sections['3'])}"
    )
    if grid_table:
        grid_step = f"{grid_text}\n\n{grid_table}"
    else:
        grid_step = f"{grid_text}\n\n그리드 결과가 없습니다."
    post_grid_title = "4) 최근 LOT 불량률 현황"
    if post_grid_table:
        post_grid_step = f"{post_grid_title}\n\n{post_grid_table}"
    else:
        post_grid_note = "최근 LOT 불량률 데이터가 없습니다."
        if not post_grid_columns:
            post_grid_note = "post_grid_lot_search 컬럼 설정이 없습니다."
            if missing_post_grid_columns:
                post_grid_note = (
                    f"{post_grid_note}\n요청 컬럼 누락: "
                    f"{', '.join(missing_post_grid_columns)}"
                )
        if post_grid_diagnostic:
            post_grid_note = f"{post_grid_note}\n{post_grid_diagnostic}"
        post_grid_step = f"{post_grid_title}\n\n{post_grid_note}"
    summary_step = f"5) {rewritten.get('5', rewrite_sections['5'])}"
    sections = [
        "브리핑 시작",
        reference_step,
        reference_detail_step,
        grid_step,
        post_grid_step,
        summary_step,
    ]
    message = "\n\n".join(sections)
    stream_blocks: list[dict] = [{"type": "text", "value": "브리핑 시작"}]
    if reference_step_text:
        stream_blocks.append({"type": "text", "value": reference_step_text})
    if reference_table:
        stream_blocks.append({"type": "table", "markdown": reference_table})
        if reference_table_note:
            stream_blocks.append({"type": "text", "value": reference_table_note})
    if reference_detail_text:
        if detail_tables:
            stream_blocks.append({"type": "text", "value": reference_detail_text})
            for table in detail_tables:
                stream_blocks.append({"type": "table", "markdown": table})
        else:
            stream_blocks.append(
                {"type": "text", "value": reference_detail_step}
            )
    if grid_text:
        stream_blocks.append({"type": "text", "value": grid_text})
        if grid_table:
            stream_blocks.append({"type": "table", "markdown": grid_table})
        else:
            stream_blocks.append({"type": "text", "value": "그리드 결과가 없습니다."})
    if post_grid_table:
        stream_blocks.append({"type": "text", "value": post_grid_title})
        stream_blocks.append({"type": "table", "markdown": post_grid_table})
    elif post_grid_step:
        stream_blocks.append({"type": "text", "value": post_grid_step})
    if summary_step:
        stream_blocks.append({"type": "text", "value": summary_step})
    process_chart_payload = _build_defect_chart_payload(
        rules,
        summary_payload,
        "공정",
        DEFECT_CHART_TABLE,
    )
    if process_chart_payload:
        message = f"{message}\n\n{INSPECTION_CHART_PROMPT}"
        stream_blocks.append({"type": "text", "value": INSPECTION_CHART_PROMPT})
    streamed = await _stream_briefing_blocks(session_id, stream_blocks)
    if process_chart_payload:
        await _emit_defect_chart(session_id, process_chart_payload)
    pipeline_store.update(
        session_id,
        briefing_text=message,
        briefing_text_mode="detail",
        briefing_text_run_id=run_id,
        briefing_summary=memory_summary,
    )
    pipeline_store.set_pending_memory_summary(
        session_id, memory_summary, label="final_briefing"
    )
    response = {"message": message, "simulation_result": simulation_result}
    if streamed:
        response["streamed"] = True
    return response


@app.post("/api/test/trigger")
async def trigger_test(request: TestRequest) -> dict:
    token = current_session_id.set(request.session_id)
    try:
        if request.query:
            await emit_lot_info(request.query, limit=1)
            result = await run_lot_simulation_impl(request.query)
        else:
            params = request.params or {
                "temperature": 120,
                "voltage": 3.7,
                "size": 12,
                "capacity": 6,
                "production_mode": "양산",
            }
            safe_params = {
                key: value
                for key, value in params.items()
                if key in {
                    "temperature",
                    "voltage",
                    "size",
                    "capacity",
                    "production_mode",
                    "chip_prod_id",
                }
            }
            collect_simulation_params(**safe_params)
            result = await execute_simulation()
    finally:
        current_session_id.reset(token)

    return {"status": "ok", "simulation": result}


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
            chip_prod_id=request.chip_prod_id,
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
