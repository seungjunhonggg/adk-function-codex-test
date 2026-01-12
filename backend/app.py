import json
import uuid
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import asyncio
import logging
from typing import Any, Awaitable, Callable

from agents import Runner, SQLiteSession
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from agents.tracing import set_tracing_disabled
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import (
    auto_message_agent,
    briefing_agent,
    chart_agent,
    conversation_agent,
    db_agent,
    edit_intent_agent,
    planner_agent,
    route_agent,
    stage_resolver_agent,
)
from .config import (
    OPENAI_API_KEY,
    SESSION_DB_PATH,
    TRACING_ENABLED,
    DEMO_LATENCY_SECONDS,
    BRIEFING_STREAM_DELAY_SECONDS,
)
from .context import current_session_id
from .db_connections import connect_and_save, get_schema, list_connections, preload_schema
from .db import init_db
from .events import event_bus
from .guardrails import guardrail_fallback_message
from .observability import WorkflowRunHooks, emit_workflow_log
from .pipeline_store import pipeline_store
from .reference_lot import (
    collect_post_grid_defects,
    collect_grid_candidate_matches,
    load_reference_rules,
    normalize_reference_rules,
    select_reference_from_params,
    get_defect_rates_by_lot_id,
    get_lot_detail_by_id,
)
from .simulation import (
    call_grid_search_api,
    extract_simulation_params_hybrid,
    recommendation_store,
    simulation_store,
)
from .tools import (
    collect_simulation_params,
    apply_chart_config_impl,
    emit_lot_info,
    emit_simulation_form,
    execute_simulation,
    get_simulation_progress_impl,
    reset_simulation_state_impl,
    run_lot_simulation_impl,
    show_simulation_stage_impl,
    STAGE_EVENT_MAP,
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
logger = logging.getLogger(__name__)


@dataclass
class WorkflowOutcome:
    message: str
    ui_event: dict | None = None
    memory_summary: str | None = None
    streamed: bool = False


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
    size: str | None = None
    capacity: float | None = None
    production_mode: str | None = None
    chip_prod_id: str | None = None


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


@app.get("/builder")
async def builder() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "builder.html")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    session = SQLiteSession(request.session_id, SESSION_DB_PATH)
    token = current_session_id.set(request.session_id)
    try:
        pending_result = await _maybe_handle_pending_action(
            request.message, request.session_id, session
        )
        if pending_result is not None:
            workflow_id, outcome = pending_result
            return await _record_workflow_outcome(
                request.session_id, request.message, workflow_id, outcome
            )
        planner_result = await _maybe_handle_planner_message(
            request.message, request.session_id, session
        )
        if planner_result is not None:
            workflow_id, outcome = planner_result
            return await _record_workflow_outcome(
                request.session_id, request.message, workflow_id, outcome
            )
        route_command = await _route_message(request.message, request.session_id)
        if route_command.get("needs_clarification") and route_command.get("clarifying_question"):
            clarifying = route_command.get("clarifying_question")
            primary_intent = route_command.get("primary_intent") or "unknown"
            pipeline_store.update(request.session_id, workflow_id=primary_intent)
            await emit_workflow_log(
                "ROUTE",
                "clarify",
                meta={
                    "primary": route_command.get("primary_intent"),
                    "secondary": route_command.get("secondary_intents"),
                    "confidence": route_command.get("confidence"),
                    "reason": route_command.get("reason"),
                },
            )
            await emit_workflow_log("FLOW", "clarification_requested")
            await _append_user_message(request.session_id, request.message)
            await _append_assistant_message(request.session_id, clarifying)
            return {"assistant_message": clarifying}

        primary_intent = route_command.get("primary_intent") or "unknown"
        pipeline_store.update(request.session_id, workflow_id=primary_intent)
        await emit_workflow_log(
            "ROUTE",
            f"primary={primary_intent}",
            meta={
                "secondary": route_command.get("secondary_intents"),
                "confidence": route_command.get("confidence"),
                "reason": route_command.get("reason"),
            },
        )
        handler = WORKFLOW_HANDLERS.get(primary_intent)
        outcome: WorkflowOutcome | None = None
        if handler:
            outcome = await handler(request.message, request.session_id, route_command)
        if outcome is None:
            if primary_intent == "chat":
                outcome = await _handle_chat_workflow(request.message, session)
            else:
                outcome = await _handle_fallback_workflow(
                    request.message, request.session_id, session
                )
        return await _record_workflow_outcome(
            request.session_id, request.message, primary_intent, outcome
        )
    finally:
        current_session_id.reset(token)


async def _maybe_update_simulation_from_message(message: str, session_id: str) -> None:
    if not simulation_store.is_active(session_id):
        return
    clear_keys = _extract_clear_simulation_keys(message)
    parsed, conflicts = await extract_simulation_params_hybrid(message)
    if clear_keys:
        for key in clear_keys:
            parsed.pop(key, None)
    if not parsed and not conflicts and not clear_keys:
        return
    for key in conflicts:
        parsed.pop(key, None)
    current = simulation_store.get(session_id)
    missing_only = {key: value for key, value in parsed.items() if key not in current}
    if not missing_only and not conflicts and not clear_keys:
        return
    combined_clears = []
    for key in conflicts + clear_keys:
        if key and key not in combined_clears:
            combined_clears.append(key)
    update_result = collect_simulation_params(
        **missing_only,
        clear_keys=combined_clears,
        extra_missing=combined_clears,
    )
    await emit_simulation_form(
        update_result.get("params", {}),
        update_result.get("missing", []),
    )


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
MEMORY_SUMMARY_LONG_THRESHOLD = 420
SUMMARY_ID_EXCLUDE_PREFIXES = ("param", "lot", "lotid", "lot_id")
MEMORY_SUMMARY_SKIP_KEYS = {
    "top_candidates",
    "grid_results",
    "defect_rates",
    "rows",
    "row",
    "columns",
    "defect_stats",
    "post_grid_defects",
    "design_blocks",
}
BRIEFING_STREAM_CHUNK_SIZE = 80


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

    post_grid_defects = payload.get("post_grid_defects")
    if isinstance(post_grid_defects, dict):
        columns = post_grid_defects.get("columns")
        if isinstance(columns, list):
            token = _join_summary_tokens(columns)
            if token:
                items.append(("post_grid_cols", token))
        pg_items = post_grid_defects.get("items")
        if isinstance(pg_items, list):
            items.append(("post_grid_items", str(len(pg_items))))
        recent = _summary_token(post_grid_defects.get("recent_months"))
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


def _extract_summary_ids(message: str, limit: int = 6) -> list[str]:
    candidates = re.findall(r"\b[a-z0-9][a-z0-9-_]{5,}\b", message, re.IGNORECASE)
    tokens: list[str] = []
    for token in candidates:
        lowered = token.lower()
        if any(lowered.startswith(prefix) for prefix in SUMMARY_ID_EXCLUDE_PREFIXES):
            continue
        if token in tokens:
            continue
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def _build_message_memory_summary(message: str, label: str) -> str:
    compact = " ".join(message.split())
    preview = compact[:160]
    items = [f"label={label}"]

    lot_id = _extract_lot_id(message) or ""
    if lot_id:
        items.append(f"lot={lot_id}")
    ids = _extract_summary_ids(message)
    if ids:
        items.append(f"ids={','.join(ids)}")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", message)
    if numbers:
        items.append(f"nums={','.join(numbers[:6])}")
    if "추천" in message:
        items.append("intent=추천")
    items.append(f"preview={preview}")

    summary = "MSG_SUMMARY " + "; ".join(items)
    if len(summary) > MEMORY_SUMMARY_MAX_LEN:
        summary = summary[: MEMORY_SUMMARY_MAX_LEN - 3].rstrip() + "..."
    return summary


def _queue_memory_summary(
    session_id: str,
    workflow_id: str,
    message: str,
    explicit_summary: str | None = None,
) -> None:
    if not message:
        return
    state = pipeline_store.get(session_id)
    if state.get("pending_memory_summary"):
        return
    summary = explicit_summary
    if not summary and len(message) >= MEMORY_SUMMARY_LONG_THRESHOLD:
        summary = _build_message_memory_summary(message, workflow_id)
    if summary:
        pipeline_store.set_pending_memory_summary(
            session_id, summary, label=workflow_id
        )


async def _record_workflow_outcome(
    session_id: str,
    message: str,
    workflow_id: str,
    outcome: WorkflowOutcome,
) -> dict:
    await _append_user_message(session_id, message)
    if outcome.message:
        _queue_memory_summary(
            session_id,
            workflow_id,
            outcome.message,
            explicit_summary=outcome.memory_summary,
        )
        await _append_assistant_message(session_id, outcome.message)
    response_payload = {"assistant_message": outcome.message}
    if outcome.ui_event:
        response_payload["ui_event"] = outcome.ui_event
    if outcome.streamed:
        response_payload["streamed"] = True
    return response_payload


async def _handle_stage_view_workflow(
    message: str, session_id: str, route_command: dict
) -> WorkflowOutcome:
    await emit_workflow_log("FLOW", "stage_view -> show_simulation_stage")
    response = await _handle_stage_view_request(message, session_id, route_command)
    return WorkflowOutcome(response)


async def _handle_chart_edit_workflow(
    message: str, session_id: str, route_command: dict
) -> WorkflowOutcome:
    await emit_workflow_log("FLOW", "chart_edit -> apply_chart_config")
    response = await _handle_chart_request(message, session_id, route_command)
    return WorkflowOutcome(response)


async def _handle_simulation_edit_workflow(
    message: str, session_id: str, route_command: dict
) -> WorkflowOutcome | None:
    await emit_workflow_log("FLOW", "simulation_edit -> edit_intent")
    edit_response = await _maybe_handle_edit_intent(message, session_id)
    if edit_response is not None:
        message_text, ui_event = _split_edit_response(edit_response)
        streamed = _extract_streamed_flag(edit_response)
        return WorkflowOutcome(message_text, ui_event, streamed=streamed)
    await emit_workflow_log("FLOW", "simulation_edit -> pipeline")
    sim_response = await _maybe_handle_simulation_message(message, session_id)
    if sim_response is None:
        return None
    message_text, ui_event = _split_edit_response(sim_response)
    streamed = _extract_streamed_flag(sim_response)
    return WorkflowOutcome(message_text, ui_event, streamed=streamed)


async def _handle_simulation_run_workflow(
    message: str, session_id: str, route_command: dict
) -> WorkflowOutcome | None:
    await emit_workflow_log("FLOW", "simulation_run -> edit_intent")
    edit_response = await _maybe_handle_edit_intent(message, session_id)
    if edit_response is not None:
        message_text, ui_event = _split_edit_response(edit_response)
        streamed = _extract_streamed_flag(edit_response)
        return WorkflowOutcome(message_text, ui_event, streamed=streamed)
    await emit_workflow_log("FLOW", "simulation_run -> pipeline")
    sim_response = await _maybe_handle_simulation_message(message, session_id)
    if sim_response is None:
        return None
    message_text, ui_event = _split_edit_response(sim_response)
    streamed = _extract_streamed_flag(sim_response)
    return WorkflowOutcome(message_text, ui_event, streamed=streamed)


async def _handle_db_query_workflow(
    message: str, session_id: str, route_command: dict
) -> WorkflowOutcome | None:
    await emit_workflow_log("FLOW", "db_query -> db_agent")
    if not OPENAI_API_KEY:
        return WorkflowOutcome("DB 조회 기능이 비활성화되어 있습니다.")
    session = SQLiteSession(session_id, SESSION_DB_PATH)
    try:
        run = await Runner.run(
            db_agent,
            input=message,
            session=session,
            hooks=WorkflowRunHooks(),
        )
    except Exception:
        return WorkflowOutcome("DB 조회 중 오류가 발생했습니다.")
    result = _normalize_db_agent_output(run.final_output)
    status = result.get("status")
    summary = result.get("summary") or ""
    next_question = result.get("next") or ""
    if status == "missing" and next_question and next_question != "none":
        return WorkflowOutcome(next_question)
    if status == "error":
        return WorkflowOutcome(summary or "DB 조회에 실패했습니다.")
    if summary:
        return WorkflowOutcome(summary)
    return WorkflowOutcome("DB 조회를 완료했습니다.")


async def _handle_chat_workflow(
    message: str, session: SQLiteSession
) -> WorkflowOutcome:
    await emit_workflow_log("FLOW", "chat -> conversation_agent")
    try:
        convo = await Runner.run(
            conversation_agent,
            input=message,
            session=session,
            hooks=WorkflowRunHooks(),
        )
    except InputGuardrailTripwireTriggered:
        response = guardrail_fallback_message(["simulation_result", "db_result"])
        return WorkflowOutcome(response)
    except OutputGuardrailTripwireTriggered as exc:
        info = exc.guardrail_result.output.output_info or {}
        response = guardrail_fallback_message(info.get("missing_events"))
        return WorkflowOutcome(response)
    response = (convo.final_output or "").strip()
    return WorkflowOutcome(response)


async def _handle_fallback_workflow(
    message: str, session_id: str, session: SQLiteSession
) -> WorkflowOutcome:
    await emit_workflow_log("FLOW", "fallback -> conversation_agent")
    try:
        fallback = await Runner.run(
            conversation_agent,
            input=message,
            session=session,
            hooks=WorkflowRunHooks(),
        )
    except InputGuardrailTripwireTriggered:
        response = guardrail_fallback_message(["simulation_result", "db_result"])
        await _maybe_update_simulation_from_message(message, session_id)
        return WorkflowOutcome(response)
    except OutputGuardrailTripwireTriggered as exc:
        info = exc.guardrail_result.output.output_info or {}
        response = guardrail_fallback_message(info.get("missing_events"))
        await _maybe_update_simulation_from_message(message, session_id)
        return WorkflowOutcome(response)
    await _maybe_update_simulation_from_message(message, session_id)
    response = (fallback.final_output or "").strip()
    return WorkflowOutcome(response)


WORKFLOW_HANDLERS: dict[str, Callable[..., Awaitable[WorkflowOutcome | None]]] = {
    "stage_view": _handle_stage_view_workflow,
    "chart_edit": _handle_chart_edit_workflow,
    "simulation_edit": _handle_simulation_edit_workflow,
    "simulation_run": _handle_simulation_run_workflow,
    "db_query": _handle_db_query_workflow,
}


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


def _split_edit_response(edit_response: object) -> tuple[str, dict | None]:
    if isinstance(edit_response, dict):
        message = str(edit_response.get("message") or "")
        ui_event = edit_response.get("ui_event")
        if not isinstance(ui_event, dict):
            ui_event = None
        return message, ui_event
    return str(edit_response), None


def _extract_streamed_flag(edit_response: object) -> bool:
    if not isinstance(edit_response, dict):
        return False
    return bool(edit_response.get("streamed"))


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
        "If result is present, summarize recommended_chip_prod_id, representative_lot, "
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


async def _generate_edit_auto_message(
    action: str, context: dict, fallback: str
) -> str:
    if not OPENAI_API_KEY:
        return fallback

    pipeline_message = context.get("pipeline_message") if isinstance(context, dict) else ""
    pipeline_message = str(pipeline_message or "")
    table_blocks = _extract_markdown_table_blocks(pipeline_message)
    if pipeline_message and action in {"rerun", "update_grid", "update_reference"}:
        return pipeline_message

    payload = {"action": action, "context": context}
    prompt = (
        "다음 JSON을 참고해 한국어로 1~2문장 답변을 작성하세요. "
        "규칙: 도구/라우팅 언급 금지, 과한 추측 금지. "
        "missing_fields가 있으면 그 항목만 질문하세요. "
        "pipeline_message가 있으면 자연스럽게 요약하세요. "
        "pipeline_message 안에 마크다운 표가 있으면 표는 수정하지 말고 마지막에 그대로 포함하세요. "
        "result가 있으면 recommended_chip_prod_id, representative_lot, param_count를 짧게 언급하세요. "
        "ask_prediction=true 일 때만 추가 시뮬레이션 여부를 질문하세요. "
        f"JSON: {json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        run = await Runner.run(
            conversation_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
        message = (run.final_output or "").strip()
    except OutputGuardrailTripwireTriggered as exc:
        info = exc.guardrail_result.output.output_info or {}
        message = guardrail_fallback_message(info.get("missing_events"))
    except Exception:
        message = ""
    if not message:
        message = fallback
    if table_blocks:
        missing = [block for block in table_blocks if block not in message]
        if missing:
            message = f"{message.rstrip()}\n\n" + "\n\n".join(missing)
    return message


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


def _keyword_hits(message: str, tokens: tuple[str, ...]) -> list[str]:
    lowered = (message or "").lower()
    hits: list[str] = []
    for token in tokens:
        if token.lower() in lowered:
            hits.append(token)
    return hits


STAGE_NOUN_KEYWORDS = (
    "추천",
    "레퍼런스",
    "reference",
    "그리드",
    "grid",
    "최종",
    "final",
)
STAGE_ACTION_KEYWORDS = (
    "show",
    "replay",
    "stage",
    "화면",
    "보여",
    "다시",
    "이전",
    "직전",
    "방금",
)
CHART_KEYWORDS = (
    "chart",
    "graph",
    "histogram",
    "bin",
    "plot",
    "bar",
    "차트",
    "그래프",
    "히스토그램",
    "분포",
    "막대",
    "빈",
    "불량률",
)
SIM_RUN_KEYWORDS = (
    "추천",
    "인접",
    "시뮬",
    "simulation",
    "simulate",
    "예측",
    "실행",
    "시작",
)
SIM_EDIT_KEYWORDS = (
    "수정",
    "바꿔",
    "변경",
    "업데이트",
    "재실행",
    "다시 계산",
    "리셋",
    "reset",
    "rerun",
)
DB_KEYWORDS = (
    "lot",
    "db",
    "조회",
    "데이터",
    "공정",
    "라인",
    "설비",
    "불량",
    "품질",
    "조회해",
    "검색",
    "기록",
    "테이블",
    "컬럼",
    "view",
    "평균",
    "트렌드",
)


def _is_chart_request(message: str) -> bool:
    return _contains_any(message, CHART_KEYWORDS)


def _is_db_request(message: str) -> bool:
    return _contains_any(message, DB_KEYWORDS)


def _as_dict(value: object) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_db_agent_output(raw: object) -> dict:
    data = _as_dict(raw)
    if not data:
        text = str(raw or "").strip()
        return {
            "status": "ok" if text else "error",
            "summary": text,
            "missing": "none",
            "next": "none",
        }
    status = str(data.get("status") or "error").lower()
    if status not in {"ok", "missing", "error"}:
        status = "error"
    summary = str(data.get("summary") or "")
    missing = str(data.get("missing") or "")
    next_question = str(data.get("next") or "")
    return {
        "status": status,
        "summary": summary,
        "missing": missing,
        "next": next_question,
    }


def _normalize_route_command(raw: object) -> dict:
    data = _as_dict(raw)
    primary = str(data.get("primary_intent") or "unknown").lower()
    allowed = {
        "simulation_edit",
        "simulation_run",
        "stage_view",
        "db_query",
        "chart_edit",
        "chat",
        "unknown",
    }
    if primary not in allowed:
        primary = "unknown"

    secondary = data.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []
    filtered_secondary = [item for item in secondary if item in allowed and item != primary]

    needs_clarification = bool(data.get("needs_clarification"))
    clarifying_question = str(data.get("clarifying_question") or "")
    confidence = data.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    reason = str(data.get("reason") or "")
    stage = str(data.get("stage") or "unknown").lower()
    allowed_stages = {"recommendation", "reference", "grid", "final", "unknown"}
    if stage not in allowed_stages:
        stage = "unknown"
    return {
        "primary_intent": primary,
        "secondary_intents": filtered_secondary,
        "stage": stage,
        "needs_clarification": needs_clarification,
        "clarifying_question": clarifying_question,
        "confidence": confidence_value,
        "reason": reason,
    }


def _normalize_stage_resolution(raw: object) -> dict:
    data = _as_dict(raw)
    stage = str(data.get("stage") or "unknown").lower()
    allowed = {"recommendation", "reference", "grid", "final", "unknown"}
    if stage not in allowed:
        stage = "unknown"
    needs_clarification = bool(data.get("needs_clarification"))
    clarifying_question = str(data.get("clarifying_question") or "")
    confidence = data.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    return {
        "stage": stage,
        "needs_clarification": needs_clarification,
        "clarifying_question": clarifying_question,
        "confidence": confidence_value,
    }


def _fallback_route(message: str) -> dict:
    if _is_chart_request(message) and _is_db_request(message):
        return {
            "primary_intent": "db_query",
            "secondary_intents": [],
            "stage": "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "confidence": 0.5,
            "reason": "db_chart_keyword",
        }
    if _is_stage_view_request(message):
        return {
            "primary_intent": "stage_view",
            "secondary_intents": [],
            "stage": _extract_stage_label(message) or "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "confidence": 0.55,
            "reason": "stage_view_keyword",
        }
    if _is_chart_request(message):
        return {
            "primary_intent": "chart_edit",
            "secondary_intents": [],
            "stage": "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "confidence": 0.55,
            "reason": "keyword_match",
        }
    if _is_recommendation_intent(message):
        return {
            "primary_intent": "simulation_run",
            "secondary_intents": [],
            "stage": "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "confidence": 0.5,
            "reason": "simulation_keyword",
        }
    if _is_db_request(message):
        return {
            "primary_intent": "db_query",
            "secondary_intents": [],
            "stage": "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "confidence": 0.45,
            "reason": "db_keyword",
        }
    return {
        "primary_intent": "unknown",
        "secondary_intents": [],
        "stage": "unknown",
        "needs_clarification": False,
        "clarifying_question": "",
        "confidence": 0.2,
        "reason": "fallback",
    }


def _normalize_chart_decision(raw: object) -> dict:
    data = _as_dict(raw)
    chart_type = str(data.get("chart_type") or "auto").lower()
    if chart_type == "auto":
        chart_type = "histogram"
    bins = data.get("bins")
    try:
        bins_value = int(bins) if bins is not None else None
    except (TypeError, ValueError):
        bins_value = None
    range_min = data.get("range_min")
    range_max = data.get("range_max")
    try:
        range_min_value = float(range_min) if range_min is not None else None
    except (TypeError, ValueError):
        range_min_value = None
    try:
        range_max_value = float(range_max) if range_max is not None else None
    except (TypeError, ValueError):
        range_max_value = None
    normalize = str(data.get("normalize") or "auto").lower()
    if normalize == "auto":
        normalize = None
    value_unit = str(data.get("value_unit") or "auto").lower()
    if value_unit == "auto":
        value_unit = None
    reset = bool(data.get("reset"))
    note = str(data.get("note") or "")
    return {
        "chart_type": chart_type,
        "bins": bins_value,
        "range_min": range_min_value,
        "range_max": range_max_value,
        "normalize": normalize,
        "value_unit": value_unit,
        "reset": reset,
        "note": note,
    }


def _normalize_edit_decision(raw: object) -> dict:
    data = _as_dict(raw)
    intent = str(data.get("intent") or "none").lower()
    allowed = {
        "none",
        "update_params",
        "update_recommendation_params",
        "update_grid",
        "update_reference",
        "show_stage",
        "show_progress",
        "reset",
        "rerun",
        "new_simulation",
    }
    if intent not in allowed:
        intent = "none"

    updates = data.get("updates")
    updates = updates if isinstance(updates, dict) else {}
    clear_fields = data.get("clear_fields")
    if not isinstance(clear_fields, list):
        clear_fields = []
    clear_fields = [str(item) for item in clear_fields if item not in (None, "")]
    grid_overrides = data.get("grid_overrides")
    grid_overrides = grid_overrides if isinstance(grid_overrides, dict) else {}
    reference_lot_id = data.get("reference_lot_id")
    reference_lot_id = str(reference_lot_id).strip() if reference_lot_id else None

    stage = str(data.get("stage") or "unknown").lower()
    if stage not in {"recommendation", "reference", "grid", "final", "any", "unknown"}:
        stage = "unknown"

    rerun = bool(data.get("rerun"))
    needs_clarification = bool(data.get("needs_clarification"))
    note = str(data.get("note") or "")
    confidence = data.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    return {
        "intent": intent,
        "updates": updates,
        "clear_fields": clear_fields,
        "grid_overrides": grid_overrides,
        "reference_lot_id": reference_lot_id,
        "stage": stage,
        "rerun": rerun,
        "needs_clarification": needs_clarification,
        "note": note,
        "confidence": confidence_value,
    }


EDIT_INTENT_ALIAS_PATTERNS = [
    (
        r"(시트\s*t|시트티|시트\s*두께|성형\s*두께|성형두께|sheet\s*t|sheet\s*thickness)",
        "sheet_t",
    ),
    (r"(레이다운|레이\s*다운|lay\s*down|laydown|도포량|코팅량)", "laydown"),
    (r"(액티브\s*레이어|액티브|활성\s*층|활성층|active\s*layer)", "active_layer"),
    (r"(온도|공정\s*온도|소성\s*온도|temp|temperature)", "temperature"),
    (r"(전압|인가\s*전압|테스트\s*전압|정격\s*전압|volt|voltage)", "voltage"),
    (r"(사이즈|크기|치수|외형|l\s*/\s*w|l\s*x\s*w|lxw|길이|폭|size)", "size"),
    (r"(용량|정전\s*용량|정전용량|cap|capacitance|capacity)", "capacity"),
    (
        r"(기종명|기종|모델명|모델|제품명|타입|품번|part|chip_prod_id|model)",
        "chip_prod_id",
    ),
]


def _normalize_edit_intent_message(message: str) -> str:
    if not message:
        return ""
    normalized = message
    normalized = re.sub(
        r"(파라미터|parameter|param|p)\s*-?\s*([0-9]{1,2})",
        r"param\2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(양산|생산|mp|mass)",
        "production_mode 양산",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(개발|샘플|시제|proto|prototype|dev|pilot)",
        "production_mode 개발",
        normalized,
        flags=re.IGNORECASE,
    )
    for pattern, replacement in EDIT_INTENT_ALIAS_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"(temperature)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)(?:\s*°?\s*c)?",
        r"\1 \2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(voltage)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)(?:\s*v|volt)?",
        r"\1 \2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(size)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)(?:\s*(mm|cm|um|µm))?",
        r"\1 \2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(capacity)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)(?:\s*(pf|nf|uf|µf))?",
        r"\1 \2",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized


def _build_simulation_summary(session_id: str) -> dict:
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    rec = recommendation_store.get(session_id)
    rec_params = rec.get("params") if isinstance(rec, dict) else {}
    rec_param_count = len(rec_params) if isinstance(rec_params, dict) else 0
    state = pipeline_store.get(session_id)
    events = pipeline_store.get_events(session_id)
    status = pipeline_store.get_status(session_id)
    reference = state.get("reference") if isinstance(state, dict) else {}
    if not isinstance(reference, dict):
        reference = {}
    grid = state.get("grid") if isinstance(state, dict) else {}
    if not isinstance(grid, dict):
        grid = {}
    reference_lot_id = reference.get("lot_id") or reference.get("selected_lot_id")
    grid_overrides = grid.get("overrides")
    if not isinstance(grid_overrides, dict):
        grid_overrides = {}
    return {
        "params": params,
        "missing": missing,
        "simulation_active": simulation_store.is_active(session_id),
        "has_recommendation": bool(rec),
        "recommendation_param_count": rec_param_count,
        "events": list(events.keys()),
        "status": status,
        "last_stage_request": state.get("last_stage_request"),
        "reference_lot_id": reference_lot_id,
        "grid_overrides": grid_overrides,
    }


def _build_route_summary(session_id: str) -> dict:
    summary = _build_simulation_summary(session_id)
    missing = summary.get("missing", [])
    summary = dict(summary)
    summary["missing_count"] = len(missing) if isinstance(missing, list) else 0
    summary.pop("missing", None)
    return summary


PLANNER_WORKFLOW_CONFIG: dict[str, dict[str, list[str]]] = {
    "simulation_run": {
        "required_memory": [
            "sim.temperature",
            "sim.voltage",
            "sim.size",
            "sim.capacity",
            "sim.production_mode",
        ],
        "optional_memory": ["sim.chip_prod_id"],
        "success_criteria": ["events.simulation_result"],
    },
    "simulation_edit": {
        "required_memory": [],
        "optional_memory": [
            "sim.temperature",
            "sim.voltage",
            "sim.size",
            "sim.capacity",
            "sim.production_mode",
            "sim.chip_prod_id",
            "events.simulation_form",
            "events.simulation_result",
        ],
        "success_criteria": ["events.simulation_result", "last_stage_request"],
    },
    "db_query": {
        "required_memory": [],
        "optional_memory": [
            "sim.chip_prod_id",
            "events.simulation_result",
            "events.lot_result",
        ],
        "success_criteria": [
            "events.db_result",
            "events.lot_result",
            "events.defect_rate_chart",
        ],
    },
    "chart_edit": {
        "required_memory": ["events.defect_rate_chart"],
        "optional_memory": [],
        "success_criteria": ["events.defect_rate_chart", "chart_config"],
    },
    "stage_view": {
        "required_memory": [],
        "optional_memory": [
            "events.simulation_result",
            "events.lot_result",
            "events.defect_rate_chart",
            "events.design_candidates",
            "events.final_briefing",
        ],
        "success_criteria": ["last_stage_request"],
    },
    "briefing": {
        "required_memory": ["events.final_briefing"],
        "optional_memory": [
            "briefing_text",
            "briefing_summary",
            "events.simulation_result",
            "events.defect_rate_chart",
            "events.design_candidates",
        ],
        "success_criteria": ["events.final_briefing", "briefing_text", "briefing_summary"],
    },
    "chat": {"required_memory": [], "optional_memory": [], "success_criteria": []},
}

PLANNER_SELF_SUFFICIENT = {
    "simulation_run",
    "simulation_edit",
    "db_query",
    "stage_view",
    "chat",
}
PLANNER_MAX_STEPS = 6
PENDING_ACTION_TTL_SECONDS = 600

PLANNER_KNOWN_MEMORY_KEYS = {
    "reference",
    "grid",
    "overrides",
    "chart_config",
    "last_stage_request",
    "briefing_text",
    "briefing_summary",
    "recommendation",
    "recommendation.params",
    "recommendation.awaiting_prediction",
    "workflow_id",
}
for _config in PLANNER_WORKFLOW_CONFIG.values():
    PLANNER_KNOWN_MEMORY_KEYS.update(_config.get("required_memory", []))
    PLANNER_KNOWN_MEMORY_KEYS.update(_config.get("optional_memory", []))
    PLANNER_KNOWN_MEMORY_KEYS.update(_config.get("success_criteria", []))


def _collect_planner_memory_keys(session_id: str) -> set[str]:
    keys: set[str] = set()
    state = pipeline_store.get(session_id)
    events = pipeline_store.get_events(session_id)
    for event_type in events.keys():
        keys.add(f"events.{event_type}")
    for key in (
        "reference",
        "grid",
        "overrides",
        "chart_config",
        "last_stage_request",
        "briefing_text",
        "briefing_summary",
        "workflow_id",
    ):
        value = state.get(key)
        if value not in (None, "", {}, []):
            keys.add(key)
    sim_params = simulation_store.get(session_id)
    for key in sim_params.keys():
        keys.add(f"sim.{key}")
    recommendation = recommendation_store.get(session_id)
    if isinstance(recommendation, dict) and recommendation:
        keys.add("recommendation")
        params = recommendation.get("params")
        if isinstance(params, dict) and params:
            keys.add("recommendation.params")
        if recommendation.get("awaiting_prediction") is True:
            keys.add("recommendation.awaiting_prediction")
    return keys


def _planner_context_payload(session_id: str) -> dict:
    memory_keys = sorted(_collect_planner_memory_keys(session_id))
    sim_params = simulation_store.get(session_id)
    sim_missing = simulation_store.missing(session_id)
    events = pipeline_store.get_events(session_id)
    return {
        "memory_keys": memory_keys,
        "missing_sim_fields": sim_missing,
        "has_chip_prod_id": bool(sim_params.get("chip_prod_id")),
        "events": sorted(events.keys()),
    }


def _normalize_pending_action_definition(value: object) -> dict | None:
    data = _as_dict(value)
    if not data:
        return None
    workflow = str(data.get("workflow") or "").lower()
    if workflow not in PLANNER_WORKFLOW_CONFIG:
        return None
    return {
        "workflow": workflow,
        "input": str(data.get("input") or data.get("message") or ""),
        "required_memory": _normalize_memory_list(data.get("required_memory")),
        "optional_memory": _normalize_memory_list(data.get("optional_memory")),
        "success_criteria": _normalize_memory_list(data.get("success_criteria")),
    }


def _normalize_planner_decision(raw: object) -> dict:
    data = _as_dict(raw)
    goal = str(data.get("goal") or "")
    steps_raw = data.get("steps") if isinstance(data.get("steps"), list) else []
    steps: list[dict] = []
    allowed_workflows = set(PLANNER_WORKFLOW_CONFIG.keys())
    for index, item in enumerate(steps_raw, start=1):
        step_data = _as_dict(item)
        workflow = str(step_data.get("workflow") or "chat").lower()
        if workflow not in allowed_workflows:
            workflow = "chat"
        step_id = str(step_data.get("step_id") or f"step-{index}")
        status = str(step_data.get("status") or "pending").lower()
        if status not in {"pending", "running", "done", "blocked", "failed"}:
            status = "pending"
        steps.append(
            {
                "step_id": step_id,
                "workflow": workflow,
                "required_memory": step_data.get("required_memory") or [],
                "optional_memory": step_data.get("optional_memory") or [],
                "success_criteria": step_data.get("success_criteria") or [],
                "status": status,
                "notes": str(step_data.get("notes") or ""),
            }
        )
    missing_inputs = data.get("missing_inputs")
    if not isinstance(missing_inputs, list):
        missing_inputs = []
    next_action = str(data.get("next_action") or "run_step").lower()
    if next_action not in {"ask_user", "run_step", "finalize", "confirm"}:
        next_action = "run_step"
    confirmation_prompt = str(data.get("confirmation_prompt") or "")
    pending_action = _normalize_pending_action_definition(data.get("pending_action"))
    return {
        "goal": goal,
        "steps": steps,
        "missing_inputs": missing_inputs,
        "next_action": next_action,
        "confirmation_prompt": confirmation_prompt,
        "pending_action": pending_action,
    }


def _ensure_briefing_step(decision: dict) -> dict:
    steps = decision.get("steps") if isinstance(decision.get("steps"), list) else []
    if not steps:
        return decision
    if all(step.get("workflow") == "chat" for step in steps):
        return decision
    if any(step.get("workflow") == "briefing" for step in steps):
        return decision
    step_id = f"step-{len(steps) + 1}"
    steps.append(
        {
            "step_id": step_id,
            "workflow": "briefing",
            "required_memory": [],
            "optional_memory": [],
            "success_criteria": [],
            "status": "pending",
            "notes": "final briefing",
        }
    )
    decision["steps"] = steps
    return decision


def _normalize_memory_list(values: object) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        raw = [chunk.strip() for chunk in values.split(",")]
    elif isinstance(values, list):
        raw = [str(item).strip() for item in values]
    else:
        return []
    return [item for item in raw if item]


def _apply_planner_defaults(decision: dict, memory_keys: set[str]) -> dict:
    allowed_keys = PLANNER_KNOWN_MEMORY_KEYS | set(memory_keys)
    normalized_steps: list[dict] = []
    for step in decision.get("steps", []):
        workflow = str(step.get("workflow") or "chat")
        config = PLANNER_WORKFLOW_CONFIG.get(workflow, {})
        required = _normalize_memory_list(step.get("required_memory"))
        optional = _normalize_memory_list(step.get("optional_memory"))
        success = _normalize_memory_list(step.get("success_criteria"))
        if not required:
            required = list(config.get("required_memory", []))
        if not optional:
            optional = list(config.get("optional_memory", []))
        if not success:
            success = list(config.get("success_criteria", []))
        required = [key for key in required if key in allowed_keys]
        optional = [key for key in optional if key in allowed_keys]
        success = [key for key in success if key in allowed_keys]
        step["required_memory"] = required
        step["optional_memory"] = optional
        step["success_criteria"] = success
        normalized_steps.append(step)
    decision["steps"] = normalized_steps
    return decision


def _planner_missing_required(step: dict, memory_keys: set[str]) -> list[str]:
    required = step.get("required_memory") if isinstance(step, dict) else []
    required_list = required if isinstance(required, list) else []
    missing = [key for key in required_list if key not in memory_keys]
    if step.get("workflow") in {"simulation_run", "simulation_edit"}:
        if "sim.chip_prod_id" in memory_keys:
            missing = [key for key in missing if not key.startswith("sim.")]
    return missing


def _planner_step_done(step: dict, memory_keys: set[str]) -> bool:
    if step.get("workflow") == "briefing":
        return False
    success = step.get("success_criteria") if isinstance(step, dict) else []
    success_list = success if isinstance(success, list) else []
    if not success_list:
        return False
    return any(key in memory_keys for key in success_list)


def _evaluate_planner_decision(decision: dict, memory_keys: set[str]) -> tuple[dict, dict | None]:
    steps = decision.get("steps") if isinstance(decision.get("steps"), list) else []
    missing_inputs: list[str] = []
    next_step: dict | None = None
    next_missing: list[str] = []
    for step in steps:
        missing = _planner_missing_required(step, memory_keys)
        if _planner_step_done(step, memory_keys):
            step["status"] = "done"
        elif missing:
            step["status"] = "blocked"
        else:
            step["status"] = "pending"
        if next_step is None and step["status"] != "done":
            next_step = step
            next_missing = missing

    if next_step is None:
        decision["missing_inputs"] = []
        decision["next_action"] = "finalize"
        return decision, None

    if next_missing and next_step.get("workflow") not in PLANNER_SELF_SUFFICIENT:
        decision["next_action"] = "ask_user"
        missing_inputs = next_missing
    else:
        decision["next_action"] = "run_step"
        missing_inputs = next_missing

    decision["missing_inputs"] = missing_inputs
    return decision, next_step


def _missing_sim_fields_from_keys(keys: list[str]) -> list[str]:
    fields: list[str] = []
    for key in keys:
        if key.startswith("sim."):
            fields.append(key.split(".", 1)[1])
    return fields


async def _planner_missing_message(
    session_id: str, step: dict | None, missing_inputs: list[str]
) -> WorkflowOutcome:
    sim_fields = _missing_sim_fields_from_keys(missing_inputs)
    if sim_fields:
        missing_text = _format_missing_fields(sim_fields)
        params = simulation_store.get(session_id)
        form_payload = await emit_simulation_form(params, sim_fields)
        message = f"추천을 실행하려면 {missing_text} 값을 알려주세요."
        return WorkflowOutcome(
            message,
            ui_event={"type": "simulation_form", "payload": form_payload},
        )
    if "events.defect_rate_chart" in missing_inputs:
        return WorkflowOutcome(
            "차트를 변경하려면 먼저 불량률 데이터가 필요합니다. 추천이나 DB 조회를 먼저 진행할까요?"
        )
    if "events.final_briefing" in missing_inputs:
        return WorkflowOutcome(
            "브리핑할 데이터가 없습니다. 먼저 추천이나 DB 조회를 진행해 주세요."
        )
    if missing_inputs:
        missing_text = ", ".join(missing_inputs)
        return WorkflowOutcome(f"추가 정보가 필요합니다: {missing_text}")
    return WorkflowOutcome("추가 정보를 알려주세요.")


async def _emit_planner_status(
    step_index: int,
    total_steps: int,
    workflow: str,
    status: str,
    done: bool = False,
) -> None:
    message = f"Planner {step_index}/{total_steps} {workflow} {status}"
    await _emit_pipeline_status("planner", message, done=done)


async def _handle_planner_briefing(session_id: str) -> WorkflowOutcome:
    state = pipeline_store.get(session_id)
    briefing_text = state.get("briefing_text") if isinstance(state, dict) else ""
    if isinstance(briefing_text, str) and briefing_text.strip():
        return WorkflowOutcome(briefing_text)

    payload = pipeline_store.get_event(session_id, "final_briefing")
    if not isinstance(payload, dict) or not payload:
        return WorkflowOutcome("브리핑할 데이터가 아직 없습니다.")

    if not OPENAI_API_KEY:
        chip_prod_id = payload.get("chip_prod_id")
        reference_lot = payload.get("reference_lot")
        candidate_total = payload.get("candidate_total")
        defect_rate = payload.get("defect_rate_overall")
        parts = ["요청하신 브리핑 요약입니다."]
        if chip_prod_id:
            parts.append(f"추천 기종={chip_prod_id}")
        if reference_lot:
            parts.append(f"레퍼런스 LOT={reference_lot}")
        if candidate_total is not None:
            parts.append(f"후보군={candidate_total}건")
        if defect_rate is not None:
            try:
                defect_value = float(defect_rate)
            except (TypeError, ValueError):
                defect_value = None
            if defect_value is not None:
                parts.append(f"평균 불량률={defect_value:.4f}")
        return WorkflowOutcome(" ".join(str(part) for part in parts))

    context_payload = {
        "final_briefing": payload,
        "simulation_result": pipeline_store.get_event(session_id, "simulation_result"),
        "db_result": pipeline_store.get_event(session_id, "db_result"),
        "defect_rate_chart": pipeline_store.get_event(session_id, "defect_rate_chart"),
        "lot_result": pipeline_store.get_event(session_id, "lot_result"),
        "design_candidates": pipeline_store.get_event(session_id, "design_candidates"),
        "reference": state.get("reference") if isinstance(state, dict) else None,
        "grid": state.get("grid") if isinstance(state, dict) else None,
    }
    prompt = (
        "다음 JSON을 근거로만 MLCC 브리핑을 작성하세요. "
        "근거 없는 수치/LOT/공정 값은 말하지 마세요. "
        "3~6문장으로 간결하게 요약하세요. "
        "JSON: "
        f"{json.dumps(context_payload, ensure_ascii=False)}"
    )
    try:
        run = await Runner.run(
            briefing_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
    except OutputGuardrailTripwireTriggered as exc:
        info = exc.guardrail_result.output.output_info or {}
        response = guardrail_fallback_message(info.get("missing_events"))
        return WorkflowOutcome(response)
    except Exception:
        return WorkflowOutcome("브리핑 생성 중 오류가 발생했습니다.")

    response = (run.final_output or "").strip()
    if response:
        memory_summary = _build_memory_summary(payload)
        pipeline_store.update(
            session_id,
            briefing_text=response,
            briefing_summary=memory_summary,
        )
        pipeline_store.set_pending_memory_summary(
            session_id, memory_summary, label="final_briefing"
        )
        return WorkflowOutcome(response, memory_summary=memory_summary)

    return WorkflowOutcome("브리핑을 생성하지 못했습니다.")


async def _run_planner_step(
    step: dict, message: str, session_id: str, session: SQLiteSession
) -> WorkflowOutcome:
    workflow = str(step.get("workflow") or "chat")
    route_command = {
        "primary_intent": workflow,
        "secondary_intents": [],
        "stage": "unknown",
        "needs_clarification": False,
        "clarifying_question": "",
        "confidence": 0.0,
        "reason": "planner",
    }
    handler = WORKFLOW_HANDLERS.get(workflow)
    if handler:
        outcome = await handler(message, session_id, route_command)
        if outcome is not None:
            return outcome
        if workflow == "chat":
            return await _handle_chat_workflow(message, session)
        return await _handle_fallback_workflow(message, session_id, session)
    if workflow == "chat":
        return await _handle_chat_workflow(message, session)
    if workflow == "briefing":
        return await _handle_planner_briefing(session_id)
    return await _handle_fallback_workflow(message, session_id, session)


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _pending_action_expired(pending: dict) -> bool:
    now = datetime.utcnow()
    expires_at = _parse_iso_timestamp(str(pending.get("expires_at") or ""))
    if expires_at:
        return expires_at <= now
    created_at = _parse_iso_timestamp(str(pending.get("created_at") or ""))
    if created_at:
        return created_at <= (now - timedelta(seconds=PENDING_ACTION_TTL_SECONDS))
    return False


def _normalize_pending_action(session_id: str) -> dict | None:
    state = pipeline_store.get(session_id)
    pending = state.get("pending_action")
    if not isinstance(pending, dict):
        return None
    if _pending_action_expired(pending):
        _clear_pending_action(session_id)
        return None
    workflow = str(pending.get("workflow") or "").strip().lower()
    if not workflow or workflow not in PLANNER_WORKFLOW_CONFIG:
        _clear_pending_action(session_id)
        return None
    required = _normalize_memory_list(pending.get("required_memory"))
    optional = _normalize_memory_list(pending.get("optional_memory"))
    success = _normalize_memory_list(pending.get("success_criteria"))
    return {
        "id": str(pending.get("id") or "pending"),
        "workflow": workflow,
        "input": str(pending.get("input") or pending.get("message") or ""),
        "required_memory": required,
        "optional_memory": optional,
        "success_criteria": success,
    }


def _clear_pending_action(session_id: str) -> None:
    pipeline_store.update(
        session_id,
        pending_action=None,
        pending_plan=None,
        pending_inputs=[],
        dialogue_state="idle",
    )


def _default_confirmation_prompt(workflow: str) -> str:
    prompts = {
        "simulation_run": "추천 시뮬레이션을 진행할까요?",
        "simulation_edit": "수정된 조건으로 다시 진행할까요?",
        "db_query": "DB 조회를 진행할까요?",
        "chart_edit": "차트를 변경해도 될까요?",
        "stage_view": "요청하신 화면을 보여드릴까요?",
        "briefing": "브리핑을 진행할까요?",
    }
    return prompts.get(workflow, "이 작업을 진행할까요?")


def _build_pending_action_from_step(step: dict | None, message: str) -> dict | None:
    if not isinstance(step, dict):
        return None
    workflow = str(step.get("workflow") or "").lower()
    if workflow not in PLANNER_WORKFLOW_CONFIG:
        return None
    return {
        "workflow": workflow,
        "input": message,
        "required_memory": _normalize_memory_list(step.get("required_memory")),
        "optional_memory": _normalize_memory_list(step.get("optional_memory")),
        "success_criteria": _normalize_memory_list(step.get("success_criteria")),
    }


def _store_pending_action(
    session_id: str,
    pending: dict,
    confirmation_prompt: str,
    plan: dict | None,
) -> None:
    created_at = datetime.utcnow()
    expires_at = created_at + timedelta(seconds=PENDING_ACTION_TTL_SECONDS)
    pending_payload = {
        "id": str(pending.get("id") or f"pending-{uuid.uuid4().hex[:8]}"),
        "workflow": pending.get("workflow"),
        "input": pending.get("input"),
        "required_memory": pending.get("required_memory") or [],
        "optional_memory": pending.get("optional_memory") or [],
        "success_criteria": pending.get("success_criteria") or [],
        "confirmation_prompt": confirmation_prompt,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }
    pipeline_store.update(
        session_id,
        pending_action=_json_safe_dict(pending_payload),
        pending_plan=_json_safe_dict(plan or {}),
        pending_inputs=[],
        dialogue_state="awaiting_confirmation",
        planner_state=_json_safe_dict(plan or {}),
        planner_goal=str((plan or {}).get("goal") or ""),
    )


def _is_rejection(message: str) -> bool:
    lowered = (message or "").strip().lower()
    if not lowered:
        return False
    tokens = {
        "no",
        "nope",
        "nah",
        "아니",
        "아니요",
        "싫어",
        "안해",
        "취소",
        "그만",
        "다음에",
        "보류",
        "패스",
    }
    if lowered in tokens:
        return True
    return any(token in lowered for token in tokens)


async def _maybe_handle_pending_action(
    message: str, session_id: str, session: SQLiteSession
) -> tuple[str, WorkflowOutcome] | None:
    pending = _normalize_pending_action(session_id)
    if not pending:
        return None
    if _is_affirmative(message):
        pipeline_store.update(session_id, dialogue_state="executing")
        workflow = pending["workflow"]
        step = {
            "step_id": pending.get("id") or "pending",
            "workflow": workflow,
            "required_memory": pending.get("required_memory") or [],
            "optional_memory": pending.get("optional_memory") or [],
            "success_criteria": pending.get("success_criteria") or [],
            "status": "running",
            "notes": "pending_action",
        }
        input_message = pending.get("input") or message
        outcome = await _run_planner_step(step, input_message, session_id, session)
        _clear_pending_action(session_id)
        return workflow, outcome
    if _is_rejection(message):
        _clear_pending_action(session_id)
        return "chat", WorkflowOutcome("알겠습니다. 다른 요청이 있으면 알려주세요.")
    _clear_pending_action(session_id)
    return None


async def _maybe_handle_planner_message(
    message: str, session_id: str, session: SQLiteSession
) -> tuple[str, WorkflowOutcome] | None:
    if not OPENAI_API_KEY:
        return None
    context_payload = _planner_context_payload(session_id)
    prompt = (
        "Return JSON only. Create a concise execution plan. "
        "Keys: goal, steps, missing_inputs, next_action, confirmation_prompt, pending_action. "
        "Each step: step_id, workflow, required_memory, optional_memory, success_criteria, status, notes. "
        "Notes must be short. "
        "If next_action=confirm, include confirmation_prompt and pending_action (workflow + memory criteria). "
        f"Context JSON: {json.dumps(context_payload, ensure_ascii=False)}\n"
        f"User message: {message}"
    )
    try:
        run = await Runner.run(
            planner_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
    except Exception:
        return None
    decision = _normalize_planner_decision(run.final_output)
    if not decision.get("steps"):
        return None

    decision = _ensure_briefing_step(decision)
    memory_keys = _collect_planner_memory_keys(session_id)
    decision = _apply_planner_defaults(decision, memory_keys)
    if decision.get("next_action") == "confirm" or decision.get("pending_action"):
        pending = decision.get("pending_action")
        if not pending:
            _, next_step = _evaluate_planner_decision(decision, memory_keys)
            pending = _build_pending_action_from_step(next_step, message)
        if pending:
            confirmation_prompt = decision.get("confirmation_prompt") or ""
            if not confirmation_prompt:
                confirmation_prompt = _default_confirmation_prompt(pending.get("workflow", ""))
            _store_pending_action(session_id, pending, confirmation_prompt, decision)
            return "planner", WorkflowOutcome(confirmation_prompt)
    pipeline_store.update(
        session_id,
        planner_state=_json_safe_dict(decision),
        planner_goal=decision.get("goal") or "",
        planner_batch=True,
    )
    last_outcome: WorkflowOutcome | None = None
    last_workflow = "planner"
    steps = decision.get("steps", [])
    total_steps = min(len(steps), PLANNER_MAX_STEPS)
    try:
        for index, step in enumerate(steps[:total_steps], start=1):
            memory_keys = _collect_planner_memory_keys(session_id)
            decision = _apply_planner_defaults(decision, memory_keys)
            workflow = str(step.get("workflow") or "planner")
            pipeline_store.update(session_id, workflow_id=workflow)

            if _planner_step_done(step, memory_keys):
                await _emit_planner_status(index, total_steps, workflow, "skip")
                continue

            missing = _planner_missing_required(step, memory_keys)
            if missing and workflow not in PLANNER_SELF_SUFFICIENT:
                await _emit_planner_status(index, total_steps, workflow, "blocked")
                last_outcome = await _planner_missing_message(session_id, step, missing)
                last_workflow = workflow
                break

            await _emit_planner_status(index, total_steps, workflow, "start")
            outcome = await _run_planner_step(step, message, session_id, session)
            last_outcome = outcome
            last_workflow = workflow

            memory_keys = _collect_planner_memory_keys(session_id)
            if _planner_step_done(step, memory_keys):
                await _emit_planner_status(
                    index, total_steps, workflow, "done", done=index == total_steps
                )
                continue

            missing_after = _planner_missing_required(step, memory_keys)
            if missing_after:
                await _emit_planner_status(index, total_steps, workflow, "blocked")
                if outcome and outcome.message:
                    last_outcome = outcome
                else:
                    last_outcome = await _planner_missing_message(
                        session_id, step, missing_after
                    )
            break
    finally:
        pipeline_store.update(session_id, planner_batch=False)

    decision, _ = _evaluate_planner_decision(
        decision, _collect_planner_memory_keys(session_id)
    )
    pipeline_store.update(session_id, planner_state=_json_safe_dict(decision))

    if last_outcome is None:
        return None
    return last_workflow, last_outcome


async def _route_message(message: str, session_id: str) -> dict:
    if not OPENAI_API_KEY:
        return _fallback_route(message)
    summary = _build_route_summary(session_id)
    keyword_hints = {
        "stage_keyword_hits": _keyword_hits(message, STAGE_NOUN_KEYWORDS),
        "stage_action_hits": _keyword_hits(message, STAGE_ACTION_KEYWORDS),
        "chart_keyword_hits": _keyword_hits(message, CHART_KEYWORDS),
        "simulation_run_hits": _keyword_hits(message, SIM_RUN_KEYWORDS),
        "simulation_edit_hits": _keyword_hits(message, SIM_EDIT_KEYWORDS),
        "db_keyword_hits": _keyword_hits(message, DB_KEYWORDS),
    }
    meta = {
        "simulation_active": simulation_store.is_active(session_id),
        "summary": summary,
        "keyword_hints": keyword_hints,
    }
    prompt = (
        "Return JSON only. Decide the primary intent and any secondary intents. "
        "Output keys: primary_intent, secondary_intents, needs_clarification, clarifying_question, confidence, reason. "
        "Intents: simulation_run, simulation_edit, stage_view, db_query, chart_edit, chat, unknown. "
        "Guidelines: "
        "- chart_edit when user wants to change or redraw charts/graphs/histograms. "
        "- simulation_edit when user wants to modify existing simulation inputs/results, "
        "replay stages, check progress, or reset. "
        "- stage_view when user asks to show/replay a stage screen "
        "(추천/레퍼런스/그리드/최종/화면). If stage_view, set stage to "
        "recommendation/reference/grid/final if possible. "
        "- simulation_run when user wants to start/restart a simulation with new inputs. "
        "- Do not set needs_clarification just because params are missing; "
        "for simulation_run, proceed and let the pipeline collect missing inputs. "
        "- db_query for LOT/process database lookups (including averages/trends/graphs from DB). "
        "- chat for casual or unrelated conversation. "
        "Use keyword_hints as soft signals: "
        "1) If stage_keyword_hits or stage_action_hits are present and chart_keyword_hits are empty, "
        "prefer stage_view. "
        "2) If both stage_* and chart_keyword_hits are present, ask for clarification. "
        "3) If simulation_active and simulation_edit_hits are present, prefer simulation_edit. "
        "4) If simulation_run_hits are present and simulation_active is false, prefer simulation_run. "
        "5) If db_keyword_hits are present and simulation keywords are weak, prefer db_query. "
        "If ambiguous between db_query and simulation_edit, ask a short Korean clarification. "
        f"Context: {json.dumps(meta, ensure_ascii=False)}\n"
        f"User message: {message}"
    )
    try:
        run = await Runner.run(
            route_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
        return _normalize_route_command(run.final_output)
    except Exception:
        return _fallback_route(message)


async def _heuristic_chart_update(message: str, session_id: str) -> str:
    if not pipeline_store.get_event(session_id, "defect_rate_chart"):
        return "\uc544\uc9c1 \ubd88\ub7c9\ub960 \ucc28\ud2b8 \ub370\uc774\ud130\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. \uba3c\uc800 \uc2dc\ubbac\ub808\uc774\uc158\uc744 \uc2e4\ud589\ud574 \uc8fc\uc138\uc694."

    reset = bool(re.search(r"\ucd08\uae30\ud654|\uae30\ubcf8|reset", message, re.IGNORECASE))
    chart_type = "histogram"
    if re.search(r"line|\uc120", message, re.IGNORECASE):
        chart_type = "line"
    elif re.search(r"scatter|\uc0c1\uc810", message, re.IGNORECASE):
        chart_type = "scatter"
    elif re.search(r"bar|\ub9c9\ub300", message, re.IGNORECASE):
        chart_type = "bar"
    bins = None
    bins_match = re.search(r"(\d+)\s*(?:bins?|bin|\uad6c\uac04|\uac1c)", message, re.IGNORECASE)
    if bins_match:
        try:
            bins = int(bins_match.group(1))
        except ValueError:
            bins = None

    range_min = None
    range_max = None
    range_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*%?\s*[-~]\s*([-+]?\d+(?:\.\d+)?)\s*%?",
        message,
        re.IGNORECASE,
    )
    if range_match:
        try:
            range_min = float(range_match.group(1))
            range_max = float(range_match.group(2))
        except ValueError:
            range_min = None
            range_max = None

    normalize = None
    if re.search(r"\ube44\uc728|percent|%|\ud37c\uc13c\ud2b8", message, re.IGNORECASE):
        normalize = "percent"
    elif re.search(r"\ube44\uc911|ratio|\uc815\uaddc\ud654", message, re.IGNORECASE):
        normalize = "ratio"
    elif re.search(r"count|\uac1c\uc218", message, re.IGNORECASE):
        normalize = "count"

    value_unit = None
    if re.search(r"%|\ud37c\uc13c\ud2b8", message, re.IGNORECASE):
        value_unit = "percent"

    result = await apply_chart_config_impl(
        chart_type=chart_type,
        bins=bins,
        range_min=range_min,
        range_max=range_max,
        normalize=normalize,
        value_unit=value_unit,
        reset=reset,
    )
    status = result.get("status")
    if status == "ok":
        return "\ubd88\ub7c9\ub960 \ucc28\ud2b8\ub97c \uc694\uccad\ud558\uc2e0 \ud615\ud0dc\ub85c \uc5c5\ub370\uc774\ud2b8\ud588\uc5b4\uc694."
    return result.get("message") or "\ucc28\ud2b8 \uc5c5\ub370\uc774\ud2b8\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4."


async def _handle_chart_request(
    message: str, session_id: str, route_command: dict
) -> str:
    if not OPENAI_API_KEY:
        return await _heuristic_chart_update(message, session_id)

    state = pipeline_store.get(session_id)
    context_payload = {
        "route": route_command,
        "chart_config": state.get("chart_config") or {},
        "has_defect_chart": bool(pipeline_store.get_event(session_id, "defect_rate_chart")),
    }
    prompt = (
        "Extract chart update intent from the user message. "
        "Return only JSON with chart_type, bins, range_min, range_max, normalize, value_unit, reset, note. "
        f"Context JSON: {json.dumps(context_payload, ensure_ascii=False)}\n"
        f"User message: {message}"
    )
    try:
        run = await Runner.run(
            chart_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
        decision = _normalize_chart_decision(run.final_output)
        result = await apply_chart_config_impl(**decision)
        if result.get("status") == "ok":
            return "차트를 업데이트했어요."
        return result.get("message") or "차트 업데이트에 실패했습니다."
    except Exception:
        pass
    return await _heuristic_chart_update(message, session_id)


async def _apply_edit_decision(
    decision: dict, message: str, session_id: str
) -> str | None:
    intent = decision.get("intent")
    if intent == "none":
        return None

    if intent == "reset":
        await reset_simulation_state_impl(reason=message)
        missing = simulation_store.missing(session_id)
        await emit_simulation_form({}, missing)
        missing_text = _format_missing_fields(missing)
        fallback = f"시뮬레이션을 초기화했어요. 필요한 값: {missing_text}"
        return await _generate_edit_auto_message(
            "reset",
            {"missing_fields": missing, "ask_prediction": False},
            fallback,
        )

    if intent == "new_simulation":
        await reset_simulation_state_impl(reason=message)
        missing = simulation_store.missing(session_id)
        form_payload = await emit_simulation_form({}, missing)
        missing_text = _format_missing_fields(missing)
        fallback = f"새 시뮬레이션을 시작했어요. 필요한 값: {missing_text}"
        message_text = await _generate_edit_auto_message(
            "new_simulation",
            {"missing_fields": missing, "ask_prediction": False},
            fallback,
        )
        return {
            "message": message_text,
            "ui_event": {"type": "simulation_form", "payload": form_payload},
        }

    if intent == "show_progress":
        progress = await get_simulation_progress_impl()
        history = progress.get("history", []) if isinstance(progress, dict) else []
        fallback = "진행 상태를 이벤트 로그에 표시했어요."
        return await _generate_edit_auto_message(
            "show_progress",
            {"history": history[-6:], "ask_prediction": False},
            fallback,
        )

    if intent == "show_stage":
        stage = decision.get("stage") or ""
        if stage in {"any", "unknown"}:
            stage = message
        result = await show_simulation_stage_impl(stage)
        fallback = result.get("message") or "요청하신 단계를 다시 표시했어요."
        return await _generate_edit_auto_message(
            "show_stage",
            {"stage": stage, "result": result, "ask_prediction": False},
            fallback,
        )

    updates = decision.get("updates") if isinstance(decision.get("updates"), dict) else {}
    clear_fields = (
        decision.get("clear_fields")
        if isinstance(decision.get("clear_fields"), list)
        else []
    )
    grid_overrides = (
        decision.get("grid_overrides")
        if isinstance(decision.get("grid_overrides"), dict)
        else {}
    )

    if intent == "update_recommendation_params":
        current = recommendation_store.get(session_id)
        if not current:
            fallback = "추천 결과가 아직 없습니다. 먼저 추천을 실행해 주세요."
            return await _generate_edit_auto_message(
                "update_recommendation_params",
                {"missing_fields": ["추천 결과"], "ask_prediction": False},
                fallback,
            )
        params = {
            key: value
            for key, value in updates.items()
            if isinstance(key, str) and key.startswith("param")
        }
        clear_param_keys = [
            key for key in clear_fields if isinstance(key, str) and key.startswith("param")
        ]
        updated = None
        if clear_param_keys:
            updated = recommendation_store.remove_params(session_id, clear_param_keys)
        if params:
            updated = recommendation_store.update_params(session_id, params)
        if updated is None:
            updated = current
        result_payload = {
            "recommended_chip_prod_id": updated.get("recommended_chip_prod_id"),
            "representative_lot": updated.get("representative_lot"),
            "param_count": len(updated.get("params") or {}),
        }
        payload = {"params": updated.get("input_params") or {}, "result": result_payload}
        await event_bus.broadcast({"type": "simulation_result", "payload": payload})
        pipeline_store.set_event(session_id, "simulation_result", payload)
        param_count = result_payload.get("param_count", 0)
        chip_prod_id = result_payload.get("recommended_chip_prod_id")
        rep_lot = result_payload.get("representative_lot")
        fallback_parts = ["추천 파라미터를 갱신했어요."]
        if chip_prod_id:
            fallback_parts.append(f"추천 기종={chip_prod_id}")
        if rep_lot:
            fallback_parts.append(f"대표 LOT={rep_lot}")
        fallback_parts.append(f"파라미터={param_count}개")
        fallback = " ".join(fallback_parts)
        return await _generate_edit_auto_message(
            "update_recommendation_params",
            {"result": result_payload, "ask_prediction": False},
            fallback,
        )

    if intent in {"update_params", "rerun"}:
        sim_updates = {
            key: value
            for key, value in updates.items()
            if key in {
                "temperature",
                "voltage",
                "size",
                "capacity",
                "production_mode",
                "chip_prod_id",
            }
        }
        clear_sim_keys = [
            key
            for key in clear_fields
            if key
            in {
                "temperature",
                "voltage",
                "size",
                "capacity",
                "production_mode",
                "chip_prod_id",
            }
        ]
        if sim_updates or clear_sim_keys:
            update_result = collect_simulation_params(
                **sim_updates,
                clear_keys=clear_sim_keys,
                extra_missing=clear_sim_keys,
            )
            await emit_simulation_form(
                update_result.get("params", {}), update_result.get("missing", [])
            )
            if update_result.get("missing") and not update_result.get("params", {}).get(
                "chip_prod_id"
            ):
                missing = update_result.get("missing", [])
                missing_text = _format_missing_fields(missing)
                fallback = f"추천을 실행하려면 {missing_text} 값을 알려주세요."
                return await _generate_edit_auto_message(
                    "update_params_missing",
                    {"missing_fields": missing, "ask_prediction": False},
                    fallback,
                )
            if decision.get("rerun"):
                state = pipeline_store.get(session_id)
                grid_state = state.get("grid") if isinstance(state, dict) else {}
                if not isinstance(grid_state, dict):
                    grid_state = {}
                stored_overrides = grid_state.get("overrides")
                if not isinstance(stored_overrides, dict):
                    stored_overrides = {}
                result = await _run_reference_pipeline(
                    session_id,
                    update_result.get("params", {}),
                    chip_prod_override=update_result.get("params", {}).get("chip_prod_id"),
                    grid_overrides=stored_overrides,
                )
                sim_result = (
                    result.get("simulation_result", {}).get("result")
                    if isinstance(result.get("simulation_result"), dict)
                    else None
                )
                result_payload = None
                if isinstance(sim_result, dict):
                    result_payload = {
                        "recommended_chip_prod_id": sim_result.get(
                            "recommended_chip_prod_id"
                        ),
                        "representative_lot": sim_result.get("representative_lot"),
                        "param_count": len(sim_result.get("params") or {}),
                    }
                fallback = result.get("message") or "요청하신 조건으로 다시 계산했습니다."
                return await _generate_edit_auto_message(
                    "rerun",
                    {
                        "pipeline_message": result.get("message", ""),
                        "result": result_payload,
                        "ask_prediction": False,
                    },
                    fallback,
                )
            missing = update_result.get("missing", [])
            missing_text = _format_missing_fields(missing)
            fallback = (
                f"입력을 갱신했어요. {missing_text} 값을 추가로 알려주세요."
                if missing
                else "입력을 갱신했어요. 이어서 추천을 진행할까요?"
            )
            return await _generate_edit_auto_message(
                "update_params",
                {"missing_fields": missing, "ask_prediction": False},
                fallback,
            )

    if intent == "update_grid":
        overrides = {}
        for key in ("sheet_t", "laydown", "active_layer"):
            value = grid_overrides.get(key)
            if value is None:
                continue
            try:
                overrides[key] = float(value)
            except (TypeError, ValueError):
                continue
        params = simulation_store.get(session_id)
        if not params:
            missing = simulation_store.missing(session_id)
            missing_text = _format_missing_fields(missing)
            fallback = f"시뮬레이션 입력이 부족합니다. {missing_text} 값을 알려주세요."
            return await _generate_edit_auto_message(
                "update_grid_missing",
                {"missing_fields": missing, "ask_prediction": False},
                fallback,
            )
        result = await _run_reference_pipeline(
            session_id,
            params,
            chip_prod_override=params.get("chip_prod_id"),
            grid_overrides=overrides,
        )
        sim_result = (
            result.get("simulation_result", {}).get("result")
            if isinstance(result.get("simulation_result"), dict)
            else None
        )
        result_payload = None
        if isinstance(sim_result, dict):
            result_payload = {
                "recommended_chip_prod_id": sim_result.get(
                    "recommended_chip_prod_id"
                ),
                "representative_lot": sim_result.get("representative_lot"),
                "param_count": len(sim_result.get("params") or {}),
            }
        fallback = result.get("message") or "그리드 조건을 반영해 다시 계산했습니다."
        return await _generate_edit_auto_message(
            "update_grid",
            {
                "pipeline_message": result.get("message", ""),
                "result": result_payload,
                "ask_prediction": False,
            },
            fallback,
        )

    if intent == "update_reference":
        reference_lot_id = decision.get("reference_lot_id")
        if not reference_lot_id:
            fallback = "레퍼런스 LOT ID를 알려주세요."
            return await _generate_edit_auto_message(
                "update_reference_missing",
                {"missing_fields": ["레퍼런스 LOT ID"], "ask_prediction": False},
                fallback,
            )
        params = simulation_store.get(session_id)
        if not params:
            missing = simulation_store.missing(session_id)
            missing_text = _format_missing_fields(missing)
            fallback = f"시뮬레이션 입력이 부족합니다. {missing_text} 값을 알려주세요."
            return await _generate_edit_auto_message(
                "update_reference_missing",
                {"missing_fields": missing, "ask_prediction": False},
                fallback,
            )
        result = await _run_reference_pipeline(
            session_id,
            params,
            chip_prod_override=params.get("chip_prod_id"),
            reference_override=reference_lot_id,
        )
        sim_result = (
            result.get("simulation_result", {}).get("result")
            if isinstance(result.get("simulation_result"), dict)
            else None
        )
        result_payload = None
        if isinstance(sim_result, dict):
            result_payload = {
                "recommended_chip_prod_id": sim_result.get(
                    "recommended_chip_prod_id"
                ),
                "representative_lot": sim_result.get("representative_lot"),
                "param_count": len(sim_result.get("params") or {}),
            }
        fallback = result.get("message") or "레퍼런스 LOT를 반영해 다시 계산했습니다."
        return await _generate_edit_auto_message(
            "update_reference",
            {
                "pipeline_message": result.get("message", ""),
                "result": result_payload,
                "ask_prediction": False,
            },
            fallback,
        )

    return None


async def _maybe_handle_edit_intent(
    message: str, session_id: str
) -> str | None:
    if not OPENAI_API_KEY:
        return None
    summary = _build_simulation_summary(session_id)
    normalized_message = _normalize_edit_intent_message(message)
    prompt = (
        "Return JSON only. "
        "Parse the user request into a structured edit intent. "
        f"Current state: {json.dumps(summary, ensure_ascii=False)}\n"
        f"User message: {message}\n"
        f"Normalized message: {normalized_message}"
    )
    try:
        run = await Runner.run(
            edit_intent_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
        decision = _normalize_edit_decision(run.final_output)
    except Exception:
        return None
    await emit_workflow_log(
        "INTENT",
        f"{decision.get('intent')}",
        meta={
            "updates": list((decision.get("updates") or {}).keys()),
            "grid_overrides": list((decision.get("grid_overrides") or {}).keys()),
            "stage": decision.get("stage"),
            "rerun": decision.get("rerun"),
            "confidence": decision.get("confidence"),
        },
    )
    if decision.get("needs_clarification") and decision.get("note"):
        return decision.get("note")
    return await _apply_edit_decision(decision, message, session_id)


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
    return _contains_any(message, SIM_RUN_KEYWORDS + ("what-if",))


def _is_progress_request(message: str) -> bool:
    tokens = (
        "progress",
        "status",
        "stage",
        "timeline",
        "history",
        "진행",
        "상태",
        "단계",
        "기록",
    )
    return _contains_any(message, tokens)


def _is_reset_request(message: str) -> bool:
    tokens = (
        "reset",
        "restart",
        "over",
        "초기화",
        "다시 처음",
        "처음부터",
        "리셋",
    )
    return _contains_any(message, tokens)


def _extract_stage_label(message: str) -> str:
    lowered = (message or "").lower()
    stage_aliases = {
        "recommendation": ("추천", "recommendation", "모델추천", "인접기종"),
        "reference": ("레퍼런스", "reference", "기준 lot", "기준lot", "대표 lot", "대표lot"),
        "grid": ("그리드", "grid", "design candidates"),
        "final": ("최종", "final", "브리핑", "요약"),
    }
    for stage, aliases in stage_aliases.items():
        for alias in aliases:
            if alias.lower() in lowered:
                return stage
    return ""


def _available_stages_from_events(events: dict) -> list[str]:
    available = []
    for stage, event_types in STAGE_EVENT_MAP.items():
        if any(event_type in events for event_type in event_types):
            available.append(stage)
    return available


async def _resolve_stage_with_llm(
    message: str, session_id: str, route_command: dict
) -> dict | None:
    if not OPENAI_API_KEY:
        return None
    events = pipeline_store.get_events(session_id)
    available = _available_stages_from_events(events)
    last_stage = pipeline_store.get(session_id).get("last_stage_request")
    payload = {
        "message": message,
        "available_stages": available,
        "last_stage_request": last_stage,
        "keyword_hints": {
            "stage_keyword_hits": _keyword_hits(message, STAGE_NOUN_KEYWORDS),
            "stage_action_hits": _keyword_hits(message, STAGE_ACTION_KEYWORDS),
            "chart_keyword_hits": _keyword_hits(message, CHART_KEYWORDS),
        },
        "route_hint": {
            "primary_intent": route_command.get("primary_intent"),
            "stage": route_command.get("stage"),
        },
    }
    prompt = (
        "Return JSON only. Map the user request to a stage UI screen. "
        "Keys: stage, needs_clarification, clarifying_question, confidence. "
        "Stage must be one of recommendation, reference, grid, final, unknown. "
        "If available_stages has exactly one entry and the user is vague, choose it. "
        "If ambiguous between chart_edit and stage_view, ask a short Korean clarification. "
        f"Context: {json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        run = await Runner.run(
            stage_resolver_agent,
            input=prompt,
            hooks=WorkflowRunHooks(),
        )
        return _normalize_stage_resolution(run.final_output)
    except Exception:
        return None


def _is_stage_view_request(message: str) -> bool:
    display_tokens = (
        "show",
        "replay",
        "stage",
        "화면",
        "보여",
        "다시",
        "단계",
        "이전",
        "중간",
    )
    if not _contains_any(message, display_tokens):
        return False
    if _extract_stage_label(message):
        return True
    fallback_tokens = ("이전", "그거", "직전", "방금")
    return _contains_any(message, fallback_tokens)


def _is_stage_request(message: str) -> bool:
    return _is_stage_view_request(message)


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


def _extract_clear_simulation_keys(message: str) -> list[str]:
    if not message:
        return []
    if not _contains_any(
        message,
        ("삭제", "제외", "빼", "지워", "없애", "remove", "delete", "exclude", "clear"),
    ):
        return []
    keys: list[str] = []
    for pattern, key in EDIT_INTENT_ALIAS_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            if key not in keys:
                keys.append(key)
    return keys


def _should_rerun_grid(message: str) -> bool:
    tokens = ("다시", "재실행", "리런", "run", "grid", "그리드", "예측")
    return _contains_any(message, tokens)


def _build_final_briefing_payload(
    chip_prod_id: str,
    reference_result: dict,
    grid_results: list[dict],
    top_candidates: list[dict],
    defect_stats: dict,
    rules: dict | None = None,
    post_grid_defects: dict | None = None,
    design_blocks_override: list[dict] | None = None,
    defect_rates: list[dict] | None = None,
    defect_rate_overall: float | None = None,
    chip_prod_id_count: int | None = None,
    chip_prod_id_samples: list[str] | None = None,
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
    if post_grid_defects is not None:
        payload["post_grid_defects"] = post_grid_defects
        payload["design_blocks"] = _build_design_blocks(post_grid_defects, rules)
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
    design: dict, design_fields: list[str], design_labels: dict
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
        label = design_labels.get(key) or key
        display.append({"key": key, "label": label, "value": value})
    return display


def _build_design_blocks(post_grid_defects: dict | None, rules: dict) -> list[dict]:
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


def _build_design_blocks_from_matches(match_payload: dict | None, rules: dict) -> list[dict]:
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
                        "label": entry.get("column"),
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


def _build_candidate_defect_chart_payload(match_payload: dict | None) -> dict | None:
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
        lots.append(
            {
                "label": entry.get("column"),
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


def _build_markdown_table(
    columns: list[str], rows: list[object], row_limit: int | None = None
) -> str | None:
    if not columns or not rows:
        return None
    safe_columns = [str(column) for column in columns]
    header = "| " + " | ".join(safe_columns) + " |"
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


def _format_defect_condition_summary(rules: dict) -> str:
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


def _format_sort_summary(rules: dict) -> str:
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
        column_text = "+".join(str(column) for column in columns if column)
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
    columns: list[str], row: dict, chunk_size: int
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
        table = _build_markdown_table(chunk, [row])
        if table:
            tables.append(table)
    return tables


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
    post_grid_defects: dict | None, include_prefix: bool = True
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
        f"{prefix}TOP3 설계조건으로 {period} LOT 불량실적({', '.join(columns)})을 조회했습니다. "
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
    emit_briefing: bool | None = None,
) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    db_config = rules.get("db", {})
    chip_prod_column = db_config.get("chip_prod_id_column") or "chip_prod_id"
    lot_id_column = db_config.get("lot_id_column") or "lot_id"

    await _emit_pipeline_status("recommendation", "조건 매칭 중...")
    await _emit_pipeline_status("reference", "조건에 맞는 LOT 조회 중...")
    reference_result = select_reference_from_params(
        params, chip_prod_override=chip_prod_override
    )

    if reference_override:
        override_detail = get_lot_detail_by_id(reference_override)
        if override_detail.get("status") == "ok":
            override_row = override_detail.get("row") or {}
            override_chip = _row_value(override_row, chip_prod_column)
            override_chip_id = str(override_chip) if override_chip is not None else ""
            if override_chip_id:
                override_result = select_reference_from_params(
                    params, chip_prod_override=override_chip_id
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

    reference_payload = {
        "status": "ok",
        "lot_id": selected_lot_id,
        "chip_prod_id": chip_prod_id,
        "row": selected_row,
        "columns": reference_result.get("columns", []),
        "source": reference_result.get("source") or "postgresql",
        "reference_columns": reference_columns,
        "reference_rows": _json_safe_rows(reference_rows),
    }
    pipeline_store.set_reference(session_id, reference_payload)
    lot_payload = {
        "lot_id": selected_lot_id,
        "columns": reference_payload.get("columns", []),
        "rows": [selected_row],
        "source": reference_payload.get("source") or "postgresql",
    }
    await event_bus.broadcast({"type": "lot_result", "payload": lot_payload})
    pipeline_store.set_event(session_id, "lot_result", lot_payload)
    await _maybe_demo_sleep()
    await _emit_pipeline_status("reference", "레퍼런스 LOT 조회 완료", done=True)

    synthetic = {
        "recommended_chip_prod_id": chip_prod_id,
        "representative_lot": selected_lot_id,
        "params": {},
    }
    await event_bus.broadcast(
        {"type": "simulation_result", "payload": {"params": params, "result": synthetic}}
    )
    pipeline_store.set_event(
        session_id,
        "simulation_result",
        {"params": params, "result": synthetic},
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
        condition_filters, chip_prod_id, "ratio"
    )
    defect_rates: list[dict] = []
    defect_stats = {}
    defect_rate_overall = None
    if condition_table:
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
            chart_lots = [
                {
                    "label": item.get("label") or item.get("key") or item.get("column"),
                    "defect_rate": item.get("defect_rate"),
                }
                for item in defect_rates
                if item.get("defect_rate") is not None
            ]
            defect_source = defect_payload.get("source") or "postgresql"
            chart_payload = {
                "lots": chart_lots,
                "filters": {"chip_prod_id": chip_prod_id},
                "stats": defect_stats,
                "source": defect_source,
                "metric_label": "불량률",
                "value_unit": "ratio",
            }
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

    candidate_matches = collect_grid_candidate_matches(top_candidates, rules)
    design_blocks_override = _build_design_blocks_from_matches(candidate_matches, rules)

    post_grid_defects = collect_post_grid_defects(
        top_candidates,
        rules,
        chip_prod_id=chip_prod_id,
    )
    pipeline_store.set_grid(
        session_id,
        {
            "chip_prod_id": chip_prod_id,
            "lot_id": selected_lot_id,
            "results": grid_results,
            "top": top_candidates,
            "overrides": grid_overrides,
        },
    )
    pipeline_store.set_event(session_id, "final_defect_chart", None)

    candidate_chart = _build_candidate_defect_chart_payload(candidate_matches)
    if candidate_chart:
        await event_bus.broadcast(
            {"type": "defect_rate_chart", "payload": candidate_chart}
        )
        pipeline_store.set_event(
            session_id,
            "defect_rate_chart",
            candidate_chart,
        )

    await event_bus.broadcast(
        {
            "type": "design_candidates",
            "payload": {
                "candidates": top_candidates,
                "total": len(grid_results),
                "offset": 0,
                "limit": len(top_candidates),
                "target": chip_prod_id,
            },
        }
    )
    pipeline_store.set_event(
        session_id,
        "design_candidates",
        {
            "candidates": top_candidates,
            "total": len(grid_results),
            "offset": 0,
            "limit": len(top_candidates),
            "target": chip_prod_id,
        },
    )
    summary_payload = _build_final_briefing_payload(
        chip_prod_id,
        reference_payload,
        grid_results,
        top_candidates,
        defect_stats,
        rules=rules,
        post_grid_defects=post_grid_defects,
        design_blocks_override=design_blocks_override or None,
        defect_rates=defect_rates,
        defect_rate_overall=defect_rate_overall,
        chip_prod_id_count=chip_prod_id_count,
        chip_prod_id_samples=chip_prod_id_samples,
    )
    memory_summary = _build_memory_summary(summary_payload)
    await event_bus.broadcast({"type": "final_briefing", "payload": summary_payload})
    pipeline_store.set_event(session_id, "final_briefing", summary_payload)
    await _emit_pipeline_status("grid", "그리드 탐색 완료", done=True)

    if emit_briefing is None:
        emit_briefing = not bool(
            pipeline_store.get(session_id).get("planner_batch")
        )
    if not emit_briefing:
        return {
            "message": "추천 결과를 저장했습니다. 브리핑 단계에서 요약을 제공합니다.",
            "simulation_result": simulation_result,
        }

    briefing_config = _resolve_final_briefing_config(rules)
    reference_table_limit = briefing_config.get("reference_table_max_rows", 10)
    detail_chunk_size = briefing_config.get("reference_detail_chunk_size", 6)

    safe_reference_rows = _json_safe_rows(reference_rows)
    reference_table = _build_markdown_table(
        reference_columns, safe_reference_rows, row_limit=reference_table_limit
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
        get_lot_detail_by_id(selected_lot_id, use_available_columns=True)
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
    detail_row_safe = _json_safe_dict(detail_row) if isinstance(detail_row, dict) else {}
    detail_tables = _build_markdown_table_chunks(
        detail_columns, detail_row_safe, detail_chunk_size
    )
    detail_table_text = (
        "\n\n".join(detail_tables) if detail_tables else "상세 정보를 찾지 못했습니다."
    )
    detail_note = ""
    if len(detail_tables) > 1:
        detail_note = "컬럼이 많아 표를 여러 개로 나눴습니다."

    grid_rows = []
    for index, candidate in enumerate(top_candidates[:3], start=1):
        if not isinstance(candidate, dict):
            continue
        design = candidate.get("design")
        if not isinstance(design, dict):
            design = {}
        grid_rows.append(
            {
                "rank": candidate.get("rank") or index,
                "electrode_c_avg": _resolve_design_value(
                    design, selected_row, ("electrode_c_avg",)
                ),
                "grinding_t_avg": _resolve_design_value(
                    design, selected_row, ("grinding_t_avg", "grinding_t_size")
                ),
                "active_layer": _resolve_design_value(
                    design, selected_row, ("active_layer",)
                ),
                "cast_dsgn_thk": _resolve_design_value(
                    design, selected_row, ("cast_dsgn_thk",)
                ),
                "ldn_avr_value": _resolve_design_value(
                    design, selected_row, ("ldn_avr_value",)
                ),
            }
        )
    grid_columns = [
        "rank",
        "electrode_c_avg",
        "grinding_t_avg",
        "active_layer",
        "cast_dsgn_thk",
        "ldn_avr_value",
    ]
    grid_table = _build_markdown_table(
        grid_columns, _json_safe_rows(grid_rows)
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
    await _emit_pipeline_stage_tables(
        "grid", grid_tables, grid_notes, session_id=session_id
    )

    input_summary = _format_input_conditions(params)
    condition_summary = _format_defect_condition_summary(rules)
    sort_summary = _format_sort_summary(rules)
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
    post_grid_body = _build_post_grid_step(post_grid_defects, include_prefix=False)
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
        "4": post_grid_body,
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
    post_grid_step = f"4) {rewritten.get('4', rewrite_sections['4'])}"
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
    if post_grid_step:
        stream_blocks.append({"type": "text", "value": post_grid_step})
    if summary_step:
        stream_blocks.append({"type": "text", "value": summary_step})
    streamed = await _stream_briefing_blocks(session_id, stream_blocks)
    pipeline_store.update(
        session_id,
        briefing_text=message,
        briefing_summary=memory_summary,
    )
    pipeline_store.set_pending_memory_summary(
        session_id, memory_summary, label="final_briefing"
    )
    response = {"message": message, "simulation_result": simulation_result}
    if streamed:
        response["streamed"] = True
    return response


async def _handle_pipeline_edit_message(
    message: str, session_id: str
) -> str | dict | None:
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
    chip_prod_override = params.get("chip_prod_id")
    result = await _run_reference_pipeline(
        session_id,
        params,
        chip_prod_override=chip_prod_override,
        reference_override=reference_override,
        grid_overrides=grid_overrides or (state.get("grid") or {}).get("overrides"),
    )
    if isinstance(result, dict) and result.get("streamed"):
        return result
    return result.get("message") if isinstance(result, dict) else result


async def _handle_pipeline_run_message(
    message: str, session_id: str
) -> str | dict | None:
    params, conflicts = await extract_simulation_params_hybrid(message)
    for key in conflicts:
        params.pop(key, None)
    has_params = bool(params)
    intent = _is_recommendation_intent(message)
    if simulation_store.is_active(session_id) and not (has_params or intent):
        if _is_affirmative(message):
            stored_params = simulation_store.get(session_id)
            chip_prod_override = stored_params.get("chip_prod_id")
            missing = simulation_store.missing(session_id)
            if missing and not chip_prod_override:
                return await _generate_simulation_auto_message(
                    stored_params, missing, None
                )
            result = await _run_reference_pipeline(
                session_id, stored_params, chip_prod_override=chip_prod_override
            )
            if isinstance(result, dict) and result.get("streamed"):
                return result
            return result.get("message") if isinstance(result, dict) else result
        return None

    if not (has_params or intent):
        return None

    if has_params:
        update_result = collect_simulation_params(
            **params,
            clear_keys=conflicts,
            extra_missing=conflicts,
        )
    elif conflicts:
        update_result = collect_simulation_params(
            clear_keys=conflicts,
            extra_missing=conflicts,
        )
    else:
        update_result = {
            "params": simulation_store.get(session_id),
            "missing": simulation_store.missing(session_id),
        }
    form_payload = await emit_simulation_form(
        update_result.get("params", {}), update_result.get("missing", [])
    )

    chip_prod_override = update_result.get("params", {}).get("chip_prod_id")
    missing = update_result.get("missing", [])
    if missing and not chip_prod_override:
        message_text = await _generate_simulation_auto_message(
            update_result.get("params", {}), missing, None
        )
        return {
            "message": message_text,
            "ui_event": {"type": "simulation_form", "payload": form_payload},
        }

    result = await _run_reference_pipeline(
        session_id,
        update_result.get("params", {}),
        chip_prod_override=chip_prod_override,
    )
    if isinstance(result, dict) and result.get("streamed"):
        return result
    return result.get("message") if isinstance(result, dict) else result


def _resolve_stage_request(
    message: str, session_id: str, route_stage: str | None
) -> str:
    if route_stage and route_stage not in {"unknown", "none"}:
        return route_stage
    stage = _extract_stage_label(message)
    if stage:
        return stage
    if _contains_any(message, ("이전", "그거", "직전", "방금", "다시")):
        stored = pipeline_store.get(session_id).get("last_stage_request")
        if isinstance(stored, str) and stored:
            return stored
    return ""


async def _handle_stage_view_request(
    message: str, session_id: str, route_command: dict
) -> str:
    stage = _resolve_stage_request(message, session_id, route_command.get("stage"))
    if not stage:
        resolved = await _resolve_stage_with_llm(message, session_id, route_command)
        if resolved:
            if resolved.get("needs_clarification") and resolved.get("clarifying_question"):
                pipeline_store.update(session_id, last_stage_request="")
                return resolved["clarifying_question"]
            stage = resolved.get("stage") if resolved.get("stage") != "unknown" else ""
    if not stage:
        pipeline_store.update(session_id, last_stage_request="")
        return "어떤 단계 화면을 보여드릴까요? (추천/레퍼런스/그리드/최종) 중 하나를 말씀해 주세요."
    pipeline_store.update(session_id, last_stage_request=stage)
    await event_bus.broadcast({"type": "stage_focus", "payload": {"stage": stage}})
    result = await show_simulation_stage_impl(stage)
    return result.get("message") or "요청하신 화면을 다시 표시했어요."


async def _maybe_handle_simulation_message(
    message: str, session_id: str
) -> str | dict | None:
    if _is_reset_request(message):
        await reset_simulation_state_impl(reason=message)
        missing = simulation_store.missing(session_id)
        await emit_simulation_form({}, missing)
        return "Simulation reset. Please enter parameters again."

    if _is_stage_request(message):
        return await _handle_stage_view_request(message, session_id, {"stage": "unknown"})

    if _is_progress_request(message):
        await get_simulation_progress_impl()
        return "Progress history is shown in the event log."

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
        chip_prod_override = params.get("chip_prod_id")
        if missing and not chip_prod_override:
            auto_message = await _generate_simulation_auto_message(params, missing, None)
            await _emit_chat_message(request.session_id, auto_message)
            return {"status": "missing", "missing": missing, "params": params}
        pipeline_result = await _run_reference_pipeline(
            request.session_id, params, chip_prod_override=chip_prod_override
        )
        await _append_system_note(
            request.session_id, "추천 파이프라인 실행 완료."
        )
        message = pipeline_result.get("message")
        if message:
            if pipeline_result.get("streamed"):
                await _append_assistant_message(request.session_id, message)
            else:
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
            "recommended_chip_prod_id": updated.get("recommended_chip_prod_id"),
            "representative_lot": updated.get("representative_lot"),
            "params": updated.get("params"),
        }
        payload = {
            "params": updated.get("input_params") or {},
            "result": result_payload,
        }
        await event_bus.broadcast({"type": "simulation_result", "payload": payload})
        pipeline_store.set_event(request.session_id, "simulation_result", payload)
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
