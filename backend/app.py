import json
from dataclasses import dataclass
from pathlib import Path
import re
import asyncio
import logging
from typing import Any, Awaitable, Callable

from agents import Runner, SQLiteSession
from agents.tracing import set_tracing_disabled
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import (
    auto_message_agent,
    chart_agent,
    conversation_agent,
    edit_intent_agent,
    route_agent,
    stage_resolver_agent,
)
from .config import (
    OPENAI_API_KEY,
    SESSION_DB_PATH,
    TRACING_ENABLED,
    DEMO_LATENCY_SECONDS,
)
from .context import current_session_id
from .db_connections import connect_and_save, get_schema, list_connections, preload_schema
from .db import init_db
from .events import event_bus
from .observability import WorkflowRunHooks, emit_workflow_log
from .pipeline_store import pipeline_store
from .reference_lot import (
    build_grid_values,
    collect_post_grid_defects,
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
    parsed, conflicts = await extract_simulation_params_hybrid(message)
    if not parsed and not conflicts:
        return
    for key in conflicts:
        parsed.pop(key, None)
    current = simulation_store.get(session_id)
    missing_only = {key: value for key, value in parsed.items() if key not in current}
    if not missing_only and not conflicts:
        return
    update_result = collect_simulation_params(
        **missing_only,
        clear_keys=conflicts,
        extra_missing=conflicts,
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
}


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
        "value1_count",
        "value2_count",
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
        return WorkflowOutcome(message_text, ui_event)
    await emit_workflow_log("FLOW", "simulation_edit -> pipeline")
    sim_response = await _maybe_handle_simulation_message(message, session_id)
    if sim_response is None:
        return None
    return WorkflowOutcome(sim_response)


async def _handle_simulation_run_workflow(
    message: str, session_id: str, route_command: dict
) -> WorkflowOutcome | None:
    await emit_workflow_log("FLOW", "simulation_run -> edit_intent")
    edit_response = await _maybe_handle_edit_intent(message, session_id)
    if edit_response is not None:
        message_text, ui_event = _split_edit_response(edit_response)
        return WorkflowOutcome(message_text, ui_event)
    await emit_workflow_log("FLOW", "simulation_run -> pipeline")
    sim_response = await _maybe_handle_simulation_message(message, session_id)
    if sim_response is None:
        return None
    return WorkflowOutcome(sim_response)


async def _handle_chat_workflow(
    message: str, session: SQLiteSession
) -> WorkflowOutcome:
    await emit_workflow_log("FLOW", "chat -> conversation_agent")
    convo = await Runner.run(
        conversation_agent,
        input=message,
        session=session,
        hooks=WorkflowRunHooks(),
    )
    response = (convo.final_output or "").strip()
    return WorkflowOutcome(response)


async def _handle_fallback_workflow(
    message: str, session_id: str, session: SQLiteSession
) -> WorkflowOutcome:
    await emit_workflow_log("FLOW", "fallback -> conversation_agent")
    fallback = await Runner.run(
        conversation_agent,
        input=message,
        session=session,
        hooks=WorkflowRunHooks(),
    )
    await _maybe_update_simulation_from_message(message, session_id)
    response = (fallback.final_output or "").strip()
    return WorkflowOutcome(response)


WORKFLOW_HANDLERS: dict[str, Callable[..., Awaitable[WorkflowOutcome | None]]] = {
    "stage_view": _handle_stage_view_workflow,
    "chart_edit": _handle_chart_edit_workflow,
    "simulation_edit": _handle_simulation_edit_workflow,
    "simulation_run": _handle_simulation_run_workflow,
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


def _split_edit_response(edit_response: object) -> tuple[str, dict | None]:
    if isinstance(edit_response, dict):
        message = str(edit_response.get("message") or "")
        ui_event = edit_response.get("ui_event")
        if not isinstance(ui_event, dict):
            ui_event = None
        return message, ui_event
    return str(edit_response), None


async def _emit_pipeline_status(stage: str, message: str, done: bool = False) -> None:
    pipeline_store.add_status(current_session_id.get(), stage, message, done)
    await event_bus.broadcast(
        {"type": "pipeline_status", "payload": {"stage": stage, "message": message, "done": done}}
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

    payload = {"action": action, "context": context}
    prompt = (
        "다음 JSON을 참고해 한국어로 1~2문장 답변을 작성하세요. "
        "규칙: 도구/라우팅 언급 금지, 과한 추측 금지. "
        "missing_fields가 있으면 그 항목만 질문하세요. "
        "pipeline_message가 있으면 자연스럽게 요약하세요. "
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
    except Exception:
        message = ""
    return message or fallback


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
    return {
        "params": params,
        "missing": missing,
        "simulation_active": simulation_store.is_active(session_id),
        "has_recommendation": bool(rec),
        "recommendation_param_count": rec_param_count,
        "events": list(events.keys()),
        "status": status,
        "last_stage_request": state.get("last_stage_request"),
    }


def _build_route_summary(session_id: str) -> dict:
    summary = _build_simulation_summary(session_id)
    missing = summary.get("missing", [])
    summary = dict(summary)
    summary["missing_count"] = len(missing) if isinstance(missing, list) else 0
    summary.pop("missing", None)
    return summary


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
        "- db_query for LOT/process database lookups. "
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
        updated = recommendation_store.update_params(session_id, params)
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
        if sim_updates:
            update_result = collect_simulation_params(**sim_updates)
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
                result = await _run_reference_pipeline(
                    session_id,
                    update_result.get("params", {}),
                    chip_prod_override=update_result.get("params", {}).get("chip_prod_id"),
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
    return _contains_any(
        message,
        (
            "추천",
            "인접",
            "시뮬",
            "simulation",
            "simulate",
            "what-if",
            "예측",
        ),
    )


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


def _should_rerun_grid(message: str) -> bool:
    tokens = ("다시", "재실행", "리런", "run", "grid", "그리드", "예측")
    return _contains_any(message, tokens)


def _build_final_briefing_payload(
    chip_prod_id: str,
    reference_result: dict,
    grid_results: list[dict],
    top_candidates: list[dict],
    defect_stats: dict,
    post_grid_defects: dict | None = None,
    value1_count: int | None = None,
    value2_count: int | None = None,
    defect_rates: list[dict] | None = None,
    defect_rate_overall: float | None = None,
    chip_prod_id_count: int | None = None,
    chip_prod_id_samples: list[str] | None = None,
) -> dict:
    preview = []
    for item in top_candidates[:5]:
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
    if value1_count is not None:
        payload["value1_count"] = value1_count
    if value2_count is not None:
        payload["value2_count"] = value2_count
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
    return payload


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
                        "value1_counts": {},
                        "value2_all_counts": {},
                        "value2_counts_by_condition": {},
                        "defect_rates": [],
                        "columns": override_detail.get("columns", []),
                        "source": override_detail.get("source"),
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

    value1_counts = reference_result.get("value1_counts") or {}
    value2_all_counts = reference_result.get("value2_all_counts") or {}
    chip_prod_ids = reference_result.get("chip_prod_ids") or []
    if not isinstance(chip_prod_ids, list):
        chip_prod_ids = []
    chip_prod_id_count = len(chip_prod_ids)
    chip_prod_id_samples = [str(item) for item in chip_prod_ids[:10]]
    value1_count = value1_counts.get(chip_prod_id, 0)
    value2_count = value2_all_counts.get(chip_prod_id, 0)

    reference_payload = {
        "status": "ok",
        "lot_id": selected_lot_id,
        "chip_prod_id": chip_prod_id,
        "row": selected_row,
        "columns": reference_result.get("columns", []),
        "source": reference_result.get("source") or "postgresql",
        "value1_count": value1_count,
        "value2_count": value2_count,
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
    defect_rates = defect_payload.get("defect_rates", []) or []
    defect_stats = {}
    defect_rate_overall = None
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
        await event_bus.broadcast(
            {
                "type": "defect_rate_chart",
                "payload": {
                    "lots": chart_lots,
                    "filters": {"chip_prod_id": chip_prod_id},
                    "stats": defect_stats,
                    "source": defect_source,
                },
            }
        )
        pipeline_store.set_event(
            session_id,
            "defect_rate_chart",
            {
                "lots": chart_lots,
                "filters": {"chip_prod_id": chip_prod_id},
                "stats": defect_stats,
                "source": defect_source,
            },
        )

    grid_overrides = grid_overrides or {}
    grid_values = build_grid_values(selected_row, rules, grid_overrides)
    grid_payload = {
        "chip_prod_id": chip_prod_id,
        "lot_id": selected_lot_id,
        "grid": grid_values,
        "max_results": rules.get("grid_search", {}).get("max_results", 100),
    }
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
        post_grid_defects=post_grid_defects,
        value1_count=value1_count,
        value2_count=value2_count,
        defect_rates=defect_rates,
        defect_rate_overall=defect_rate_overall,
        chip_prod_id_count=chip_prod_id_count,
        chip_prod_id_samples=chip_prod_id_samples,
    )
    memory_summary = _build_memory_summary(summary_payload)
    await event_bus.broadcast({"type": "final_briefing", "payload": summary_payload})
    pipeline_store.set_event(session_id, "final_briefing", summary_payload)
    await _emit_pipeline_status("grid", "그리드 탐색 완료", done=True)

    if simulation_result and simulation_result.get("result"):
        summary = _format_simulation_result(simulation_result.get("result"))
        post_grid_columns = post_grid_defects.get("columns", []) if isinstance(post_grid_defects, dict) else []
        post_grid_items = post_grid_defects.get("items", []) if isinstance(post_grid_defects, dict) else []
        if post_grid_columns:
            if post_grid_items:
                post_grid_step = (
                    f"3) TOP3 설계조건으로 동일 조건 LOT 불량실적({', '.join(post_grid_columns)})을 조회했습니다."
                )
            else:
                post_grid_step = (
                    "3) TOP3 설계조건으로 동일 조건 LOT 불량실적을 조회했지만 매칭 LOT가 없습니다."
                )
        else:
            post_grid_step = (
                "3) TOP3 설계조건 기반 불량실적 컬럼 설정이 없어 해당 단계는 생략했습니다."
            )
        message = (
            f"1) chip_prod_id {chip_prod_id} 기준으로 설계조건+불량조건을 함께 적용해 LOT 후보를 필터링했습니다. "
            f"2) 최신 LOT {selected_lot_id}를 기준으로 설계조건을 흔들어 그리드 탐색을 실행했습니다. "
            f"{post_grid_step} "
            f"4) 최종 요약: {summary}. 상위 후보를 확인해 주세요."
        )
        pipeline_store.update(
            session_id,
            briefing_text=message,
            briefing_summary=memory_summary,
        )
        pipeline_store.set_pending_memory_summary(
            session_id, memory_summary, label="final_briefing"
        )
        return {"message": message, "simulation_result": simulation_result}
    message = (
        f"1) chip_prod_id {chip_prod_id} 기준으로 설계조건+불량조건을 함께 적용해 LOT 후보를 필터링했습니다. "
        f"2) 최신 LOT {selected_lot_id}를 기준으로 설계조건을 흔들어 그리드 탐색을 실행했습니다. "
        "3) TOP3 설계조건 기반 불량실적 조회를 진행했습니다. "
        "4) 최종 요약: 추천 결과를 확인해 주세요."
    )
    pipeline_store.update(
        session_id,
        briefing_text=message,
        briefing_summary=memory_summary,
    )
    pipeline_store.set_pending_memory_summary(
        session_id, memory_summary, label="final_briefing"
    )
    return {
        "message": message,
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
    chip_prod_override = params.get("chip_prod_id")
    result = await _run_reference_pipeline(
        session_id,
        params,
        chip_prod_override=chip_prod_override,
        reference_override=reference_override,
        grid_overrides=grid_overrides or (state.get("grid") or {}).get("overrides"),
    )
    return result.get("message")


async def _handle_pipeline_run_message(
    message: str, session_id: str
) -> str | None:
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
            return result.get("message")
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
    await emit_simulation_form(
        update_result.get("params", {}), update_result.get("missing", [])
    )

    chip_prod_override = update_result.get("params", {}).get("chip_prod_id")
    missing = update_result.get("missing", [])
    if missing and not chip_prod_override:
        return await _generate_simulation_auto_message(
            update_result.get("params", {}), missing, None
        )

    result = await _run_reference_pipeline(
        session_id,
        update_result.get("params", {}),
        chip_prod_override=chip_prod_override,
    )
    return result.get("message")


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
) -> str | None:
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
