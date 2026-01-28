import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core import agents, db_production, demo, schemas, state
from agents import SQLiteSession

router = APIRouter()


def _format_sse(event: str, payload: dict) -> str:
    # SSE 포맷 문자열을 만든다.
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _log_briefing_debug(
    note: str,
    session_id: str,
    route: str,
    action: str | None,
    blocks: list[dict[str, Any]],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
) -> None:
    # 테이블 키를 수집한다.
    table_keys = [key for key in (tables or {}).keys() if isinstance(key, str)]
    # 차트 ID를 수집한다.
    chart_ids = [
        chart.get("chart_id")
        for chart in (charts or [])
        if isinstance(chart, dict) and isinstance(chart.get("chart_id"), str)
    ]
    # 블록에서 table_ref를 수집한다.
    block_table_refs = [
        block.get("table_key")
        for block in (blocks or [])
        if isinstance(block, dict) and block.get("type") == "table_ref"
    ]
    # 블록에서 chart_ref를 수집한다.
    block_chart_refs = [
        block.get("chart_id")
        for block in (blocks or [])
        if isinstance(block, dict) and block.get("type") == "chart_ref"
    ]
    # 누락된 table_ref를 계산한다.
    missing_tables = [key for key in block_table_refs if key not in table_keys]
    # 누락된 chart_ref를 계산한다.
    missing_charts = [key for key in block_chart_refs if key not in chart_ids]
    # 디버그 페이로드를 만든다.
    payload = {
        "note": note,
        "session_id": session_id,
        "route": route,
        "action": action,
        "block_count": len(blocks or []),
        "table_count": len(table_keys),
        "chart_count": len(chart_ids),
        "table_keys": table_keys,
        "chart_ids": chart_ids,
        "block_table_refs": block_table_refs,
        "block_chart_refs": block_chart_refs,
        "missing_tables": missing_tables,
        "missing_charts": missing_charts,
    }
    # 디버그 로그를 출력한다.
    print("[BRIEFING_DEBUG]", json.dumps(payload, ensure_ascii=False))


_ACTION_MESSAGE_TYPES = {"select_candidates"}


def _parse_action_payload(message: str) -> dict[str, Any] | None:
    # 액션 페이로드를 파싱한다.
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _is_action_payload(payload: dict[str, Any] | None) -> bool:
    # 액션 페이로드인지 확인한다.
    if not payload:
        return False
    action = payload.get("action")
    return action in _ACTION_MESSAGE_TYPES


def _extract_pending_selection(
    payload: dict[str, Any] | None,
    pending_action: dict[str, Any] | None,
) -> list[str] | None:
    # pending_action에 맞는 선택값을 추출한다.
    if not payload or not pending_action:
        return None
    if payload.get("action") != pending_action.get("action"):
        return None
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        return None
    selection_field = pending_action.get("selection_field")
    if not selection_field:
        return None
    selected = selection.get(selection_field)
    if isinstance(selected, list) and selected:
        return [str(item) for item in selected if item]
    return None


def _build_table_select_block(
    pending_action: dict[str, Any],
    selected_ids: list[str] | None = None,
) -> dict[str, Any]:
    # 테이블 선택 블록을 만든다.
    return {
        "type": "table_select",
        "table_key": pending_action.get("table_key"),
        "id_field": pending_action.get("id_field", "chip_type_id"),
        "selection_field": pending_action.get("selection_field", "chip_type_ids"),
        "allow_multi": pending_action.get("allow_multi", True),
        "action": pending_action.get("action", "select_candidates"),
        "title": pending_action.get("title", "후보 선택"),
        "description": pending_action.get("description", ""),
        "submit_label": pending_action.get("submit_label", "해당 기종으로 진행"),
        "selected_ids": selected_ids or [],
    }


def _load_pending_tables(
    session_state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # pending 상태에서 사용할 raw 테이블을 불러온다.
    raw_path = session_state.get("raw_refs", {}).get("stage_outputs_path")
    return state._load_raw_outputs(raw_path)


def _build_pending_question(pending_action: dict[str, Any]) -> str:
    # pending_action에 저장된 질문을 꺼낸다.
    question = pending_action.get("question")
    # 질문이 없으면 기본 문구로 보완한다.
    return question or "후보를 선택해 진행할까요?"


def _build_pending_blocks(
    question: str,
    pending_action: dict[str, Any],
    selected_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    # 질문 텍스트 블록을 만든다.
    question_block = {"type": "text", "section": "summary", "value": question}
    # 선택된 ID를 준비한다.
    resolved_ids = selected_ids or []
    # 테이블 선택 블록을 만든다.
    select_block = _build_table_select_block(
        pending_action,
        selected_ids=resolved_ids,
    )
    # 두 블록을 묶어서 반환한다.
    return [question_block, select_block]


def _build_pending_repeat_response(
    session_state: dict[str, Any],
    pending_action: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # 재질문에 사용할 기본 질문을 만든다.
    question = _build_pending_question(pending_action)
    # 기존 선택값을 블록에 채운다.
    selected_ids = session_state.get("selections", {}).get("chip_type_ids") or []
    # 질문 + 선택 UI 블록을 만든다.
    blocks = _build_pending_blocks(
        question,
        pending_action,
        selected_ids=selected_ids,
    )
    # 이전 단계 테이블/차트를 불러온다.
    tables, charts = _load_pending_tables(session_state)
    # 선택 강조 표시를 적용한다.
    state._apply_table_highlights(tables, session_state.get("selections", {}))
    # 히스토리에 재질문 기록을 남긴다.
    session_state["history"].append(
        {
            "action": "pending_repeat",
            "payload": {"action": pending_action.get("action")},
            "at": state._utc_now(),
        }
    )
    # 응답에 사용할 블록/데이터를 반환한다.
    return blocks, tables, charts


def _extract_candidate_ids(
    rows: list[dict[str, Any]] | None,
    id_field: str,
) -> list[str]:
    # 후보 ID 목록을 만든다.
    candidate_ids: list[str] = []
    # 행을 하나씩 확인한다.
    for row in rows or []:
        # dict가 아니면 건너뛴다.
        if not isinstance(row, dict):
            continue
        # ID 값을 꺼낸다.
        value = row.get(id_field)
        # 값이 없으면 건너뛴다.
        if value is None:
            continue
        # 문자열로 변환해 저장한다.
        candidate_ids.append(str(value))
    # 후보 ID 목록을 반환한다.
    return candidate_ids


def _build_gap_context(gap: dict[str, Any]) -> dict[str, Any]:
    # gap 정보를 질문 생성용 컨텍스트로 정리한다.
    return {
        "stage": gap.get("stage"),
        "reason": gap.get("reason"),
        "fallback_summary": gap.get("fallback_summary"),
        "candidate_count": gap.get("candidate_count", 0),
    }


def _build_gap_pending_payload(
    gap: dict[str, Any],
    question: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # pending_action 페이로드를 만든다.
    pending_action = {
        "action": "select_candidates",
        "target_stage": gap.get("stage"),
        "table_key": gap.get("table_key"),
        "id_field": gap.get("id_field", "chip_type_id"),
        "selection_field": gap.get("selection_field", "chip_type_ids"),
        "allow_multi": gap.get("allow_multi", True),
        "question": question,
        "submit_label": "해당 기종으로 진행",
        "requested_at": state._utc_now(),
    }
    # 질문 + 선택 UI 블록을 만든다.
    blocks = _build_pending_blocks(question, pending_action)
    # pending_action과 블록을 함께 반환한다.
    return pending_action, blocks


def _build_input_form_blocks(merged_params: schemas.InputParams) -> list[dict[str, Any]]:
    # 입력 폼에 필요한 필드 목록을 준비한다.
    target_keys = ["temperature", "size", "capacity", "voltage"]
    # 폼 필드를 담을 리스트를 만든다.
    fields: list[dict[str, Any]] = []
    # 각 입력 항목을 순서대로 구성한다.
    for key in target_keys:
        current_val = merged_params.dict().get(key)
        field_def = {
            "key": key,
            "label": state.INPUT_LABEL_MAP.get(key, key),
            "type": "text",
            "value": current_val or "",
        }
        # 필드별 UI 설정을 추가한다.
        if key == "temperature":
            field_def["type"] = "select"
            field_def["options"] = ["A", "B", "D"]
            field_def["unit"] = "특성"
        elif key == "voltage":
            field_def["type"] = "number"
            field_def["unit"] = "V"
        elif key == "size":
            field_def["type"] = "select"
            field_def["options"] = ["1005", "1608", "2012", "3216"]
        elif key == "capacity":
            field_def["type"] = "number"
            field_def["unit"] = "pF"
            field_def["unit_options"] = ["pF", "nF", "uF"]
        # 구성된 필드를 목록에 추가한다.
        fields.append(field_def)
    # 입력 폼 블록을 반환한다.
    return [
        {
            "type": "input_form",
            "form_id": "mlcc_basic_params",
            "title": "시뮬레이션 조건 입력",
            "description": "다음 핵심 정보를 입력해주세요.",
            "fields": fields,
            "submit_label": "시뮬레이션 시작",
            "submitted": False,
        }
    ]


async def _handle_explain_stage_request(
    session_state: dict[str, Any],
    command: schemas.CommandDecision,
    request_message: str,
    route: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # 설명 요청 진행 로그를 만든다.
    progress_logs = state._build_progress_logs(
        route,
        "explain_stage",
        session_state.get("stage_status"),
        is_final=False,
    )
    # 설명 대상 단계를 확정한다.
    target_stage = state._normalize_stage(
        command.target_stage or session_state.get("last_explain_stage"),
        session_state["stage_status"],
    )
    # LLM 설명 응답을 생성한다.
    blocks, tables, charts = await state._build_explain_response(
        session_state, target_stage, request_message
    )
    # 마지막 설명 단계 기록을 갱신한다.
    if target_stage:
        session_state["last_explain_stage"] = target_stage
    # 설명 요청 히스토리를 남긴다.
    session_state["history"].append(
        {
            "action": "explain_stage",
            "payload": {"target_stage": target_stage},
            "at": state._utc_now(),
        }
    )
    # 진행 로그와 결과 블록을 함께 반환한다.
    return progress_logs, blocks, tables, charts


async def _resolve_input_and_update(
    session_state: dict[str, Any],
    request_message: str,
    pending_selection: list[str] | None,
    had_results: bool,
) -> tuple[schemas.InputParams, schemas.UpdateDecision, list[str]]:
    # pending 선택이 있으면 해당 선택만 업데이트한다.
    if pending_selection:
        input_params = schemas.InputParams(**session_state["input_params"])
        update = schemas.UpdateDecision(
            selections=schemas.UpdateSelections(chip_type_ids=pending_selection)
        )
        session_state["pending_action"] = None
    else:
        # 일반 메시지는 LLM 파싱으로 입력/업데이트를 만든다.
        input_params = await agents._parse_input_with_llm(request_message)
        update = await agents._parse_update_with_llm(request_message)
    # 누락 요청 필드를 수집한다.
    missing_update = update.missing_fields or []
    # 결과가 없던 상태라면 누락 요청은 무시한다.
    if not had_results:
        missing_update = []
    # 입력/업데이트/누락 정보를 반환한다.
    return input_params, update, missing_update


def _apply_update_and_dirty(
    session_state: dict[str, Any],
    input_params: schemas.InputParams,
    update: schemas.UpdateDecision,
    missing_update: list[str],
) -> tuple[schemas.InputParams, list[str]]:
    # 입력 병합과 dirty 계산을 동시에 수행한다.
    merged_params, dirty_stages = state._merge_update_and_collect_dirty(
        session_state,
        input_params,
        update,
        missing_update,
    )
    # 병합된 입력을 세션에 반영한다.
    session_state["input_params"] = merged_params.dict()
    # 업데이트 필드를 세션에 적용한다.
    state._apply_update_fields(session_state, update)
    # dirty 상태를 기록한다.
    state._mark_dirty(session_state, dirty_stages)
    # 병합 결과와 dirty 정보를 반환한다.
    return merged_params, dirty_stages


def _handle_missing_update_request(
    session_state: dict[str, Any],
    missing_update: list[str],
) -> tuple[bool, list[str], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    # 누락 업데이트가 없으면 처리하지 않는다.
    if not missing_update:
        return False, [], [], {}, [], None
    # 누락 요청 pending_action을 만든다.
    pending_action = state._build_pending_action(missing_update)
    session_state["pending_action"] = pending_action
    # 누락 요청 히스토리를 기록한다.
    session_state["history"].append(
        {
            "action": "update_input_pending",
            "payload": {"missing": missing_update},
            "at": state._utc_now(),
        }
    )
    # 사용자 안내 블록을 만든다.
    blocks = [
        {
            "type": "text",
            "section": "summary",
            "value": state._format_update_missing(missing_update),
        }
    ]
    # 누락 요청 응답용 테이블/차트를 초기화한다.
    tables, charts = {}, []
    # 누락 처리 결과를 반환한다.
    return True, missing_update, blocks, tables, charts, pending_action


def _handle_missing_targets_form(
    missing: list[str],
    merged_params: schemas.InputParams,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    # 입력 폼에 필요한 필드를 고른다.
    target_keys = ["temperature", "size", "capacity", "voltage"]
    # 누락된 주요 입력 필드를 찾는다.
    missing_targets = [key for key in missing if key in target_keys]
    # 주요 필드가 없으면 입력 폼을 만들지 않는다.
    if not missing_targets:
        return False, [], {}, [], {}
    # 입력 폼 블록을 만든다.
    blocks = _build_input_form_blocks(merged_params)
    # 입력 폼은 테이블/차트가 없다.
    tables, charts = {}, []
    # 입력 폼 단계 노트를 비워둔다.
    stage_notes: dict[str, str] = {}
    # 입력 폼 처리 결과를 반환한다.
    return True, blocks, tables, charts, stage_notes


async def _build_briefing_blocks_for_run(
    route: str,
    action: str | None,
    session_state: dict[str, Any],
    had_results: bool,
    dirty_stages: list[str],
    stage_notes: dict[str, str],
    llm_tables: dict[str, Any],
    llm_charts: list[dict[str, Any]],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # 브리핑 진행 로그를 만든다.
    progress_logs = state._build_progress_logs(
        route,
        action,
        session_state.get("stage_status"),
        current_stage="1-8",
        is_final=False,
    )
    # 브리핑 시작 단계와 힌트를 결정한다.
    briefing_start = None
    briefing_hint = None
    if had_results and dirty_stages:
        briefing_start = state._pick_briefing_start_stage(dirty_stages)
        briefing_hint = state._build_briefing_hint(briefing_start)
    # 브리핑 시퀀스를 구성한다.
    briefing_sequence = state._build_briefing_sequence(stage_notes, briefing_start)
    # 브리핑 대상 테이블/차트를 추린다.
    briefing_tables, briefing_charts, _ = state._filter_briefing_outputs(
        llm_tables, llm_charts, briefing_start
    )
    # LLM 브리핑 블록을 생성한다.
    blocks = await agents._build_briefing_blocks(
        briefing_tables,
        briefing_charts,
        briefing_hint,
        briefing_sequence,
    )
    # 블록 참조를 정규화한다.
    blocks = state._normalize_block_refs(blocks, tables, charts)
    # 참조 누락 블록을 보정한다.
    blocks = state._ensure_block_refs(
        blocks,
        tables,
        charts,
        briefing_sequence,
    )
    # 브리핑 블록과 진행 로그를 반환한다.
    return blocks, progress_logs


def _post_process_run_tables(
    session_state: dict[str, Any],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    dirty_stages: list[str],
    missing: list[str],
    pending_action: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # 누락이나 pending이 있으면 후처리를 하지 않는다.
    if missing or pending_action:
        return tables, charts
    # raw 출력과 히스토리를 병합한다.
    tables, charts = state._merge_raw_outputs_with_history(
        session_state,
        tables,
        charts,
        dirty_stages,
    )
    # 병합 후에도 강조 표시를 유지한다.
    state._apply_table_highlights(
        tables,
        session_state["selections"],
    )
    # 후처리 결과를 반환한다.
    return tables, charts


def _update_run_state(
    session_state: dict[str, Any],
    merged_params: schemas.InputParams,
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    stage_notes: dict[str, str],
    missing: list[str],
    demo: bool,
    llm_tables: dict[str, Any],
    llm_charts: list[dict[str, Any]],
    dirty_stages: list[str],
    pending_action: dict[str, Any] | None,
) -> None:
    # 세션 상태를 업데이트한다.
    state._update_state(
        session_state,
        merged_params,
        tables,
        charts,
        blocks,
        stage_notes,
        missing,
        demo,
        llm_tables=llm_tables,
        llm_charts=llm_charts,
        dirty_stages=dirty_stages,
        pending_action=pending_action,
    )
    # 누락과 pending이 없으면 dirty 상태를 해제한다.
    if not missing and not pending_action:
        state._mark_clean(session_state, dirty_stages)


def _run_simulation_with_progress(
    route: str,
    action: str | None,
    session_state: dict[str, Any],
    merged_params: schemas.InputParams,
    dirty_stages: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, Any] | None, list[list[dict[str, Any]]]]:
    # 진행 로그 이벤트를 담을 리스트를 준비한다.
    progress_events: list[list[dict[str, Any]]] = []
    # 단계별 진행 로그를 수집하는 콜백을 정의한다.
    def progress_cb(stage: str) -> None:
        logs = state._build_progress_logs(
            route,
            action,
            session_state.get("stage_status"),
            current_stage=stage,
            is_final=False,
        )
        if logs:
            progress_events.append(logs)
    # DB 시뮬레이션을 실행한다.
    tables, charts, stage_notes, gap = db_production.build_simulation_from_db(
        merged_params,
        session_state["configs"],
        session_state["selections"],
        session_state["user_prefs"],
        dirty_stages=dirty_stages,
        progress_cb=progress_cb,
    )
    # 결과와 진행 로그를 함께 반환한다.
    return tables, charts, stage_notes, gap, progress_events


def _finalize_gap_stream_response(
    route: str,
    action: str | None,
    session_state: dict[str, Any],
    merged_params: schemas.InputParams,
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    stage_notes: dict[str, str],
    missing: list[str],
    dirty_stages: list[str],
    pending_action: dict[str, Any],
    demo: bool,
) -> dict[str, Any]:
    # 선택 강조 표시를 적용한다.
    state._apply_table_highlights(tables, session_state.get("selections", {}))
    # 상태를 업데이트한다.
    state._update_state(
        session_state,
        merged_params,
        tables,
        charts,
        blocks,
        stage_notes,
        missing,
        demo,
        dirty_stages=dirty_stages,
        pending_action=pending_action,
    )
    # 최종 응답 페이로드를 만든다.
    return _build_final_payload(
        route,
        action,
        session_state,
        blocks,
        tables,
        charts,
        debug_note="final_stream_response",
    )

def _build_final_payload(
    route: str,
    action: str | None,
    session_state: dict[str, Any],
    blocks: list[dict[str, Any]],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    debug_note: str | None = None,
) -> dict[str, Any]:
    # 진행 로그를 만든다.
    final_logs = state._build_progress_logs(
        route,
        action,
        session_state.get("stage_status"),
    )
    # 진행 로그가 있으면 블록 앞에 붙인다.
    if final_logs:
        blocks = [{"type": "progress_log", "logs": final_logs}] + blocks
    # 디버그 로그가 필요하면 기록한다.
    if debug_note and route == "simulation":
        _log_briefing_debug(
            debug_note,
            session_state.get("session_id", "-"),
            route,
            action,
            blocks,
            tables,
            charts,
        )
    # 최종 응답 페이로드를 반환한다.
    return {"route": route, "blocks": blocks, "tables": tables, "charts": charts}



@router.post("/api/chat/stream")
async def api_chat_stream(request: schemas.ChatRequest) -> StreamingResponse:
    # SSE 스트림 응답을 만든다.
    session = SQLiteSession(request.session_id, "conversation_123")

    async def event_stream():
        # 라우팅을 먼저 수행한다.
        action_payload = _parse_action_payload(request.message)
        route = (
            "simulation"
            if _is_action_payload(action_payload)
            else await agents._route_with_llm(session, request.message)
        )
        # 세션 상태를 가져온다.
        session_state = state._get_session_state(request.session_id)
        action: str | None = None
        missing: list[str] = []
        blocks: list[dict[str, Any]] = []
        tables: dict[str, Any] = {}
        charts: list[dict[str, Any]] = []

        if route == "simulation":
            pending_action = (
                session_state.get("pending_action")
                if isinstance(session_state.get("pending_action"), dict)
                else None
            )
            pending_selection = _extract_pending_selection(
                action_payload, pending_action
            )
            # 커맨드를 결정한다.
            command_hint = state._build_command_hint(session_state)
            command_message = f"{command_hint}\n\n[사용자 메시지]\n{request.message}"
            if _is_action_payload(action_payload):
                command = schemas.CommandDecision(action="run", target_stage=None)
            else:
                command = await agents._decide_command_with_llm(session, command_message)
            if command.action == "reset":
                # 리셋 액션을 기록한다.
                action = "reset"
                # 세션 상태를 초기화한다.
                session_state = state._reset_session_state(request.session_id)
                # 리셋 안내 블록을 만든다.
                blocks = [
                    {
                        "type": "text",
                        "section": "summary",
                        "value": "시뮬레이션 상태를 초기화했어. 새로 시작해줘.",
                    }
                ]
                # 표/차트는 비운다.
                tables, charts = {}, []
                # 최종 응답을 만든다.
                final_payload = _build_final_payload(
                    route,
                    action,
                    session_state,
                    blocks,
                    tables,
                    charts,
                    debug_note="final_stream_response",
                )
                # 최종 응답을 전송하고 종료한다.
                yield _format_sse("final", final_payload)
                return
            if pending_action and pending_action.get("action") == "select_candidates" and not pending_selection:
                # 후보 테이블을 불러온다.
                pending_tables, _ = _load_pending_tables(session_state)
                # 테이블 키를 꺼낸다.
                table_key = pending_action.get("table_key")
                # ID 필드를 꺼낸다.
                id_field = pending_action.get("id_field", "chip_type_id")
                # 후보 행을 꺼낸다.
                candidate_rows = pending_tables.get(table_key) if table_key else []
                # 후보 ID 목록을 만든다.
                candidate_ids = _extract_candidate_ids(candidate_rows, id_field)
                # LLM으로 선택을 추출한다.
                pending_selection = await agents._select_candidates_with_llm(
                    request.message,
                    candidate_ids,
                )
                # 선택이 없으면 UI를 다시 보낸다.
                if not pending_selection:
                    action = "run"
                    blocks, tables, charts = _build_pending_repeat_response(
                        session_state, pending_action
                    )
                    # 최종 응답을 만든다.
                    final_payload = _build_final_payload(
                        route,
                        action,
                        session_state,
                        blocks,
                        tables,
                        charts,
                        debug_note="final_stream_response",
                    )
                    # 최종 응답을 전송하고 종료한다.
                    yield _format_sse("final", final_payload)
                    return
            if command.action == "explain_stage":
                action = "explain_stage"
                (
                    progress_logs,
                    blocks,
                    tables,
                    charts,
                ) = await _handle_explain_stage_request(
                    session_state,
                    command,
                    request.message,
                    route,
                )
                yield _format_sse("progress", {"logs": progress_logs})
            else:
                action = "run"
                progress_logs = state._build_progress_logs(
                    route,
                    action,
                    session_state.get("stage_status"),
                    current_stage="1-1",
                    is_final=False,
                )
                yield _format_sse("progress", {"logs": progress_logs})
                # ?? ??? ?? ??? ????.
                had_results = bool(
                    session_state["stage_status"].get("1-8", {}).get("done")
                )
                # ??? ?? ??? ????.
                input_params, update, missing_update = await _resolve_input_and_update(
                    session_state,
                    request.message,
                    pending_selection,
                    had_results,
                )
                # ?? ??? dirty ??? ????.
                merged_params, dirty_stages = _apply_update_and_dirty(
                    session_state,
                    input_params,
                    update,
                    missing_update,
                )
                # ???? ?? ??? ?? ????.
                handled_missing, missing, blocks, tables, charts, pending_action = (
                    _handle_missing_update_request(session_state, missing_update)
                )
                if not handled_missing:
                    # ??? ??? ????.
                    missing = state._get_missing_fields(merged_params)
                    pending_action = None
                    # ?? ? ?? ??? ????.
                    handled_form, blocks, tables, charts, stage_notes = (
                        _handle_missing_targets_form(missing, merged_params)
                    )
                    if not handled_form:
                        if request.demo:
                            blocks, tables, charts, stage_notes = (
                                demo._build_simulation_stub(
                                    request,
                                    merged_params,
                                    session_state["configs"],
                                    session_state["selections"],
                                    session_state["user_prefs"],
                                )
                            )
                        else:
                            # ?? ??? ????? ??? ?? ????.
                            (
                                tables,
                                charts,
                                stage_notes,
                                gap,
                                progress_events,
                            ) = _run_simulation_with_progress(
                                route,
                                action,
                                session_state,
                                merged_params,
                                dirty_stages,
                            )
                            for logs in progress_events:
                                yield _format_sse("progress", {"logs": logs})
                            if gap:
                                # notice 타입 gap이면 안내만 하고 종료한다.
                                if gap.get("type") == "notice":
                                    # 안내 문구를 준비한다.
                                    notice_message = gap.get("message") or "해당 단계 결과를 찾을 수 없어 진행할 수 없어."
                                    # 안내 블록을 만든다.
                                    blocks = [
                                        {
                                            "type": "text",
                                            "section": "summary",
                                            "value": notice_message,
                                        }
                                    ]
                                    # 마지막 gap 정보를 저장한다.
                                    session_state["last_gap"] = {
                                        "type": "notice",
                                        "stage": gap.get("stage"),
                                        "message": notice_message,
                                    }
                                    # 최종 응답을 만든다.
                                    final_payload = {
                                        "route": route,
                                        "blocks": blocks,
                                        "tables": {},
                                        "charts": [],
                                    }
                                    # 최종 응답을 전송하고 종료한다.
                                    yield _format_sse("final", final_payload)
                                    return
                                # gap 질문 컨텍스트를 만든다.
                                gap_context = _build_gap_context(gap)
                                # gap 질문을 생성한다.
                                question = await agents._build_gap_question(gap_context)
                                # pending_action과 블록을 만든다.
                                pending_action, blocks = _build_gap_pending_payload(
                                    gap, question
                                )
                                # 마지막 gap 정보를 저장한다.
                                session_state["last_gap"] = gap_context
                                # gap?? ?? ??? ???? ????.
                                final_payload = _finalize_gap_stream_response(
                                    route,
                                    action,
                                    session_state,
                                    merged_params,
                                    tables,
                                    charts,
                                    blocks,
                                    stage_notes,
                                    missing,
                                    dirty_stages,
                                    pending_action,
                                    request.demo,
                                )
                                yield _format_sse("final", final_payload)
                                return
                            else:
                                pending_action = None
                                # ??? ?? ?? ??? ? ??? ????.
                                blocks = []
                    # ??? ?? ??? ????.
                    state._apply_table_highlights(tables, session_state["selections"])
                    # LLM? ??? ???? ???.
                    llm_tables, llm_charts = state._build_llm_payload(
                        tables, charts, session_state["configs"]
                    )
                    if not missing and not pending_action:
                        blocks, progress_logs = await _build_briefing_blocks_for_run(
                            route,
                            action,
                            session_state,
                            had_results,
                            dirty_stages,
                            stage_notes,
                            llm_tables,
                            llm_charts,
                            tables,
                            charts,
                        )
                        yield _format_sse("progress", {"logs": progress_logs})
                    if not missing and not pending_action:
                        tables, charts = _post_process_run_tables(
                            session_state,
                            tables,
                            charts,
                            dirty_stages,
                            missing,
                            pending_action,
                        )
                    _update_run_state(
                        session_state,
                        merged_params,
                        tables,
                        charts,
                        blocks,
                        stage_notes,
                        missing,
                        request.demo,
                        llm_tables,
                        llm_charts,
                        dirty_stages,
                        pending_action,
                    )
        else:
            # 캐주얼 응답 로그를 전송한다.
            progress_logs = state._build_progress_logs(
                route,
                None,
                session_state.get("stage_status"),
                is_final=False,
            )
            yield _format_sse("progress", {"logs": progress_logs})
            # 캐주얼 응답을 LLM으로 생성한다.
            blocks = await agents._build_casual_blocks(session, request.message)
            # 캐주얼 응답에는 표/차트가 없다.
            tables, charts = {}, []

        # 최종 진행 로그를 만든다.
        final_payload = _build_final_payload(
            route,
            action,
            session_state,
            blocks,
            tables,
            charts,
            debug_note="final_stream_response",
        )
        # 최종 응답을 전송한다.
        yield _format_sse("final", final_payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
