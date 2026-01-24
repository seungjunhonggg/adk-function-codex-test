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
            if pending_action and pending_action.get("action") == "select_candidates" and not pending_selection:
                action = "run"
                blocks, tables, charts = _build_pending_repeat_response(
                    session_state, pending_action
                )
            elif command.action == "explain_stage":
                # 설명 요청은 별도로 처리한다.
                action = "explain_stage"
                # 설명 단계 진행 로그를 전송한다.
                progress_logs = state._build_progress_logs(
                    route,
                    action,
                    session_state.get("stage_status"),
                    is_final=False,
                )
                yield _format_sse("progress", {"logs": progress_logs})
                # 요청 단계가 없으면 마지막 설명 단계를 재사용한다.
                target_stage = state._normalize_stage(
                    command.target_stage or session_state.get("last_explain_stage"),
                    session_state["stage_status"],
                )
                # 설명용 LLM을 호출해 답변을 만든다.
                blocks, tables, charts = await state._build_explain_response(
                    session_state, target_stage, request.message
                )
                # 마지막 설명 단계를 저장한다.
                if target_stage:
                    session_state["last_explain_stage"] = target_stage
                session_state["history"].append(
                    {
                        "action": "explain_stage",
                        "payload": {"target_stage": target_stage},
                        "at": state._utc_now(),
                    }
                )
            else:
                # 설명 요청이 아니면 모두 run으로 처리한다.
                action = "run"
                # 입력 수집 단계 로그를 전송한다.
                progress_logs = state._build_progress_logs(
                    route,
                    action,
                    session_state.get("stage_status"),
                    current_stage="1-1",
                    is_final=False,
                )
                yield _format_sse("progress", {"logs": progress_logs})
                # 기존 브리핑 완료 여부를 확인한다.
                had_results = bool(
                    session_state["stage_status"].get("1-8", {}).get("done")
                )
                # 입력과 변경 요청을 함께 파싱한다.
                if pending_selection:
                    input_params = schemas.InputParams(**session_state["input_params"])
                    update = schemas.UpdateDecision(
                        selections=schemas.UpdateSelections(
                            chip_type_ids=pending_selection
                        )
                    )
                    session_state["pending_action"] = None
                else:
                    input_params = await agents._parse_input_with_llm(request.message)
                    update = await agents._parse_update_with_llm(request.message)
                missing_update = update.missing_fields or []
                if not had_results:
                    # 결과가 없으면 업데이트 누락 처리를 건너뛴다.
                    missing_update = []
                # 입력 병합과 dirty 계산을 한번에 처리한다.
                merged_params, dirty_stages = state._merge_update_and_collect_dirty(
                    session_state,
                    input_params,
                    update,
                    missing_update,
                )
                # 입력 변경에 따라 dirty를 표시한다.
                session_state["input_params"] = merged_params.dict()
                state._apply_update_fields(session_state, update)
                state._mark_dirty(session_state, dirty_stages)
                # 업데이트 누락이 있으면 요청만 반환한다.
                if missing_update:
                    missing = missing_update
                    session_state["pending_action"] = state._build_pending_action(
                        missing_update
                    )
                    session_state["history"].append(
                        {
                            "action": "update_input_pending",
                            "payload": {"missing": missing_update},
                            "at": state._utc_now(),
                        }
                    )
                    blocks = [
                        {
                            "type": "text",
                            "section": "summary",
                            "value": state._format_update_missing(missing_update),
                        }
                    ]
                    tables, charts = {}, []
                else:
                    # 누락된 입력을 확인한다.
                    missing = state._get_missing_fields(merged_params)
                    pending_action = None
                    
                    # 스트림에서도 동일한 위젯 로직 적용
                    target_keys = ["temperature", "size", "capacity", "voltage"]
                    missing_targets = [k for k in missing if k in target_keys]

                    if missing_targets:
                         # 폼 필드를 구성한다.
                        fields = []
                        for key in target_keys:
                            current_val = merged_params.dict().get(key)
                            field_def = {
                                "key": key,
                                "label": state.INPUT_LABEL_MAP.get(key, key),
                                "type": "text",
                                "value": current_val or ""
                            }
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
                            fields.append(field_def)

                        blocks = [
                            {
                                "type": "input_form",
                                "form_id": "mlcc_basic_params",
                                "title": "시뮬레이션 조건 입력",
                                "description": "다음 핵심 정보를 입력해주세요.",
                                "fields": fields,
                                "submit_label": "시뮬레이션 시작",
                                "submitted": False
                            }
                        ]
                        tables, charts = {}, []
                        stage_notes = {}
                    else:
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
                            # 진행 로그 이벤트를 모아둔다.
                            progress_events: list[list[dict[str, Any]]] = []

                            def progress_cb(stage: str) -> None:
                                # 단계 진행 로그를 수집한다.
                                logs = state._build_progress_logs(
                                    route,
                                    action,
                                    session_state.get("stage_status"),
                                    current_stage=stage,
                                    is_final=False,
                                )
                                if logs:
                                    progress_events.append(logs)

                            tables, charts, stage_notes, gap = (
                                db_production.build_simulation_from_db(
                                    merged_params,
                                    session_state["configs"],
                                    session_state["selections"],
                                    session_state["user_prefs"],
                                    dirty_stages=dirty_stages,
                                    progress_cb=progress_cb,
                                )
                            )
                            # 수집한 진행 로그를 순서대로 전송한다.
                            for logs in progress_events:
                                yield _format_sse("progress", {"logs": logs})
                            if gap:
                                gap_context = _build_gap_context(gap)
                                question = await agents._build_gap_question(
                                    gap_context
                                )
                                pending_action, blocks = _build_gap_pending_payload(
                                    gap, question
                                )
                                session_state["last_gap"] = gap_context
                                # gap이면 최종 응답만 전송하고 종료한다.
                                state._apply_table_highlights(
                                    tables, session_state.get("selections", {})
                                )
                                state._update_state(
                                    session_state,
                                    merged_params,
                                    tables,
                                    charts,
                                    blocks,
                                    stage_notes,
                                    missing,
                                    request.demo,
                                    dirty_stages=dirty_stages,
                                    pending_action=pending_action,
                                )
                                final_payload = _build_final_payload(
                                    route,
                                    action,
                                    session_state,
                                    blocks,
                                    tables,
                                    charts,
                                    debug_note="final_stream_response",
                                )
                                yield _format_sse("final", final_payload)
                                return
                            else:
                                pending_action = None
                                # 브리핑 생성 전에 사용할 빈 블록을 준비한다.
                                blocks = []
                    # 테이블 강조 표시를 적용한다.
                    state._apply_table_highlights(tables, session_state["selections"])
                    # LLM에 전달할 요약본을 만든다.
                    llm_tables, llm_charts = state._build_llm_payload(
                        tables, charts, session_state["configs"]
                    )
                    if not missing and not pending_action:
                        # 브리핑 작성 단계 로그를 전송한다.
                        progress_logs = state._build_progress_logs(
                            route,
                            action,
                            session_state.get("stage_status"),
                            current_stage="1-8",
                            is_final=False,
                        )
                        yield _format_sse("progress", {"logs": progress_logs})
                        # 변경 시작 단계를 정리한다.
                        briefing_start = None
                        briefing_hint = None
                        if had_results and dirty_stages:
                            briefing_start = state._pick_briefing_start_stage(
                                dirty_stages
                            )
                            briefing_hint = state._build_briefing_hint(briefing_start)
                        # 브리핑 순서를 단계 기준으로 만든다.
                        briefing_sequence = state._build_briefing_sequence(
                            stage_notes, briefing_start
                        )
                        briefing_tables, briefing_charts, _ = (
                            state._filter_briefing_outputs(
                                llm_tables, llm_charts, briefing_start
                            )
                        )
                        blocks = await agents._build_briefing_blocks(
                            briefing_tables,
                            briefing_charts,
                            briefing_hint,
                            briefing_sequence,
                        )
                        # 블록 참조 키를 실제 데이터 키로 정리한다.
                        blocks = state._normalize_block_refs(blocks, tables, charts)
                    if not missing and not pending_action:
                        # 이전 raw 출력과 병합해 누락된 표/차트를 보정한다.
                        tables, charts = state._merge_raw_outputs_with_history(
                            session_state,
                            tables,
                            charts,
                            dirty_stages,
                        )
                        # 병합된 테이블에 강조 표시를 다시 적용한다.
                        state._apply_table_highlights(
                            tables,
                            session_state["selections"],
                        )
                    state._update_state(
                        session_state,
                        merged_params,
                        tables,
                        charts,
                        blocks,
                        stage_notes,
                        missing,
                        request.demo,
                        llm_tables=llm_tables,
                        llm_charts=llm_charts,
                        dirty_stages=dirty_stages,
                        pending_action=pending_action,
                    )
                    # 재실행 완료 단계의 dirty를 해소한다.
                    if not missing and not pending_action:
                        state._mark_clean(session_state, dirty_stages)
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
