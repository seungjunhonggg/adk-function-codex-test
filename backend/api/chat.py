import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core import agents, demo, schemas, state
from agents import SQLiteSession

router = APIRouter()


def _format_sse(event: str, payload: dict) -> str:
    # SSE 포맷 문자열을 만든다.
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/chat", response_model=schemas.ChatResponse)
async def api_chat(request: schemas.ChatRequest) -> schemas.ChatResponse:
    # 요청을 라우팅한다.
    session = SQLiteSession(request.session_id, "conversation_123")
    route = await agents._route_with_llm(session, request.message)
    # 세션 상태를 가져온다.
    session_state = state._get_session_state(request.session_id)
    action: str | None = None
    missing: list[str] = []
    # 라우트에 맞는 기본 응답을 만든다.
    if route == "simulation":
        # 상태 힌트를 포함해 커맨드를 결정한다.
        command_hint = state._build_command_hint(session_state)
        command_message = f"{command_hint}\n\n[사용자 메시지]\n{request.message}"
        command = await agents._decide_command_with_llm(session, command_message)
        action = command.action
        print("route = ", route, "command_action = ",command.action)
        if command.action == "explain_stage":
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
        elif command.action == "update_input":
            # 변경 요청을 파싱한다.
            update = await agents._parse_update_with_llm(request.message)
            print("update 제대로됐나", update.selections.reference_lot_id)
            missing = update.missing_fields or []
            # 변경 전 상태를 보관한다.
            current_input = schemas.InputParams(**session_state["input_params"])
            current_selections = dict(session_state["selections"])
            current_configs = dict(session_state["configs"])
            current_prefs = dict(session_state["user_prefs"])
            # 변경 후 입력값을 만든다.
            incoming_params = update.input_params or schemas.InputParams()
            merged_params = state._merge_input_params(
                current_input.dict(), incoming_params
            )
            # 실제 변경된 항목만 추린다.
            changed_input = state._extract_changed_keys(
                current_input.dict(), merged_params.dict()
            )
            changed_selections = state._extract_changed_keys(
                current_selections,
                update.selections.dict() if update.selections else {},
            )
            changed_configs = state._extract_changed_keys(
                current_configs,
                update.configs.dict() if update.configs else {},
                skip_empty=False,
            )
            changed_prefs = state._extract_changed_keys(
                current_prefs,
                update.user_prefs.dict() if update.user_prefs else {},
            )
            changed_fields = (
                changed_input + changed_selections + changed_configs + changed_prefs
            )
            dirty_stages = state._collect_dirty_stages(changed_fields + missing)
            # 변경값을 상태에 반영한다.
            session_state["input_params"] = merged_params.dict()
            state._apply_update_fields(session_state, update)
            # dirty 상태를 표시한다.
            state._mark_dirty(session_state, dirty_stages)
            # 값이 없는 변경 요청이면 pending_action으로 보류한다.
            if missing:
                session_state["pending_action"] = state._build_pending_action(missing)
                session_state["history"].append(
                    {
                        "action": "update_input_pending",
                        "payload": {"missing": missing},
                        "at": state._utc_now(),
                    }
                )
                blocks = [
                    {
                        "type": "text",
                        "section": "summary",
                        "value": state._format_update_missing(missing),
                    }
                ]
                tables, charts = {}, []
            else:
                # 변경 후 누락된 입력을 확인한다.
                missing = state._get_missing_fields(merged_params)
                # 변경 후 시뮬레이션 결과를 만든다.
                blocks, tables, charts, stage_notes = demo._build_simulation_stub(
                    request,
                    merged_params,
                    session_state["configs"],
                    session_state["selections"],
                    session_state["user_prefs"],
                )
                # 테이블 강조 표시를 적용한다.
                state._apply_table_highlights(tables, session_state["selections"])
                # LLM에 전달할 요약본을 만든다.
                llm_tables, llm_charts = state._build_llm_payload(
                    tables, charts, session_state["configs"]
                )
                if not missing:
                    briefing_start = state._pick_briefing_start_stage(dirty_stages)
                    # 브리핑 범위를 단계별로 정리한다.
                    briefing_tables, briefing_charts, _ = state._filter_briefing_outputs(
                        llm_tables, llm_charts, briefing_start
                    )
                    # 브리핑 순서를 단계 기준으로 만든다.
                    briefing_sequence = state._build_briefing_sequence(
                        stage_notes, briefing_start
                    )
                    # 변경 반영 문구를 준비한다.
                    briefing_hint = state._build_briefing_hint(briefing_start)
                    blocks = await agents._build_briefing_blocks(
                        briefing_tables,
                        briefing_charts,
                        briefing_hint,
                        briefing_sequence,
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
                )
                # 재실행 완료 단계의 dirty를 해소한다.
                if not missing:
                    state._mark_clean(session_state, dirty_stages)
        else:
            input_params = await agents._parse_input_with_llm(request.message)
            # 입력 변경 여부를 확인한다.
            current_input = schemas.InputParams(**session_state["input_params"])
            merged_params = state._merge_input_params(
                session_state["input_params"], input_params
            )
            changed_input = state._extract_changed_keys(
                current_input.dict(), merged_params.dict()
            )
            dirty_stages = state._collect_dirty_stages(changed_input)
            # 입력 변경에 따라 dirty를 표시한다.
            state._mark_dirty(session_state, dirty_stages)
            missing = state._get_missing_fields(merged_params)
            blocks, tables, charts, stage_notes = demo._build_simulation_stub(
                request,
                merged_params,
                session_state["configs"],
                session_state["selections"],
                session_state["user_prefs"],
            )
            # 테이블 강조 표시를 적용한다.
            state._apply_table_highlights(tables, session_state["selections"])
            # LLM에 전달할 요약본을 만든다.
            llm_tables, llm_charts = state._build_llm_payload(
                tables, charts, session_state["configs"]
            )
            if not missing:
                # 브리핑 순서를 단계 기준으로 만든다.
                briefing_sequence = state._build_briefing_sequence(stage_notes, None)
                briefing_tables, briefing_charts, _ = state._filter_briefing_outputs(
                    llm_tables, llm_charts, None
                )
                blocks = await agents._build_briefing_blocks(
                    briefing_tables, briefing_charts, None, briefing_sequence
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
            )
            # 재실행 완료 단계의 dirty를 해소한다.
            if not missing:
                state._mark_clean(session_state, dirty_stages)
    else:
        # 캐주얼 응답을 LLM으로 생성한다.
        blocks = await agents._build_casual_blocks(session,request.message)
        # 캐주얼 응답에는 표/차트가 없다.
        tables, charts = {}, []
    # 진행 로그를 블록 앞에 추가한다.
    progress_logs = state._build_progress_logs(
        route,
        action,
        session_state.get("stage_status"),
        missing,
        session_state.get("last_error"),
    )
    if progress_logs:
        blocks = [{"type": "progress_log", "logs": progress_logs}] + blocks
    # 응답을 구성한다.
    return schemas.ChatResponse(route=route, blocks=blocks, tables=tables, charts=charts)


@router.post("/api/chat/stream")
async def api_chat_stream(request: schemas.ChatRequest) -> StreamingResponse:
    # SSE 스트림 응답을 만든다.
    session = SQLiteSession(request.session_id, "conversation_123")

    async def event_stream():
        # 라우팅을 먼저 수행한다.
        route = await agents._route_with_llm(session, request.message)
        # 세션 상태를 가져온다.
        session_state = state._get_session_state(request.session_id)
        action: str | None = None
        missing: list[str] = []
        blocks: list[dict[str, Any]] = []
        tables: dict[str, Any] = {}
        charts: list[dict[str, Any]] = []

        if route == "simulation":
            # 커맨드를 결정한다.
            command_hint = state._build_command_hint(session_state)
            command_message = f"{command_hint}\n\n[사용자 메시지]\n{request.message}"
            command = await agents._decide_command_with_llm(session, command_message)
            action = command.action
            if command.action == "explain_stage":
                # 설명 단계 진행 로그를 전송한다.
                progress_logs = state._build_progress_logs(
                    route,
                    action,
                    session_state.get("stage_status"),
                    None,
                    None,
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
            elif command.action == "update_input":
                # 입력 변경 단계 로그를 전송한다.
                progress_logs = state._build_progress_logs(
                    route,
                    action,
                    session_state.get("stage_status"),
                    None,
                    None,
                    current_stage="1-1",
                    is_final=False,
                )
                yield _format_sse("progress", {"logs": progress_logs})
                # 변경 요청을 파싱한다.
                update = await agents._parse_update_with_llm(request.message)
                missing = update.missing_fields or []
                # 변경 전 상태를 보관한다.
                current_input = schemas.InputParams(**session_state["input_params"])
                current_selections = dict(session_state["selections"])
                current_configs = dict(session_state["configs"])
                current_prefs = dict(session_state["user_prefs"])
                # 변경 후 입력값을 만든다.
                incoming_params = update.input_params or schemas.InputParams()
                merged_params = state._merge_input_params(
                    current_input.dict(), incoming_params
                )
                # 실제 변경된 항목만 추린다.
                changed_input = state._extract_changed_keys(
                    current_input.dict(), merged_params.dict()
                )
                changed_selections = state._extract_changed_keys(
                    current_selections,
                    update.selections.dict() if update.selections else {},
                )
                changed_configs = state._extract_changed_keys(
                    current_configs,
                    update.configs.dict() if update.configs else {},
                    skip_empty=False,
                )
                changed_prefs = state._extract_changed_keys(
                    current_prefs,
                    update.user_prefs.dict() if update.user_prefs else {},
                )
                changed_fields = (
                    changed_input + changed_selections + changed_configs + changed_prefs
                )
                dirty_stages = state._collect_dirty_stages(changed_fields + missing)
                # 변경값을 상태에 반영한다.
                session_state["input_params"] = merged_params.dict()
                state._apply_update_fields(session_state, update)
                # dirty 상태를 표시한다.
                state._mark_dirty(session_state, dirty_stages)
                # 값이 없는 변경 요청이면 pending_action으로 보류한다.
                if missing:
                    session_state["pending_action"] = state._build_pending_action(
                        missing
                    )
                    session_state["history"].append(
                        {
                            "action": "update_input_pending",
                            "payload": {"missing": missing},
                            "at": state._utc_now(),
                        }
                    )
                    blocks = [
                        {
                            "type": "text",
                            "section": "summary",
                            "value": state._format_update_missing(missing),
                        }
                    ]
                    tables, charts = {}, []
                else:
                    # 변경 후 누락된 입력을 확인한다.
                    missing = state._get_missing_fields(merged_params)
                    # 변경 후 시뮬레이션 결과를 만든다.
                    blocks, tables, charts, stage_notes = demo._build_simulation_stub(
                        request,
                        merged_params,
                        session_state["configs"],
                        session_state["selections"],
                        session_state["user_prefs"],
                    )
                    # 테이블 강조 표시를 적용한다.
                    state._apply_table_highlights(tables, session_state["selections"])
                    # LLM에 전달할 요약본을 만든다.
                    llm_tables, llm_charts = state._build_llm_payload(
                        tables, charts, session_state["configs"]
                    )
                    if not missing:
                        # 브리핑 작성 단계 로그를 전송한다.
                        progress_logs = state._build_progress_logs(
                            route,
                            action,
                            session_state.get("stage_status"),
                            None,
                            None,
                            current_stage="1-8",
                            is_final=False,
                        )
                        yield _format_sse("progress", {"logs": progress_logs})
                        briefing_start = state._pick_briefing_start_stage(dirty_stages)
                        # 브리핑 범위를 단계별로 정리한다.
                        briefing_tables, briefing_charts, _ = (
                            state._filter_briefing_outputs(
                                llm_tables, llm_charts, briefing_start
                            )
                        )
                        # 브리핑 순서를 단계 기준으로 만든다.
                        briefing_sequence = state._build_briefing_sequence(
                            stage_notes, briefing_start
                        )
                        # 변경 반영 문구를 준비한다.
                        briefing_hint = state._build_briefing_hint(briefing_start)
                        blocks = await agents._build_briefing_blocks(
                            briefing_tables,
                            briefing_charts,
                            briefing_hint,
                            briefing_sequence,
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
                    )
                    # 재실행 완료 단계의 dirty를 해소한다.
                    if not missing:
                        state._mark_clean(session_state, dirty_stages)
            else:
                # 입력 수집 단계 로그를 전송한다.
                progress_logs = state._build_progress_logs(
                    route,
                    action,
                    session_state.get("stage_status"),
                    None,
                    None,
                    current_stage="1-1",
                    is_final=False,
                )
                yield _format_sse("progress", {"logs": progress_logs})
                input_params = await agents._parse_input_with_llm(request.message)
                # 입력 변경 여부를 확인한다.
                current_input = schemas.InputParams(**session_state["input_params"])
                merged_params = state._merge_input_params(
                    session_state["input_params"], input_params
                )
                changed_input = state._extract_changed_keys(
                    current_input.dict(), merged_params.dict()
                )
                dirty_stages = state._collect_dirty_stages(changed_input)
                # 입력 변경에 따라 dirty를 표시한다.
                state._mark_dirty(session_state, dirty_stages)
                missing = state._get_missing_fields(merged_params)
                blocks, tables, charts, stage_notes = demo._build_simulation_stub(
                    request,
                    merged_params,
                    session_state["configs"],
                    session_state["selections"],
                    session_state["user_prefs"],
                )
                # 테이블 강조 표시를 적용한다.
                state._apply_table_highlights(tables, session_state["selections"])
                # LLM에 전달할 요약본을 만든다.
                llm_tables, llm_charts = state._build_llm_payload(
                    tables, charts, session_state["configs"]
                )
                if not missing:
                    # 브리핑 작성 단계 로그를 전송한다.
                    progress_logs = state._build_progress_logs(
                        route,
                        action,
                        session_state.get("stage_status"),
                        None,
                        None,
                        current_stage="1-8",
                        is_final=False,
                    )
                    yield _format_sse("progress", {"logs": progress_logs})
                    # 브리핑 순서를 단계 기준으로 만든다.
                    briefing_sequence = state._build_briefing_sequence(stage_notes, None)
                    briefing_tables, briefing_charts, _ = (
                        state._filter_briefing_outputs(
                            llm_tables, llm_charts, None
                        )
                    )
                    blocks = await agents._build_briefing_blocks(
                        briefing_tables, briefing_charts, None, briefing_sequence
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
                )
                # 재실행 완료 단계의 dirty를 해소한다.
                if not missing:
                    state._mark_clean(session_state, dirty_stages)
        else:
            # 캐주얼 응답 로그를 전송한다.
            progress_logs = state._build_progress_logs(
                route,
                None,
                session_state.get("stage_status"),
                None,
                None,
                is_final=False,
            )
            yield _format_sse("progress", {"logs": progress_logs})
            # 캐주얼 응답을 LLM으로 생성한다.
            blocks = await agents._build_casual_blocks(session, request.message)
            # 캐주얼 응답에는 표/차트가 없다.
            tables, charts = {}, []

        # 최종 진행 로그를 만든다.
        final_logs = state._build_progress_logs(
            route,
            action,
            session_state.get("stage_status"),
            missing,
            session_state.get("last_error"),
        )
        if final_logs:
            blocks = [{"type": "progress_log", "logs": final_logs}] + blocks
        # 최종 응답을 전송한다.
        yield _format_sse(
            "final",
            {"route": route, "blocks": blocks, "tables": tables, "charts": charts},
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
