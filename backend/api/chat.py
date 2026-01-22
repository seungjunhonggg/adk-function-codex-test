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
        print("route = ", route, "command_action = ",command.action)
        if command.action == "explain_stage":
            # 설명 요청은 별도로 처리한다.
            action = "explain_stage"
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
            # 기존 브리핑 완료 여부를 확인한다.
            had_results = bool(session_state["stage_status"].get("1-8", {}).get("done"))
            # 입력과 변경 요청을 함께 파싱한다.
            input_params = await agents._parse_input_with_llm(request.message)
            update = await agents._parse_update_with_llm(request.message)
            missing_update = update.missing_fields or []
            if not had_results:
                # 결과가 없으면 업데이트 누락 처리를 건너뛴다.
                missing_update = []
            # 변경 전 상태를 보관한다.
            current_input = schemas.InputParams(**session_state["input_params"])
            current_selections = dict(session_state["selections"])
            current_configs = dict(session_state["configs"])
            current_prefs = dict(session_state["user_prefs"])
            # 입력값을 병합한다.
            merged_params = state._merge_input_params(
                session_state["input_params"], input_params
            )
            if update.input_params:
                merged_params = state._merge_input_params(
                    merged_params.dict(), update.input_params
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
            dirty_stages = state._collect_dirty_stages(changed_fields + missing_update)
            # 변경값을 상태에 반영한다.
            session_state["input_params"] = merged_params.dict()
            state._apply_update_fields(session_state, update)
            # dirty 상태를 표시한다.
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
                
                # 4가지 핵심 파라미터가 누락된 경우 input_form 블록을 생성한다.
                # 사용자 요청: 온도, 크기, 용량, 전압
                target_keys = ["temperature", "size", "capacity", "voltage"]
                missing_targets = [k for k in missing if k in target_keys]
                
                if missing_targets:
                    # 폼 필드를 구성한다.
                    fields = []
                    # 순서대로 필드를 추가한다.
                    for key in target_keys:
                        current_val = merged_params.dict().get(key)
                        field_def = {
                            "key": key,
                            "label": state.INPUT_LABEL_MAP.get(key, key),
                            "type": "text", # 기본값 text
                            "value": current_val or ""
                        }
                        
                        # 각 필드별 특화 설정
                        if key == "temperature":
                            field_def["type"] = "select"
                            field_def["options"] = ["A", "B", "O"]
                            field_def["unit"] = "℃"
                        elif key == "voltage":
                            field_def["type"] = "number"
                            field_def["unit"] = "V"
                        elif key == "size":
                            field_def["type"] = "select"
                            field_def["options"] = ["1005", "1608", "2012", "3216"]
                        elif key == "capacity":
                            field_def["type"] = "number"
                            field_def["unit"] = "pF" # 기본 단위 표시
                            # 단위 선택 옵션 추가 (프론트엔드에서 렌더링 및 자동 변환 처리)
                            field_def["unit_options"] = ["pF", "nF", "uF"]
                            
                        fields.append(field_def)

                    blocks = [
                        {
                            "type": "input_form",
                            "form_id": "mlcc_basic_params",
                            "title": "시뮬레이션 조건 입력",
                            "description": "MLCC 시뮬레이션을 위해 다음 핵심 정보를 입력해주세요.",
                            "fields": fields,
                            "submit_label": "시뮬레이션 시작",
                            "submitted": False
                        }
                    ]
                    tables, charts = {}, []
                    stage_notes = {}
                else:
                    if request.demo:
                        # 시뮬레이션 결과를 만든다(데모).
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
                        # 시뮬레이션 결과를 만든다(DB).
                        tables, charts, stage_notes = (
                            db_production.build_simulation_from_db(
                                merged_params,
                                session_state["configs"],
                                session_state["selections"],
                                session_state["user_prefs"],
                                dirty_stages=dirty_stages,
                            )
                        )
                        # 브리핑 생성 전에 사용할 빈 블록을 준비한다.
                        blocks = []
                # 테이블 강조 표시를 적용한다.
                state._apply_table_highlights(tables, session_state["selections"])
                # LLM에 전달할 요약본을 만든다.
                llm_tables, llm_charts = state._build_llm_payload(
                    tables, charts, session_state["configs"]
                )
                if not missing:
                    # 변경 시작 단계를 정리한다.
                    briefing_start = None
                    briefing_hint = None
                    if had_results and dirty_stages:
                        briefing_start = state._pick_briefing_start_stage(dirty_stages)
                        briefing_hint = state._build_briefing_hint(briefing_start)
                    # 브리핑 범위를 단계별로 정리한다.
                    briefing_tables, briefing_charts, _ = state._filter_briefing_outputs(
                        llm_tables, llm_charts, briefing_start
                    )
                    # 브리핑 순서를 단계 기준으로 만든다.
                    briefing_sequence = state._build_briefing_sequence(
                        stage_notes, briefing_start
                    )
                    blocks = await agents._build_briefing_blocks(
                        briefing_tables,
                        briefing_charts,
                        briefing_hint,
                        briefing_sequence,
                    )
                    # 블록 참조 키를 실제 데이터 키로 정리한다.
                    blocks = state._normalize_block_refs(blocks, tables, charts)
                if not missing:
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
    )
    if progress_logs:
        blocks = [{"type": "progress_log", "logs": progress_logs}] + blocks
    # 시뮬레이션 응답 디버그 로그를 남긴다.
    if route == "simulation":
        _log_briefing_debug(
            "final_response",
            request.session_id,
            route,
            action,
            blocks,
            tables,
            charts,
        )
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
            if command.action == "explain_stage":
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
                input_params = await agents._parse_input_with_llm(request.message)
                update = await agents._parse_update_with_llm(request.message)
                missing_update = update.missing_fields or []
                if not had_results:
                    # 결과가 없으면 업데이트 누락 처리를 건너뛴다.
                    missing_update = []
                # 입력 변경 여부를 확인한다.
                current_input = schemas.InputParams(**session_state["input_params"])
                current_selections = dict(session_state["selections"])
                current_configs = dict(session_state["configs"])
                current_prefs = dict(session_state["user_prefs"])
                # 입력값을 병합한다.
                merged_params = state._merge_input_params(
                    session_state["input_params"], input_params
                )
                if update.input_params:
                    merged_params = state._merge_input_params(
                        merged_params.dict(), update.input_params
                    )
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
                dirty_stages = state._collect_dirty_stages(
                    changed_fields + missing_update
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

                            tables, charts, stage_notes = (
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
                            # 브리핑 생성 전에 사용할 빈 블록을 준비한다.
                            blocks = []
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
                    if not missing:
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
        )
        if final_logs:
            blocks = [{"type": "progress_log", "logs": final_logs}] + blocks
        # 시뮬레이션 스트림 응답 디버그 로그를 남긴다.
        if route == "simulation":
            _log_briefing_debug(
                "final_stream_response",
                request.session_id,
                route,
                action,
                blocks,
                tables,
                charts,
            )
        # 최종 응답을 전송한다.
        yield _format_sse(
            "final",
            {"route": route, "blocks": blocks, "tables": tables, "charts": charts},
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
