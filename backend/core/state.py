from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable

from .agents import _build_explain_answer
from . import db_production
from .schemas import InputParams, UpdateDecision


# 입력 라벨 맵을 정의한다.
INPUT_LABEL_MAP = {
    "temperature": "온도",
    "voltage": "전압",
    "size": "크기",
    "capacity": "용량",
    "chip_prod_id": "CHIP 기종",
}

# 변경 라벨 맵을 정의한다.
UPDATE_LABEL_MAP = {
    "reference_lot_id": "레퍼런스 LOT",
    "chip_type_ids": "칩기종",
    "top_k": "top-k",
    "chart_type": "차트 타입",
}

# pending_action 규칙을 정의한다.
PENDING_ACTION_RULES = {
    "reference_lot_id": ("update_reference_lot", "1-3"),
    "chip_type_ids": ("update_chip_type", "1-2"),
    "top_k": ("update_top_k", "1-5"),
    "chart_type": ("update_chart_type", "1-7"),
}

# 단계 순서를 정의한다.
STAGE_ORDER = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]

# dirty 규칙을 정의한다.
DIRTY_STAGE_RULES = {
    "input_params": ["1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"],
    "chip_type_ids": ["1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"],
    "reference_lot_id": ["1-4", "1-5", "1-6", "1-7", "1-8"],
    "top_k": ["1-5", "1-6", "1-7", "1-8"],
    "chart_type": ["1-7", "1-8"],
}

# 단계별 표 키를 정의한다.
_STAGE_TABLE_KEYS = {
    "1-1": ["input_params_table"],
    "1-2": ["chip_type_candidates_table"],
    "1-3": ["reference_lot_candidates_table"],
    "1-4": ["reference_lot_table"],
    "1-5": ["top_k_table"],
    "1-6": ["recent_similar_table"],
    "1-7": ["defect_rate_table"],
}

# 단계별 차트 키를 정의한다.
_STAGE_CHART_IDS = {
    "1-7": ["defect_rate_summary"],
}

# 진행 로그 문구를 정의한다.
_PROGRESS_ROUTE_TEXT = {
    "casual": "답변 생성하는 중",
    "update_input": "변경 요청 반영하는 중",
    "explain_stage": "단계 근거 설명하는 중",
}

_PROGRESS_STAGE_TEXT = {
    "1-1": "입력 조건 확인하는 중",
    "1-2": "칩기종 후보 찾는 중",
    "1-3": "레퍼런스 LOT 선정하는 중",
    "1-4": "시뮬레이션 payload 만드는 중",
    "1-5": "top-k 후보 생성하는 중",
    "1-6": "최근 유사 설계 조회하는 중",
    "1-7": "불량률 지표 집계하는 중",
    "1-8": "브리핑 작성하는 중",
}

# LLM 입력 요약 한도를 정의한다.
_LLM_PAYLOAD_MAX_CHARS = 8000
_LLM_MAX_ROWS_DEFAULT = 8
_LLM_MAX_COLS_DEFAULT = 6
_LLM_MAX_SERIES_DEFAULT = 3
_LLM_MAX_POINTS_DEFAULT = 6
_RAW_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_outputs"
_CORE_PARAM_LABEL_MAP = {
    "active_powder_base": "활성파우더베이스",
    "active_powder_additives": "활성파우더첨가제",
    "ldn_avr_value": "LDN 평균값",
    "cast_dsgn_thk": "캐스팅 설계 두께",
}


# 세션 스토어를 준비한다.
_SESSION_STORE: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    # 현재 시각을 ISO 문자열로 만든다.
    return datetime.utcnow().isoformat() + "Z"


def _init_stage_status() -> dict[str, dict[str, Any]]:
    # 단계 상태를 초기화한다.
    status: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        status[stage] = {"done": False, "dirty": True, "updated_at": None}
    return status


def _init_session_state(session_id: str) -> dict[str, Any]:
    # 세션 상태를 초기화한다.
    return {
        "session_id": session_id,
        "input_params": {
            "temperature": None,
            "voltage": None,
            "size": None,
            "capacity": None,
        },
        "selections": {
            "chip_type_ids": None,
            "reference_lot_id": None,
        },
        "configs": {
            "top_k": 5,
            "top_k_sort": "rank",
            "core_match_params": [
                "active_powder_base",
                "active_powder_additives",
                "ldn_avr_value",
                "cast_dsgn_thk",
            ],
        },
        "stage_status": _init_stage_status(),
        "stage_outputs": {"tables": {}, "charts": [], "briefing_blocks": []},
        "stage_notes": {},
        "raw_refs": {
            "top_k_raw_id": None,
            "defect_raw_id": None,
            "stage_outputs_path": None,
            "stage_outputs_saved_at": None,
        },
        "pending_action": None,
        "last_gap": None,
        "last_explain_stage": None,
        "history": [],
        "user_prefs": {"chart_type": "bar", "language": "ko"},
        "last_error": None,
    }


def _merge_session_state(
    base_state: dict[str, Any],
    stored_state: dict[str, Any],
) -> dict[str, Any]:
    # 기본 상태에 저장된 값을 덮어쓴다.
    for key, value in stored_state.items():
        # 중첩 dict는 얕게 병합한다.
        if isinstance(value, dict) and isinstance(base_state.get(key), dict):
            base_state[key].update(value)
            continue
        # 그 외에는 그대로 덮어쓴다.
        base_state[key] = value
    # 병합된 상태를 반환한다.
    return base_state


def _get_session_state(session_id: str, use_db: bool = False) -> dict[str, Any]:
    # DB 사용 여부에 따라 분기한다.
    if use_db:
        # DB에서 상태를 읽는다.
        stored_state = db_production.fetch_session_state(session_id)
        # 저장된 상태가 있으면 병합해서 반환한다.
        if stored_state:
            base_state = _init_session_state(session_id)
            return _merge_session_state(base_state, stored_state)
        # 상태가 없으면 초기 상태를 저장한다.
        fresh_state = _init_session_state(session_id)
        db_production.upsert_session_state(session_id, fresh_state)
        return fresh_state
    # 인메모리 스토어를 사용한다.
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = _init_session_state(session_id)
    return _SESSION_STORE[session_id]


def _reset_session_state(session_id: str, use_db: bool = False) -> dict[str, Any]:
    # 새 상태를 만든다.
    fresh_state = _init_session_state(session_id)
    # DB 사용 여부에 따라 저장한다.
    if use_db:
        db_production.upsert_session_state(session_id, fresh_state)
        return fresh_state
    # 인메모리 스토어에 저장한다.
    _SESSION_STORE[session_id] = fresh_state
    return _SESSION_STORE[session_id]


def _save_session_state(session_state: dict[str, Any], use_db: bool = False) -> None:
    # DB 사용 여부에 따라 저장한다.
    if use_db:
        session_id = session_state.get("session_id")
        if not session_id:
            return
        db_production.upsert_session_state(session_id, session_state)
        return
    # 인메모리 스토어에 저장한다.
    session_id = session_state.get("session_id")
    if session_id:
        _SESSION_STORE[session_id] = session_state


def _build_command_hint(state: dict[str, Any]) -> str:
    # 커맨드 에이전트 힌트를 만든다.
    stage_status = state.get("stage_status", {})
    # 브리핑 완료 여부를 확인한다.
    has_results = bool(stage_status.get("1-8", {}).get("done"))
    # 입력값 완성 여부를 확인한다.
    input_params = InputParams(**state.get("input_params", {}))
    missing = _get_missing_fields(input_params)
    has_input_complete = not missing
    # 보류된 액션을 확인한다.
    pending = state.get("pending_action")
    pending_action = pending.get("action") if isinstance(pending, dict) else None
    # 최근 액션을 확인한다.
    history = state.get("history", [])
    last_action = history[-1].get("action") if history else None
    # 힌트 텍스트를 구성한다.
    return (
        "[STATE_HINT]\n"
        f"- has_results: {str(has_results).lower()}\n"
        f"- has_input_complete: {str(has_input_complete).lower()}\n"
        f"- pending_action: {pending_action or 'none'}\n"
        f"- last_action: {last_action or 'none'}"
    )


def _build_progress_logs(
    route: str,
    action: str | None,
    stage_status: dict[str, Any] | None,
    current_stage: str | None = None,
    is_final: bool = True,
) -> list[dict[str, Any]]:
    # 진행 로그를 만든다(완료/진행중만 사용).
    logs: list[dict[str, Any]] = []
    status = "done" if is_final else "in_progress"
    if route == "casual":
        return [{"text": _PROGRESS_ROUTE_TEXT["casual"], "status": status}]
    if action == "explain_stage":
        return [{"text": _PROGRESS_ROUTE_TEXT["explain_stage"], "status": status}]
    if route != "simulation":
        return []
    status_map = stage_status or {}
    for stage in STAGE_ORDER:
        text = _PROGRESS_STAGE_TEXT.get(stage)
        if not text:
            continue
        done = bool(status_map.get(stage, {}).get("done"))
        if done:
            logs.append({"text": text, "status": "done"})
            continue
        if current_stage and stage == current_stage:
            logs.append({"text": text, "status": "in_progress"})
    if current_stage and not logs:
        text = _PROGRESS_STAGE_TEXT.get(current_stage)
        if text:
            logs.append({"text": text, "status": "in_progress"})
    return logs


def _merge_input_params(
    current: dict[str, Any], incoming: InputParams
) -> InputParams:
    # 입력값을 병합한다.
    merged = dict(current or {})
    for key, value in incoming.dict().items():
        if value not in (None, ""):
            merged[key] = value
    return InputParams(**merged)


def _merge_update_and_collect_dirty(
    session_state: dict[str, Any],
    input_params: InputParams,
    update: UpdateDecision,
    missing_update: list[str] | None = None,
) -> tuple[InputParams, list[str]]:
    # 현재 입력값을 준비한다.
    current_input = InputParams(**session_state["input_params"])
    # 현재 선택값을 복사한다.
    current_selections = dict(session_state["selections"])
    # 현재 설정값을 복사한다.
    current_configs = dict(session_state["configs"])
    # 현재 사용자 설정을 복사한다.
    current_prefs = dict(session_state["user_prefs"])
    # 입력값을 병합한다.
    merged_params = _merge_input_params(session_state["input_params"], input_params)
    # 업데이트 입력값이 있으면 추가 병합한다.
    if update.input_params:
        merged_params = _merge_input_params(merged_params.dict(), update.input_params)
    # 입력 변경점을 계산한다.
    changed_input = _extract_changed_keys(
        current_input.dict(), merged_params.dict()
    )
    # 선택 변경점을 계산한다.
    changed_selections = _extract_changed_keys(
        current_selections,
        update.selections.dict() if update.selections else {},
    )
    # 설정 변경점을 계산한다.
    changed_configs = _extract_changed_keys(
        current_configs,
        update.configs.dict() if update.configs else {},
        skip_empty=False,
    )
    # 사용자 설정 변경점을 계산한다.
    changed_prefs = _extract_changed_keys(
        current_prefs,
        update.user_prefs.dict() if update.user_prefs else {},
    )
    # 변경 필드를 하나로 모은다.
    changed_fields = (
        changed_input + changed_selections + changed_configs + changed_prefs
    )
    # 누락 업데이트를 포함해 dirty 단계를 계산한다.
    dirty_stages = _collect_dirty_stages(
        changed_fields + (missing_update or [])
    )
    # 병합 결과와 dirty 단계를 반환한다.
    return merged_params, dirty_stages


def _update_stage_status(state: dict[str, Any], has_missing: bool, demo: bool) -> None:
    # 단계 상태를 업데이트한다.
    status = state["stage_status"]
    now = _utc_now()
    if has_missing:
        # 1-1 입력이 부족하면 여기서 멈춘다.
        status["1-1"]["done"] = False
        status["1-1"]["dirty"] = True
        status["1-1"]["updated_at"] = now
        return
    # 1-1 입력 완료 상태를 기록한다.
    status["1-1"]["done"] = True
    status["1-1"]["dirty"] = False
    status["1-1"]["updated_at"] = now
    if demo:
        # 데모는 기본적으로 완료 처리하되 dirty는 보존한다.
        for stage in STAGE_ORDER[1:]:
            if status[stage]["dirty"]:
                continue
            status[stage]["done"] = True
            status[stage]["dirty"] = False
            status[stage]["updated_at"] = now


def _update_state(
    state: dict[str, Any],
    input_params: InputParams,
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    stage_notes: dict[str, str],
    missing: list[str],
    demo: bool,
    llm_tables: dict[str, Any] | None = None,
    llm_charts: list[dict[str, Any]] | None = None,
    dirty_stages: list[str] | None = None,
    pending_action: dict[str, Any] | None = None,
) -> None:
    # 입력값을 저장한다.
    state["input_params"] = input_params.dict()
    # 원본 출력물을 외부에 저장한다.
    raw_refs = _store_raw_outputs(state, tables, charts)
    if raw_refs:
        state["raw_refs"].update(raw_refs)
    # LLM 요약 출력물을 준비한다.
    if llm_tables is None or llm_charts is None:
        llm_tables, llm_charts = _build_llm_payload(
            tables, charts, state.get("configs", {})
        )
    # dirty 단계가 있으면 기존 요약본과 병합한다.
    if dirty_stages:
        # 이전 요약본을 꺼낸다.
        previous_outputs = state.get("stage_outputs", {})
        previous_tables = previous_outputs.get("tables", {})
        previous_charts = previous_outputs.get("charts", [])
        # 이전 근거 노트를 꺼낸다.
        previous_notes = state.get("stage_notes", {})
        # 변경된 단계만 덮어쓰도록 병합한다.
        llm_tables, llm_charts, stage_notes = _merge_stage_outputs(
            previous_tables,
            previous_charts,
            previous_notes,
            llm_tables,
            llm_charts,
            stage_notes,
            dirty_stages,
        )
    # 출력물을 저장한다.
    state["stage_outputs"] = {
        "tables": llm_tables,
        "charts": llm_charts,
        "briefing_blocks": blocks,
    }
    # 단계 근거를 저장한다.
    state["stage_notes"] = stage_notes
    # 단계 상태를 갱신한다.
    _update_stage_status(state, bool(missing), demo)
    # pending_action을 저장한다.
    if pending_action is not None:
        state["pending_action"] = pending_action
    elif missing:
        state["pending_action"] = {
            "action": "collect_input",
            "target_stage": "1-1",
            "missing_fields": missing,
            "requested_at": _utc_now(),
        }
    else:
        state["pending_action"] = None
    # 히스토리를 기록한다.
    state["history"].append(
        {"action": "update_state", "payload": {"missing": missing}, "at": _utc_now()}
    )


def _normalize_stage(target_stage: str | None, stage_status: dict[str, Any]) -> str | None:
    # 요청 단계가 유효한지 확인한다.
    if not target_stage:
        return None
    if target_stage in stage_status:
        return target_stage
    return None


async def _build_explain_response(
    state: dict[str, Any], target_stage: str | None, question: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # 단계 근거 설명 응답을 만든다.
    if not target_stage:
        blocks = [
            {
                "type": "text",
                "section": "explain",
                "value": "어느 단계인지 알려줘. 예: 1-4",
            }
        ]
        return blocks, {}, []
    status = state["stage_status"].get(target_stage)
    if not status or not status.get("done"):
        blocks = [
            {
                "type": "text",
                "section": "explain",
                "value": f"{target_stage} 단계가 아직 실행되지 않았어요. 먼저 시뮬레이션을 진행할까요?",
            }
        ]
        return blocks, {}, []
    note = state.get("stage_notes", {}).get(target_stage)
    if not note:
        blocks = [
            {
                "type": "text",
                "section": "explain",
                "value": f"{target_stage} 단계 근거 요약이 아직 없어요. 필요한 항목을 알려줘.",
            }
        ]
        return blocks, {}, []
    # 단계별 표/차트를 고른다.
    tables = state.get("stage_outputs", {}).get("tables", {})
    charts = state.get("stage_outputs", {}).get("charts", [])
    table_keys = _STAGE_TABLE_KEYS.get(target_stage, [])
    chart_ids = _STAGE_CHART_IDS.get(target_stage, [])
    selected_tables = {key: tables[key] for key in table_keys if key in tables}
    selected_charts = [
        chart for chart in charts if chart.get("chart_id") in chart_ids
    ]
    # 설명용 payload를 하드캡으로 줄인다.
    selected_tables, selected_charts = _apply_payload_budget(
        selected_tables, selected_charts
    )
    # 설명용 컨텍스트를 만든다.
    context = {
        "question": question,
        "stage": target_stage,
        "stage_notes": note,
        "tables": selected_tables,
        "charts": selected_charts,
    }
    # LLM으로 설명을 생성한다.
    answer = await _build_explain_answer(context)
    # 설명 블록을 구성한다.
    blocks = [{"type": "text", "section": "explain", "value": answer}]
    for key in table_keys:
        if key in selected_tables:
            blocks.append({"type": "table_ref", "table_key": key})
    for chart_id in chart_ids:
        if any(chart.get("chart_id") == chart_id for chart in selected_charts):
            blocks.append({"type": "chart_ref", "chart_id": chart_id})
    return blocks, selected_tables, selected_charts


def _get_missing_fields(input_params: InputParams) -> list[str]:
    # 칩 기종 입력 여부를 확인한다.
    has_chip = bool(input_params.chip_prod_id)
    # 칩 기종이 있으면 누락을 비운다.
    if has_chip:
        return []
    # 기본 입력 필드만 누락 체크한다.
    required_fields = ["temperature", "voltage", "size", "capacity"]
    return [name for name in required_fields if not getattr(input_params, name, None)]


def _format_missing_summary(missing: list[str]) -> str:
    # 누락된 입력 안내 문구를 만든다.
    missing_labels = [INPUT_LABEL_MAP.get(name, name) for name in missing]
    return f"다음 입력이 필요합니다: {', '.join(missing_labels)}"


def _format_update_missing(missing: list[str]) -> str:
    # 변경 요청 누락 안내 문구를 만든다.
    missing_labels = [
        UPDATE_LABEL_MAP.get(name, INPUT_LABEL_MAP.get(name, name)) for name in missing
    ]
    return f"다음 값을 알려줘: {', '.join(missing_labels)}"


def _build_pending_action(missing: list[str]) -> dict[str, Any]:
    # pending_action을 만든다.
    if not missing:
        return {}
    primary = missing[0]
    action, target_stage = PENDING_ACTION_RULES.get(
        primary, ("update_input_params", "1-1")
    )
    return {
        "action": action,
        "target_stage": target_stage,
        "missing_fields": missing,
        "requested_at": _utc_now(),
    }


def _extract_changed_keys(
    current: dict[str, Any], incoming: dict[str, Any], skip_empty: bool = True
) -> list[str]:
    # 변경된 필드만 추린다.
    changed: list[str] = []
    for key, value in incoming.items():
        if value is None:
            continue
        if skip_empty and value == "":
            continue
        if current.get(key) != value:
            changed.append(key)
    return changed


def _collect_dirty_stages(fields: list[str]) -> list[str]:
    # 필드 목록으로 dirty 단계를 만든다.
    dirty: set[str] = set()
    for field in fields:
        rule_key = "input_params" if field in INPUT_LABEL_MAP else field
        for stage in DIRTY_STAGE_RULES.get(rule_key, []):
            dirty.add(stage)
    order_map = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    return sorted(dirty, key=lambda stage: order_map.get(stage, 99))


def _pick_briefing_start_stage(dirty_stages: list[str]) -> str | None:
    # 변경된 단계 중 가장 앞 단계를 고른다.
    if not dirty_stages:
        return None
    return dirty_stages[0]


def _collect_stage_range(start_stage: str | None) -> list[str]:
    # 브리핑 범위에 포함될 단계를 만든다.
    if not start_stage or start_stage not in STAGE_ORDER:
        return list(STAGE_ORDER)
    start_index = STAGE_ORDER.index(start_stage)
    return STAGE_ORDER[start_index:]


def _filter_briefing_outputs(
    tables: dict[str, Any], charts: list[dict[str, Any]], start_stage: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    # 브리핑 범위에 맞는 표/차트를 고른다.
    stages = _collect_stage_range(start_stage)
    table_keys: list[str] = []
    chart_ids: list[str] = []
    for stage in stages:
        table_keys.extend(_STAGE_TABLE_KEYS.get(stage, []))
        chart_ids.extend(_STAGE_CHART_IDS.get(stage, []))
    selected_tables = {key: tables[key] for key in table_keys if key in tables}
    if not chart_ids:
        selected_charts = list(charts)
    else:
        selected_charts = [
            chart for chart in charts if chart.get("chart_id") in chart_ids
        ]
    return selected_tables, selected_charts, stages


def _extract_reason_note(note: str | None) -> str:
    # 단계 근거만 추린다.
    if not note:
        return ""
    return note.splitlines()[0]


def _build_briefing_sequence(
    stage_notes: dict[str, str], start_stage: str | None
) -> list[dict[str, Any]]:
    # 브리핑 순서 목록을 만든다.
    stages = _collect_stage_range(start_stage)
    sequence: list[dict[str, Any]] = []
    for stage in stages:
        table_keys = list(_STAGE_TABLE_KEYS.get(stage, []))
        sequence.append(
            {
                "stage": stage,
                "note": _extract_reason_note(stage_notes.get(stage)),
                "table_keys": table_keys,
                "chart_ids": _STAGE_CHART_IDS.get(stage, []),
            }
        )
    return sequence


def _build_briefing_hint(start_stage: str | None) -> str | None:
    # 변경 반영 안내 문구를 만든다.
    if not start_stage:
        return None
    return f"{start_stage} 단계 변경사항을 반영했습니다. 첫 문장에서 짧게 언급하세요."


def _merge_chart_outputs(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # chart_id 기준으로 차트를 병합한다.
    merged = list(existing)
    # 새 차트를 순회한다.
    for chart in incoming:
        if not isinstance(chart, dict):
            continue
        # chart_id가 있으면 기존 차트를 제거한다.
        chart_id = chart.get("chart_id")
        if chart_id:
            merged = [
                item for item in merged if item.get("chart_id") != chart_id
            ]
        # 새 차트를 추가한다.
        merged.append(chart)
    return merged


def _merge_stage_outputs(
    previous_tables: dict[str, Any],
    previous_charts: list[dict[str, Any]],
    previous_notes: dict[str, str],
    new_tables: dict[str, Any],
    new_charts: list[dict[str, Any]],
    new_notes: dict[str, str],
    dirty_stages: list[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    # dirty 단계가 없으면 새 결과만 사용한다.
    if not dirty_stages:
        return new_tables, new_charts, new_notes
    # 기존 결과를 복사한다.
    merged_tables = dict(previous_tables or {})
    merged_charts = list(previous_charts or [])
    merged_notes = dict(previous_notes or {})
    # dirty 단계에 해당하는 기존 결과를 제거한다.
    for stage in dirty_stages:
        for key in _STAGE_TABLE_KEYS.get(stage, []):
            merged_tables.pop(key, None)
        chart_ids = set(_STAGE_CHART_IDS.get(stage, []))
        if chart_ids:
            merged_charts = [
                chart
                for chart in merged_charts
                if chart.get("chart_id") not in chart_ids
            ]
        merged_notes.pop(stage, None)
    # 새 결과를 덮어쓴다.
    merged_tables.update(new_tables or {})
    merged_charts = _merge_chart_outputs(merged_charts, new_charts or [])
    merged_notes.update(new_notes or {})
    return merged_tables, merged_charts, merged_notes


def _extract_row_value(row: dict[str, Any], keys: list[str]) -> Any:
    # 여러 키 중 값이 있는 첫 번째를 찾는다.
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _parse_number(value: Any) -> int | float | None:
    # 숫자 값으로 변환한다.
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return None


def _find_min_rank(rows: list[dict[str, Any]], keys: list[str]) -> int | float | None:
    # 최소 rank 값을 찾는다.
    values: list[int | float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = _extract_row_value(row, keys)
        parsed = _parse_number(raw)
        if parsed is not None:
            values.append(parsed)
    return min(values) if values else None


def _mark_selected_rows(
    rows: list[dict[str, Any]], is_selected: Callable[[dict[str, Any]], bool]
) -> None:
    # 선택된 행을 강조 표시한다.
    for row in rows:
        if not isinstance(row, dict):
            continue
        if is_selected(row):
            row["__row_state"] = "selected"
        else:
            if row.get("__row_state") == "selected":
                row.pop("__row_state", None)


def _apply_table_highlights(tables: dict[str, Any], selections: dict[str, Any]) -> None:
    # 테이블 강조 표시를 적용한다.
    # 칩기종 후보 강조 처리.
    chip_rows = tables.get("chip_type_candidates_table", [])
    selected_chip_ids = selections.get("chip_type_ids") or []
    if isinstance(selected_chip_ids, list) and isinstance(chip_rows, list):
        _mark_selected_rows(
            chip_rows,
            lambda row: _extract_row_value(row, ["chip_type_id", "칩기종 ID"])
            in selected_chip_ids,
        )
    # 레퍼런스 LOT 후보 강조 처리.
    ref_rows = tables.get("reference_lot_candidates_table", [])
    selected_ref_id = selections.get("reference_lot_id")
    if not selected_ref_id:
        ref_selected = tables.get("reference_lot_table", [])
        if isinstance(ref_selected, list) and ref_selected:
            selected_ref_id = _extract_row_value(
                ref_selected[0], ["lot_id", "LOT ID"]
            )
    if selected_ref_id and isinstance(ref_rows, list):
        _mark_selected_rows(
            ref_rows,
            lambda row: _extract_row_value(row, ["lot_id", "LOT ID"])
            == selected_ref_id,
        )
    # top-k 표의 rank 1 강조 처리.
    top_rows = tables.get("top_k_table", [])
    if isinstance(top_rows, list) and top_rows:
        min_rank = _find_min_rank(top_rows, ["rank", "순위"])
        if min_rank is not None:
            _mark_selected_rows(
                top_rows,
                lambda row: _parse_number(_extract_row_value(row, ["rank", "순위"]))
                == min_rank,
            )
    # 최근 유사 설계 표의 rank 1 강조 처리.
    recent_rows = tables.get("recent_similar_table", [])
    if isinstance(recent_rows, list) and recent_rows:
        min_rank = _find_min_rank(recent_rows, ["rank", "순위"])
        if min_rank is not None:
            _mark_selected_rows(
                recent_rows,
                lambda row: _parse_number(
                    _extract_row_value(row, ["rank", "순위"])
                )
                == min_rank,
            )
    # 불량률 표의 rank 1 강조 처리.
    defect_rows = tables.get("defect_rate_table", [])
    if isinstance(defect_rows, list) and defect_rows:
        min_rank = _find_min_rank(defect_rows, ["rank", "순위"])
        if min_rank is not None:
            _mark_selected_rows(
                defect_rows,
                lambda row: _parse_number(_extract_row_value(row, ["rank", "순위"]))
                == min_rank,
            )


def _safe_path_component(value: str) -> str:
    # 경로에 사용할 문자열을 정리한다.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _store_raw_outputs(
    state: dict[str, Any],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
) -> dict[str, Any]:
    # 원본 출력물을 파일로 저장한다.
    if not tables and not charts:
        return {}
    session_id = _safe_path_component(state.get("session_id", "session"))
    raw_dir = _RAW_OUTPUTS_DIR / session_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_id = f"{session_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}.json"
    raw_path = raw_dir / raw_id
    payload = {"tables": tables, "charts": charts}
    # UTF-8로 저장한다.
    raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {
        "stage_outputs_path": str(raw_path),
        "stage_outputs_saved_at": _utc_now(),
    }


def _load_raw_outputs(raw_path: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # raw 출력 경로가 없으면 빈 값을 반환한다.
    if not raw_path:
        return {}, []
    # 파일 경로를 준비한다.
    path = Path(raw_path)
    # 파일이 없으면 빈 값을 반환한다.
    if not path.exists():
        return {}, []
    # 파일 내용을 읽는다.
    raw_text = path.read_text(encoding="utf-8")
    # JSON payload로 파싱한다.
    payload = json.loads(raw_text)
    # 테이블/차트를 꺼낸다.
    tables = payload.get("tables", {}) if isinstance(payload, dict) else {}
    charts = payload.get("charts", []) if isinstance(payload, dict) else []
    # 타입을 정리한다.
    if not isinstance(tables, dict):
        tables = {}
    if not isinstance(charts, list):
        charts = []
    return tables, charts


def _merge_raw_outputs(
    previous_tables: dict[str, Any],
    previous_charts: list[dict[str, Any]],
    new_tables: dict[str, Any],
    new_charts: list[dict[str, Any]],
    dirty_stages: list[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # dirty 단계가 없으면 새 결과만 사용한다.
    if not dirty_stages:
        return new_tables, new_charts
    # 기존 결과를 복사한다.
    merged_tables = dict(previous_tables or {})
    merged_charts = list(previous_charts or [])
    # dirty 단계에 해당하는 기존 결과를 제거한다.
    for stage in dirty_stages:
        for key in _STAGE_TABLE_KEYS.get(stage, []):
            merged_tables.pop(key, None)
        chart_ids = set(_STAGE_CHART_IDS.get(stage, []))
        if chart_ids:
            merged_charts = [
                chart
                for chart in merged_charts
                if chart.get("chart_id") not in chart_ids
            ]
    # 새 결과를 덮어쓴다.
    merged_tables.update(new_tables or {})
    merged_charts = _merge_chart_outputs(merged_charts, new_charts or [])
    return merged_tables, merged_charts


def _merge_raw_outputs_with_history(
    state: dict[str, Any],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    dirty_stages: list[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # 이전 raw 출력 경로를 꺼낸다.
    raw_path = state.get("raw_refs", {}).get("stage_outputs_path")
    # 이전 raw 출력이 없으면 현재 결과를 반환한다.
    if not raw_path:
        return tables, charts
    # 이전 raw 출력 내용을 읽는다.
    previous_tables, previous_charts = _load_raw_outputs(raw_path)
    # dirty 단계 기준으로 병합한다.
    merged_tables, merged_charts = _merge_raw_outputs(
        previous_tables,
        previous_charts,
        tables,
        charts,
        dirty_stages,
    )
    return merged_tables, merged_charts


def _normalize_block_refs(
    blocks: list[dict[str, Any]],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # 참조 키 후보를 준비한다.
    table_keys = [key for key in (tables or {}).keys() if isinstance(key, str)]
    chart_ids = [
        chart.get("chart_id")
        for chart in (charts or [])
        if isinstance(chart, dict) and isinstance(chart.get("chart_id"), str)
    ]
    # 정규화용 맵을 만든다.
    table_key_map = {key.strip().lower(): key for key in table_keys}
    chart_id_map = {key.strip().lower(): key for key in chart_ids}
    # 블록을 순회하며 참조 키를 정리한다.
    normalized: list[dict[str, Any]] = []
    for block in blocks or []:
        # dict가 아니면 그대로 유지한다.
        if not isinstance(block, dict):
            normalized.append(block)
            continue
        # 블록을 복사한다.
        next_block = dict(block)
        block_type = next_block.get("type")
        # table_ref 키를 정규화한다.
        if block_type == "table_ref":
            raw_key = next_block.get("table_key")
            if isinstance(raw_key, str):
                cleaned = raw_key.strip()
                mapped = table_key_map.get(cleaned.lower())
                if mapped:
                    next_block["table_key"] = mapped
                else:
                    next_block["table_key"] = cleaned
        # chart_ref 키를 정규화한다.
        if block_type == "chart_ref":
            raw_key = next_block.get("chart_id")
            if isinstance(raw_key, str):
                cleaned = raw_key.strip()
                mapped = chart_id_map.get(cleaned.lower())
                if mapped:
                    next_block["chart_id"] = mapped
                else:
                    next_block["chart_id"] = cleaned
        # 정리된 블록을 추가한다.
        normalized.append(next_block)
    return normalized


def _ensure_block_refs(
    blocks: list[dict[str, Any]],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    stage_sequence: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    # 참조할 데이터가 없으면 그대로 반환한다.
    if not tables and not charts:
        return blocks
    # 참조 블록이 이미 있으면 그대로 반환한다.
    has_ref = any(
        isinstance(block, dict)
        and block.get("type") in ("table_ref", "chart_ref")
        for block in (blocks or [])
    )
    if has_ref:
        return blocks
    # 차트 ID 집합을 만든다.
    chart_id_set = {
        chart.get("chart_id")
        for chart in (charts or [])
        if isinstance(chart, dict) and chart.get("chart_id")
    }
    # 단계 순서가 있으면 그 순서대로 참조 블록을 만든다.
    if stage_sequence:
        refs: list[dict[str, Any]] = []
        for stage in stage_sequence:
            for key in stage.get("table_keys", []):
                if key in (tables or {}):
                    refs.append({"type": "table_ref", "table_key": key})
            for chart_id in stage.get("chart_ids", []):
                if chart_id in chart_id_set:
                    refs.append({"type": "chart_ref", "chart_id": chart_id})
        if refs:
            return (blocks or []) + refs
    # 순서 정보가 없으면 키 순서대로 참조 블록을 붙인다.
    fallback_refs: list[dict[str, Any]] = []
    for key in (tables or {}).keys():
        if isinstance(key, str):
            fallback_refs.append({"type": "table_ref", "table_key": key})
    for chart_id in chart_id_set:
        fallback_refs.append({"type": "chart_ref", "chart_id": chart_id})
    return (blocks or []) + fallback_refs


def _payload_size(tables: dict[str, Any], charts: list[dict[str, Any]]) -> int:
    # LLM payload 크기를 추정한다.
    payload = {"tables": tables, "charts": charts}
    return len(json.dumps(payload, ensure_ascii=False))


def _strip_meta_fields(row: dict[str, Any]) -> dict[str, Any]:
    # 메타 필드를 제거한다.
    return {key: value for key, value in row.items() if not key.startswith("__")}


def _prioritize_keys(row: dict[str, Any], max_cols: int) -> list[str]:
    # 우선 순위가 높은 키를 먼저 배치한다.
    priority = [
        "rank",
        "순위",
        "lot_id",
        "LOT ID",
        "chip_type_id",
        "칩기종 ID",
        "candidate_rank",
        "후보 순위",
    ]
    ordered: list[str] = []
    for key in priority:
        if key in row and key not in ordered:
            ordered.append(key)
    for key in row.keys():
        if key not in ordered:
            ordered.append(key)
    return ordered[:max_cols]


def _select_keys(
    row: dict[str, Any], preferred_keys: list[str] | None, max_cols: int
) -> dict[str, Any]:
    # 필요한 컬럼만 추린다.
    if preferred_keys:
        keys = [key for key in preferred_keys if key in row]
    else:
        keys = []
    if not keys:
        keys = _prioritize_keys(row, max_cols)
    return {key: row.get(key) for key in keys[:max_cols]}


def _project_table_rows(
    rows: list[dict[str, Any]],
    preferred_keys: list[str] | None,
    max_rows: int,
    max_cols: int,
) -> list[dict[str, Any]]:
    # 테이블 행을 요약한다.
    projected: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, dict):
            continue
        clean = _strip_meta_fields(row)
        projected.append(_select_keys(clean, preferred_keys, max_cols))
    return projected


def _extract_chart_metric_keys(charts: list[dict[str, Any]]) -> list[str]:
    # 차트에서 metric 키를 추출한다.
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        if chart.get("chart_id") != "defect_rate_summary":
            continue
        series = chart.get("series", [])
        if not isinstance(series, list) or not series:
            continue
        points = series[0].get("points", [])
        if not isinstance(points, list):
            continue
        metrics: list[str] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            metric = point.get("x")
            if metric and metric not in metrics:
                metrics.append(metric)
        if metrics:
            return metrics
    return []


def _pick_rank_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    # rank가 가장 낮은 행을 고른다.
    min_rank = _find_min_rank(rows, ["rank", "순위"])
    if min_rank is None:
        return rows[0] if rows else None
    for row in rows:
        raw = _extract_row_value(row, ["rank", "순위"])
        if _parse_number(raw) == min_rank:
            return row
    return rows[0] if rows else None


def _project_defect_rate_table(
    rows: list[dict[str, Any]],
    charts: list[dict[str, Any]],
    max_metrics: int,
) -> list[dict[str, Any]]:
    # 불량률 표를 요약한다.
    target_row = _pick_rank_row(rows)
    if not target_row:
        return []
    clean = _strip_meta_fields(target_row)
    metric_keys = _extract_chart_metric_keys(charts)
    if not metric_keys:
        metric_keys = [
            key for key in clean.keys() if key not in ("rank", "순위")
        ][:max_metrics]
    preferred = ["rank", "순위"] + metric_keys
    return [_select_keys(clean, preferred, len(preferred))]


def _project_tables(
    tables: dict[str, Any],
    configs: dict[str, Any],
    charts: list[dict[str, Any]],
) -> dict[str, Any]:
    # 표 요약본을 만든다.
    projected: dict[str, Any] = {}
    core_params = configs.get("core_match_params") or []
    core_labels = [
        _CORE_PARAM_LABEL_MAP.get(key) for key in core_params if key in _CORE_PARAM_LABEL_MAP
    ]
    top_k_limit = int(configs.get("top_k") or 5)
    for key, rows in tables.items():
        if not isinstance(rows, list):
            projected[key] = rows
            continue
        if key == "input_params_table":
            preferred = ["항목", "값", "item", "value"]
            projected[key] = _project_table_rows(rows, preferred, _LLM_MAX_ROWS_DEFAULT, 2)
            continue
        if key in ("chip_type_candidates_table", "reference_lot_candidates_table", "reference_lot_table"):
            preferred = [
                "chip_type_id",
                "칩기종 ID",
                "chip_type_name",
                "칩기종명",
                "match_count",
                "매칭수",
                "lot_id",
                "LOT ID",
                "defect_score",
                "불량률 점수",
                "defect_metrics_summary",
                "불량률 요약",
                "notes",
                "비고",
            ]
            projected[key] = _project_table_rows(rows, preferred, 5, _LLM_MAX_COLS_DEFAULT)
            continue
        if key == "top_k_table":
            preferred = (
                ["rank", "순위", "predicted_capacity", "예상 용량", "total_layer", "총 레이어"]
                + core_params
                + [label for label in core_labels if label]
            )
            projected[key] = _project_table_rows(
                rows, preferred, min(top_k_limit, _LLM_MAX_ROWS_DEFAULT), _LLM_MAX_COLS_DEFAULT
            )
            continue
        if key == "recent_similar_table":
            preferred = [
                "rank",
                "순위",
                "match_count",
                "매칭수",
                "representative_lot_id",
                "대표 LOT",
                "date_range_start",
                "기간 시작",
                "date_range_end",
                "기간 종료",
            ]
            projected[key] = _project_table_rows(rows, preferred, 5, _LLM_MAX_COLS_DEFAULT)
            continue
        if key == "defect_rate_table":
            projected[key] = _project_defect_rate_table(
                rows, charts, _LLM_MAX_COLS_DEFAULT
            )
            continue
        projected[key] = _project_table_rows(
            rows, None, _LLM_MAX_ROWS_DEFAULT, _LLM_MAX_COLS_DEFAULT
        )
    return projected


def _project_charts(
    charts: list[dict[str, Any]],
    max_series: int = _LLM_MAX_SERIES_DEFAULT,
    max_points: int = _LLM_MAX_POINTS_DEFAULT,
) -> list[dict[str, Any]]:
    # 차트 요약본을 만든다.
    projected: list[dict[str, Any]] = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        summary = {
            key: chart.get(key)
            for key in ("chart_id", "type", "title", "subtitle", "x_label", "y_label", "unit")
            if key in chart
        }
        series = chart.get("series", [])
        if isinstance(series, list):
            trimmed_series: list[dict[str, Any]] = []
            for item in series[:max_series]:
                if not isinstance(item, dict):
                    continue
                points = item.get("points", [])
                trimmed_points = points[:max_points] if isinstance(points, list) else []
                trimmed_series.append({"name": item.get("name"), "points": trimmed_points})
            summary["series"] = trimmed_series
        projected.append(summary)
    return projected


def _trim_tables_for_budget(
    tables: dict[str, Any], max_rows: int, max_cols: int
) -> dict[str, Any]:
    # 테이블을 추가로 줄인다.
    trimmed: dict[str, Any] = {}
    for key, rows in tables.items():
        if not isinstance(rows, list):
            trimmed[key] = rows
            continue
        sliced = rows[:max_rows]
        if sliced and isinstance(sliced[0], dict):
            trimmed_rows = []
            for row in sliced:
                clean = _strip_meta_fields(row)
                keys = _prioritize_keys(clean, max_cols)
                trimmed_rows.append({key: clean.get(key) for key in keys})
            trimmed[key] = trimmed_rows
        else:
            trimmed[key] = sliced
    return trimmed


def _trim_charts_for_budget(
    charts: list[dict[str, Any]], max_series: int, max_points: int
) -> list[dict[str, Any]]:
    # 차트를 추가로 줄인다.
    trimmed: list[dict[str, Any]] = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        series = chart.get("series", [])
        if isinstance(series, list):
            clipped_series = []
            for item in series[:max_series]:
                if not isinstance(item, dict):
                    continue
                points = item.get("points", [])
                clipped_points = points[:max_points] if isinstance(points, list) else []
                clipped_series.append({"name": item.get("name"), "points": clipped_points})
            chart = dict(chart)
            chart["series"] = clipped_series
        trimmed.append(chart)
    return trimmed


def _apply_payload_budget(
    tables: dict[str, Any], charts: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # 하드캡을 적용해 payload를 줄인다.
    if _payload_size(tables, charts) <= _LLM_PAYLOAD_MAX_CHARS:
        return tables, charts
    row_budget = _LLM_MAX_ROWS_DEFAULT
    col_budget = _LLM_MAX_COLS_DEFAULT
    series_budget = _LLM_MAX_SERIES_DEFAULT
    points_budget = _LLM_MAX_POINTS_DEFAULT
    trimmed_tables = tables
    trimmed_charts = charts
    for _ in range(3):
        trimmed_tables = _trim_tables_for_budget(trimmed_tables, row_budget, col_budget)
        trimmed_charts = _trim_charts_for_budget(trimmed_charts, series_budget, points_budget)
        if _payload_size(trimmed_tables, trimmed_charts) <= _LLM_PAYLOAD_MAX_CHARS:
            break
        row_budget = max(1, row_budget // 2)
        col_budget = max(2, col_budget // 2)
        series_budget = max(1, series_budget // 2)
        points_budget = max(2, points_budget // 2)
    return trimmed_tables, trimmed_charts


def _build_llm_payload(
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    configs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # LLM에 전달할 요약 payload를 만든다.
    projected_tables = _project_tables(tables, configs, charts)
    projected_charts = _project_charts(charts)
    return _apply_payload_budget(projected_tables, projected_charts)


def _mark_dirty(state: dict[str, Any], stages: list[str]) -> None:
    # dirty 상태를 기록한다.
    if not stages:
        return
    now = _utc_now()
    for stage in stages:
        if stage in state["stage_status"]:
            state["stage_status"][stage]["done"] = False
            state["stage_status"][stage]["dirty"] = True
            state["stage_status"][stage]["updated_at"] = now


def _mark_clean(state: dict[str, Any], stages: list[str]) -> None:
    # 재실행 완료 단계를 정리한다.
    if not stages:
        return
    now = _utc_now()
    for stage in stages:
        if stage in state["stage_status"]:
            state["stage_status"][stage]["done"] = True
            state["stage_status"][stage]["dirty"] = False
            state["stage_status"][stage]["updated_at"] = now


def _apply_update_fields(state: dict[str, Any], update: UpdateDecision) -> None:
    # 선택값을 업데이트한다.
    if update.selections:
        for key, value in update.selections.dict().items():
            if value not in (None, ""):
                state["selections"][key] = value
    # 설정값을 업데이트한다.
    if update.configs:
        for key, value in update.configs.dict().items():
            if value is not None:
                state["configs"][key] = value
    # 사용자 설정을 업데이트한다.
    if update.user_prefs:
        for key, value in update.user_prefs.dict().items():
            if value not in (None, ""):
                state["user_prefs"][key] = value
