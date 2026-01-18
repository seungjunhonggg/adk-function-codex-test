from datetime import datetime
from typing import Any

from .agents import _build_explain_answer
from .schemas import InputParams, UpdateDecision


# 입력 라벨 맵을 정의한다.
INPUT_LABEL_MAP = {
    "temperature": "온도",
    "voltage": "전압",
    "size": "크기",
    "capacity": "용량",
    "dev_flag": "개발품여부",
    "powder_size": "파우더사이즈",
    "chip_type": "칩기종",
}

# 변경 라벨 맵을 정의한다.
UPDATE_LABEL_MAP = {
    "reference_lot_id": "레퍼런스 LOT",
    "chip_type_id": "칩기종",
    "top_k": "top-k",
    "chart_type": "차트 타입",
}

# pending_action 규칙을 정의한다.
PENDING_ACTION_RULES = {
    "reference_lot_id": ("update_reference_lot", "1-3"),
    "chip_type_id": ("update_chip_type", "1-2"),
    "top_k": ("update_top_k", "1-5"),
    "chart_type": ("update_chart_type", "1-7"),
}

# 단계 순서를 정의한다.
STAGE_ORDER = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]

# dirty 규칙을 정의한다.
DIRTY_STAGE_RULES = {
    "input_params": ["1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"],
    "chip_type_id": ["1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"],
    "reference_lot_id": ["1-4", "1-5", "1-6", "1-7", "1-8"],
    "top_k": ["1-5", "1-6", "1-7", "1-8"],
    "chart_type": ["1-7", "1-8"],
}

# 단계별 표 키를 정의한다.
_STAGE_TABLE_KEYS = {
    "1-1": ["input_params_table"],
    "1-2": ["chip_type_candidates_table"],
    "1-3": ["reference_lot_candidates_table", "reference_lot_table"],
    "1-5": ["top_k_table"],
    "1-6": ["recent_similar_table"],
    "1-7": ["defect_rate_table"],
}

# 단계별 차트 키를 정의한다.
_STAGE_CHART_IDS = {
    "1-7": ["defect_rate_summary"],
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
            "dev_flag": None,
            "powder_size": None,
            "chip_type": None,
        },
        "selections": {
            "chip_type_id": None,
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
        "raw_refs": {"top_k_raw_id": None, "defect_raw_id": None},
        "pending_action": None,
        "last_explain_stage": None,
        "history": [],
        "user_prefs": {"chart_type": "bar", "language": "ko"},
        "last_error": None,
    }


def _get_session_state(session_id: str) -> dict[str, Any]:
    # 세션 상태를 가져오거나 만든다.
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = _init_session_state(session_id)
    return _SESSION_STORE[session_id]


def _merge_input_params(
    current: dict[str, Any], incoming: InputParams
) -> InputParams:
    # 입력값을 병합한다.
    merged = dict(current or {})
    for key, value in incoming.dict().items():
        if value not in (None, ""):
            merged[key] = value
    return InputParams(**merged)


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


def _apply_chip_type_skip(state: dict[str, Any], input_params: InputParams) -> None:
    # chip_type 입력이면 1-2 단계를 생략 처리한다.
    if not input_params.chip_type:
        return
    status = state["stage_status"]
    now = _utc_now()
    status["1-2"]["done"] = True
    status["1-2"]["dirty"] = False
    status["1-2"]["updated_at"] = now


def _update_state(
    state: dict[str, Any],
    input_params: InputParams,
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    stage_notes: dict[str, str],
    missing: list[str],
    demo: bool,
) -> None:
    # 입력값을 저장한다.
    state["input_params"] = input_params.dict()
    # 출력물을 저장한다.
    state["stage_outputs"] = {
        "tables": tables,
        "charts": charts,
        "briefing_blocks": blocks,
    }
    # 단계 근거를 저장한다.
    state["stage_notes"] = stage_notes
    # 단계 상태를 갱신한다.
    _update_stage_status(state, bool(missing), demo)
    # chip_type 입력 시 1-2 단계를 생략 처리한다.
    _apply_chip_type_skip(state, input_params)
    # pending_action을 저장한다.
    if missing:
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
    # 누락된 입력 필드를 찾는다.
    return [
        name
        for name, value in input_params.dict().items()
        if name != "chip_type" and not value
    ]


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


def _build_briefing_hint(start_stage: str | None) -> str | None:
    # 변경 반영 안내 문구를 만든다.
    if not start_stage:
        return None
    return f"{start_stage} 단계 변경사항을 반영했습니다. 첫 문장에서 짧게 언급하세요."


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
