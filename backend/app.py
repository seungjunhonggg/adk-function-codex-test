import json
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel

from agents import Agent, Runner

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    overrides: dict[str, Any] | None = None
    demo: bool = False


class ChatResponse(BaseModel):
    route: str
    blocks: list[dict[str, Any]]
    tables: dict[str, Any]
    charts: list[dict[str, Any]]


class RouteDecision(BaseModel):
    route: Literal["casual", "simulation"]


class CommandDecision(BaseModel):
    action: Literal["run", "update_input", "explain_stage"]
    target_stage: str | None = None


router_agent = Agent(
    name="RouteAgent",
    instructions=(
        "사용자 메시지를 보고 route를 분류해.\n"
        "- simulation: 칩 설계/시뮬레이션/추천을 요청하는 경우\n"
        "- casual: 그 외 일반 대화\n"
        "반드시 route만 출력해."
    ),
    output_type=RouteDecision,
)


command_agent = Agent(
    name="CommandAgent",
    instructions=(
        "사용자 메시지를 보고 action을 결정해.\n"
        "- run: 시뮬레이션 시작/진행/결과 요청\n"
        "- update_input: 입력값/선택값 변경 요청(바꿔/수정/다시)\n"
        "- explain_stage: 특정 단계 근거/이유 요청\n"
        "단계가 명시되면 target_stage에 1-4 형식으로 넣어.\n"
        "단계가 없으면 target_stage는 null.\n"
        "action과 target_stage만 출력해."
    ),
    output_type=CommandDecision,
)


class InputParams(BaseModel):
    temperature: str | None = None
    voltage: str | None = None
    size: str | None = None
    capacity: str | None = None
    dev_flag: str | None = None
    powder_size: str | None = None
    chip_type: str | None = None


class UpdateSelections(BaseModel):
    chip_type_id: str | None = None
    reference_lot_id: str | None = None


class UpdateConfigs(BaseModel):
    top_k: int | None = None


class UpdateUserPrefs(BaseModel):
    chart_type: Literal["bar", "line", "scatter"] | None = None


class UpdateDecision(BaseModel):
    input_params: InputParams | None = None
    selections: UpdateSelections | None = None
    configs: UpdateConfigs | None = None
    user_prefs: UpdateUserPrefs | None = None
    missing_fields: list[str] = []


class BriefingBlock(BaseModel):
    type: Literal["text", "table_ref", "chart_ref"]
    section: str | None = None
    value: str | None = None
    table_key: str | None = None
    chart_id: str | None = None


class BriefingOutput(BaseModel):
    blocks: list[BriefingBlock]


class ExplainOutput(BaseModel):
    answer: str


_SESSION_STORE: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    # 현재 시각을 ISO 문자열로 만든다.
    return datetime.utcnow().isoformat() + "Z"


def _init_stage_status() -> dict[str, dict[str, Any]]:
    # 단계 상태를 초기화한다.
    status: dict[str, dict[str, Any]] = {}
    for stage in ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]:
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
        status["1-1"]["done"] = False
        status["1-1"]["dirty"] = True
        status["1-1"]["updated_at"] = now
        return
    status["1-1"]["done"] = True
    status["1-1"]["dirty"] = False
    status["1-1"]["updated_at"] = now
    if demo:
        for stage in ["1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]:
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
input_agent = Agent(
    name="InputAgent",
    instructions=(
        "메시지에서 다음 필드만 추출해: temperature, voltage, size, capacity, dev_flag, powder_size, chip_type.\n"
        "없으면 null. 추측 금지. 위 필드만 출력."
    ),
    output_type=InputParams,
)


update_agent = Agent(
    name="UpdateAgent",
    instructions=(
        "사용자 메시지에서 변경 요청을 추출해.\n"
        "- input_params: temperature, voltage, size, capacity, dev_flag, powder_size, chip_type\n"
        "- selections: chip_type_id, reference_lot_id\n"
        "- configs: top_k\n"
        "- user_prefs: chart_type(bar|line|scatter)\n"
        "규칙:\n"
        "- 변경 의도만 있고 값이 없으면 missing_fields에 해당 키를 넣어.\n"
        "- 값이 있는 항목만 채워. 나머지는 null.\n"
        "missing_fields 포함해서 출력해."
    ),
    output_type=UpdateDecision,
)


explain_agent = Agent(
    name="ExplainAgent",
    instructions=(
        "다음 JSON을 보고 사용자의 질문에 답해.\n"
        "- question: 사용자 질문\n"
        "- stage: 단계\n"
        "- stage_notes: 단계 근거 요약\n"
        "- tables/charts: 필요한 증거 데이터\n"
        "규칙:\n"
        "- stage_notes와 tables/charts 내용만 사용해.\n"
        "- 모르는 내용은 추측하지 말고 되물어.\n"
        "- 3~6문장 한국어로 간결하게 답해.\n"
        "answer만 출력해."
    ),
    output_type=ExplainOutput,
)


briefing_agent = Agent(
    name="BriefingAgent",
    instructions=(
        "아래 표/차트 데이터를 보고 브리핑 블록을 생성해.\n"
        "- 출력 형식: blocks 배열만\n"
        "- block.type은 text|table_ref|chart_ref만 사용\n"
        "- text는 한국어로 작성\n"
        "- 표/차트 값만 인용\n"
        "- children 지표는 언급하지 않음\n"
        "- 길이 목표: 2k~3k 토큰\n"
        "필수 table_key: input_params_table, chip_type_candidates_table, "
        "reference_lot_candidates_table, reference_lot_table, top_k_table, "
        "recent_similar_table, defect_rate_table\n"
        "필수 chart_id: defect_rate_summary\n"
    ),
    output_type=BriefingOutput,
)


def _get_demo_label_mapping() -> dict[str, str]:
    # 데모용 한글 라벨 매핑을 준비한다.
    return {
        "chip_type_id": "칩기종 ID",
        "chip_type_name": "칩기종명",
        "match_count": "매칭수",
        "notes": "비고",
        "lot_id": "LOT ID",
        "defect_score": "불량률 점수",
        "defect_metrics_summary": "불량률 요약",
        "rank": "순위",
        "active_powder_base": "활성파우더 베이스",
        "active_powder_additives": "활성파우더 첨가제",
        "ldn_avr_value": "LDN 평균값",
        "cast_dsgn_thk": "캐스팅 설계 두께",
        "grinding_l_avg": "연마 L 평균",
        "grinding_w_avg": "연마 W 평균",
        "grinding_t_avg": "연마 T 평균",
        "total_layer": "총 레이어",
        "predicted_capacity": "예상 용량",
        "candidate_rank": "후보 순위",
        "date_range_start": "기간 시작",
        "date_range_end": "기간 종료",
        "representative_lot_id": "대표 LOT",
        "defect_metric": "불량률 지표",
        "defect_avg": "평균",
        "defect_min": "최소",
        "defect_max": "최대",
    }


INPUT_LABEL_MAP = {
    "temperature": "온도",
    "voltage": "전압",
    "size": "크기",
    "capacity": "용량",
    "dev_flag": "개발품여부",
    "powder_size": "파우더사이즈",
    "chip_type": "칩기종",
}

UPDATE_LABEL_MAP = {
    "reference_lot_id": "레퍼런스 LOT",
    "chip_type_id": "칩기종",
    "top_k": "top-k",
    "chart_type": "차트 타입",
}

PENDING_ACTION_RULES = {
    "reference_lot_id": ("update_reference_lot", "1-3"),
    "chip_type_id": ("update_chip_type", "1-2"),
    "top_k": ("update_top_k", "1-5"),
    "chart_type": ("update_chart_type", "1-7"),
}


def _get_db_label_mapping() -> dict[str, str]:
    # DB 연결 시 한글 라벨 매핑을 조회한다.
    return {}


def _get_label_mapping(demo: bool) -> dict[str, str]:
    # 데모 여부에 따라 한글 라벨 매핑을 결정한다.
    if demo:
        return _get_demo_label_mapping()
    return _get_db_label_mapping()


def _map_row_labels(row: dict[str, Any], label_map: dict[str, str]) -> dict[str, Any]:
    # 행의 컬럼명을 한글 라벨로 변환한다.
    return {label_map.get(key, key): value for key, value in row.items()}


def _map_table_labels(
    tables: dict[str, Any], label_map: dict[str, str]
) -> dict[str, Any]:
    # 모든 테이블 컬럼명을 한글 라벨로 변환한다.
    mapped: dict[str, Any] = {}
    for table_key, rows in tables.items():
        if isinstance(rows, list):
            mapped_rows = []
            for row in rows:
                if isinstance(row, dict):
                    mapped_rows.append(_map_row_labels(row, label_map))
                else:
                    mapped_rows.append(row)
            mapped[table_key] = mapped_rows
        else:
            mapped[table_key] = rows
    return mapped


async def _route_with_llm(message: str) -> str:
    # LLM으로 라우팅을 결정한다.
    result = await Runner.run(router_agent, message)
    decision = result.final_output
    return decision.route


async def _decide_command_with_llm(message: str) -> CommandDecision:
    # LLM으로 커맨드 액션을 결정한다.
    result = await Runner.run(command_agent, message)
    return result.final_output


async def _parse_input_with_llm(message: str) -> InputParams:
    # LLM으로 입력값을 추출한다.
    result = await Runner.run(input_agent, message)
    return result.final_output


async def _parse_update_with_llm(message: str) -> UpdateDecision:
    # LLM으로 변경 요청을 추출한다.
    result = await Runner.run(update_agent, message)
    return result.final_output


async def _build_briefing_blocks(
    tables: dict[str, Any], charts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # 브리핑 입력을 만든다.
    payload = json.dumps({"tables": tables, "charts": charts}, ensure_ascii=False)
    # LLM으로 브리핑 블록을 만든다.
    result = await Runner.run(briefing_agent, payload)
    # Pydantic 객체를 dict로 변환한다.
    return [block.dict() for block in result.final_output.blocks]


_STAGE_TABLE_KEYS = {
    "1-1": ["input_params_table"],
    "1-2": ["chip_type_candidates_table"],
    "1-3": ["reference_lot_candidates_table", "reference_lot_table"],
    "1-5": ["top_k_table"],
    "1-6": ["recent_similar_table"],
    "1-7": ["defect_rate_table"],
}

_STAGE_CHART_IDS = {
    "1-7": ["defect_rate_summary"],
}


def _build_stage_notes(
    input_params: InputParams,
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    configs: dict[str, Any],
    selections: dict[str, Any],
    user_prefs: dict[str, Any],
) -> dict[str, str]:
    # 단계 근거 요약을 만든다.
    notes: dict[str, str] = {}
    # 1-1 입력 근거를 만든다.
    filled_keys = [key for key, value in input_params.dict().items() if value]
    filled_labels = [INPUT_LABEL_MAP.get(key, key) for key in filled_keys]
    missing_fields = _get_missing_fields(input_params)
    if missing_fields:
        line1 = "근거: 필수 입력이 누락됨"
        line3 = "출력: 입력 보완 필요"
    else:
        line1 = "근거: 입력값 확보 완료"
        line3 = "출력: 입력 확정"
    line2 = f"입력: {', '.join(filled_labels) if filled_labels else '-'}"
    notes["1-1"] = "\n".join([line1, line2, line3])
    # 1-2 칩기종 후보 근거를 만든다.
    chip_rows = tables.get("chip_type_candidates_table", [])
    chip_count = len(chip_rows) if isinstance(chip_rows, list) else 0
    chip_top = chip_rows[0] if chip_count else {}
    chip_id = chip_top.get("chip_type_id", "-")
    chip_match = chip_top.get("match_count", "-")
    line1 = "근거: match_count 높은 순 정렬" if chip_count else "근거: 후보 데이터 없음"
    line2 = f"입력: 후보 {chip_count}개"
    line3 = (
        f"출력: {chip_id} (match_count={chip_match})" if chip_count else "출력: -"
    )
    notes["1-2"] = "\n".join([line1, line2, line3])
    # 1-3 레퍼런스 LOT 근거를 만든다.
    ref_rows = tables.get("reference_lot_candidates_table", [])
    ref_count = len(ref_rows) if isinstance(ref_rows, list) else 0
    ref_selected = tables.get("reference_lot_table", [])
    ref_top = ref_selected[0] if isinstance(ref_selected, list) and ref_selected else {}
    ref_id = ref_top.get("lot_id", "-")
    ref_score = ref_top.get("defect_score", "-")
    line1 = "근거: defect_score 낮은 LOT 선정" if ref_top else "근거: 후보 데이터 없음"
    line2 = f"입력: 후보 {ref_count}개"
    line3 = f"출력: {ref_id} (defect_score={ref_score})" if ref_top else "출력: -"
    notes["1-3"] = "\n".join([line1, line2, line3])
    # 1-4 API payload 근거를 만든다.
    ref_lot_id = selections.get("reference_lot_id") or ref_id or "-"
    chip_type_value = selections.get("chip_type_id") or input_params.chip_type or "-"
    line1 = "근거: 입력값 + ref LOT로 payload 구성"
    line2 = f"입력: ref_lot={ref_lot_id}, chip_type={chip_type_value}"
    line3 = "출력: API payload 구성"
    notes["1-4"] = "\n".join([line1, line2, line3])
    # 1-5 top-k 근거를 만든다.
    top_k_rows = tables.get("top_k_table", [])
    top_k_count = len(top_k_rows) if isinstance(top_k_rows, list) else 0
    top_k_row = top_k_rows[0] if top_k_count else {}
    top_k_value = configs.get("top_k")
    top_k_sort = configs.get("top_k_sort", "rank")
    top_rank = top_k_row.get("rank", "-")
    top_capacity = top_k_row.get("predicted_capacity", "-")
    line1 = f"근거: {top_k_sort} 오름차순 정렬 + top_k={top_k_value}"
    line2 = f"입력: top_k={top_k_value}"
    line3 = (
        f"출력: rank1={top_rank}, predicted_capacity={top_capacity}"
        if top_k_count
        else "출력: -"
    )
    notes["1-5"] = "\n".join([line1, line2, line3])
    # 1-6 최근 유사 설계 근거를 만든다.
    recent_rows = tables.get("recent_similar_table", [])
    recent_count = len(recent_rows) if isinstance(recent_rows, list) else 0
    recent_top = recent_rows[0] if recent_count else {}
    date_start = recent_top.get("date_range_start", "-")
    date_end = recent_top.get("date_range_end", "-")
    rep_lot = recent_top.get("representative_lot_id", "-")
    match_count = recent_top.get("match_count", "-")
    core_params = configs.get("core_match_params", [])
    core_text = ", ".join(core_params) if core_params else "-"
    line1 = "근거: 최근 6개월 + 핵심 파라미터 매칭"
    line2 = f"입력: 기간={date_start}~{date_end}, core_params={core_text}"
    line3 = (
        f"출력: 대표 LOT={rep_lot}, match_count={match_count}"
        if recent_count
        else "출력: -"
    )
    notes["1-6"] = "\n".join([line1, line2, line3])
    # 1-7 불량률 집계 근거를 만든다.
    defect_rows = tables.get("defect_rate_table", [])
    metric_set = {
        row.get("defect_metric")
        for row in defect_rows
        if isinstance(row, dict) and row.get("defect_metric")
    }
    metric_count = len(metric_set)
    chart_type = user_prefs.get("chart_type", "bar")
    chart = next(
        (item for item in charts if item.get("chart_id") == "defect_rate_summary"),
        None,
    )
    series_name = "-"
    first_value = "-"
    if chart and chart.get("series"):
        series = chart["series"][0]
        series_name = series.get("name", "-")
        points = series.get("points", [])
        if points:
            first_value = points[0].get("y", "-")
    line1 = f"근거: metric {metric_count}개 집계 + chart_type={chart_type}"
    line2 = f"입력: chart_type={chart_type}"
    line3 = f"출력: {series_name} rank1={first_value}"
    notes["1-7"] = "\n".join([line1, line2, line3])
    # 1-8 브리핑 근거를 만든다.
    line1 = "근거: 표/차트 요약 기반 브리핑 생성"
    line2 = "입력: stage_outputs 표/차트"
    line3 = "출력: briefing_blocks"
    notes["1-8"] = "\n".join([line1, line2, line3])
    return notes


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
    payload = json.dumps(context, ensure_ascii=False)
    result = await Runner.run(explain_agent, payload)
    answer = result.final_output.answer
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


def _build_casual_stub() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # 캐주얼 응답 블록을 만든다.
    blocks = [
        {
            "type": "text",
            "section": "casual",
            "value": "안녕하세요. 시뮬레이션이 필요하면 '시뮬레이션'이라고 말해 주세요.",
        }
    ]
    return blocks, {}, []


def _build_simulation_stub(
    request: ChatRequest,
    input_params: InputParams,
    configs: dict[str, Any],
    selections: dict[str, Any],
    user_prefs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    # 시뮬레이션 더미 응답을 만든다.
    summary = "시뮬레이션을 시작합니다. 필요한 입력을 확인 중입니다."
    if request.demo:
        summary = "데모 모드로 시뮬레이션을 시작합니다. 필요한 입력을 확인 중입니다."
    # 누락된 입력을 확인한다.
    missing = _get_missing_fields(input_params)
    if missing:
        summary = _format_missing_summary(missing)
    blocks = [
        {"type": "text", "section": "summary", "value": summary},
        {"type": "table_ref", "table_key": "input_params_table"},
    ]
    # 입력 요약 표를 만든다.
    input_rows = [
        {"항목": INPUT_LABEL_MAP[key], "값": value or ""}
        for key, value in input_params.dict().items()
    ]
    tables = {"input_params_table": input_rows}
    charts: list[dict[str, Any]] = []
    # 데모 모드일 때만 간단한 표/차트를 채운다.
    if request.demo:
        blocks.append({"type": "table_ref", "table_key": "chip_type_candidates_table"})
        blocks.append({"type": "table_ref", "table_key": "reference_lot_candidates_table"})
        blocks.append({"type": "table_ref", "table_key": "reference_lot_table"})
        blocks.append({"type": "table_ref", "table_key": "top_k_table"})
        blocks.append({"type": "table_ref", "table_key": "recent_similar_table"})
        blocks.append({"type": "table_ref", "table_key": "defect_rate_table"})
        blocks.append({"type": "chart_ref", "chart_id": "defect_rate_summary"})
        # 칩기종 후보 표를 만든다.
        tables["chip_type_candidates_table"] = [
            {"chip_type_id": "CT-001", "chip_type_name": "MLCC-A", "match_count": 12, "notes": "고온/고전압"},
            {"chip_type_id": "CT-002", "chip_type_name": "MLCC-B", "match_count": 9, "notes": "용량 우선"},
            {"chip_type_id": "CT-003", "chip_type_name": "MLCC-C", "match_count": 7, "notes": "소형화"},
            {"chip_type_id": "CT-004", "chip_type_name": "MLCC-D", "match_count": 5, "notes": "개발품"},
            {"chip_type_id": "CT-005", "chip_type_name": "MLCC-E", "match_count": 4, "notes": "표준형"},
        ]
        # 레퍼런스 LOT 후보 표(10개)를 만든다.
        tables["reference_lot_candidates_table"] = [
            {"lot_id": "LOT-CAND-001", "chip_type_id": "CT-001", "defect_score": 0.08, "defect_metrics_summary": "ci_def_rate 0.08%, fr_defect_rate 80ppm"},
            {"lot_id": "LOT-CAND-002", "chip_type_id": "CT-001", "defect_score": 0.10, "defect_metrics_summary": "ci_def_rate 0.10%, fr_defect_rate 95ppm"},
            {"lot_id": "LOT-CAND-003", "chip_type_id": "CT-001", "defect_score": 0.12, "defect_metrics_summary": "ci_def_rate 0.12%, fr_defect_rate 110ppm"},
            {"lot_id": "LOT-CAND-004", "chip_type_id": "CT-001", "defect_score": 0.14, "defect_metrics_summary": "ci_def_rate 0.14%, fr_defect_rate 125ppm"},
            {"lot_id": "LOT-CAND-005", "chip_type_id": "CT-001", "defect_score": 0.16, "defect_metrics_summary": "ci_def_rate 0.16%, fr_defect_rate 140ppm"},
            {"lot_id": "LOT-CAND-006", "chip_type_id": "CT-001", "defect_score": 0.18, "defect_metrics_summary": "ci_def_rate 0.18%, fr_defect_rate 155ppm"},
            {"lot_id": "LOT-CAND-007", "chip_type_id": "CT-001", "defect_score": 0.20, "defect_metrics_summary": "ci_def_rate 0.20%, fr_defect_rate 170ppm"},
            {"lot_id": "LOT-CAND-008", "chip_type_id": "CT-001", "defect_score": 0.22, "defect_metrics_summary": "ci_def_rate 0.22%, fr_defect_rate 185ppm"},
            {"lot_id": "LOT-CAND-009", "chip_type_id": "CT-001", "defect_score": 0.24, "defect_metrics_summary": "ci_def_rate 0.24%, fr_defect_rate 200ppm"},
            {"lot_id": "LOT-CAND-010", "chip_type_id": "CT-001", "defect_score": 0.26, "defect_metrics_summary": "ci_def_rate 0.26%, fr_defect_rate 215ppm"},
        ]
        # 레퍼런스 LOT 표(최종 선정 1개)를 만든다.
        tables["reference_lot_table"] = [
            {
                "lot_id": "LOT-CAND-001",
                "chip_type_id": "CT-001",
                "defect_score": 0.08,
                "defect_metrics_summary": "ci_def_rate 0.08%, fr_defect_rate 80ppm",
            }
        ]
        # top-k 표를 만든다.
        tables["top_k_table"] = [
            {
                "rank": 1,
                "active_powder_base": "A",
                "active_powder_additives": "X1",
                "ldn_avr_value": 1.1,
                "cast_dsgn_thk": 2.2,
                "grinding_l_avg": 0.31,
                "grinding_w_avg": 0.29,
                "grinding_t_avg": 0.28,
                "total_layer": 320,
                "predicted_capacity": 10.5,
            },
            {
                "rank": 2,
                "active_powder_base": "B",
                "active_powder_additives": "X2",
                "ldn_avr_value": 1.0,
                "cast_dsgn_thk": 2.0,
                "grinding_l_avg": 0.30,
                "grinding_w_avg": 0.28,
                "grinding_t_avg": 0.27,
                "total_layer": 310,
                "predicted_capacity": 10.1,
            },
            {
                "rank": 3,
                "active_powder_base": "C",
                "active_powder_additives": "X3",
                "ldn_avr_value": 1.05,
                "cast_dsgn_thk": 2.1,
                "grinding_l_avg": 0.29,
                "grinding_w_avg": 0.27,
                "grinding_t_avg": 0.26,
                "total_layer": 300,
                "predicted_capacity": 9.9,
            },
            {
                "rank": 4,
                "active_powder_base": "D",
                "active_powder_additives": "X4",
                "ldn_avr_value": 1.2,
                "cast_dsgn_thk": 2.3,
                "grinding_l_avg": 0.32,
                "grinding_w_avg": 0.30,
                "grinding_t_avg": 0.29,
                "total_layer": 330,
                "predicted_capacity": 10.8,
            },
            {
                "rank": 5,
                "active_powder_base": "E",
                "active_powder_additives": "X5",
                "ldn_avr_value": 0.98,
                "cast_dsgn_thk": 1.95,
                "grinding_l_avg": 0.28,
                "grinding_w_avg": 0.26,
                "grinding_t_avg": 0.25,
                "total_layer": 295,
                "predicted_capacity": 9.7,
            },
        ]
        # 최근 6개월 유사 설계 표를 만든다.
        tables["recent_similar_table"] = [
            {
                "candidate_rank": 1,
                "match_count": 8,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-071",
            },
            {
                "candidate_rank": 2,
                "match_count": 6,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-088",
            },
            {
                "candidate_rank": 3,
                "match_count": 5,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-103",
            },
            {
                "candidate_rank": 4,
                "match_count": 4,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-120",
            },
            {
                "candidate_rank": 5,
                "match_count": 3,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-134",
            },
        ]
        # 불량률 요약 표를 만든다(모든 metric 포함).
        metric_specs = [
            {"metric": "ci_def_rate", "base": 0.12, "step": 0.02, "delta": 0.02},
            {"metric": "fr_defect_rate", "base": 120, "step": 15, "delta": 10},
            {"metric": "gr_short_defect_rate", "base": 0.08, "step": 0.01, "delta": 0.01},
            {"metric": "tvi_defect_rate_f", "base": 0.06, "step": 0.01, "delta": 0.01},
            {"metric": "tr_short_defect_rate", "base": 0.05, "step": 0.01, "delta": 0.01},
            {"metric": "df_def_rate", "base": 0.09, "step": 0.01, "delta": 0.01},
            {"metric": "soul_defect_rate_f", "base": 0.04, "step": 0.01, "delta": 0.01},
            {"metric": "gm_defect_rate_f", "base": 0.03, "step": 0.01, "delta": 0.01},
            {"metric": "pi_def_rate", "base": 0.11, "step": 0.02, "delta": 0.02},
            {"metric": "mf_def_rate", "base": 0.07, "step": 0.01, "delta": 0.01},
            {"metric": "ttm_defect_rate_f", "base": 0.05, "step": 0.01, "delta": 0.01},
            {"metric": "sum_burn_ppm", "base": 90, "step": 12, "delta": 8},
            {"metric": "sum_8585_ppm", "base": 110, "step": 14, "delta": 9},
            {"metric": "fail_halt_ppm", "base": 70, "step": 10, "delta": 7},
        ]
        defect_rows = []
        for rank in range(1, 6):
            for spec in metric_specs:
                avg = spec["base"] + spec["step"] * (rank - 1)
                min_value = avg - spec["delta"]
                max_value = avg + spec["delta"]
                if min_value < 0:
                    min_value = 0
                defect_rows.append(
                    {
                        "candidate_rank": rank,
                        "defect_metric": spec["metric"],
                        "defect_avg": avg,
                        "defect_min": min_value,
                        "defect_max": max_value,
                    }
                )
        tables["defect_rate_table"] = defect_rows
        # 불량률 차트를 만든다.
        charts = [
            {
                "chart_id": "defect_rate_summary",
                "type": "bar",
                "title": "불량률 비교",
                "x_label": "후보",
                "y_label": "불량률",
                "series": [
                    {
                        "name": "ci_def_rate",
                        "points": [
                            {"x": "rank_1", "y": 0.12},
                            {"x": "rank_2", "y": 0.18},
                            {"x": "rank_3", "y": 0.16},
                            {"x": "rank_4", "y": 0.20},
                            {"x": "rank_5", "y": 0.14},
                        ],
                    }
                ],
                "notes": "",
            }
        ]
    # 단계 근거 요약을 만든다.
    stage_notes = _build_stage_notes(
        input_params, tables, charts, configs, selections, user_prefs
    )
    # 한글 라벨 매핑을 적용한다.
    label_map = _get_label_mapping(request.demo)
    tables = _map_table_labels(tables, label_map)
    return blocks, tables, charts, stage_notes


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest) -> ChatResponse:
    # 요청을 라우팅한다.
    route = await _route_with_llm(request.message)
    # 세션 상태를 가져온다.
    state = _get_session_state(request.session_id)
    # 라우트에 맞는 기본 응답을 만든다.
    if route == "simulation":
        command = await _decide_command_with_llm(request.message)
        if command.action == "explain_stage":
            # 요청 단계가 없으면 마지막 설명 단계를 재사용한다.
            target_stage = _normalize_stage(
                command.target_stage or state.get("last_explain_stage"),
                state["stage_status"],
            )
            # 설명용 LLM을 호출해 답변을 만든다.
            blocks, tables, charts = await _build_explain_response(
                state, target_stage, request.message
            )
            # 마지막 설명 단계를 저장한다.
            if target_stage:
                state["last_explain_stage"] = target_stage
            state["history"].append(
                {
                    "action": "explain_stage",
                    "payload": {"target_stage": target_stage},
                    "at": _utc_now(),
                }
            )
        elif command.action == "update_input":
            # 변경 요청을 파싱한다.
            update = await _parse_update_with_llm(request.message)
            missing = update.missing_fields or []
            # 값이 없는 변경 요청이면 pending_action으로 보류한다.
            if missing:
                current_params = InputParams(**state["input_params"])
                incoming_params = update.input_params or InputParams()
                merged_params = _merge_input_params(
                    current_params.dict(), incoming_params
                )
                state["input_params"] = merged_params.dict()
                _apply_update_fields(state, update)
                state["pending_action"] = _build_pending_action(missing)
                state["history"].append(
                    {
                        "action": "update_input_pending",
                        "payload": {"missing": missing},
                        "at": _utc_now(),
                    }
                )
                blocks = [
                    {
                        "type": "text",
                        "section": "summary",
                        "value": _format_update_missing(missing),
                    }
                ]
                tables, charts = {}, []
            else:
                # 변경값을 상태에 반영한다.
                _apply_update_fields(state, update)
                current_params = InputParams(**state["input_params"])
                incoming_params = update.input_params or InputParams()
                merged_params = _merge_input_params(
                    current_params.dict(), incoming_params
                )
                missing = _get_missing_fields(merged_params)
                # 변경 후 시뮬레이션 결과를 만든다.
                blocks, tables, charts, stage_notes = _build_simulation_stub(
                    request,
                    merged_params,
                    state["configs"],
                    state["selections"],
                    state["user_prefs"],
                )
                if not missing:
                    blocks = await _build_briefing_blocks(tables, charts)
                _update_state(
                    state,
                    merged_params,
                    tables,
                    charts,
                    blocks,
                    stage_notes,
                    missing,
                    request.demo,
                )
        else:
            input_params = await _parse_input_with_llm(request.message)
            merged_params = _merge_input_params(state["input_params"], input_params)
            missing = _get_missing_fields(merged_params)
            blocks, tables, charts, stage_notes = _build_simulation_stub(
                request,
                merged_params,
                state["configs"],
                state["selections"],
                state["user_prefs"],
            )
            if not missing:
                blocks = await _build_briefing_blocks(tables, charts)
            _update_state(
                state,
                merged_params,
                tables,
                charts,
                blocks,
                stage_notes,
                missing,
                request.demo,
            )
    else:
        blocks, tables, charts = _build_casual_stub()
    # 응답을 구성한다.
    return ChatResponse(route=route, blocks=blocks, tables=tables, charts=charts)
