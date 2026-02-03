import json
import os
from typing import Any, AsyncIterator, Callable

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.callbacks import CallbackContext
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from . import db_production, demo, schemas, state

# 앱 상태 키를 정의한다.
_APP_STATE_KEY = "app:mlcc_state"
# 런타임 임시 키를 정의한다.
_RUN_STATE_KEY = "temp:run_state"
# 멈춤 플래그 키를 정의한다.
_HALT_KEY = "temp:halt"
# 단계 결과 키를 정의한다.
_RUN_TABLES_KEY = "temp:run_tables"
_RUN_CHARTS_KEY = "temp:run_charts"
_RUN_NOTES_KEY = "temp:run_notes"
_RUN_BLOCKS_KEY = "temp:run_blocks"
# gap 정보를 저장할 키를 정의한다.
_GAP_KEY = "temp:gap"
# 데모 출력물을 저장할 키를 정의한다.
_DEMO_OUTPUTS_KEY = "temp:demo_outputs"

# 기본 모델명을 정의한다.
MODEL_NAME = os.getenv("ADK_MODEL", "gemini-2.0-flash")

# 단계 카탈로그를 정의한다.
STAGE_CATALOG = [
    {
        "id": "1-1",
        "name": "입력 수집",
        "keywords": ["입력", "조건", "파라미터", "스펙", "온도", "전압", "용량", "치수", "분말"],
    },
    {
        "id": "1-2",
        "name": "칩기종 후보",
        "keywords": ["칩기종", "후보", "매칭", "유사"],
    },
    {
        "id": "1-3",
        "name": "REF LOT 선정",
        "keywords": ["ref lot", "레퍼런스", "LOT", "선정", "후보"],
    },
    {
        "id": "1-4",
        "name": "API payload",
        "keywords": ["payload", "API", "전송", "요청", "ref", "sim"],
    },
    {
        "id": "1-5",
        "name": "Top-K",
        "keywords": ["top-k", "rank", "순위", "예측", "용량", "capacity"],
    },
    {
        "id": "1-6",
        "name": "최근 유사 LOT",
        "keywords": ["최근", "유사", "6개월", "match", "대표 LOT"],
    },
    {
        "id": "1-7",
        "name": "불량률/차트",
        "keywords": ["불량률", "defect", "차트", "metric", "ppm", "percent"],
    },
    {
        "id": "1-8",
        "name": "브리핑",
        "keywords": ["브리핑", "요약", "결론", "설명"],
    },
]


def _format_stage_catalog() -> str:
    # 단계 카탈로그 텍스트를 만든다.
    lines = []
    for item in STAGE_CATALOG:
        keywords = ", ".join(item["keywords"])
        lines.append(f"{item['id']} - {item['name']} | keywords: {keywords}")
    return "\n".join(lines)


# 단계 카탈로그 텍스트를 준비한다.
STAGE_CATALOG_TEXT = _format_stage_catalog()

# 커맨드 에이전트 힌트를 정의한다.
COMMAND_STAGE_HINT = (
    "\n\n[단계 카탈로그]\n"
    f"{STAGE_CATALOG_TEXT}\n"
    "규칙: 사용자 질문에서 단계 의미를 추론해 target_stage를 지정해."
)

# 업데이트 에이전트 힌트를 정의한다.
UPDATE_STAGE_HINT = (
    "\n\n[필드 매핑 힌트]\n"
    "- ref lot/레퍼런스/LOT 변경 -> selections.reference_lot_id\n"
    "- 칩기종/칩 타입 변경 -> selections.chip_type_ids (리스트)\n"
    "- top-k/순위 변경 -> configs.top_k\n"
    "\n[단계 카탈로그]\n"
    f"{STAGE_CATALOG_TEXT}"
)

# 라우팅 에이전트 지침을 정의한다.
ROUTE_AGENT_INSTRUCTIONS = (
    "사용자 메시지를 보고 route를 분류해.\n"
    "- simulation: 칩 설계/시뮬레이션/추천을 요청하는 경우, 산출된 산출물(REF LOT/ 칩기종/ 차트/ 불량률/ 선정기준) 에 대한 근거 자료를 요청하는경우, 시뮬레이션 리셋/처음부터 다시 요청\n"
    "- casual: 그 외 일반 대화\n"
    "반드시 route만 출력해."
)

# 커맨드 에이전트 지침을 정의한다.
COMMAND_AGENT_INSTRUCTIONS = (
    "사용자 메시지를 보고 action을 결정해.\n"
    "- run: 시뮬레이션 시작/진행/결과 요청과 모든 변경 요청\n"
    "- reset: 시뮬레이션 상태 초기화 요청(처음부터 다시/리셋)\n"
    "- explain_stage: 특정 단계 근거/이유 요청\n"
    "사용자 메시지에 [STATE_HINT]가 포함되면 참고해.\n"
    "- has_results=false면 run 우선\n"
    "- reset 요청이면 pending_action과 무관하게 reset\n"
    "- pending_action이 있어도 action은 run\n"
    "단계가 명시되면 target_stage에 1-4 형식으로 넣어.\n"
    "단계가 없으면 target_stage는 null.\n"
    "action과 target_stage만 출력해.\n"
    "\n[예시]\n"
    "[STATE_HINT]\n"
    "- has_results: false\n"
    "- has_input_complete: false\n"
    "- pending_action: none\n"
    "- last_action: none\n"
    "[사용자 메시지]\n"
    '"온도 25, 전압 6, 용량 10uF로 시뮬레이션 해줘"\n'
    "=> action: run, target_stage: null\n"
    "\n"
    "[STATE_HINT]\n"
    "- has_results: true\n"
    "- has_input_complete: true\n"
    "- pending_action: none\n"
    "- last_action: update_state\n"
    "[사용자 메시지]\n"
    '"레퍼런스 LOT 바꿔줘"\n'
    "=> action: run, target_stage: null\n"
    "\n"
    "[STATE_HINT]\n"
    "- has_results: true\n"
    "- has_input_complete: true\n"
    "- pending_action: none\n"
    "- last_action: update_state\n"
    "[사용자 메시지]\n"
    '"1-6 단계 근거 설명해줘"\n'
    "=> action: explain_stage, target_stage: 1-6\n"
) + COMMAND_STAGE_HINT

# 입력 파싱 에이전트 지침을 정의한다.
INPUT_AGENT_INSTRUCTIONS = (
    "메시지에서 다음 필드만 추출해 JSON으로 출력해: "
    "temperature, voltage, size, capacity, chip_prod_id.\n"
    "없으면 null. 추측 금지. 위 5개 필드 외에는 출력 금지.\n"
    "출력은 반드시 JSON 객체 1개만.\n"
    "사용자가 '예제', '샘플', 'test data', 'example'등 예시 데이터 사용을 요청한 경우에 "
    "아래 <example_output>을 그대로 출력.\n"
    "<example_output>\n"
    "{\n"
    "  \"temperature\": 55,\n"
    "  \"voltage\": 6,\n"
    "  \"size\": \"1005\",\n"
    "  \"capacity\": 10000000,\n"
    "  \"chip_prod_id\": null\n"
    "}\n"
    "</example_output>"
)

# 변경 요청 파싱 에이전트 지침을 정의한다.
UPDATE_AGENT_INSTRUCTIONS = (
    "사용자 메시지에서 변경 요청을 추출해.\n"
    "- input_params: temperature, voltage, size, capacity, chip_prod_id\n"
    "- selections: chip_type_ids(리스트), reference_lot_id\n"
    "- configs: top_k\n"
    "- user_prefs: chart_type(bar|line|scatter)\n"
    "규칙:\n"
    "- 변경 의도만 있고 값이 없으면 missing_fields에 해당 키를 넣어.\n"
    "- 값이 있는 항목만 채워. 나머지는 null.\n"
    "missing_fields 포함해서 출력해."
) + UPDATE_STAGE_HINT

# 설명 에이전트 지침을 정의한다.
EXPLAIN_AGENT_INSTRUCTIONS = (
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
)

# 갭 질문 에이전트 지침을 정의한다.
GAP_AGENT_INSTRUCTIONS = (
    "다음 JSON을 보고 사용자에게 확인 질문을 만들어.\n"
    "- stage: 단계\n"
    "- reason: 데이터 공백 이유\n"
    "- fallback_summary: 대체조건 요약\n"
    "- candidate_count: 후보 개수\n"
    "규칙:\n"
    "- 항상 확인 질문으로 끝내.\n"
    "- 짧고 간결한 한국어 2~3문장.\n"
    "- fallback_summary를 한 번 언급.\n"
    "question만 출력해."
)

# 선택 에이전트 지침을 정의한다.
SELECTION_AGENT_INSTRUCTIONS = (
    "다음 JSON을 보고 후보 선택 결과를 만들어.\n"
    "- message: 사용자 메시지\n"
    "- candidate_ids: 후보 ID 리스트\n"
    "규칙:\n"
    "- candidate_ids에 있는 값만 선택해.\n"
    "- 전체/전부/다 선택 요청이면 전부 선택해.\n"
    "- 'XX로 시작' 또는 'XX로 시작하는'은 접두사 매칭으로 처리해.\n"
    "- 부분 일치 요청이면 포함되는 후보를 모두 선택해.\n"
    "- 매칭이 없으면 빈 리스트를 반환해.\n"
    "selected_ids만 출력해."
)

# 브리핑 에이전트 지침을 정의한다.
BRIEFING_AGENT_INSTRUCTIONS = (
    "아래 표/차트 데이터를 보고 각 단계별 브리핑 텍스트를 생성해.\n"
    "- 출력 형식: texts 배열 (단계별 텍스트만)\n"
    "- table_ref/chart_ref는 생성하지 않음 (코드에서 자동 삽입됨)\n"
    "- text는 한국어로 친절한 문장으로 작성\n"
    "- 문장마다 줄바꿈(\\n)으로 끝내고, 한 줄에 문장 1개만 작성\n"
    "- 빈 줄 금지\n"
    "- 표/차트 값을 인용하여 설명\n"
    "- briefing_hint가 있으면 첫 텍스트(summary)에 반영\n"
    "- stage_sequence 순서대로 작성\n"
    "- stage_sequence.note(근거)를 활용하되, '근거'라는 단어는 직접 언급하지 않음\n"
    "- 테이블에서 __로 시작하는 메타 필드는 무시\n"
    "- children 지표는 언급하지 않음\n"
    "\n[출력 템플릿 예시]\n"
    "{\n"
    "  \"texts\": [\n"
    "    {\"section\": \"summary\", \"value\": \"전체 요약 문장1\\n전체 요약 문장2\"},\n"
    "    {\"section\": \"1-2\", \"value\": \"칩기종 후보 설명\"},\n"
    "    {\"section\": \"1-3\", \"value\": \"레퍼런스 LOT 선정 설명\"},\n"
    "    {\"section\": \"1-5\", \"value\": \"top-k 후보 요약\"},\n"
    "    {\"section\": \"1-6\", \"value\": \"최근 유사 설계 요약\"},\n"
    "    {\"section\": \"1-7\", \"value\": \"불량률 요약\"},\n"
    "    {\"section\": \"conclusion\", \"value\": \"최종 결론\"}\n"
    "  ]\n"
    "}\n"
)

# 캐주얼 에이전트 지침을 정의한다.
CASUAL_AGENT_INSTRUCTIONS = (
    "사용자 메시지에 대해 캐주얼하게 한국어로 답변해.\n"
    "- 2~4문장으로 짧게\n"
    "- 필요하면 질문 1개만\n"
    "- 시뮬레이션이 필요해 보이면 '시뮬레이션'이라고 말해 달라고 안내\n"
    "answer만 출력해."
)


def _text_content(text: str, role: str = "user") -> types.Content:
    # 텍스트 콘텐츠를 만든다.
    return types.Content(role=role, parts=[types.Part(text=text)])


def _content_to_text(content: types.Content | None) -> str:
    # 콘텐츠에서 텍스트를 추출한다.
    if not content or not content.parts:
        return ""
    return "\n".join(part.text or "" for part in content.parts if isinstance(part, types.Part))


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    # JSON 텍스트를 파싱한다.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_json_model(text: str, model_cls: type[Any]) -> Any:
    # 텍스트를 모델로 파싱한다.
    try:
        return model_cls.model_validate_json(text)
    except Exception:
        # JSON 블록만 추출해 재시도한다.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            sliced = text[start : end + 1]
            return model_cls.model_validate_json(sliced)
        raise


def _merge_instructions(base: str, memory_context: str | None) -> str:
    # 메모리 컨텍스트를 지침에 결합한다.
    if not memory_context:
        return base
    return f"{base}\n\n{memory_context}"


def _get_session_id(ctx: InvocationContext) -> str:
    # 세션 ID를 추출한다.
    session = getattr(ctx, "session", None)
    if session is None:
        return "default"
    return getattr(session, "session_id", None) or getattr(session, "id", None) or "default"


def _get_app_state(ctx: InvocationContext) -> dict[str, Any]:
    # 앱 상태를 가져온다.
    stored = ctx.session.state.get(_APP_STATE_KEY)
    if isinstance(stored, dict):
        return stored
    return {}


def _set_app_state(ctx: InvocationContext, session_state: dict[str, Any]) -> None:
    # 앱 상태를 저장한다.
    ctx.session.state[_APP_STATE_KEY] = session_state


def _get_run_state(ctx: InvocationContext) -> dict[str, Any]:
    # 런타임 상태를 준비한다.
    stored = ctx.session.state.get(_RUN_STATE_KEY)
    if isinstance(stored, dict):
        return stored
    fresh: dict[str, Any] = {}
    ctx.session.state[_RUN_STATE_KEY] = fresh
    return fresh


def _reset_run_state(ctx: InvocationContext) -> dict[str, Any]:
    # 런타임 상태를 초기화한다.
    fresh: dict[str, Any] = {}
    ctx.session.state[_RUN_STATE_KEY] = fresh
    ctx.session.state[_HALT_KEY] = False
    ctx.session.state[_GAP_KEY] = None
    ctx.session.state[_RUN_TABLES_KEY] = {}
    ctx.session.state[_RUN_CHARTS_KEY] = []
    ctx.session.state[_RUN_NOTES_KEY] = {}
    ctx.session.state[_RUN_BLOCKS_KEY] = []
    return fresh


def _should_skip_stage(callback_state: dict[str, Any], stage_id: str) -> bool:
    # 멈춤 플래그를 확인한다.
    if callback_state.get(_HALT_KEY):
        return True
    # dirty 단계가 있으면 포함 여부를 확인한다.
    dirty = callback_state.get("temp:dirty_stages")
    if isinstance(dirty, list) and dirty:
        return stage_id not in dirty
    return False


def _build_progress_event(author: str, logs: list[dict[str, Any]]) -> Event:
    # progress 이벤트를 만든다.
    payload = {"type": "progress", "logs": logs}
    content = _text_content(json.dumps(payload, ensure_ascii=False), role="model")
    return Event(author=author, content=content, actions=EventActions())


def _build_final_event(author: str, payload: dict[str, Any]) -> Event:
    # final 이벤트를 만든다.
    body = {"type": "final", "payload": payload}
    content = _text_content(json.dumps(body, ensure_ascii=False), role="model")
    return Event(author=author, content=content, actions=EventActions())


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
    return payload.get("action") == "select_candidates"


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


def _build_pending_blocks(
    question: str,
    pending_action: dict[str, Any],
    selected_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    # 질문 텍스트 블록을 만든다.
    question_block = {"type": "text", "section": "summary", "value": question}
    # 선택 블록을 만든다.
    select_block = _build_table_select_block(
        pending_action,
        selected_ids=selected_ids or [],
    )
    return [question_block, select_block]


def _build_pending_repeat_response(
    session_state: dict[str, Any],
    pending_action: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # 기본 질문을 준비한다.
    question = pending_action.get("question") or "후보를 선택해 진행할까요?"
    # 기존 선택값을 꺼낸다.
    selected_ids = session_state.get("selections", {}).get("chip_type_ids") or []
    # 질문 블록을 만든다.
    blocks = _build_pending_blocks(question, pending_action, selected_ids=selected_ids)
    # 이전 테이블/차트를 불러온다.
    tables, charts = state._load_raw_outputs(
        session_state.get("raw_refs", {}).get("stage_outputs_path")
    )
    # 선택 강조 표시를 적용한다.
    state._apply_table_highlights(tables, session_state.get("selections", {}))
    # 히스토리를 기록한다.
    session_state["history"].append(
        {
            "action": "pending_repeat",
            "payload": {"action": pending_action.get("action")},
            "at": state._utc_now(),
        }
    )
    return blocks, tables, charts


def _extract_candidate_ids(rows: list[dict[str, Any]] | None, id_field: str) -> list[str]:
    # 후보 ID 목록을 만든다.
    candidate_ids: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        value = row.get(id_field)
        if value is None:
            continue
        candidate_ids.append(str(value))
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
    return pending_action, blocks


def _build_input_form_blocks(merged_params: schemas.InputParams) -> list[dict[str, Any]]:
    # 왼쪽 입력 필드 목록을 준비한다.
    left_keys = ["temperature", "size", "capacity", "voltage"]
    # 오른쪽 입력 필드 목록을 준비한다.
    right_keys = ["chip_prod_id"]
    # 폼 필드를 담을 리스트를 만든다.
    fields: list[dict[str, Any]] = []
    # 왼쪽 입력 항목을 구성한다.
    for key in left_keys:
        current_val = merged_params.dict().get(key)
        field_def = {
            "key": key,
            "label": state.INPUT_LABEL_MAP.get(key, key),
            "type": "text",
            "value": current_val or "",
            "column": "left",
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
    # 오른쪽 입력 항목을 구성한다.
    for key in right_keys:
        current_val = merged_params.dict().get(key)
        fields.append(
            {
                "key": key,
                "label": state.INPUT_LABEL_MAP.get(key, key),
                "type": "text",
                "value": current_val or "",
                "placeholder": "예: CL32Y106KCBNB",
                "column": "right",
            }
        )
    # 입력 폼 블록을 반환한다.
    return [
        {
            "type": "input_form",
            "form_id": "mlcc_basic_params",
            "title": "시뮬레이션 조건 입력",
            "description": "4개 조건 또는 CHIP 기종 중 하나를 입력해주세요.",
            "fields": fields,
            "submit_label": "시뮬레이션 시작",
            "submitted": False,
        }
    ]


async def _run_llm_text(
    agent: LlmAgent,
    ctx: InvocationContext,
    message_text: str | None = None,
    instruction_override: str | None = None,
) -> str:
    # 입력 메시지를 준비한다.
    if message_text is None:
        content = ctx.user_content
    else:
        content = _text_content(message_text)
    # 호출용 컨텍스트를 복제한다.
    try:
        child_ctx = ctx.model_copy(update={"user_content": content})
    except Exception:
        child_ctx = ctx.copy(update={"user_content": content})
    # 지침을 덮어쓸 에이전트를 준비한다.
    target_agent = agent
    if instruction_override:
        target_agent = agent.clone(update={"instruction": instruction_override})
    # 응답 텍스트를 모은다.
    final_text = ""
    async for event in target_agent.run_async(child_ctx):
        text = _content_to_text(event.content)
        if text:
            final_text = text
    return final_text


class StageAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        stage_id: str,
        run_fn: Callable[[InvocationContext, dict[str, Any]], Any],
    ) -> None:
        # 베이스 에이전트를 초기화한다.
        super().__init__(name=name, description=f"stage-{stage_id}")
        self.stage_id = stage_id
        self._run_fn = run_fn
        # 단계 스킵 콜백을 등록한다.
        self.before_agent_callback = self._before_agent_callback

    def _before_agent_callback(self, callback_context: CallbackContext):
        # 콜백 상태를 확인한다.
        callback_state = callback_context.state
        # 스킵 조건을 확인한다.
        if _should_skip_stage(callback_state, self.stage_id):
            return _text_content("", role="model")
        return None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        # 세션 상태를 준비한다.
        session_state = _get_app_state(ctx)
        # 진행 로그를 만든다.
        run_state = _get_run_state(ctx)
        logs = state._build_progress_logs(
            run_state.get("route"),
            run_state.get("action"),
            session_state.get("stage_status"),
            current_stage=self.stage_id,
            is_final=False,
        )
        # 진행 로그 이벤트를 전송한다.
        if logs:
            yield _build_progress_event(self.name, logs)
        # 단계 실행 함수를 호출한다.
        result = await self._run_fn(ctx, session_state)
        if not isinstance(result, dict):
            return
        # 실행 결과를 꺼낸다.
        tables = result.get("tables") or {}
        charts = result.get("charts") or []
        notes = result.get("stage_notes") or {}
        blocks = result.get("blocks") or []
        gap = result.get("gap")
        # 단계 결과를 누적한다.
        run_tables = ctx.session.state.get(_RUN_TABLES_KEY, {})
        run_charts = ctx.session.state.get(_RUN_CHARTS_KEY, [])
        run_notes = ctx.session.state.get(_RUN_NOTES_KEY, {})
        run_blocks = ctx.session.state.get(_RUN_BLOCKS_KEY, [])
        if isinstance(run_tables, dict):
            run_tables.update(tables)
            ctx.session.state[_RUN_TABLES_KEY] = run_tables
        if isinstance(run_charts, list):
            run_charts = state._merge_chart_outputs(run_charts, charts)
            ctx.session.state[_RUN_CHARTS_KEY] = run_charts
        if isinstance(run_notes, dict):
            run_notes.update(notes)
            ctx.session.state[_RUN_NOTES_KEY] = run_notes
        if blocks:
            run_blocks = list(run_blocks) if isinstance(run_blocks, list) else []
            run_blocks = blocks
            ctx.session.state[_RUN_BLOCKS_KEY] = run_blocks
        # gap이 있으면 중단 플래그를 세운다.
        if gap:
            ctx.session.state[_GAP_KEY] = gap
            ctx.session.state[_HALT_KEY] = True


class RootAgent(BaseAgent):
    def __init__(self, use_db: bool) -> None:
        # 베이스 에이전트를 초기화한다.
        super().__init__(name="RootAgent", description="mlcc-root")
        self.use_db = use_db
        # LLM 에이전트를 준비한다.
        self.router_agent = LlmAgent(
            name="RouteAgent",
            model=MODEL_NAME,
            instruction=ROUTE_AGENT_INSTRUCTIONS,
            output_schema=schemas.RouteDecision,
        )
        self.command_agent = LlmAgent(
            name="CommandAgent",
            model=MODEL_NAME,
            instruction=COMMAND_AGENT_INSTRUCTIONS,
            output_schema=schemas.CommandDecision,
        )
        self.input_agent = LlmAgent(
            name="InputAgent",
            model=MODEL_NAME,
            instruction=INPUT_AGENT_INSTRUCTIONS,
            output_schema=schemas.InputParams,
        )
        self.update_agent = LlmAgent(
            name="UpdateAgent",
            model=MODEL_NAME,
            instruction=UPDATE_AGENT_INSTRUCTIONS,
            output_schema=schemas.UpdateDecision,
        )
        self.explain_agent = LlmAgent(
            name="ExplainAgent",
            model=MODEL_NAME,
            instruction=EXPLAIN_AGENT_INSTRUCTIONS,
            output_schema=schemas.ExplainOutput,
        )
        self.gap_agent = LlmAgent(
            name="GapAgent",
            model=MODEL_NAME,
            instruction=GAP_AGENT_INSTRUCTIONS,
            output_schema=schemas.GapQuestionOutput,
        )
        self.selection_agent = LlmAgent(
            name="SelectionAgent",
            model=MODEL_NAME,
            instruction=SELECTION_AGENT_INSTRUCTIONS,
            output_schema=schemas.SelectionDecision,
        )
        self.briefing_agent = LlmAgent(
            name="BriefingAgent",
            model=MODEL_NAME,
            instruction=BRIEFING_AGENT_INSTRUCTIONS,
            output_schema=schemas.StageBriefingOutput,
        )
        self.casual_agent = LlmAgent(
            name="CasualAgent",
            model=MODEL_NAME,
            instruction=CASUAL_AGENT_INSTRUCTIONS,
            output_schema=schemas.CasualOutput,
        )
        # 단계 에이전트를 준비한다.
        self.stage_1_1 = StageAgent("Stage-1-1", "1-1", self._run_stage_1_1)
        self.stage_1_2 = StageAgent("Stage-1-2", "1-2", self._run_stage_1_2)
        self.stage_1_3 = StageAgent("Stage-1-3", "1-3", self._run_stage_1_3)
        self.stage_1_4 = StageAgent("Stage-1-4", "1-4", self._run_stage_1_4)
        self.stage_1_5 = StageAgent("Stage-1-5", "1-5", self._run_stage_1_5)
        self.stage_1_6 = StageAgent("Stage-1-6", "1-6", self._run_stage_1_6)
        self.stage_1_7 = StageAgent("Stage-1-7", "1-7", self._run_stage_1_7)
        self.stage_1_8 = StageAgent("Stage-1-8", "1-8", self._run_stage_1_8)
        # SequentialAgent를 준비한다.
        self.simulation_agent = SequentialAgent(
            name="SimulationSequential",
            sub_agents=[
                self.stage_1_1,
                self.stage_1_2,
                self.stage_1_3,
                self.stage_1_4,
                self.stage_1_5,
                self.stage_1_6,
                self.stage_1_7,
                self.stage_1_8,
            ],
        )

    async def _run_stage_1_1(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 입력 테이블을 만든다.
        input_params = schemas.InputParams(**session_state.get("input_params", {}))
        rows = [
            {"item": "temperature", "value": input_params.temperature or ""},
            {"item": "voltage", "value": input_params.voltage or ""},
            {"item": "size", "value": input_params.size or ""},
            {"item": "capacity", "value": input_params.capacity or ""},
            {"item": "chip_prod_id", "value": input_params.chip_prod_id or ""},
        ]
        tables = {"input_params_table": rows}
        # 단계 노트를 만든다.
        stage_notes = {"1-1": "입력값을 확인했습니다."}
        return {"tables": tables, "charts": [], "stage_notes": stage_notes, "gap": None}

    async def _run_stage_1_2(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 데모 출력이 있으면 데모 데이터를 사용한다.
        demo_outputs = ctx.session.state.get(_DEMO_OUTPUTS_KEY)
        if isinstance(demo_outputs, dict):
            tables = demo_outputs.get("tables", {})
            charts = demo_outputs.get("charts", [])
            notes = demo_outputs.get("stage_notes", {})
            return {
                "tables": {
                    key: tables.get(key)
                    for key in state._STAGE_TABLE_KEYS.get("1-2", [])
                    if key in tables
                },
                "charts": [
                    chart
                    for chart in charts
                    if chart.get("chart_id")
                    in state._STAGE_CHART_IDS.get("1-2", [])
                ],
                "stage_notes": {"1-2": notes.get("1-2", "")},
                "gap": None,
            }
        # DB 시뮬레이션을 실행한다.
        input_params = schemas.InputParams(**session_state.get("input_params", {}))
        tables, charts, stage_notes, gap = db_production.build_simulation_from_db(
            input_params,
            session_state.get("configs", {}),
            session_state.get("selections", {}),
            session_state.get("user_prefs", {}),
            dirty_stages=["1-2"],
        )
        return {
            "tables": tables,
            "charts": charts,
            "stage_notes": stage_notes,
            "gap": gap,
        }

    async def _run_stage_1_3(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 데모 출력이 있으면 데모 데이터를 사용한다.
        demo_outputs = ctx.session.state.get(_DEMO_OUTPUTS_KEY)
        if isinstance(demo_outputs, dict):
            tables = demo_outputs.get("tables", {})
            notes = demo_outputs.get("stage_notes", {})
            return {
                "tables": {
                    key: tables.get(key)
                    for key in state._STAGE_TABLE_KEYS.get("1-3", [])
                    if key in tables
                },
                "charts": [],
                "stage_notes": {"1-3": notes.get("1-3", "")},
                "gap": None,
            }
        # DB 시뮬레이션을 실행한다.
        input_params = schemas.InputParams(**session_state.get("input_params", {}))
        tables, charts, stage_notes, gap = db_production.build_simulation_from_db(
            input_params,
            session_state.get("configs", {}),
            session_state.get("selections", {}),
            session_state.get("user_prefs", {}),
            dirty_stages=["1-3"],
        )
        return {
            "tables": tables,
            "charts": charts,
            "stage_notes": stage_notes,
            "gap": gap,
        }

    async def _run_stage_1_4(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 데모 출력이 있으면 데모 데이터를 사용한다.
        demo_outputs = ctx.session.state.get(_DEMO_OUTPUTS_KEY)
        if isinstance(demo_outputs, dict):
            notes = demo_outputs.get("stage_notes", {})
            return {
                "tables": {},
                "charts": [],
                "stage_notes": {"1-4": notes.get("1-4", "")},
                "gap": None,
            }
        # DB 시뮬레이션을 실행한다.
        input_params = schemas.InputParams(**session_state.get("input_params", {}))
        tables, charts, stage_notes, gap = db_production.build_simulation_from_db(
            input_params,
            session_state.get("configs", {}),
            session_state.get("selections", {}),
            session_state.get("user_prefs", {}),
            dirty_stages=["1-4"],
        )
        return {
            "tables": tables,
            "charts": charts,
            "stage_notes": stage_notes,
            "gap": gap,
        }

    async def _run_stage_1_5(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 데모 출력이 있으면 데모 데이터를 사용한다.
        demo_outputs = ctx.session.state.get(_DEMO_OUTPUTS_KEY)
        if isinstance(demo_outputs, dict):
            tables = demo_outputs.get("tables", {})
            notes = demo_outputs.get("stage_notes", {})
            return {
                "tables": {
                    key: tables.get(key)
                    for key in state._STAGE_TABLE_KEYS.get("1-5", [])
                    if key in tables
                },
                "charts": [],
                "stage_notes": {"1-5": notes.get("1-5", "")},
                "gap": None,
            }
        # DB 시뮬레이션을 실행한다.
        input_params = schemas.InputParams(**session_state.get("input_params", {}))
        tables, charts, stage_notes, gap = db_production.build_simulation_from_db(
            input_params,
            session_state.get("configs", {}),
            session_state.get("selections", {}),
            session_state.get("user_prefs", {}),
            dirty_stages=["1-5"],
        )
        return {
            "tables": tables,
            "charts": charts,
            "stage_notes": stage_notes,
            "gap": gap,
        }

    async def _run_stage_1_6(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 데모 출력이 있으면 데모 데이터를 사용한다.
        demo_outputs = ctx.session.state.get(_DEMO_OUTPUTS_KEY)
        if isinstance(demo_outputs, dict):
            tables = demo_outputs.get("tables", {})
            notes = demo_outputs.get("stage_notes", {})
            return {
                "tables": {
                    key: tables.get(key)
                    for key in state._STAGE_TABLE_KEYS.get("1-6", [])
                    if key in tables
                },
                "charts": [],
                "stage_notes": {"1-6": notes.get("1-6", "")},
                "gap": None,
            }
        # DB 시뮬레이션을 실행한다.
        input_params = schemas.InputParams(**session_state.get("input_params", {}))
        tables, charts, stage_notes, gap = db_production.build_simulation_from_db(
            input_params,
            session_state.get("configs", {}),
            session_state.get("selections", {}),
            session_state.get("user_prefs", {}),
            dirty_stages=["1-6"],
        )
        return {
            "tables": tables,
            "charts": charts,
            "stage_notes": stage_notes,
            "gap": gap,
        }

    async def _run_stage_1_7(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 데모 출력이 있으면 데모 데이터를 사용한다.
        demo_outputs = ctx.session.state.get(_DEMO_OUTPUTS_KEY)
        if isinstance(demo_outputs, dict):
            tables = demo_outputs.get("tables", {})
            charts = demo_outputs.get("charts", [])
            notes = demo_outputs.get("stage_notes", {})
            return {
                "tables": {
                    key: tables.get(key)
                    for key in state._STAGE_TABLE_KEYS.get("1-7", [])
                    if key in tables
                },
                "charts": [
                    chart
                    for chart in charts
                    if chart.get("chart_id")
                    in state._STAGE_CHART_IDS.get("1-7", [])
                ],
                "stage_notes": {"1-7": notes.get("1-7", "")},
                "gap": None,
            }
        # DB 시뮬레이션을 실행한다.
        input_params = schemas.InputParams(**session_state.get("input_params", {}))
        tables, charts, stage_notes, gap = db_production.build_simulation_from_db(
            input_params,
            session_state.get("configs", {}),
            session_state.get("selections", {}),
            session_state.get("user_prefs", {}),
            dirty_stages=["1-7"],
        )
        return {
            "tables": tables,
            "charts": charts,
            "stage_notes": stage_notes,
            "gap": gap,
        }

    async def _run_stage_1_8(
        self, ctx: InvocationContext, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        # 이미 멈췄으면 아무 것도 하지 않는다.
        if ctx.session.state.get(_HALT_KEY):
            return {"tables": {}, "charts": [], "stage_notes": {}, "gap": None}
        # 브리핑용 테이블/차트를 준비한다.
        tables = ctx.session.state.get(_RUN_TABLES_KEY, {})
        charts = ctx.session.state.get(_RUN_CHARTS_KEY, [])
        # LLM 요약본을 만든다.
        llm_tables, llm_charts = state._build_llm_payload(
            tables, charts, session_state.get("configs", {})
        )
        # 브리핑 힌트와 시퀀스를 만든다.
        stage_notes = ctx.session.state.get(_RUN_NOTES_KEY, {})
        dirty_stages = ctx.session.state.get("temp:dirty_stages") or []
        briefing_start = state._pick_briefing_start_stage(dirty_stages)
        briefing_hint = state._build_briefing_hint(briefing_start)
        briefing_sequence = state._build_briefing_sequence(stage_notes, briefing_start)
        briefing_tables, briefing_charts, _ = state._filter_briefing_outputs(
            llm_tables, llm_charts, briefing_start
        )
        payload_obj: dict[str, Any] = {
            "tables": briefing_tables,
            "charts": briefing_charts,
            "stage_sequence": briefing_sequence,
        }
        if briefing_hint:
            payload_obj["briefing_hint"] = briefing_hint
        payload = json.dumps(payload_obj, ensure_ascii=False)
        # 브리핑 텍스트를 생성한다.
        output_text = await _run_llm_text(self.briefing_agent, ctx, message_text=payload)
        output = _parse_json_model(output_text, schemas.StageBriefingOutput)
        # 텍스트를 section별로 매핑한다.
        text_map: dict[str, str] = {}
        for text_block in output.texts:
            text_map[text_block.section] = text_block.value
        # 차트 ID 집합을 만든다.
        chart_id_set = {
            chart.get("chart_id")
            for chart in (briefing_charts or [])
            if isinstance(chart, dict) and chart.get("chart_id")
        }
        # 최종 블록을 조립한다.
        blocks: list[dict[str, Any]] = []
        if "summary" in text_map:
            blocks.append({"type": "text", "section": "summary", "value": text_map["summary"]})
        if briefing_sequence:
            for stage_info in briefing_sequence:
                stage_id = stage_info.get("stage", "")
                if stage_id in text_map:
                    blocks.append({"type": "text", "section": stage_id, "value": text_map[stage_id]})
                for table_key in stage_info.get("table_keys", []):
                    if table_key in (briefing_tables or {}):
                        blocks.append({"type": "table_ref", "table_key": table_key})
                for chart_id in stage_info.get("chart_ids", []):
                    if chart_id in chart_id_set:
                        blocks.append({"type": "chart_ref", "chart_id": chart_id})
        if "conclusion" in text_map:
            blocks.append({"type": "text", "section": "conclusion", "value": text_map["conclusion"]})
        return {"tables": {}, "charts": [], "stage_notes": {}, "gap": None, "blocks": blocks}

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        # 세션 ID를 가져온다.
        session_id = _get_session_id(ctx)
        # 세션 상태를 로드한다.
        session_state = state._get_session_state(session_id, use_db=self.use_db)
        # 앱 상태를 컨텍스트에 저장한다.
        _set_app_state(ctx, session_state)
        # 런타임 상태를 초기화한다.
        _reset_run_state(ctx)
        # 사용자 메시지를 추출한다.
        user_message = _content_to_text(ctx.user_content)
        # 액션 페이로드를 파싱한다.
        action_payload = _parse_action_payload(user_message)
        # 라우팅을 결정한다.
        route = "simulation" if _is_action_payload(action_payload) else "casual"
        if route == "casual":
            # 라우팅 LLM을 실행한다.
            memory_context = state.build_memory_context(session_state, "route")
            instruction = _merge_instructions(ROUTE_AGENT_INSTRUCTIONS, memory_context)
            route_text = await _run_llm_text(self.router_agent, ctx, instruction_override=instruction)
            try:
                route_output = _parse_json_model(route_text, schemas.RouteDecision)
                route = route_output.route
            except Exception:
                route = "casual"
        # 캐주얼 응답을 처리한다.
        if route == "casual":
            memory_context = state.build_memory_context(session_state, "casual")
            instruction = _merge_instructions(CASUAL_AGENT_INSTRUCTIONS, memory_context)
            casual_text = await _run_llm_text(self.casual_agent, ctx, instruction_override=instruction)
            casual_output = _parse_json_model(casual_text, schemas.CasualOutput)
            blocks = [
                {"type": "text", "section": "casual", "value": casual_output.answer}
            ]
            payload = {"route": "casual", "blocks": blocks, "tables": {}, "charts": []}
            # 세션 상태를 저장한다.
            state._save_session_state(session_state, use_db=self.use_db)
            # final 이벤트를 전송한다.
            yield _build_final_event(self.name, payload)
            return
        # 시뮬레이션 루트를 처리한다.
        run_state = _get_run_state(ctx)
        run_state["route"] = "simulation"
        # 보류 액션을 준비한다.
        pending_action = (
            session_state.get("pending_action")
            if isinstance(session_state.get("pending_action"), dict)
            else None
        )
        pending_selection = _extract_pending_selection(action_payload, pending_action)
        # 커맨드를 결정한다.
        command_hint = state._build_command_hint(session_state)
        memory_context = state.build_memory_context(session_state, "command")
        instruction = _merge_instructions(COMMAND_AGENT_INSTRUCTIONS, memory_context)
        if _is_action_payload(action_payload):
            command = schemas.CommandDecision(action="run", target_stage=None)
        else:
            command_text = await _run_llm_text(
                self.command_agent,
                ctx,
                instruction_override=f"{instruction}\n\n{command_hint}",
            )
            try:
                command = _parse_json_model(command_text, schemas.CommandDecision)
            except Exception:
                command = schemas.CommandDecision(action="run", target_stage=None)
        # 리셋 요청을 처리한다.
        if command.action == "reset":
            run_state["action"] = "reset"
            session_state = state._reset_session_state(session_id, use_db=self.use_db)
            _set_app_state(ctx, session_state)
            blocks = [
                {
                    "type": "text",
                    "section": "summary",
                    "value": "시뮬레이션 상태를 초기화했어. 새로 시작해줘.",
                }
            ]
            payload = {"route": "simulation", "blocks": blocks, "tables": {}, "charts": []}
            state._save_session_state(session_state, use_db=self.use_db)
            yield _build_final_event(self.name, payload)
            return
        # explain_stage 요청을 처리한다.
        if command.action == "explain_stage":
            run_state["action"] = "explain_stage"
            # 최근 설명 단계를 보완한다.
            target_stage = command.target_stage or session_state.get("last_explain_stage")
            blocks, tables, charts = await state._build_explain_response(
                session_state, target_stage, user_message
            )
            if target_stage:
                session_state["last_explain_stage"] = target_stage
            # 히스토리를 기록한다.
            session_state["history"].append(
                {
                    "action": "explain_stage",
                    "payload": {"target_stage": target_stage},
                    "at": state._utc_now(),
                }
            )
            # 최종 응답을 만든다.
            payload = _build_final_payload(
                "simulation",
                "explain_stage",
                session_state,
                blocks,
                tables,
                charts,
            )
            state._save_session_state(session_state, use_db=self.use_db)
            yield _build_final_event(self.name, payload)
            return
        # pending 선택을 처리한다.
        if pending_action and pending_action.get("action") == "select_candidates" and not pending_selection:
            tables, _ = state._load_raw_outputs(
                session_state.get("raw_refs", {}).get("stage_outputs_path")
            )
            table_key = pending_action.get("table_key")
            id_field = pending_action.get("id_field", "chip_type_id")
            candidate_rows = tables.get(table_key) if table_key else []
            candidate_ids = _extract_candidate_ids(candidate_rows, id_field)
            selection_payload = {
                "message": user_message,
                "candidate_ids": candidate_ids,
            }
            selection_text = await _run_llm_text(
                self.selection_agent,
                ctx,
                message_text=json.dumps(selection_payload, ensure_ascii=False),
            )
            selection_output = _parse_json_model(selection_text, schemas.SelectionDecision)
            pending_selection = selection_output.selected_ids
            if not pending_selection:
                blocks, tables, charts = _build_pending_repeat_response(
                    session_state, pending_action
                )
                payload = _build_final_payload(
                    "simulation",
                    "run",
                    session_state,
                    blocks,
                    tables,
                    charts,
                )
                state._save_session_state(session_state, use_db=self.use_db)
                yield _build_final_event(self.name, payload)
                return
        # 입력/변경을 파싱한다.
        if pending_selection:
            input_params = schemas.InputParams(**session_state.get("input_params", {}))
            update = schemas.UpdateDecision(
                selections=schemas.UpdateSelections(chip_type_ids=pending_selection)
            )
            session_state["pending_action"] = None
        else:
            input_text = await _run_llm_text(self.input_agent, ctx)
            input_params = _parse_json_model(input_text, schemas.InputParams)
            update_text = await _run_llm_text(self.update_agent, ctx)
            update = _parse_json_model(update_text, schemas.UpdateDecision)
        # 누락 업데이트를 계산한다.
        missing_update = update.missing_fields or []
        had_results = bool(session_state.get("stage_status", {}).get("1-8", {}).get("done"))
        if not had_results:
            missing_update = []
        # 입력/업데이트를 병합하고 dirty 단계를 계산한다.
        merged_params, dirty_stages = state._merge_update_and_collect_dirty(
            session_state,
            input_params,
            update,
            missing_update,
        )
        # 입력값과 업데이트를 저장한다.
        session_state["input_params"] = merged_params.dict()
        state._apply_update_fields(session_state, update)
        state._mark_dirty(session_state, dirty_stages)
        # 누락 업데이트가 있으면 보류 처리한다.
        if missing_update:
            pending_action = state._build_pending_action(missing_update)
            session_state["pending_action"] = pending_action
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
            payload = _build_final_payload(
                "simulation",
                "run",
                session_state,
                blocks,
                {},
                [],
            )
            state._save_session_state(session_state, use_db=self.use_db)
            yield _build_final_event(self.name, payload)
            return
        # 필수 입력 누락을 확인한다.
        missing_inputs = state._get_missing_fields(merged_params)
        if missing_inputs:
            session_state["pending_action"] = {
                "action": "collect_input",
                "target_stage": "1-1",
                "missing_fields": missing_inputs,
                "requested_at": state._utc_now(),
            }
            blocks = [
                {
                    "type": "text",
                    "section": "summary",
                    "value": state._format_missing_summary(missing_inputs),
                }
            ]
            blocks += _build_input_form_blocks(merged_params)
            payload = _build_final_payload(
                "simulation",
                "run",
                session_state,
                blocks,
                {},
                [],
            )
            state._save_session_state(session_state, use_db=self.use_db)
            yield _build_final_event(self.name, payload)
            return
        # dirty 단계가 없으면 전체 실행한다.
        if not dirty_stages:
            dirty_stages = list(state.STAGE_ORDER)
        # 입력 테이블이 없으면 1-1을 추가한다.
        if "input_params_table" not in session_state.get("stage_outputs", {}).get("tables", {}):
            if "1-1" not in dirty_stages:
                dirty_stages = ["1-1"] + dirty_stages
        # 런타임 상태를 저장한다.
        run_state["action"] = "run"
        ctx.session.state["temp:dirty_stages"] = dirty_stages
        # 데모 모드는 데모 출력물을 준비한다.
        if not self.use_db:
            dummy_request = type("Dummy", (), {"demo": True})()
            blocks, tables, charts, stage_notes = demo._build_simulation_stub(
                dummy_request,
                merged_params,
                session_state.get("configs", {}),
                session_state.get("selections", {}),
                session_state.get("user_prefs", {}),
            )
            ctx.session.state[_DEMO_OUTPUTS_KEY] = {
                "blocks": blocks,
                "tables": tables,
                "charts": charts,
                "stage_notes": stage_notes,
            }
        # SequentialAgent를 실행한다.
        async for event in self.simulation_agent.run_async(ctx):
            yield event
        # gap 여부를 확인한다.
        gap = ctx.session.state.get(_GAP_KEY)
        if isinstance(gap, dict):
            gap_context = _build_gap_context(gap)
            gap_text = await _run_llm_text(
                self.gap_agent,
                ctx,
                message_text=json.dumps(gap_context, ensure_ascii=False),
            )
            gap_output = _parse_json_model(gap_text, schemas.GapQuestionOutput)
            pending_action, blocks = _build_gap_pending_payload(gap, gap_output.question)
            session_state["pending_action"] = pending_action
            session_state["last_gap"] = gap
            tables = ctx.session.state.get(_RUN_TABLES_KEY, {})
            charts = ctx.session.state.get(_RUN_CHARTS_KEY, [])
            state._apply_table_highlights(tables, session_state.get("selections", {}))
            payload = _build_final_payload(
                "simulation",
                "run",
                session_state,
                blocks,
                tables,
                charts,
            )
            state._save_session_state(session_state, use_db=self.use_db)
            yield _build_final_event(self.name, payload)
            return
        # 결과를 병합한다.
        tables = ctx.session.state.get(_RUN_TABLES_KEY, {})
        charts = ctx.session.state.get(_RUN_CHARTS_KEY, [])
        stage_notes = ctx.session.state.get(_RUN_NOTES_KEY, {})
        blocks = ctx.session.state.get(_RUN_BLOCKS_KEY, [])
        # 후처리를 수행한다.
        tables, charts = _post_process_run_tables(
            session_state,
            tables,
            charts,
            dirty_stages,
            [],
            None,
        )
        # 상태를 업데이트한다.
        state._update_state(
            session_state,
            merged_params,
            tables,
            charts,
            blocks,
            stage_notes,
            [],
            demo=not self.use_db,
            dirty_stages=dirty_stages,
            pending_action=None,
        )
        state._mark_clean(session_state, dirty_stages)
        payload = _build_final_payload(
            "simulation",
            "run",
            session_state,
            blocks,
            tables,
            charts,
        )
        state._save_session_state(session_state, use_db=self.use_db)
        yield _build_final_event(self.name, payload)


async def _build_explain_answer(context: dict[str, Any]) -> str:
    # 설명 에이전트를 사용해 답변을 생성한다.
    agent = LlmAgent(
        name="ExplainAgent",
        model=MODEL_NAME,
        instruction=EXPLAIN_AGENT_INSTRUCTIONS,
        output_schema=schemas.ExplainOutput,
    )
    payload = json.dumps(context, ensure_ascii=False)
    # 임시 세션 서비스를 만든다.
    session_service = InMemorySessionService()
    # 러너를 만든다.
    runner = Runner(agent=agent, app_name="mlcc_explain", session_service=session_service)
    # 세션을 생성한다.
    try:
        await session_service.create_session(
            app_name="mlcc_explain",
            user_id="explain",
            session_id="explain",
        )
    except Exception:
        # 세션이 이미 있으면 그대로 진행한다.
        pass
    # 입력 콘텐츠를 준비한다.
    content = _text_content(payload)
    # LLM 응답을 수집한다.
    final_text = ""
    async for event in runner.run_async(
        user_id="explain",
        session_id="explain",
        new_message=content,
    ):
        text = _content_to_text(event.content)
        if text:
            final_text = text
    output = _parse_json_model(final_text, schemas.ExplainOutput)
    return output.answer


def _build_final_payload(
    route: str,
    action: str | None,
    session_state: dict[str, Any],
    blocks: list[dict[str, Any]],
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
) -> dict[str, Any]:
    # 진행 로그를 만든다.
    final_logs = state._build_progress_logs(
        route,
        action,
        session_state.get("stage_status"),
    )
    if final_logs:
        blocks = [{"type": "progress_log", "logs": final_logs}] + blocks
    return {"route": route, "blocks": blocks, "tables": tables, "charts": charts}


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
    # 강조 표시를 적용한다.
    state._apply_table_highlights(tables, session_state["selections"])
    return tables, charts


__all__ = ["RootAgent", "_build_explain_answer"]
