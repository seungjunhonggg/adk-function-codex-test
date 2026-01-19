import json
from typing import Any

from agents import Agent, Runner
import os
from .schemas import (
    BriefingOutput,
    CasualOutput,
    CommandDecision,
    ExplainOutput,
    InputParams,
    RouteDecision,
    UpdateDecision,
)
from agents import ModelSettings
from dotenv import load_dotenv

load_dotenv()
model_setting = ModelSettings(temperature=0.1)
MODEL_NAME = "gpt-5-mini"
MODEL_KWARGS = {"model": MODEL_NAME} if MODEL_NAME else {}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
        "keywords": ["칩기종","후보", "매칭", "유사"],
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
    # 단계 카탈로그를 텍스트로 만든다.
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
    "- 칩기종/칩 타입 변경 -> selections.chip_type_id\n"
    "- top-k/순위 변경 -> configs.top_k\n"
    "\n[단계 카탈로그]\n"
    f"{STAGE_CATALOG_TEXT}"
)
# 라우팅 에이전트를 정의한다.
router_agent = Agent(
    name="RouteAgent",
    instructions=(
        "사용자 메시지를 보고 route를 분류해.\n"
        "- simulation: 칩 설계/시뮬레이션/추천을 요청하는 경우, 산출된 산출물(REF LOT/ 칩기종/ 차트/ 불량률/ 선정기준) 에 대한 근거 자료를 요청하는경우\n"
        "- casual: 그 외 일반 대화\n"
        "반드시 route만 출력해."
    ),
    output_type=RouteDecision,
    **MODEL_KWARGS,
)


# 커맨드 에이전트를 정의한다.
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
        "ex) 테스트 데이터로 시뮬레이션 해줘 -> run"
    ) + COMMAND_STAGE_HINT,
    output_type=CommandDecision,
    **MODEL_KWARGS,
)


# 입력 파싱 에이전트를 정의한다.
input_agent = Agent(
    name="InputAgent",
    instructions=(
        "메시지에서 다음 필드만 추출해: temperature, voltage, size, capacity, dev_flag, powder_size, chip_type.\n"
        "없으면 null. 추측 금지. 위 필드만 출력."
        "사용자가 예시/테스트 데이터를 원할경우, temperature : 55, voltage : 6, size : 1005, capacity : 10000000, dev_Flag : 양산, powder_sie : 100 으로 반환한다."
    ),
    output_type=InputParams,
    **MODEL_KWARGS,
)


# 변경 요청 파싱 에이전트를 정의한다.
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
    ) + UPDATE_STAGE_HINT,
    output_type=UpdateDecision,
    **MODEL_KWARGS,
)


# 설명 에이전트를 정의한다.
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
    **MODEL_KWARGS,
)


# 브리핑 에이전트를 정의한다.
briefing_agent = Agent(
    name="BriefingAgent",
    instructions=(
        "아래 표/차트 데이터를 보고 브리핑 블록을 생성해.\n"
        "- 출력 형식: blocks 배열만\n"
        "- block.type은 text|table_ref|chart_ref만 사용\n"
        "- text는 한국어로 친절한 문장으로 작성.\n"
        "- 표/차트 값만 인용\n"
        "- briefing_hint가 있으면 첫 문장에 반영\n"
        "- stage_sequence가 있으면 그 순서대로 작성\n"
        "- stage_sequence.note(근거)만 사용하며, 근거로 사용하였다는 말을 직접적으로 언급하지 않음.\n"
        "- 각 stage는 text 1개 + 관련 table_ref/chart_ref를 바로 배치\n"
        "- 전달된 tables/charts 안에서만 block을 만든다\n"
        "- 테이블에서 __로 시작하는 메타 필드는 무시\n"
        "- children 지표는 언급하지 않음\n"
        "- 길이 목표: 500~1k 토큰\n"
        "필수 table_key: input_params_table, chip_type_candidates_table, "
        "reference_lot_candidates_table, top_k_table, "
        "recent_similar_table, defect_rate_table\n"
        "필수 chart_id: defect_rate_summary\n"
    ),
    output_type=BriefingOutput,
    **MODEL_KWARGS,
)


# 캐주얼 응답 에이전트를 정의한다.
casual_agent = Agent(
    name="CasualAgent",
    instructions=(
        "사용자 메시지에 대해 캐주얼하게 한국어로 답변해.\n"
        "- 2~4문장으로 짧게\n"
        "- 필요하면 질문 1개만\n"
        "- 시뮬레이션이 필요해 보이면 '시뮬레이션'이라고 말해 달라고 안내\n"
        "answer만 출력해."
    ),
    output_type=CasualOutput,
    **MODEL_KWARGS,
)



async def _route_with_llm(session, message: str) -> str:
    # LLM으로 라우팅을 결정한다.
    result = await Runner.run(router_agent, message, session=session)
    decision = result.final_output
    return decision.route


async def _decide_command_with_llm(session, message: str) -> CommandDecision:
    # LLM으로 커맨드 액션을 결정한다.
    result = await Runner.run(command_agent, message, session=session)
    return result.final_output


async def _parse_input_with_llm(message: str) -> InputParams:
    # LLM으로 입력값을 추출한다.
    print(message)
    result = await Runner.run(input_agent, message)
    return result.final_output


async def _parse_update_with_llm(message: str) -> UpdateDecision:
    # LLM으로 변경 요청을 추출한다.
    result = await Runner.run(update_agent, message)
    return result.final_output


async def _build_briefing_blocks(
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    briefing_hint: str | None = None,
    stage_sequence: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # 브리핑 입력을 만든다.
    payload_obj: dict[str, Any] = {"tables": tables, "charts": charts}
    if briefing_hint:
        payload_obj["briefing_hint"] = briefing_hint
    if stage_sequence:
        payload_obj["stage_sequence"] = stage_sequence
    payload = json.dumps(payload_obj, ensure_ascii=False)
    # LLM으로 브리핑 블록을 만든다.
    result = await Runner.run(briefing_agent, payload)
    # Pydantic 객체를 dict로 변환한다.
    return [block.dict() for block in result.final_output.blocks]


async def _build_explain_answer(context: dict[str, Any]) -> str:
    # 설명용 컨텍스트를 직렬화한다.
    payload = json.dumps(context, ensure_ascii=False)
    # LLM으로 설명을 생성한다.
    result = await Runner.run(explain_agent, payload)
    return result.final_output.answer


async def _build_casual_blocks(session, message: str) -> list[dict[str, Any]]:
    # 캐주얼 응답을 생성한다.
    result = await Runner.run(casual_agent, message, session=session)
    # 응답 블록을 구성한다.
    return [
        {
            "type": "text",
            "section": "casual",
            "value": result.final_output.answer,
        }
    ]
