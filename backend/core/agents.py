import json
from typing import Any

from agents import Agent, Runner
import os
from .schemas import (
    BriefingOutput,
    CasualOutput,
    CommandDecision,
    ExplainOutput,
    GapQuestionOutput,
    InputParams,
    RouteDecision,
    SelectionDecision,
    StageBriefingOutput,
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
    "- 칩기종/칩 타입 변경 -> selections.chip_type_ids (리스트)\n"
    "- top-k/순위 변경 -> configs.top_k\n"
    "\n[단계 카탈로그]\n"
    f"{STAGE_CATALOG_TEXT}"
)
# 라우팅 에이전트를 정의한다.
router_agent = Agent(
    name="RouteAgent",
    instructions=(
        "사용자 메시지를 보고 route를 분류해.\n"
        "- simulation: 칩 설계/시뮬레이션/추천을 요청하는 경우, 산출된 산출물(REF LOT/ 칩기종/ 차트/ 불량률/ 선정기준) 에 대한 근거 자료를 요청하는경우, 시뮬레이션 리셋/처음부터 다시 요청\n"
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
        "\"온도 25, 전압 6, 용량 10uF로 시뮬레이션 해줘\"\n"
        "=> action: run, target_stage: null\n"
        "\n"
        "[STATE_HINT]\n"
        "- has_results: true\n"
        "- has_input_complete: true\n"
        "- pending_action: none\n"
        "- last_action: update_state\n"
        "[사용자 메시지]\n"
        "\"레퍼런스 LOT 바꿔줘\"\n"
        "=> action: run, target_stage: null\n"
        "\n"
        "[STATE_HINT]\n"
        "- has_results: true\n"
        "- has_input_complete: true\n"
        "- pending_action: none\n"
        "- last_action: update_state\n"
        "[사용자 메시지]\n"
        "\"1-6 단계 근거 설명해줘\"\n"
        "=> action: explain_stage, target_stage: 1-6\n"
    ) + COMMAND_STAGE_HINT,
    output_type=CommandDecision,
    **MODEL_KWARGS,
)


# 입력 파싱 에이전트를 정의한다.
input_agent = Agent(
    name="InputAgent",
    instructions = (
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
),
    output_type=InputParams,
    **MODEL_KWARGS,
)


# 변경 요청 파싱 에이전트를 정의한다.
update_agent = Agent(
    name="UpdateAgent",
    instructions=(
        "사용자 메시지에서 변경 요청을 추출해.\n"
        "- input_params: temperature, voltage, size, capacity, chip_prod_id\n"
        "- selections: chip_type_ids(리스트), reference_lot_id\n"
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


gap_agent = Agent(
    name="GapAgent",
    instructions=(
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
    ),
    output_type=GapQuestionOutput,
    **MODEL_KWARGS,
)


# 선택 에이전트를 정의한다.
selection_agent = Agent(
    name="SelectionAgent",
    instructions=(
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
    ),
    output_type=SelectionDecision,
    **MODEL_KWARGS,
)


# 브리핑 에이전트를 정의한다 (text만 생성, ref는 코드에서 삽입).
briefing_agent = Agent(
    name="BriefingAgent",
    instructions=(
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
    ),
    output_type=StageBriefingOutput,
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
    # LLM으로 브리핑 텍스트만 생성한다.
    result = await Runner.run(briefing_agent, payload)
    # 텍스트를 section별로 매핑한다.
    text_map: dict[str, str] = {}
    for text_block in result.final_output.texts:
        text_map[text_block.section] = text_block.value
    # 차트 ID 집합을 만든다.
    chart_id_set = {
        chart.get("chart_id")
        for chart in (charts or [])
        if isinstance(chart, dict) and chart.get("chart_id")
    }
    # 최종 블록을 조립한다: text + ref를 stage_sequence 순서대로 배치.
    blocks: list[dict[str, Any]] = []
    # summary가 있으면 먼저 추가한다.
    if "summary" in text_map:
        blocks.append({"type": "text", "section": "summary", "value": text_map["summary"]})
    # stage_sequence에 따라 text와 ref를 배치한다.
    if stage_sequence:
        for stage_info in stage_sequence:
            stage = stage_info.get("stage", "")
            # 해당 단계의 text가 있으면 추가한다.
            if stage in text_map:
                blocks.append({"type": "text", "section": stage, "value": text_map[stage]})
            # 해당 단계의 table_ref를 추가한다 (실제 존재하는 것만).
            for table_key in stage_info.get("table_keys", []):
                if table_key in (tables or {}):
                    blocks.append({"type": "table_ref", "table_key": table_key})
            # 해당 단계의 chart_ref를 추가한다 (실제 존재하는 것만).
            for chart_id in stage_info.get("chart_ids", []):
                if chart_id in chart_id_set:
                    blocks.append({"type": "chart_ref", "chart_id": chart_id})
    # conclusion이 있으면 마지막에 추가한다.
    if "conclusion" in text_map:
        blocks.append({"type": "text", "section": "conclusion", "value": text_map["conclusion"]})
    return blocks


async def _build_explain_answer(context: dict[str, Any]) -> str:
    # 설명용 컨텍스트를 직렬화한다.
    payload = json.dumps(context, ensure_ascii=False)
    # LLM으로 설명을 생성한다.
    result = await Runner.run(explain_agent, payload)
    return result.final_output.answer


async def _build_gap_question(context: dict[str, Any]) -> str:
    # 데이터 공백 질문 컨텍스트를 직렬화한다.
    payload = json.dumps(context, ensure_ascii=False)
    # LLM으로 확인 질문을 생성한다.
    result = await Runner.run(gap_agent, payload)
    return result.final_output.question


async def _select_candidates_with_llm(
    message: str,
    candidate_ids: list[str],
) -> list[str]:
    # 선택용 컨텍스트를 만든다.
    payload = json.dumps(
        {"message": message, "candidate_ids": candidate_ids},
        ensure_ascii=False,
    )
    # LLM으로 선택을 추출한다.
    result = await Runner.run(selection_agent, payload)
    # 선택 결과를 반환한다.
    return result.final_output.selected_ids


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
