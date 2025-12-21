from agents import Agent

from .config import MODEL_NAME
from .tools import get_process_data, run_simulation, update_simulation_params


def _build_agent(**kwargs: object) -> Agent:
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("model", None)
        return Agent(**kwargs)


MODEL_KWARGS = {"model": MODEL_NAME} if MODEL_NAME else {}


db_agent = _build_agent(
    name="DB 에이전트",
    instructions=(
        "당신은 공정 모니터링 데모의 DB 조회 에이전트입니다. "
        "사용자가 공정 데이터를 요청하면 get_process_data를 호출하세요. "
        "라인/상태 같은 필터가 있으면 query로 전달합니다. "
        "도구 호출 후 결과를 간단히 요약하고 UI 패널이 갱신되었다고 알려주세요. "
        "모든 응답은 한국어로 작성합니다."
    ),
    tools=[get_process_data],
    **MODEL_KWARGS,
)

simulation_agent = _build_agent(
    name="시뮬레이션 에이전트",
    instructions=(
        "당신은 예측 시뮬레이션을 담당합니다. "
        "temperature, voltage, size, capacity 네 가지 입력을 수집하세요. "
        "추출 가능한 값은 항상 update_simulation_params로 전달합니다. "
        "누락된 값이 있으면 필요한 항목만 명확히 질문한 뒤 멈추세요. "
        "네 가지가 모두 모이면 run_simulation을 호출하고 간단히 요약한 뒤 "
        "UI 패널이 갱신되었다고 알려주세요. "
        "모든 응답은 한국어로 작성합니다."
    ),
    tools=[update_simulation_params, run_simulation],
    **MODEL_KWARGS,
)

triage_agent = _build_agent(
    name="분류 에이전트",
    instructions=(
        "요청을 올바른 에이전트로 라우팅합니다. "
        "공정 데이터, 최근 기록, 라인 상태 요청이면 DB 에이전트로 핸드오프합니다. "
        "예측, 시뮬레이션, what-if 요청이면 시뮬레이션 에이전트로 핸드오프합니다. "
        "불명확하면 어떤 워크플로우를 원하는지 질문하세요. "
        "모든 응답은 한국어로 작성합니다."
    ),
    handoffs=[db_agent, simulation_agent],
    **MODEL_KWARGS,
)
