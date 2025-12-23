from agents import Agent

from .config import MODEL_NAME
from .tools import (
    get_lot_info,
    get_process_data,
    open_simulation_form,
    run_lot_simulation,
    run_simulation,
    update_simulation_params,
)


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
        "당신은 LOT 모니터링 데모의 DB 조회 에이전트입니다. "
        "사용자가 LOT 정보를 요청하면 get_lot_info를 호출하세요. "
        "LOT ID를 명확히 추출해서 lot_id로 전달합니다. "
        "도구 호출 후 결과를 간단히 요약하고 UI 패널이 갱신되었다고 알려주세요. "
        "모든 응답은 한국어로 작성합니다."
    ),
    tools=[get_lot_info, get_process_data],
    **MODEL_KWARGS,
)

simulation_agent = _build_agent(
    name="시뮬레이션 에이전트",
    instructions=(
        "당신은 LOT 기반 예측 시뮬레이션을 담당합니다. "
        "LOT ID가 주어지면 run_lot_simulation을 호출하세요. "
        "사용자가 시뮬레이션을 원하면 open_simulation_form을 먼저 호출해 UI를 엽니다. "
        "필수 입력은 temperature, voltage, size, capacity, production_mode(양산/개발) 입니다. "
        "사용자가 일부 값을 제공하면 반드시 update_simulation_params로 저장한 뒤 "
        "missing 항목만 다시 질문하세요. "
        "값 추출이 애매하면 update_simulation_params(message=사용자_메시지)를 호출해 "
        "메시지를 그대로 전달하세요. "
        "update_simulation_params를 호출하지 않고 missing을 질문하지 마세요. "
        "다섯 가지 값이 모두 모이면 run_simulation을 호출하고 간단히 요약한 뒤 "
        "UI 패널이 갱신되었다고 알려주세요. "
        "모든 응답은 한국어로 작성합니다."
    ),
    tools=[
        open_simulation_form,
        update_simulation_params,
        run_simulation,
        run_lot_simulation,
    ],
    **MODEL_KWARGS,
)

triage_agent = _build_agent(
    name="분류 에이전트",
    instructions=(
        "요청을 올바른 에이전트로 라우팅합니다. "
        "LOT 정보, LOT 상태 요청이면 DB 에이전트로 핸드오프합니다. "
        "예측, 시뮬레이션, what-if 요청이면 시뮬레이션 에이전트로 핸드오프합니다. "
        "불명확하면 어떤 워크플로우를 원하는지 질문하세요. "
        "모든 응답은 한국어로 작성합니다."
    ),
    handoffs=[db_agent, simulation_agent],
    **MODEL_KWARGS,
)
