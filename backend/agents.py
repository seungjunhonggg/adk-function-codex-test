from agents import Agent

from .config import MODEL_NAME
from .observability import WorkflowRunHooks
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

auto_message_agent = _build_agent(
    name="패널 요약 에이전트",
    instructions=(
        "You compose short Korean assistant messages for UI-triggered events. "
        "Do not mention tools, routing, or system logs. "
        "Write 1-2 sentences. "
        "If missing_fields are provided, ask for those fields in one clear question. "
        "If result is present, summarize key metrics like predicted_yield, risk_band, "
        "and estimated_throughput, then suggest one optional next step."
    ),
    **MODEL_KWARGS,
)


db_agent = _build_agent(
    name="DB 에이전트",
    instructions=(
        "You are an internal DB helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "If a LOT ID is present, call get_lot_info(lot_id, limit=12). "
        "If no LOT ID but the user mentions line/status/conditions, call "
        "get_process_data(query=original_message, limit=12). "
        "If required info is missing, do not ask the user; report missing fields. "
        "Do not invent LOT IDs or data. Prefer tool calls over direct answers. "
        "Return exactly four lines: "
        "status: <ok|missing|error>. "
        "summary: <Korean 1 sentence>. "
        "missing: <comma-separated fields or 'none'>. "
        "next: <Korean follow-up question or 'none'>."
    ),
    tools=[get_lot_info, get_process_data],
    **MODEL_KWARGS,
)

simulation_agent = _build_agent(
    name="시뮬레이션 에이전트",
    instructions=(
        "You are an internal simulation helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "When the user requests a simulation, call open_simulation_form first "
        "to open the UI panel (unless it is already open). "
        "If a LOT ID is provided, call run_lot_simulation(lot_id). "
        "Otherwise collect the five required params: temperature, voltage, "
        "size, capacity, production_mode. "
        "If any params are provided, call update_simulation_params with those "
        "values or with message=original_message to extract. "
        "Do not ask the user directly; report missing fields. "
        "Never ask for values that are already filled. "
        "When all five params are available, call run_simulation. "
        "Return exactly four lines: "
        "status: <ok|missing|error>. "
        "summary: <Korean 1 sentence>. "
        "missing: <comma-separated fields or 'none'>. "
        "next: <Korean follow-up question or 'none'>."
    ),
    tools=[
        open_simulation_form,
        update_simulation_params,
        run_simulation,
        run_lot_simulation,
    ],
    **MODEL_KWARGS,
)

db_agent_tool = db_agent.as_tool(
    tool_name="db_agent",
    tool_description=("Process/LOT lookup helper. Input: lot id or line/status query. Output: 4 lines (status/summary/missing/next)."),
    hooks=WorkflowRunHooks(),
)

simulation_agent_tool = simulation_agent.as_tool(
    tool_name="simulation_agent",
    tool_description=("Simulation helper. Input: lot id or params. Output: 4 lines (status/summary/missing/next)."),
    hooks=WorkflowRunHooks(),
)

triage_agent = _build_agent(
    name="오케스트레이터",
    instructions=(
        "You are the conversation lead for a manufacturing monitoring demo. "
        "Always respond in Korean. "
        "Do not mention internal routing, tools, or intent labels. "
        "Decide whether the user wants process/LOT data or a simulation. "
        "If data lookup, call the db_agent tool with the user message. "
        "If simulation/prediction, call the simulation_agent tool. "
        "If both are requested, call db_agent first, then simulation_agent. "
        "After tool output, respond naturally in 1-3 sentences. Do not expose the tool report format; translate it into natural Korean. "
        "If the tool reports missing fields, ask a single clear question for those fields. "
        "If status is ok, summarize the result and suggest one optional next step. "
        "Do not invent LOT IDs or parameters."
    ),
    tools=[db_agent_tool, simulation_agent_tool],
    **MODEL_KWARGS,
)
