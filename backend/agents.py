from agents import Agent

from .config import MODEL_NAME
from .observability import WorkflowRunHooks
from .db_view_profile import build_db_agent_prompt, load_view_profile, normalize_view_profile
from .tools import (
    get_lot_info,
    get_process_data,
    open_simulation_form,
    run_lot_simulation,
    run_simulation,
    run_prediction_simulation,
    filter_lots_by_defect_rate,
    run_design_grid_search,
    get_design_candidates,
    get_defect_rate_stats,
    reset_test_simulation,
    update_simulation_params,
    query_view_table,
)


def _build_agent(**kwargs: object) -> Agent:
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("model", None)
        return Agent(**kwargs)


MODEL_KWARGS = {"model": MODEL_NAME} if MODEL_NAME else {}

auto_message_agent = _build_agent(
    name="\uc790\ub3d9 \uc694\uc57d \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "You compose short Korean assistant messages for UI-triggered events. "
        "Do not mention tools, routing, or system logs. "
        "Write 1-2 sentences. "
        "If missing_fields are provided, ask for those fields in one clear question. "
        "If result is present, summarize recommended_model, representative_lot, "
        "and the count of params, then ask whether to run a prediction simulation."
    ),
    **MODEL_KWARGS,
)


db_agent = _build_agent(
    name="DB \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "You are an internal DB helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "Use query_view_table to query the configured view. "
        "Map user requests to allowed filter columns only. "
        "If a LOT ID is present, prefer that column. "
        "If required info is missing, do not ask the user; report missing fields. "
        "Do not invent LOT IDs or data. Prefer tool calls over direct answers. "
        "Return a compact JSON object with keys: status, summary, missing, next. "
        "status must be ok|missing|error. "
        "missing should be a comma-separated string or 'none'. "
        "next should be a short Korean follow-up question or 'none'."
        "\n\n"
        + build_db_agent_prompt(normalize_view_profile(load_view_profile()))
    ),
    tools=[query_view_table, get_lot_info, get_process_data],
    **MODEL_KWARGS,
)

simulation_agent = _build_agent(
    name="\uc2dc\ubbac\ub808\uc774\uc158 \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "You are an internal recommendation helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "When the user requests an adjacent model recommendation, call open_simulation_form first "
        "to open the UI panel (unless it is already open). "
        "If a LOT ID is provided, call run_lot_simulation(lot_id). "
        "Otherwise collect the five required params: temperature, voltage, "
        "size, capacity, production_mode. "
        "If any params are provided, call update_simulation_params with those "
        "values or with message=original_message to extract. "
        "Do not ask the user directly; report missing fields. "
        "Never ask for values that are already filled. "
        "When all five params are available, call run_simulation. If the user confirms a prediction simulation (including short affirmatives like 네/응/그래/ok/yes) and a recommendation exists, call run_prediction_simulation. "
        "Return a compact JSON object with keys: status, summary, missing, next. "
        "status must be ok|missing|error. "
        "missing should be a comma-separated string or 'none'. "
        "next should be a short Korean follow-up question or 'none'."
    ),
    tools=[
        open_simulation_form,
        update_simulation_params,
        run_simulation,
        run_lot_simulation,
        run_prediction_simulation,
    ],
    **MODEL_KWARGS,
)

test_simulation_agent = _build_agent(
    name="\ud14c\uc2a4\ud2b8 \uc2dc\ubbac\ub808\uc774\uc158 \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "You are an internal optimization helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "Follow the same input collection flow as the simulation agent: "
        "open_simulation_form \u2192 update_simulation_params \u2192 run_simulation (or run_lot_simulation if a LOT ID is provided). "
        "After run_simulation succeeds, call filter_lots_by_defect_rate to filter LOTs by defect rate. "
        "Then call run_design_grid_search to generate 100 design candidates and show the top 10. "
        "If the user asks for other candidates (e.g. \ub2e4\ub978 10\uac1c/\ub098\uba38\uc9c0 90\uac1c), call get_design_candidates with offset/limit or rank. "
        "If the user asks about defect-rate stats, call get_defect_rate_stats. "
        "If the user wants to rerun with new conditions, call reset_test_simulation and then rerun the needed steps. "
        "Do not ask the user directly; report missing fields. "
        "Never ask for values that are already filled. "
        "Return a compact JSON object with keys: status, summary, missing, next. "
        "status must be ok|missing|error. "
        "missing should be a comma-separated string or 'none'. "
        "next should be a short Korean follow-up question or 'none'."
    ),
    tools=[
        open_simulation_form,
        update_simulation_params,
        run_simulation,
        # run_lot_simulation,
        filter_lots_by_defect_rate,
        run_design_grid_search,
        get_design_candidates,
        get_defect_rate_stats,
        reset_test_simulation,
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
    tool_description=("Adjacent model recommendation helper. Input: lot id or params. Output: 4 lines (status/summary/missing/next)."),
    hooks=WorkflowRunHooks(),
)

test_simulation_agent_tool = test_simulation_agent.as_tool(
    tool_name="test_simulation_agent",
    tool_description=("Optimization helper with defect-rate filtering and grid search. Input: lot id or params. Output: 4 lines (status/summary/missing/next)."),
    hooks=WorkflowRunHooks(),
)

triage_agent = _build_agent(
    name="\uc624\ucf00\uc2a4\ud2b8\ub808\uc774\ud130",
    instructions=(
        "You are the conversation lead for a manufacturing monitoring demo. "
        "Always respond in Korean with a warm, friendly tone. "
        "For casual chat (greetings, thanks, small talk, unrelated topics), "
        "reply naturally and briefly, and optionally offer help with LOT lookup or recommendation. "
        "Do not mention internal routing, tools, or intent labels. "
        "If the intent is unclear, ask one short clarifying question before calling tools. "
        "When the user wants process or LOT data, gather the minimum required info with a concise, "
        "professional question (for example LOT ID or key filters like line, status, date range, or conditions). "
        "Tool routing rules (follow exactly): "
        "1) If the user asks for LOT/process lookup or data, call db_agent with the user message. "
        "2) If the user asks for adjacent model recommendation/simulation, call simulation_agent with the user message. "
        "3) If both are requested, call db_agent first, then simulation_agent. "
        "4) If required info is missing, ask one short question for the missing fields, then stop. "
        "5) Do not answer with analysis when a tool should be called. "
        "After tool output, respond naturally in 1-3 sentences; translate the tool report into natural Korean. "
        "If the tool reports missing fields, ask a single clear question only for those fields. "
        "If the recommendation is complete, briefly summarize it and ask whether to run a prediction simulation. "
        "If the user agrees to run the prediction simulation (including short affirmatives like 네/응/그래/ok/yes), "
        "call the simulation_agent tool again with the confirmation so it can run the prediction step. "
        "Do not invent LOT IDs or parameters."
        "\n\n"
        "Examples (add 2 examples here):\n"
        "Example 1:\n"
        "- User: <fill>\n"
        "- Assistant: <fill>\n"
        "Example 2:\n"
        "- User: <fill>\n"
        "- Assistant: <fill>\n"
    ),
    tools=[db_agent_tool, simulation_agent_tool],
    **MODEL_KWARGS,
)
