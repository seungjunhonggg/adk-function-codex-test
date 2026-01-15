from typing import Literal

from agents import Agent, AgentOutputSchema
from pydantic import BaseModel

from .config import MODEL_NAME
from .guardrails import mlcc_input_guardrail, mlcc_output_guardrail
from .observability import WorkflowRunHooks
from .db_view_profile import build_db_agent_prompt, load_view_profile, normalize_view_profile
from .tools import (
    get_lot_info,
    get_process_data,
    resolve_view_columns,
    open_simulation_form,
    run_lot_simulation,
    run_simulation,
    run_prediction_simulation,
    run_simulation_workflow,
    run_detailed_briefing,
    filter_lots_by_defect_rate,
    run_design_grid_search,
    get_design_candidates,
    get_defect_rate_stats,
    reset_test_simulation,
    show_simulation_stage,
    get_simulation_progress,
    reset_simulation_state,
    update_simulation_params,
    apply_chart_config,
    query_view_metrics,
    query_view_table,
)


class RouteDecision(BaseModel):
    primary_intent: Literal[
        "simulation_edit",
        "simulation_run",
        "db_query",
        "chart_edit",
        "chat",
        "unknown",
    ] = "unknown"
    secondary_intents: list[
        Literal["simulation_edit", "simulation_run", "db_query", "chart_edit"]
    ] = []
    stage: Literal["recommendation", "reference", "grid", "final", "unknown"] = "unknown"
    needs_clarification: bool = False
    clarifying_question: str = ""
    confidence: float = 0.0
    reason: str = ""


class StageResolveDecision(BaseModel):
    stage: Literal["recommendation", "reference", "grid", "final", "unknown"] = "unknown"
    needs_clarification: bool = False
    clarifying_question: str = ""
    confidence: float = 0.0


class ChartDecision(BaseModel):
    chart_type: Literal["auto", "histogram", "bar", "line", "scatter", "area"] = "auto"
    bins: int | None = None
    range_min: float | None = None
    range_max: float | None = None
    normalize: Literal["auto", "count", "percent", "ratio"] = "auto"
    value_unit: Literal["auto", "raw", "percent"] = "auto"
    reset: bool = False
    note: str = ""


class PlannerStep(BaseModel):
    step_id: str
    workflow: Literal[
        "simulation_run",
        "simulation_edit",
        "db_query",
        "chart_edit",
        "chat",
        "briefing",
    ]
    required_memory: list[str] = []
    optional_memory: list[str] = []
    success_criteria: list[str] = []
    status: Literal["pending", "running", "done", "blocked", "failed"] = "pending"
    notes: str = ""


class PlannerDecision(BaseModel):
    goal: str = ""
    steps: list[PlannerStep] = []
    missing_inputs: list[str] = []
    next_action: Literal["ask_user", "run_step", "finalize", "confirm"] = "run_step"
    confirmation_prompt: str = ""
    pending_action: dict | None = None
    open_form_fields: list[str] = []


class EditIntent(BaseModel):
    intent: Literal[
        "none",
        "update_params",
        "update_recommendation_params",
        "update_grid",
        "update_reference",
        "update_selection",
        "show_progress",
        "reset",
        "rerun",
        "new_simulation",
    ] = "none"
    updates: dict[str, float | str] = {}
    clear_fields: list[str] = []
    grid_overrides: dict[str, float] = {}
    reference_lot_id: str | None = None
    selection_overrides: dict[str, int] = {}
    stage: Literal["recommendation", "reference", "grid", "final", "any", "unknown"] = "unknown"
    rerun: bool = False
    needs_clarification: bool = False
    note: str = ""
    confidence: float = 0.0


def _build_agent(**kwargs: object) -> Agent:
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("model", None)
        return Agent(**kwargs)


MODEL_KWARGS = {"model": MODEL_NAME} if MODEL_NAME else {}
MLCC_GLOSSARY_HINTS = (
    "Glossary (KR/EN): "
    "MLCC=적층 세라믹 커패시터/적층세라믹콘덴서; "
    "LOT=로트/lot id; "
    "chip_prod_id=기종/모델/품번/part number; "
    "capacity=용량/정전용량/capacitance; "
    "voltage=전압/정격전압/rated voltage; "
    "size=사이즈/치수/L/W/LxW/length/width; "
    "temperature=온도/소성온도/sintering temp; "
    "sheet_t=시트 두께/green sheet thickness; "
    "laydown=적층/레이다운/stacking; "
    "active_layer=활성층/유효층/active layer; "
    "defect rate=불량률; "
    "grid search=그리드 탐색."
)

auto_message_agent = _build_agent(
    name="\uc790\ub3d9 \uc694\uc57d \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "Always respond in Korean. "
        "You compose short Korean assistant messages for UI-triggered events. "
        "Do not mention tools, routing, or system logs. "
        "Write 1-2 sentences. "
        "If missing_fields are provided, ask for those fields in one clear question. "
        "If result is present, summarize recommended_chip_prod_id, representative_lot, "
        "and the count of params, then ask whether to run a prediction simulation."
    ),
    **MODEL_KWARGS,
)

route_agent = _build_agent(
    name="Route Agent",
    instructions=(
        "Always respond in Korean. "
        "Decide the best intent for the user message. "
        "Return JSON with keys: primary_intent, secondary_intents, needs_clarification, "
        "clarifying_question, confidence, reason. "
        "Use Korean for any natural-language fields (clarifying_question, reason). "
        "Intents: simulation_run, simulation_edit, db_query, chart_edit, chat, unknown. "
        "Use chart_edit for requests to change or redraw charts/graphs/histograms. "
        "Use simulation_edit for edits to existing simulation results, params, grid overrides, "
        "progress check, or reset. "
        "Use simulation_run when the user wants to start a recommendation/simulation run. "
        "If the intent is simulation_run, do not ask for clarification just because "
        "params are missing; route to simulation_run and let the pipeline collect inputs. "
        "You will receive keyword_hints in context. Use them as soft signals: "
        "1) simulation_active + simulation_edit_hits => simulation_edit, "
        "2) simulation_run_hits + inactive => simulation_run, "
        "3) db_keyword_hits + weak simulation hits => db_query. "
        "Use db_query for LOT/process database lookups. "
        "Boundary examples: "
        "\"불량률 히스토그램 bin 20으로 바꿔줘\" => chart_edit. "
        "\"그래프를 선그래프로 바꿔줘\" => chart_edit. "
        "\"분포를 막대그래프로 보여줘\" => chart_edit. "
        "\"차트 범위 0~5%로 맞춰줘\" => chart_edit. "
        "\"시뮬레이션 시작해줘\" => simulation_run. "
        "\"온도만 130으로 바꿔서 다시 계산\" => simulation_edit. "
        "\"이전 조건으로 다시 실행\" => simulation_edit. "
        "\"LOT-123 조회해줘\" => db_query. "
        "\"공정 데이터 좀 보여줘\" => db_query. "
        "\"이 LOT 기반으로 추천해줘\" => simulation_run. "
        "If multiple intents are present, set primary_intent to the most urgent one and "
        "add the rest in secondary_intents. "
        "If ambiguous (e.g., user could mean DB or simulation edit), set needs_clarification=true "
        "and ask one short Korean question in clarifying_question. "
        f"{MLCC_GLOSSARY_HINTS}"
    ),
    output_type=AgentOutputSchema(RouteDecision, strict_json_schema=False),
    **MODEL_KWARGS,
)

stage_resolver_agent = _build_agent(
    name="Stage Resolver Agent",
    instructions=(
        "Always respond in Korean. "
        "Map the user request to a stage UI screen. "
        "Return JSON with keys: stage, needs_clarification, clarifying_question, confidence. "
        "Use Korean for clarifying_question. "
        "stage must be recommendation|reference|grid|final|unknown. "
        "Use available_stages to pick the closest match. "
        "If available_stages has exactly one option and the user is vague, choose it. "
        "If you cannot decide, set stage=unknown and needs_clarification=true."
    ),
    output_type=AgentOutputSchema(StageResolveDecision, strict_json_schema=False),
    **MODEL_KWARGS,
)

chart_agent = _build_agent(
    name="Chart Agent",
    instructions=(
        "Always respond in Korean. "
        "Extract chart update intent from the user message. "
        "Return JSON with keys: chart_type, bins, range_min, range_max, normalize, value_unit, reset, note. "
        "Use Korean for note when provided. "
        "chart_type: histogram|bar|line|scatter|area|auto. "
        "normalize: count|percent|ratio|auto. "
        "value_unit: raw|percent|auto. "
        "Use reset=true for 'reset' or 'default' requests."
    ),
    output_type=AgentOutputSchema(ChartDecision, strict_json_schema=False),
    **MODEL_KWARGS,
)

planner_agent = _build_agent(
    name="Planner Agent",
    instructions=(
        "You are a planning agent. Produce a step-by-step execution plan as JSON only. "
        "Do NOT include chain-of-thought. Notes must be short (<= 20 words). "
        "Use Korean for notes and confirmation_prompt. "
        "Always respond in Korean. "
        "Allowed workflows: simulation_run, simulation_edit, db_query, chart_edit, briefing. "
        "MLCC workflow hints: simulation_run=인접기종 추천/신규 조건 입력, "
        "simulation_edit=기존 조건 수정/재실행/리셋/진행조회, "
        "db_query=LOT/공정 DB 조회(평균/트렌드 포함), "
        "chart_edit=불량률 차트 형식/범위/빈 변경, "
        "briefing=최종 브리핑 요약. "
        "Use required_memory for hard prerequisites, optional_memory for helpful context, "
        "success_criteria for completion keys. "
        "If the user asks for multiple tasks, split them into ordered steps. "
        "If the user explicitly asks for confirmation before execution, set next_action=confirm, "
        "provide confirmation_prompt, and include pending_action with workflow + memory criteria. "
        "If the user intent is simulation_edit and they mention param names without values, "
        "set open_form_fields to the field names that should be edited "
        "(temperature/voltage/size/capacity/production_mode/chip_prod_id). "
        "Use the provided context JSON for available memory keys and session state."
    ),
    output_type=AgentOutputSchema(PlannerDecision, strict_json_schema=False),
    **MODEL_KWARGS,
)

edit_intent_agent = _build_agent(
    name="Edit Intent Agent",
    instructions=(
        "Always respond in Korean. "
        "Parse the user request into a structured edit intent for the simulation pipeline. "
        "Return JSON with keys: intent, updates, clear_fields, grid_overrides, reference_lot_id, "
        "selection_overrides, stage, "
        "rerun, needs_clarification, note, confidence. "
        "Use Korean for note when clarification is needed. "
        "intent values: update_params (temperature/voltage/size/capacity/production_mode/chip_prod_id), "
        "update_recommendation_params (param1..param30), update_grid (sheet_t/laydown/active_layer), "
        "update_reference (reference lot change), update_selection (top_k/max_blocks overrides), "
        "show_progress, reset, rerun, new_simulation, none. "
        "If the user asks to change selection criteria like '상위 5개' or 'TOP 3', "
        "set intent=update_selection and set selection_overrides.top_k. "
        "If the user asks to change the number of briefing blocks, set selection_overrides.max_blocks. "
        "Alias map examples: sheet_t=시트티/시트 T/시트두께/성형두께/성형 두께/sheet thickness; "
        "laydown=레이다운/레이 다운/lay down/도포량/코팅량; "
        "active_layer=액티브/활성층/활성 레이어/active layer; "
        "temperature=온도/공정온도/소성온도/temp; "
        "voltage=전압/인가전압/테스트전압/정격전압/volt; "
        "size=사이즈/크기/치수/외형/L/W/LxW/길이/폭; "
        "capacity=용량/정전용량/cap/capacitance; "
        "chip_prod_id=기종/모델/제품명/타입/품번/part; "
        "production_mode=양산/생산/MP/mass=양산, 개발/샘플/시제/proto/dev=개발; "
        "paramN=파라미터 5/파라미터5/param 5/p5. "
        "If capacity units are specified (uF/nF/pF/F), convert to pF. "
        "If the user asks to remove/exclude/delete a parameter, list it in clear_fields "
        "(e.g., temperature/voltage/size/capacity/production_mode/chip_prod_id or paramN). "
        "Bilingual alias hints: sheet_t=시트 두께/시트T/그린시트 두께/green sheet thickness; "
        "laydown=레이다운/적층/stacking/stack count; "
        "active_layer=활성층/유효층/active layer; "
        "temperature=온도/소성온도/sintering temp/temp; "
        "voltage=전압/정격전압/rated voltage/DC voltage/volt; "
        "size=사이즈/치수/규격/L/W/LxW/length/width; "
        "capacity=용량/정전용량/capacitance/C; "
        "chip_prod_id=기종/모델/제품명/품번/part number; "
        "production_mode=양산/개발/시제/샘플/MP/mass=양산, proto/dev=개발; "
        "paramN=파라미터 5/param5/param 5/p5. "
        "If the user want to simulation or <인접기종 추천 or 특성예측>, set intent=new_simulation. "
        "If the user asks to start a new simulation or adjacent recommendation, set intent=new_simulation. "
        "If the user asks to restart from scratch, set intent=reset. "
        "If the user asks to rerun after changes, set rerun=true. "
        "If you need a clarification, set needs_clarification=true and put a short Korean question in note."
    ),
    output_type=AgentOutputSchema(EditIntent, strict_json_schema=False),
    **MODEL_KWARGS,
)

conversation_agent = _build_agent(
    name="Conversation Agent",
    instructions=(
        "Always respond in Korean. "
        "You are a friendly Korean conversation agent for casual chat. "
        "Be warm and concise. "
        "If the user asks about unrelated topics, reply naturally without routing or tool mentions. "
        "If the user asks for help, suggest that you can assist with simulation or LOT lookup. "
        "If the user mentions MLCC/LOT/기종/품번, keep the response aligned with MLCC support."
    ),
    input_guardrails=[mlcc_input_guardrail],
    output_guardrails=[mlcc_output_guardrail],
    **MODEL_KWARGS,
)

discussion_agent = _build_agent(
    name="Discussion Agent",
    instructions=(
        "Always respond in Korean. "
        "You are a single assistant continuing a conversation about the latest simulation results. "
        "Use ONLY the provided JSON context to answer questions. "
        "Do not invent LOTs, defect rates, or parameters. "
        "If the user asks for a change (e.g., parameter update, reference LOT change, rerun), "
        "respond briefly that you can update it and ask which values to apply. "
        "Keep answers concise and helpful."
    ),
    output_guardrails=[mlcc_output_guardrail],
    **MODEL_KWARGS,
)

briefing_agent = _build_agent(
    name="Briefing Agent",
    instructions=(
        "Always respond in Korean. "
        "You summarize MLCC analysis using ONLY the provided JSON data. "
        "Do not invent LOTs, defect rates, or parameters. "
        "If required data is missing, ask a short Korean question about what is missing. "
        "Keep the response concise (3-6 sentences). "
        "Include chip_prod_id, reference_lot, and defect_rate_overall if present. "
        "If design candidates exist, mention the top-ranked design briefly. "
        "Never mention internal tool names or routing."
    ),
    output_guardrails=[mlcc_output_guardrail],
    **MODEL_KWARGS,
)

briefing_choice_agent = _build_agent(
    name="Briefing Choice Agent",
    instructions=(
        "Always respond in Korean. "
        "Ask the user to choose between a detailed briefing and a brief summary. "
        "Write in Korean, 1-2 sentences. "
        "Always include the words '상세' and '간단'. "
        "You may include numbered options like '1) 상세, 2) 간단'. "
        "Do not mention tools, routing, or system logs."
    ),
    **MODEL_KWARGS,
)



db_agent = _build_agent(
    name="DB \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "You are an internal DB helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "Use query_view_table to query the configured view. "
        "If the user asks for a specific metric/value, call resolve_view_columns first "
        "and pass the chosen columns to query_view_table. "
        "If the user asks for average/min/max/sum/count or trend/graph, "
        "use query_view_metrics with metrics and recent_months/time_bucket. "
        "If column choice is ambiguous, mark it missing and ask one short clarification in next. "
        "Map user requests to allowed filter columns only. "
        "Interpret Korean/English terms for LOT/로트/lot id and process/공정. "
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
    tools=[resolve_view_columns, query_view_metrics, query_view_table, get_lot_info, get_process_data],
    **MODEL_KWARGS,
)

simulation_agent = _build_agent(
    name="\uc2dc\ubbac\ub808\uc774\uc158 \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "You are an internal recommendation helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "Users may mix Korean/English MLCC terms (온도/temperature, 전압/voltage, 용량/capacity, 사이즈/size, 양산/개발, 기종/모델). "
        "When the user requests an adjacent model recommendation, call open_simulation_form first "
        "to open the UI panel (unless it is already open). "
        "If a LOT ID is provided, call run_lot_simulation(lot_id). "
        "Otherwise collect the five required params: temperature, voltage, "
        "size, capacity, production_mode. "
        "If any params are provided, call update_simulation_params with those "
        "values or with message=original_message to extract. "
        "Do not ask the user directly; report missing fields. "
        "Never ask for values that are already filled. "
        "When all five params are available, call run_simulation. "
        "If the user asks to re-display a stage UI (recommendation/reference/grid/final), call show_simulation_stage. "
        "If the user asks for progress/status, call get_simulation_progress. "
        "If the user wants to restart/reset, call reset_simulation_state and then open_simulation_form. "
        "If the user confirms a prediction simulation (including short affirmatives like yes/ok) and a recommendation exists, call run_prediction_simulation. "
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
        show_simulation_stage,
        get_simulation_progress,
        reset_simulation_state,
    ],
    **MODEL_KWARGS,
)

test_simulation_agent = _build_agent(
    name="\ud14c\uc2a4\ud2b8 \uc2dc\ubbac\ub808\uc774\uc158 \uc5d0\uc774\uc804\ud2b8",
    instructions=(
        "You are an internal optimization helper used by the orchestrator. "
        "Do not address the user directly. Always respond in Korean. "
        "Follow the same input collection flow as the simulation agent: "
        "open_simulation_form ? update_simulation_params ? run_simulation (or run_lot_simulation if a LOT ID is provided). "
        "After run_simulation succeeds, call filter_lots_by_defect_rate to filter LOTs by defect rate. "
        "Then call run_design_grid_search to generate 100 design candidates and show the top 10. "
        "If the user asks for other candidates (e.g. ?? 10?/??? 90?), call get_design_candidates with offset/limit or rank. "
        "If the user asks about defect-rate stats, call get_defect_rate_stats. "
        "If the user asks to re-display a stage UI (recommendation/reference/grid/final), call show_simulation_stage. "
        "If the user asks for progress/status, call get_simulation_progress. "
        "If the user wants to restart/reset, call reset_simulation_state and then open_simulation_form. "
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
        show_simulation_stage,
        get_simulation_progress,
        reset_simulation_state,
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

chart_agent_tool = chart_agent.as_tool(
    tool_name="chart_agent",
    tool_description=("Defect-rate chart helper. Input: chart change request. Output: short Korean confirmation."),
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
        "2) If the user asks for adjacent model recommendation/simulation, call simulation_agent with the user message. 2-1) If the user asks to show a previous stage UI, call simulation_agent. "
        "2-2) If the user asks to change a chart/graph/histogram, call chart_agent. "
        "3) If both are requested, call db_agent first, then simulation_agent. "
        "4) If required info is missing, ask one short question for the missing fields, then stop. "
        "5) Do not answer with analysis when a tool should be called. "
        "After tool output, respond naturally in 1-3 sentences; translate the tool report into natural Korean. "
        "If the tool reports missing fields, ask a single clear question only for those fields. "
        "If the recommendation is complete, briefly summarize it and ask whether to run a prediction simulation. "
        "If the user agrees to run the prediction simulation (including short affirmatives like yes/ok), "
        "call the simulation_agent tool again with the confirmation so it can run the prediction step. "
        "Do not invent LOT IDs or parameters."
    ),
    tools=[db_agent_tool, simulation_agent_tool, test_simulation_agent_tool, chart_agent_tool],
    **MODEL_KWARGS,
)


simulation_flow_agent = _build_agent(
    name="시뮬레이션 에이전트",
    instructions=(
        "You are helping MLCC Developer Agent"
        "Always respond in Korean. "
        "You lead the simulation workflow and can ask the user for missing inputs. "
        "Required params: temperature, voltage, size, capacity, production_mode. "
        "Use run_simulation_workflow with the user message to collect params and run the pipeline. "
        "If the user requests a change to selection conditions (e.g., TOP 3, 상위 5개, max_blocks), "
        "call run_simulation_workflow so the grid is rerun with the new selection. "
        "If the tool returns missing fields, ask only for those fields. "
        "If the user requests a detailed briefing, call run_detailed_briefing. "
        "If the user asks to change a chart, call chart_agent first, then apply_chart_config "
        "using the returned fields (chart_type/bins/range_min/range_max/normalize/value_unit/reset). "
        "If the request is unrelated to simulation/chart, handoff to the orchestrator."
    ),
    tools=[
        run_simulation_workflow,
        run_detailed_briefing,
        chart_agent_tool,
        apply_chart_config,
    ],
    handoffs=[],
    **MODEL_KWARGS,
)

orchestrator_agent = _build_agent(
    name="오케스트레이터",
    instructions=(
        "You are helping MLCC Developer Agent"
        "Always respond in Korean. "
        "Handle casual chat naturally and briefly. "
        "If the user wants simulation/recommendation/grid/briefing/chart changes, "
        "handoff to the simulation agent. "
        "Do not mention tools or internal routing."
    ),
    handoffs=[simulation_flow_agent],
    **MODEL_KWARGS,
)

simulation_flow_agent.handoffs = [orchestrator_agent]
