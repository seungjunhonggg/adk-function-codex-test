from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents import Agent, Runner, function_tool

from .agents import db_agent, simulation_agent, test_simulation_agent, triage_agent
from .config import MODEL_NAME, WORKFLOW_PATH, WORKFLOW_STORE_PATH
from .context import current_session_id
from .db_connections import execute_table_query
from .observability import WorkflowRunHooks, emit_workflow_log
from .simulation import extract_simulation_params, recommendation_store
from .tools import (
    collect_simulation_params,
    emit_frontend_trigger,
    emit_simulation_form,
    emit_process_data,
    execute_simulation,
    file_search,
    frontend_trigger,
    get_process_data,
    open_simulation_form,
    run_prediction_simulation,
    run_lot_simulation,
    run_simulation,
    update_simulation_params,
)


ALLOWED_TYPES = {
    "start",
    "agent",
    "guardrail",
    "mcp",
    "function_tool",
    "file_search",
    "end",
    "note",
    "if_else",
    "while",
    "user_approval",
    "transform",
    "state",
}
LEGACY_TYPE_MAP = {"user": "start", "function": "mcp"}
LEGACY_TOOLSET_MAP = {
    "db_function": "process_db",
    "api_function": "simulation",
    "frontend_trigger": "frontend_trigger",
}
ALLOWED_AGENT_PROFILES = {
    "custom",
    "orchestrator",
    "db_agent",
    "simulation_agent",
    "test_simulation_agent",
}
ALLOWED_AGENT_EXECUTION = {"handoff", "as_tool"}
TOOL_NODE_TYPES = {"mcp", "file_search", "function_tool"}
PORT_RULES = {
    "start": {"out": ["out"]},
    "agent": {"out": ["out"]},
    "guardrail": {"out": ["pass", "fail"]},
    "if_else": {"out": ["true", "false"]},
    "while": {"out": ["loop", "done"]},
    "user_approval": {"out": ["approved", "rejected"]},
    "mcp": {"out": ["out"]},
    "function_tool": {"out": ["out"]},
    "file_search": {"out": ["out"]},
    "transform": {"out": ["out"]},
    "state": {"out": ["out"]},
    "end": {"out": []},
    "note": {"out": []},
}


@dataclass
class WorkflowRun:
    assistant_message: str
    used_llm: bool
    path: list[str]


@dataclass
class WorkflowContext:
    input_as_text: str
    state: dict[str, Any]
    last_output: str | None = None
    used_llm: bool = False
    tool_messages: list[str] | None = None


@dataclass
class TraversalResult:
    path: list[str]
    context: WorkflowContext
    end_node_id: str | None = None


def ensure_workflow() -> dict:
    path = Path(WORKFLOW_PATH)
    if not path.exists():
        save_workflow(default_workflow())
    return load_workflow()


def load_workflow() -> dict:
    path = Path(WORKFLOW_PATH)
    if not path.exists():
        return normalize_workflow(default_workflow())
    with path.open("r", encoding="utf-8") as handle:
        try:
            return normalize_workflow(json.load(handle))
        except json.JSONDecodeError:
            return normalize_workflow(default_workflow())


def save_workflow(workflow: dict) -> dict:
    normalized = normalize_workflow(workflow)
    normalized.setdefault("meta", {})
    normalized["meta"]["updated_at"] = _now_iso()
    path = Path(WORKFLOW_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2)
    return normalized


def ensure_workflow_store() -> dict:
    path = Path(WORKFLOW_STORE_PATH)
    if not path.exists():
        store = {"active_id": None, "items": []}
        return save_workflow_store(store)
    return load_workflow_store()


def load_workflow_store() -> dict:
    path = Path(WORKFLOW_STORE_PATH)
    if not path.exists():
        return {"active_id": None, "items": []}
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return {"active_id": None, "items": []}
    if not isinstance(data, dict):
        return {"active_id": None, "items": []}
    items = data.get("items")
    if not isinstance(items, list):
        items = []
    return {"active_id": data.get("active_id"), "items": items}


def save_workflow_store(store: dict) -> dict:
    path = Path(WORKFLOW_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=2)
    return store


def list_saved_workflows() -> dict:
    store = load_workflow_store()
    return {
        "active_id": store.get("active_id"),
        "workflows": [_workflow_summary(item) for item in store.get("items", [])],
    }


def save_workflow_entry(workflow: dict) -> dict:
    normalized = normalize_workflow(workflow)
    normalized.setdefault("meta", {})
    name = normalized["meta"].get("name") or "워크플로우"
    normalized["meta"]["name"] = name
    normalized["meta"]["updated_at"] = _now_iso()

    store = load_workflow_store()
    items = store.get("items", [])
    existing = next(
        (item for item in items if (item.get("meta") or {}).get("name") == name),
        None,
    )
    if existing:
        workflow_id = existing.get("id") or _new_workflow_id(name)
        existing["id"] = workflow_id
        existing["workflow"] = normalized
        existing["meta"] = normalized["meta"]
    else:
        workflow_id = _new_workflow_id(name)
        items.insert(
            0,
            {"id": workflow_id, "meta": normalized["meta"], "workflow": normalized},
        )

    saved_active = save_workflow(normalized)
    store["items"] = items
    store["active_id"] = workflow_id
    _sync_store_item(items, workflow_id, saved_active)
    save_workflow_store(store)
    return {
        "active_id": store.get("active_id"),
        "workflow": saved_active,
        "item": _workflow_summary(_find_store_item(items, workflow_id)),
        "workflows": [_workflow_summary(item) for item in items],
    }


def apply_saved_workflow(workflow_id: str) -> dict | None:
    store = load_workflow_store()
    item = _find_store_item(store.get("items", []), workflow_id)
    if not item:
        return None
    saved_active = save_workflow(item.get("workflow") or {})
    store["active_id"] = workflow_id
    save_workflow_store(store)
    return {"active_id": workflow_id, "workflow": saved_active, "item": item}


def delete_saved_workflow(workflow_id: str) -> bool:
    store = load_workflow_store()
    items = store.get("items", [])
    filtered = [item for item in items if item.get("id") != workflow_id]
    if len(filtered) == len(items):
        return False
    store["items"] = filtered
    if store.get("active_id") == workflow_id:
        store["active_id"] = None
    save_workflow_store(store)
    return True


def validate_workflow(workflow: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    nodes = workflow.get("nodes")
    edges = workflow.get("edges")

    if not isinstance(nodes, list) or not nodes:
        errors.append("nodes가 비어있습니다.")
        return False, errors

    if not isinstance(edges, list):
        errors.append("edges가 올바르지 않습니다.")
        edges = []

    node_ids = []
    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        if not node_id:
            errors.append("id가 없는 노드가 있습니다.")
        else:
            node_ids.append(node_id)
        if node_type not in ALLOWED_TYPES:
            errors.append(f"허용되지 않는 type: {node_type}")

    if len(node_ids) != len(set(node_ids)):
        errors.append("노드 id가 중복되었습니다.")

    node_id_set = set(node_ids)
    for edge in edges:
        if edge.get("from") not in node_id_set or edge.get("to") not in node_id_set:
            errors.append("존재하지 않는 노드로 연결된 edge가 있습니다.")

    node_map = {node.get("id"): node for node in nodes if node.get("id")}
    incoming = _build_incoming(edges)
    outgoing = _build_outgoing(edges)

    start_nodes = [node for node in nodes if node.get("type") == "start"]
    if not start_nodes:
        errors.append("Start 노드가 필요합니다.")
    if len(start_nodes) > 1:
        errors.append("Start 노드는 1개만 허용됩니다.")
    for start_node in start_nodes:
        start_id = start_node.get("id")
        if start_id and incoming.get(start_id):
            errors.append("Start 노드에는 입력 연결을 할 수 없습니다.")

    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        if not node_id:
            continue

        if node_type == "agent":
            execution_mode = node.get("execution_mode") or "handoff"
            if execution_mode not in ALLOWED_AGENT_EXECUTION:
                errors.append(f"허용되지 않는 execution_mode: {execution_mode}")
            if execution_mode == "as_tool" and not _has_agent_ancestor(
                node_id, node_map, incoming
            ):
                errors.append("as_tool 에이전트는 상위 에이전트에 연결해야 합니다.")
            profile = (node.get("config") or {}).get("agent_profile")
            if profile and profile not in ALLOWED_AGENT_PROFILES:
                errors.append(f"허용되지 않는 agent profile: {profile}")

        if node_type in TOOL_NODE_TYPES:
            if not _has_agent_ancestor(node_id, node_map, incoming):
                errors.append("도구 노드는 에이전트 뒤에 연결해야 합니다.")

        if node_type == "end" and outgoing.get(node_id):
            errors.append("End 노드에는 출력 연결을 할 수 없습니다.")

        allowed_ports = PORT_RULES.get(node_type, {}).get("out", [])
        if outgoing.get(node_id) and not allowed_ports:
            errors.append(f"{node_type} 노드는 출력 포트가 없습니다.")
        if allowed_ports:
            used_ports = set()
            for edge in outgoing.get(node_id, []):
                from_port = edge.get("from_port") or "out"
                used_ports.add(from_port)
                if from_port not in allowed_ports:
                    errors.append(
                        f"{node.get('label') or node_id} 노드의 포트가 올바르지 않습니다: {from_port}"
                    )
            if node_type in {"guardrail", "if_else", "while", "user_approval"}:
                missing = set(allowed_ports) - used_ports
                if missing:
                    missing_text = ", ".join(sorted(missing))
                    errors.append(
                        f"{node.get('label') or node_id} 노드에 누락된 분기 포트가 있습니다: {missing_text}"
                    )

    return not errors, errors


def normalize_workflow(workflow: dict) -> dict:
    meta = workflow.get("meta") or {}
    nodes = workflow.get("nodes") or []
    edges = workflow.get("edges") or []

    normalized_nodes = []
    for node in nodes:
        raw_type = node.get("type")
        config = node.get("config") or {}
        subtype = node.get("subtype") or ""
        node_type = _normalize_node_type(raw_type)

        if raw_type == "function":
            toolset = LEGACY_TOOLSET_MAP.get(subtype)
            if toolset:
                config.setdefault("toolset", toolset)
                subtype = toolset

        if node_type == "agent":
            profile = config.get("agent_profile") or subtype or "custom"
            if profile in ALLOWED_AGENT_PROFILES:
                config.setdefault("agent_profile", profile)
                subtype = profile
            else:
                subtype = "custom"
            execution_mode = node.get("execution_mode") or "handoff"
        else:
            execution_mode = ""

        if node_type == "mcp":
            toolset = config.get("toolset") or subtype
            if toolset:
                config["toolset"] = toolset
                subtype = toolset
        if node_type == "function_tool":
            tool_type = config.get("tool_type") or subtype or "db_query"
            config["tool_type"] = tool_type
            subtype = tool_type
        if node_type == "start" and not subtype:
            subtype = "start"
        if node_type == "file_search" and not subtype:
            subtype = "file_search"
        normalized_nodes.append(
            {
                "id": node.get("id"),
                "type": node_type,
                "subtype": subtype,
                "execution_mode": execution_mode,
                "label": node.get("label") or node_type,
                "keywords": _normalize_keywords(node.get("keywords")),
                "input_format": node.get("input_format") or "",
                "output_format": node.get("output_format") or "",
                "position": node.get("position") or {"x": 0, "y": 0},
                "config": config,
            }
        )

    node_type_by_id = {
        node["id"]: node["type"] for node in normalized_nodes if node.get("id")
    }
    normalized_edges = []
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        from_port = edge.get("from_port") or "out"
        to_port = edge.get("to_port") or "in"
        kind = edge.get("kind") or _infer_edge_kind(
            node_type_by_id.get(src), node_type_by_id.get(dst)
        )
        normalized_edges.append(
            {
                "from": src,
                "to": dst,
                "from_port": from_port,
                "to_port": to_port,
                "kind": kind,
            }
        )

    return {
        "meta": {
            "name": meta.get("name") or "기본 워크플로우",
            "description": meta.get("description") or "",
            "updated_at": meta.get("updated_at") or _now_iso(),
        },
        "nodes": normalized_nodes,
        "edges": normalized_edges,
    }


async def execute_workflow(
    workflow: dict,
    message: str,
    session_id: str,
    session: Any,
    allow_llm: bool,
) -> WorkflowRun | None:
    normalized = normalize_workflow(workflow)
    valid, _ = validate_workflow(normalized)
    if not valid:
        return None

    node_map = {node["id"]: node for node in normalized["nodes"] if node.get("id")}
    edges = normalized["edges"]
    flow_edges = _build_outgoing([edge for edge in edges if edge.get("kind") == "flow"])
    tool_adjacency = _build_adjacency(edges, kind="tool")
    handoff_adjacency = _build_adjacency(edges, kind="handoff")

    traversal = await _traverse_workflow(
        node_map,
        flow_edges,
        tool_adjacency,
        handoff_adjacency,
        message,
        session,
        allow_llm,
        simulate=False,
    )
    if traversal is None:
        return None

    assistant_message = _format_final_output(traversal, node_map)
    return WorkflowRun(assistant_message, traversal.context.used_llm, traversal.path)


def preview_workflow(workflow: dict, message: str) -> dict | None:
    normalized = normalize_workflow(workflow)
    valid, _ = validate_workflow(normalized)
    if not valid:
        return None

    node_map = {node["id"]: node for node in normalized["nodes"] if node.get("id")}
    edges = normalized["edges"]
    flow_edges = _build_outgoing([edge for edge in edges if edge.get("kind") == "flow"])
    tool_adjacency = _build_adjacency(edges, kind="tool")
    handoff_adjacency = _build_adjacency(edges, kind="handoff")

    traversal = _run_preview(
        node_map,
        flow_edges,
        tool_adjacency,
        handoff_adjacency,
        message,
    )
    if traversal is None:
        return None

    return {
        "selected_id": traversal["selected_id"],
        "selected_node": _node_summary(node_map.get(traversal["selected_id"], {})),
        "path": traversal["path"],
        "path_nodes": [
            _node_summary(node_map[node_id])
            for node_id in traversal["path"]
            if node_id in node_map
        ],
    }


def default_workflow() -> dict:
    now = _now_iso()
    return {
        "meta": {
            "name": "OpenAI Agent Builder 스타일 워크플로우",
            "description": "Start → Guardrail → Agent → End, MCP/File Search는 에이전트 도구로 연결",
            "updated_at": now,
        },
        "nodes": [
            {
                "id": "node-start",
                "type": "start",
                "subtype": "start",
                "label": "Start",
                "keywords": [],
                "input_format": "",
                "output_format": "",
                "position": {"x": 80, "y": 160},
                "config": {
                    "input_variables": ["input_as_text"],
                    "state_variables": ["conversation_history:text"],
                },
            },
            {
                "id": "node-guardrail",
                "type": "guardrail",
                "subtype": "guardrail",
                "label": "Guardrail",
                "keywords": [],
                "input_format": "",
                "output_format": "",
                "position": {"x": 280, "y": 160},
                "config": {
                    "input": "input_as_text",
                    "pii": True,
                    "moderation": True,
                    "jailbreak": True,
                },
            },
            {
                "id": "node-orchestrator",
                "type": "agent",
                "subtype": "orchestrator",
                "execution_mode": "handoff",
                "label": "Orchestrator Agent",
                "keywords": [
                    "lot",
                    "로트",
                    "line",
                    "라인",
                    "status",
                    "상태",
                    "data",
                    "데이터",
                    "조회",
                    "예측",
                    "시뮬레이션",
                    "simulation",
                    "forecast",
                    "what-if",
                    "risk",
                    "수율",
                ],
                "input_format": "user message (Korean)",
                "output_format": "routing decision + tool/agent call",
                "position": {"x": 520, "y": 160},
                "config": {
                    "agent_profile": "orchestrator",
                    "instructions": (
                        "You are the conversation lead for a manufacturing monitoring demo. Always respond in Korean with a warm, friendly tone. For casual chat (greetings, thanks, small talk, unrelated topics), reply naturally and briefly, and optionally offer help with LOT lookup or recommendation. Do not mention internal routing, tools, or intent labels. If the intent is unclear, ask one short clarifying question before calling tools. When the user wants process or LOT data, gather the minimum required info with a concise, professional question (for example LOT ID or key filters like line, status, date range, or conditions). If you have enough info, call the db_agent tool with the user message. When the user wants an adjacent model recommendation, call the simulation_agent tool with the user message to open the UI and extract any params. If both are requested, call db_agent first, then simulation_agent. After tool output, respond naturally in 1-3 sentences; translate the tool report into natural Korean. If the tool reports missing fields, ask a single clear question only for those fields. If the recommendation is complete, briefly summarize it and ask whether to run a prediction simulation. If the user agrees to run the prediction simulation (including short affirmatives like 네/응/그래/ok/yes), call the simulation_agent tool again with the confirmation so it can run the prediction step. Do not invent LOT IDs or parameters."
                    ),
                    "include_chat_history": True,
                    "output_format": "text",
                },
            },
            {
                "id": "node-end",
                "type": "end",
                "subtype": "end",
                "label": "End",
                "keywords": [],
                "input_format": "",
                "output_format": "",
                "position": {"x": 760, "y": 160},
                "config": {"return_type": "text"},
            },
            {
                "id": "node-db-agent",
                "type": "agent",
                "subtype": "db_agent",
                "execution_mode": "as_tool",
                "label": "DB Agent",
                "keywords": [
                    "lot",
                    "로트",
                    "line",
                    "라인",
                    "status",
                    "상태",
                    "data",
                    "데이터",
                    "조회",
                    "process",
                    "공정",
                ],
                "input_format": "lot_id or line/status query",
                "output_format": "process rows + UI update",
                "position": {"x": 520, "y": 20},
                "config": {
                    "agent_profile": "db_agent",
                    "instructions": (
                                                "You are an internal DB helper used by the orchestrator. "
                        "Do not address the user directly. Always respond in Korean. "
                        "If a LOT ID is present, call get_lot_info(lot_id, limit=12). "
                        "If no LOT ID but the user mentions line/status/conditions, call "
                        "get_process_data(query=original_message, limit=12). "
                        "If required info is missing, do not ask the user; report missing fields. "
                        "Do not invent LOT IDs or data. Prefer tool calls over direct answers. "
                        "Return a compact JSON object with keys: status, summary, missing, next. "
                        "status must be ok|missing|error. "
                        "missing should be a comma-separated string or 'none'. "
                        "next should be a short Korean follow-up question or 'none'."

                    ),
                },
            },
            {
                "id": "node-sim-agent",
                "type": "agent",
                "subtype": "simulation_agent",
                "execution_mode": "as_tool",
                "label": "Simulation Agent",
                "keywords": [
                    "예측",
                    "시뮬레이션",
                    "simulation",
                    "forecast",
                    "what-if",
                    "risk",
                    "리스크",
                    "수율",
                    "throughput",
                ],
                "input_format": "lot_id or temperature/voltage/size/capacity/production_mode",
                "output_format": "simulation result + UI update",
                "position": {"x": 520, "y": 300},
                "config": {
                    "agent_profile": "simulation_agent",
                                        "instructions": (
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

                },
            },
            {
                "id": "node-file-search",
                "type": "file_search",
                "subtype": "file_search",
                "label": "File Search",
                "keywords": [],
                "input_format": "query",
                "output_format": "results",
                "position": {"x": 520, "y": 420},
                "config": {"top_k": 3},
            },
            {
                "id": "node-mcp-db",
                "type": "mcp",
                "subtype": "process_db",
                "label": "MCP: Process DB",
                "keywords": [],
                "input_format": "query",
                "output_format": "rows",
                "position": {"x": 760, "y": 20},
                "config": {
                    "toolset": "process_db",
                    "server_name": "process_db",
                    "server_url": "local",
                    "auth_type": "none",
                },
            },
            {
                "id": "node-mcp-sim",
                "type": "mcp",
                "subtype": "simulation",
                "label": "MCP: Simulation",
                "keywords": [],
                "input_format": "params",
                "output_format": "result",
                "position": {"x": 760, "y": 300},
                "config": {
                    "toolset": "simulation",
                    "server_name": "simulation_api",
                    "server_url": "local",
                    "auth_type": "none",
                },
            },
            {
                "id": "node-mcp-ui",
                "type": "mcp",
                "subtype": "frontend_trigger",
                "label": "MCP: Frontend Trigger",
                "keywords": [],
                "input_format": "message",
                "output_format": "ui_event",
                "position": {"x": 760, "y": 420},
                "config": {
                    "toolset": "frontend_trigger",
                    "server_name": "frontend_trigger",
                    "server_url": "local",
                    "auth_type": "none",
                },
            },
        ],
        "edges": [
            {
                "from": "node-start",
                "to": "node-guardrail",
                "from_port": "out",
                "to_port": "in",
                "kind": "flow",
            },
            {
                "from": "node-guardrail",
                "to": "node-orchestrator",
                "from_port": "pass",
                "to_port": "in",
                "kind": "flow",
            },
            {
                "from": "node-guardrail",
                "to": "node-end",
                "from_port": "fail",
                "to_port": "in",
                "kind": "flow",
            },
            {
                "from": "node-orchestrator",
                "to": "node-end",
                "from_port": "out",
                "to_port": "in",
                "kind": "flow",
            },
            {
                "from": "node-orchestrator",
                "to": "node-db-agent",
                "from_port": "out",
                "to_port": "in",
                "kind": "handoff",
            },
            {
                "from": "node-orchestrator",
                "to": "node-sim-agent",
                "from_port": "out",
                "to_port": "in",
                "kind": "handoff",
            },
            {
                "from": "node-orchestrator",
                "to": "node-file-search",
                "from_port": "out",
                "to_port": "in",
                "kind": "tool",
            },
            {
                "from": "node-db-agent",
                "to": "node-mcp-db",
                "from_port": "out",
                "to_port": "in",
                "kind": "tool",
            },
            {
                "from": "node-db-agent",
                "to": "node-mcp-ui",
                "from_port": "out",
                "to_port": "in",
                "kind": "tool",
            },
            {
                "from": "node-sim-agent",
                "to": "node-mcp-sim",
                "from_port": "out",
                "to_port": "in",
                "kind": "tool",
            },
            {
                "from": "node-sim-agent",
                "to": "node-mcp-ui",
                "from_port": "out",
                "to_port": "in",
                "kind": "tool",
            },
        ],
    }


MODEL_KWARGS = {"model": MODEL_NAME} if MODEL_NAME else {}
DEFAULT_AGENT_INSTRUCTIONS = (
    "You are a workflow agent. "
    "Always respond in Korean. "
    "Use the available tools or handoffs to complete the task. "
    "If required information is missing, ask one clear question to obtain it. "
    "Do not invent data or tool results."
)


def _build_agent(**kwargs: object) -> Agent:
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("model", None)
        kwargs.pop("tool_use_behavior", None)
        return Agent(**kwargs)


def _resolve_agent_by_node(node: dict) -> Agent:
    config = node.get("config") or {}
    profile = config.get("agent_profile") or node.get("subtype")
    if profile == "db_agent":
        return db_agent
    if profile == "simulation_agent":
        return simulation_agent
    if profile == "test_simulation_agent":
        return test_simulation_agent
    if profile == "orchestrator":
        return triage_agent
    model_name = config.get("model") or MODEL_NAME
    model_kwargs = {"model": model_name} if model_name else {}
    return _build_agent(
        name=node.get("label") or "워크플로우 에이전트",
        instructions=config.get("instructions") or DEFAULT_AGENT_INSTRUCTIONS,
        **model_kwargs,
    )


def _build_handoff_description(node: dict) -> str:
    label = node.get("label") or node.get("subtype") or "에이전트"
    input_format = node.get("input_format") or "입력"
    output_format = node.get("output_format") or "출력"
    return f"{label}: {input_format} → {output_format}"


def _build_agent_instance(
    node: dict,
    node_map: dict[str, dict],
    tool_adjacency: dict[str, list[str]],
    handoff_adjacency: dict[str, list[str]],
    session: Any,
    visited: set[str] | None = None,
) -> Agent:
    base_agent = _resolve_agent_by_node(node)
    node_id = node.get("id")
    if not node_id:
        return base_agent
    if visited is None:
        visited = set()
    if node_id in visited:
        return base_agent
    next_visited = set(visited)
    next_visited.add(node_id)

    tool_nodes = _collect_tool_nodes(node_id, node_map, tool_adjacency)
    sub_agents = _collect_sub_agents(node_id, node_map, handoff_adjacency)

    tools = _resolve_tools(tool_nodes)
    base_tools = list(base_agent.tools or [])
    if base_tools:
        tools = tools + base_tools

    agent_tools = []
    handoffs = []
    for sub_agent in sub_agents:
        mode = (sub_agent.get("execution_mode") or "handoff").lower()
        if mode == "as_tool":
            agent_tool = _build_agent_tool(
                sub_agent,
                node_map,
                tool_adjacency,
                handoff_adjacency,
                session,
                next_visited,
            )
            if agent_tool:
                agent_tools.append(agent_tool)
        else:
            handoff_agent = _build_agent_instance(
                sub_agent,
                node_map,
                tool_adjacency,
                handoff_adjacency,
                session,
                next_visited,
            )
            if handoff_agent:
                handoffs.append(handoff_agent)

    config = node.get("config") or {}
    clone_kwargs = {
        "name": node.get("label") or base_agent.name,
        "handoff_description": _build_handoff_description(node),
        "tools": _dedupe_tools(tools + agent_tools),
        "handoffs": handoffs,
    }
    if config.get("instructions"):
        clone_kwargs["instructions"] = config["instructions"]

    return base_agent.clone(**clone_kwargs)


def _build_agent_tool(
    node: dict,
    node_map: dict[str, dict],
    tool_adjacency: dict[str, list[str]],
    handoff_adjacency: dict[str, list[str]],
    session: Any,
    visited: set[str],
) -> Any | None:
    agent = _build_agent_instance(
        node, node_map, tool_adjacency, handoff_adjacency, session, visited
    )
    tool_name = _safe_tool_name(node.get("label") or node.get("id"))
    tool_description = _build_tool_description(node)
    return agent.as_tool(
        tool_name=tool_name,
        tool_description=tool_description,
        session=session,
        hooks=WorkflowRunHooks(),
    )


async def _execute_agent_path(
    node_map: dict[str, dict],
    tool_adjacency: dict[str, list[str]],
    handoff_adjacency: dict[str, list[str]],
    path: list[str],
    message: str,
    session: Any,
    allow_llm: bool,
) -> WorkflowRun:
    hooks = WorkflowRunHooks() if allow_llm else None
    agent_nodes = [
        node_id for node_id in path if node_map[node_id].get("type") == "agent"
    ]
    if not agent_nodes:
        return await _execute_function_path(node_map, path, message)

    agent_node = node_map[agent_nodes[0]]
    has_tool_in_path = any(
        node_map[node_id].get("type") in TOOL_NODE_TYPES for node_id in path
    )

    if not allow_llm:
        if has_tool_in_path:
            return await _execute_function_path(node_map, path, message)
        return WorkflowRun(
            assistant_message="OPENAI_API_KEY가 설정되지 않았습니다. 키를 설정한 뒤 다시 시도해주세요.",
            used_llm=False,
            path=path,
        )

    configured_agent = _build_agent_instance(
        agent_node, node_map, tool_adjacency, handoff_adjacency, session
    )
    execution_mode = agent_node.get("execution_mode") or "handoff"

    if execution_mode == "as_tool":
        tool_name = _safe_tool_name(agent_node.get("label") or agent_node.get("id"))
        tool_description = _build_tool_description(agent_node)
        agent_tool = configured_agent.as_tool(
            tool_name=tool_name,
            tool_description=tool_description,
            session=session,
        )
        host_agent = _build_agent(
            name="오케스트레이터-도구 실행기",
            instructions=(
                "항상 제공된 도구를 호출해 사용자의 요청을 처리하세요. "
                "도구 호출 후 결과를 간단히 요약해 한국어로 답변하세요."
            ),
            tools=[agent_tool],
            tool_use_behavior="stop_on_first_tool",
            **MODEL_KWARGS,
        )
        result = await Runner.run(
            host_agent,
            input=message,
            session=session,
            hooks=hooks,
        )
        return WorkflowRun(result.final_output, True, path)

    result = await Runner.run(
        configured_agent,
        input=message,
        session=session,
        hooks=hooks,
    )
    return WorkflowRun(result.final_output, True, path)


def _collect_sub_agents(
    start_id: str,
    node_map: dict[str, dict],
    handoff_adjacency: dict[str, list[str]],
) -> list[dict]:
    sub_agents: list[dict] = []
    for node_id in handoff_adjacency.get(start_id, []):
        node = node_map.get(node_id)
        if node and node.get("type") == "agent":
            sub_agents.append(node)
    return sub_agents


def _collect_tool_nodes(
    start_id: str,
    node_map: dict[str, dict],
    tool_adjacency: dict[str, list[str]],
) -> list[dict]:
    tool_nodes: list[dict] = []
    for node_id in tool_adjacency.get(start_id, []):
        node = node_map.get(node_id)
        if node and node.get("type") in TOOL_NODE_TYPES:
            tool_nodes.append(node)
    return tool_nodes


def _resolve_tools(tool_nodes: list[dict]) -> list:
    tools = []
    for node in tool_nodes:
        node_type = node.get("type")
        if node_type == "file_search":
            tools.append(file_search)
            continue
        if node_type == "function_tool":
            tool = _build_function_tool(node)
            if tool:
                tools.append(tool)
            continue
        if node_type != "mcp":
            continue
        toolset = (node.get("config") or {}).get("toolset") or node.get("subtype")
        if toolset == "process_db":
            tools.append(get_process_data)
        elif toolset == "simulation":
            tools.extend(
                [
                    open_simulation_form,
                    update_simulation_params,
                    run_simulation,
                    run_lot_simulation,
                ]
            )
        elif toolset == "prediction":
            tools.append(run_prediction_simulation)
        elif toolset == "frontend_trigger":
            tools.append(frontend_trigger)
    return _dedupe_tools(tools)


def _dedupe_tools(tools: list) -> list:
    seen = set()
    unique = []
    for tool in tools:
        name = getattr(tool, "name", None) or id(tool)
        if name in seen:
            continue
        seen.add(name)
        unique.append(tool)
    return unique


def _safe_tool_name(value: str | None) -> str:
    if not value:
        return "agent_tool"
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).lower()
    cleaned = cleaned.strip("_")
    if not cleaned:
        return "agent_tool"
    if cleaned[0].isdigit():
        cleaned = f"agent_{cleaned}"
    return cleaned


def _build_tool_description(node: dict) -> str:
    input_format = node.get("input_format") or "Input"
    output_format = node.get("output_format") or "Output"
    return f"Input: {input_format}. Output: {output_format}."


def _build_function_tool(node: dict) -> Any | None:
    config = node.get("config") or {}
    tool_type = config.get("tool_type") or "db_query"
    if tool_type == "db_query":
        return _build_db_query_tool(node)
    return None


def _build_db_query_tool(node: dict) -> Any | None:
    config = node.get("config") or {}
    connection_id = config.get("connection_id")
    schema_name = config.get("schema")
    table_name = config.get("table")
    if not connection_id or not schema_name or not table_name:
        return None
    columns = config.get("columns") or []
    filter_column = config.get("filter_column")
    filter_operator = config.get("filter_operator") or "ilike"
    base_limit = int(config.get("limit") or 50)
    tool_name = _safe_tool_name(
        config.get("tool_name") or node.get("label") or node.get("id")
    )
    description = config.get("description") or _build_tool_description(node)

    @function_tool
    def _db_query_tool(query: str = "", limit: int | None = None) -> dict:
        try:
            limit_value = int(limit) if limit is not None else base_limit
            limit_value = max(1, min(limit_value, 500))
            return execute_table_query(
                connection_id=connection_id,
                schema_name=schema_name,
                table_name=table_name,
                columns=columns,
                filter_column=filter_column,
                filter_operator=filter_operator,
                filter_value=query,
                limit=limit_value,
            )
        except Exception as error:
            return {"error": str(error)}

    _db_query_tool.__name__ = tool_name
    _db_query_tool.__doc__ = description
    return _db_query_tool


def _node_summary(node: dict) -> dict:
    return {
        "id": node.get("id"),
        "label": node.get("label"),
        "type": node.get("type"),
        "subtype": node.get("subtype"),
        "execution_mode": node.get("execution_mode"),
    }


def _find_start_node(node_map: dict[str, dict]) -> dict | None:
    for node in node_map.values():
        if node.get("type") == "start":
            return node
    return None


def _parse_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [chunk.strip() for chunk in re.split(r"[,\n]", value) if chunk.strip()]
    return [str(value).strip()]


def _parse_state_variables(value: Any) -> list[str]:
    names: list[str] = []
    for item in _parse_list(value):
        name = item.split(":", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _resolve_variable(key: str, context: WorkflowContext) -> Any:
    if key == "input_as_text":
        return context.input_as_text
    if key.startswith("state."):
        return context.state.get(key.split(".", 1)[1])
    if key in context.state:
        return context.state.get(key)
    return None


def _render_template(template: str, context: WorkflowContext) -> str:
    if not template:
        return ""
    pattern = re.compile(r"\{\{\s*([^}]+)\s*\}\}")

    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        value = _resolve_variable(key, context)
        return "" if value is None else str(value)

    return pattern.sub(replace, template)


def _apply_start(node: dict, context: WorkflowContext, message: str) -> None:
    config = node.get("config") or {}
    context.input_as_text = message
    context.state.setdefault("input_as_text", message)
    for name in _parse_state_variables(config.get("state_variables")):
        context.state.setdefault(name, None)


def _apply_state(node: dict, context: WorkflowContext) -> None:
    config = node.get("config") or {}
    assignments = config.get("assignments")
    updates: dict[str, Any] = {}
    if isinstance(assignments, dict):
        updates = assignments
    elif isinstance(assignments, list):
        for item in assignments:
            if isinstance(item, dict):
                key = item.get("key")
                value = item.get("value")
                if key:
                    updates[key] = value
            elif isinstance(item, str) and "=" in item:
                key, value = item.split("=", 1)
                updates[key.strip()] = value.strip()
    elif isinstance(assignments, str):
        for line in assignments.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            updates[key.strip()] = value.strip()

    for key, value in updates.items():
        if isinstance(value, str):
            context.state[key] = _render_template(value, context)
        else:
            context.state[key] = value


def _apply_transform(node: dict, context: WorkflowContext) -> None:
    config = node.get("config") or {}
    template = config.get("template") or config.get("expression") or ""
    transformed = _render_template(template, context)
    if transformed:
        context.input_as_text = transformed
        context.last_output = transformed


def _guardrail_matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def _evaluate_guardrail(node: dict, context: WorkflowContext) -> tuple[bool, list[str]]:
    config = node.get("config") or {}
    input_key = config.get("input") or "input_as_text"
    text = _resolve_variable(input_key, context) or context.input_as_text or ""
    issues: list[str] = []

    if config.get("pii"):
        if _guardrail_matches(text, r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"):
            issues.append("PII(email)")
        if _guardrail_matches(text, r"\b\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}\b"):
            issues.append("PII(phone)")

    if config.get("moderation"):
        banned = ["hate", "kill", "폭탄", "살인", "혐오"]
        if any(word in text.lower() for word in banned):
            issues.append("moderation")

    if config.get("jailbreak"):
        jailbreak = ["ignore previous", "system prompt", "jailbreak", "탈옥"]
        if any(word in text.lower() for word in jailbreak):
            issues.append("jailbreak")

    if config.get("hallucination"):
        if not config.get("vector_store_id"):
            issues.append("hallucination")

    return not issues, issues


class _StateView:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, key: str) -> Any:
        return self._data.get(key)


def _safe_eval(expr: str, context: WorkflowContext) -> bool:
    if not expr:
        return True
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    allowed_calls = {"contains", "matches", "exists"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                return False
        elif isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "state":
                return False
        elif not isinstance(
            node,
            (
                ast.Expression,
                ast.BoolOp,
                ast.UnaryOp,
                ast.Compare,
                ast.Name,
                ast.Load,
                ast.Constant,
                ast.And,
                ast.Or,
                ast.Not,
                ast.Eq,
                ast.NotEq,
                ast.Gt,
                ast.GtE,
                ast.Lt,
                ast.LtE,
                ast.Call,
                ast.Attribute,
            ),
        ):
            return False

    def contains(value: Any, needle: Any) -> bool:
        return str(needle) in str(value or "")

    def matches(value: Any, pattern: str) -> bool:
        return re.search(pattern, str(value or ""), re.IGNORECASE) is not None

    def exists(value: Any) -> bool:
        return value is not None and value != ""

    local_ctx = {
        "input_as_text": context.input_as_text,
        "state": _StateView(context.state),
        "contains": contains,
        "matches": matches,
        "exists": exists,
    }
    try:
        return bool(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, local_ctx))
    except Exception:
        return False


def _select_next_node(
    current_id: str,
    flow_edges: dict[str, list[dict]],
    port: str,
) -> str | None:
    for edge in flow_edges.get(current_id, []):
        if (edge.get("from_port") or "out") == port:
            return edge.get("to")
    if flow_edges.get(current_id):
        return flow_edges[current_id][0].get("to")
    return None


async def _execute_tool_node(
    node: dict,
    context: WorkflowContext,
) -> str | None:
    config = node.get("config") or {}
    node_type = node.get("type")
    tool_label = node.get("label") or node.get("subtype") or node_type

    if node_type == "file_search":
        top_k = int(config.get("top_k") or 3)
        result = file_search(query=context.input_as_text, top_k=top_k)
        context.state["file_search"] = result
        await emit_workflow_log(
            "TOOL",
            f"{tool_label} result {result.get('count', 0)}",
        )
        return f"File Search 결과 {result.get('count', 0)}건을 찾았습니다."

    if node_type == "function_tool":
        tool = _build_function_tool(node)
        if not tool:
            return "Function Tool 설정이 필요합니다."
        result = _invoke_function_tool(node, tool, context.input_as_text)
        if hasattr(result, "__await__"):
            result = await result
        context.state["function_tool"] = result
        if isinstance(result, dict) and result.get("error"):
            await emit_workflow_log(
                "TOOL",
                f"{tool_label} error: {result.get('error')}",
            )
            return f"Function Tool 오류: {result.get('error')}"
        rows = result.get("rows") if isinstance(result, dict) else None
        if rows is not None:
            await emit_workflow_log(
                "TOOL",
                f"{tool_label} rows {len(rows)}",
            )
            return f"Function Tool 결과 {len(rows)}건을 조회했습니다."
        await emit_workflow_log("TOOL", f"{tool_label} done")
        return "Function Tool을 실행했습니다."

    toolset = config.get("toolset") or node.get("subtype")
    if toolset == "process_db":
        query = _extract_db_query(context.input_as_text)
        result = await emit_process_data(query, limit=12)
        context.state["db_result"] = result
        rows = result.get("rows", [])
        await emit_workflow_log(
            "TOOL",
            f"{tool_label} rows {len(rows)}",
        )
        return f"LOT 조회를 실행했습니다. UI에서 {len(rows)}건을 확인하세요."

    if toolset == "simulation":
        params = extract_simulation_params(context.input_as_text)
        update_result = collect_simulation_params(**params)
        await emit_simulation_form(update_result.get("params", {}), update_result.get("missing", []))
        missing = update_result.get("missing", [])
        if missing:
            missing_text = ", ".join(_to_korean_field(m) for m in missing)
            await emit_workflow_log(
                "TOOL",
                f"{tool_label} missing: {', '.join(missing)}",
            )
            return f"\ucd94\ucc9c \uc785\ub825\uc774 \ubd80\uc871\ud569\ub2c8\ub2e4: {missing_text}"
        result = await execute_simulation()
        context.state["simulation_result"] = result
        await emit_workflow_log("TOOL", f"{tool_label} done")
        return "\ucd94\ucc9c \ud568\uc218\ub97c \uc2e4\ud589\ud588\uc2b5\ub2c8\ub2e4. \uacb0\uacfc\uac00 UI\uc5d0 \ubc18\uc601\ub418\uc5c8\uc2b5\ub2c8\ub2e4."

    if toolset == "prediction":
        result = await run_prediction_simulation()
        context.state["prediction_result"] = result
        await emit_workflow_log("TOOL", f"{tool_label} done")
        if isinstance(result, dict) and result.get("status") == "missing":
            return "\uc608\uce21 \uc2dc\ubbac\ub808\uc774\uc158\uc744 \uc2e4\ud589\ud558\ub824\uba74 \ucd94\ucc9c \uacb0\uacfc\uac00 \ud544\uc694\ud569\ub2c8\ub2e4."
        return "\uc608\uce21 \uc2dc\ubbac\ub808\uc774\uc158\uc744 \uc2e4\ud589\ud588\uc2b5\ub2c8\ub2e4. \uacb0\uacfc\uac00 UI\uc5d0 \ubc18\uc601\ub418\uc5c8\uc2b5\ub2c8\ub2e4."

    if toolset == "frontend_trigger":
        trigger_message = config.get("message") or _summarize_result(
            context.state.get("prediction_result")
            or context.state.get("simulation_result")
            or context.state.get("db_result")
        )
        await emit_frontend_trigger(trigger_message, context.state)
        await emit_workflow_log("TOOL", f"{tool_label} sent")
        return "프론트 트리거를 전송했습니다."

    return None


def _format_final_output(result: TraversalResult, node_map: dict[str, dict]) -> str:
    context = result.context
    end_node = node_map.get(result.end_node_id or "")
    config = end_node.get("config") if end_node else {}
    return_type = (config or {}).get("return_type") or "text"

    if return_type == "json":
        payload = {
            "response": context.last_output or context.input_as_text,
            "state": context.state,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if context.last_output:
        return context.last_output
    if context.tool_messages:
        return context.tool_messages[-1]
    if context.state.get("llm_unavailable"):
        return "OPENAI_API_KEY가 설정되지 않았습니다. 키를 설정한 뒤 다시 시도해주세요."
    return "워크플로우를 실행했습니다."


async def _traverse_workflow(
    node_map: dict[str, dict],
    flow_edges: dict[str, list[dict]],
    tool_adjacency: dict[str, list[str]],
    handoff_adjacency: dict[str, list[str]],
    message: str,
    session: Any,
    allow_llm: bool,
    simulate: bool,
) -> TraversalResult | None:
    start_node = _find_start_node(node_map)
    if not start_node:
        return None

    context = WorkflowContext(
        input_as_text=message, state={}, tool_messages=[], used_llm=False
    )
    hooks = WorkflowRunHooks() if allow_llm and not simulate else None
    path: list[str] = []
    current_id = start_node.get("id")
    loop_counts: dict[str, int] = {}
    end_node_id: str | None = None

    while current_id:
        node = node_map.get(current_id)
        if not node:
            break
        path.append(current_id)
        node_type = node.get("type")
        next_port = "out"

        if node_type == "start":
            _apply_start(node, context, message)
        elif node_type == "guardrail":
            passed, issues = _evaluate_guardrail(node, context)
            context.state["guardrail"] = {"passed": passed, "issues": issues}
            next_port = "pass" if passed else "fail"
        elif node_type == "if_else":
            condition = (node.get("config") or {}).get("condition") or ""
            next_port = "true" if _safe_eval(condition, context) else "false"
        elif node_type == "while":
            condition = (node.get("config") or {}).get("condition") or ""
            max_iter = int((node.get("config") or {}).get("max_iterations") or 3)
            count = loop_counts.get(current_id, 0)
            if count < max_iter and _safe_eval(condition, context):
                loop_counts[current_id] = count + 1
                next_port = "loop"
            else:
                next_port = "done"
        elif node_type == "user_approval":
            config = node.get("config") or {}
            if simulate:
                next_port = config.get("default_decision") or "approved"
            elif config.get("auto_approve"):
                next_port = "approved"
            else:
                context.tool_messages.append(
                    config.get("approval_message") or "승인이 필요합니다."
                )
                end_node_id = current_id
                break
        elif node_type == "agent":
            if not allow_llm:
                context.state["llm_unavailable"] = True
                if not simulate:
                    session_id = current_session_id.get()
                    if _should_run_prediction(context.input_as_text, session_id):
                        prediction_result = await run_prediction_simulation()
                        prediction_message = _summarize_result(prediction_result)
                        context.tool_messages.append(prediction_message)
                        context.last_output = prediction_message
                        context.input_as_text = prediction_message
                        context.state["prediction_ran"] = True
                    else:
                        candidates = handoff_adjacency.get(current_id, [])
                        selected_id = (
                            _select_candidate(node_map, candidates, context.input_as_text)
                            if candidates
                            else None
                        )
                        if selected_id:
                            selected_node = node_map.get(selected_id)
                            if selected_node:
                                await emit_workflow_log(
                                    "HANDOFF",
                                    f"{node.get('label') or current_id} -> {selected_node.get('label') or selected_id}",
                                )
                        tool_owner = node_map.get(selected_id) if selected_id else node
                        if tool_owner and tool_owner.get("id"):
                            tool_nodes = _collect_tool_nodes(
                                tool_owner.get("id"), node_map, tool_adjacency
                            )
                            for tool_node in tool_nodes:
                                tool_message = await _execute_tool_node(tool_node, context)
                                if tool_message:
                                    context.tool_messages.append(tool_message)
                                    context.last_output = tool_message
                                    context.input_as_text = tool_message
                        if selected_id:
                            path.append(selected_id)
            elif not simulate:
                agent = _build_agent_instance(
                    node, node_map, tool_adjacency, handoff_adjacency, session
                )
                result = await Runner.run(
                    agent,
                    input=context.input_as_text,
                    session=session,
                    context=context,
                    hooks=hooks,
                )
                context.last_output = result.final_output
                context.input_as_text = result.final_output
                context.used_llm = True
        elif node_type in TOOL_NODE_TYPES:
            if not simulate and (not allow_llm or not context.used_llm):
                tool_message = await _execute_tool_node(node, context)
                if tool_message:
                    context.tool_messages.append(tool_message)
                    context.last_output = tool_message
                    context.input_as_text = tool_message
        elif node_type == "transform":
            if not simulate:
                _apply_transform(node, context)
        elif node_type == "state":
            if not simulate:
                _apply_state(node, context)
        elif node_type == "end":
            end_node_id = current_id
            break
        elif node_type == "note":
            if not flow_edges.get(current_id):
                end_node_id = current_id
                break

        next_id = _select_next_node(current_id, flow_edges, next_port)
        if not next_id or next_id in path and len(path) > 60:
            end_node_id = end_node_id or current_id
            break
        current_id = next_id

    return TraversalResult(path=path, context=context, end_node_id=end_node_id)


def _select_preview_node(
    node_map: dict[str, dict],
    handoff_adjacency: dict[str, list[str]],
    message: str,
    path: list[str],
) -> str | None:
    for node_id in path:
        node = node_map.get(node_id)
        if not node or node.get("type") != "agent":
            continue
        candidates = handoff_adjacency.get(node_id, [])
        if candidates:
            selected_id = _select_candidate(node_map, candidates, message)
            if selected_id:
                return selected_id
        return node_id
    for node_id in path:
        node = node_map.get(node_id)
        if node and node.get("type") in TOOL_NODE_TYPES:
            return node_id
    return path[-1] if path else None


def _run_preview(
    node_map: dict[str, dict],
    flow_edges: dict[str, list[dict]],
    tool_adjacency: dict[str, list[str]],
    handoff_adjacency: dict[str, list[str]],
    message: str,
) -> dict | None:
    start_node = _find_start_node(node_map)
    if not start_node:
        return None

    context = WorkflowContext(
        input_as_text=message, state={}, tool_messages=[], used_llm=False
    )
    path: list[str] = []
    current_id = start_node.get("id")
    loop_counts: dict[str, int] = {}

    while current_id:
        node = node_map.get(current_id)
        if not node:
            break
        path.append(current_id)
        node_type = node.get("type")
        next_port = "out"

        if node_type == "start":
            _apply_start(node, context, message)
        elif node_type == "guardrail":
            passed, _ = _evaluate_guardrail(node, context)
            next_port = "pass" if passed else "fail"
        elif node_type == "if_else":
            condition = (node.get("config") or {}).get("condition") or ""
            next_port = "true" if _safe_eval(condition, context) else "false"
        elif node_type == "while":
            condition = (node.get("config") or {}).get("condition") or ""
            max_iter = int((node.get("config") or {}).get("max_iterations") or 3)
            count = loop_counts.get(current_id, 0)
            if count < max_iter and _safe_eval(condition, context):
                loop_counts[current_id] = count + 1
                next_port = "loop"
            else:
                next_port = "done"
        elif node_type == "user_approval":
            config = node.get("config") or {}
            next_port = config.get("default_decision") or "approved"
        elif node_type == "end":
            break

        next_id = _select_next_node(current_id, flow_edges, next_port)
        if not next_id or next_id in path and len(path) > 60:
            break
        current_id = next_id

    selected_id = _select_preview_node(node_map, handoff_adjacency, message, path)
    if selected_id and selected_id not in path:
        path.append(selected_id)
    return {"path": path, "selected_id": selected_id}


def _invoke_function_tool(node: dict, tool: Any, message: str) -> Any:
    config = node.get("config") or {}
    tool_type = config.get("tool_type") or "db_query"
    if tool_type == "db_query":
        query = config.get("query") or _extract_db_query(message)
        limit = config.get("limit") or 50
        return tool(query=query, limit=limit)
    try:
        return tool(message=message)
    except TypeError:
        return tool()


async def _execute_function_path(
    node_map: dict[str, dict],
    path: list[str],
    message: str,
) -> WorkflowRun:
    last_result: dict | None = None
    assistant_message = "워크플로우를 실행했습니다."

    for node_id in path:
        node = node_map[node_id]
        node_type = node.get("type")

        if node_type == "file_search":
            last_result = file_search(query=message, top_k=3)
            assistant_message = f"File Search 결과 {last_result.get('count', 0)}건을 찾았습니다."
            continue

        if node_type == "function_tool":
            tool = _build_function_tool(node)
            if not tool:
                assistant_message = "Function Tool 설정이 필요합니다."
                continue
            result = _invoke_function_tool(node, tool, message)
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, dict) and result.get("error"):
                assistant_message = f"Function Tool 오류: {result.get('error')}"
            else:
                rows = result.get("rows") if isinstance(result, dict) else None
                if rows is not None:
                    assistant_message = f"Function Tool 결과 {len(rows)}건을 조회했습니다."
                else:
                    assistant_message = "Function Tool을 실행했습니다."
            if isinstance(result, dict):
                last_result = result
            continue

        if node_type == "mcp":
            toolset = (node.get("config") or {}).get("toolset") or node.get("subtype")
        else:
            toolset = node.get("subtype")

        if toolset == "db_function" or toolset == "process_db":
            query = _extract_db_query(message)
            last_result = await emit_process_data(query, limit=12)
            rows = last_result.get("rows", [])
            assistant_message = f"LOT 조회를 실행했습니다. UI에서 {len(rows)}건을 확인하세요."

        elif toolset == "api_function" or toolset == "simulation":
            params = extract_simulation_params(message)
            update_result = collect_simulation_params(**params)
            await emit_simulation_form(
                update_result.get("params", {}),
                update_result.get("missing", []),
            )
            missing = update_result.get("missing", [])
            if missing:
                missing_text = ", ".join(_to_korean_field(m) for m in missing)
                return WorkflowRun(
                    assistant_message=f"\ucd94\ucc9c \uc785\ub825\uc774 \ubd80\uc871\ud569\ub2c8\ub2e4: {missing_text}",
                    used_llm=False,
                    path=path,
                )
            last_result = await execute_simulation()
            assistant_message = "\ucd94\ucc9c \ud568\uc218\ub97c \uc2e4\ud589\ud588\uc2b5\ub2c8\ub2e4. \uacb0\uacfc\uac00 UI\uc5d0 \ubc18\uc601\ub418\uc5c8\uc2b5\ub2c8\ub2e4."

        elif toolset == "prediction":
            last_result = await run_prediction_simulation()
            if isinstance(last_result, dict) and last_result.get("status") == "missing":
                assistant_message = "\uc608\uce21 \uc2dc\ubbac\ub808\uc774\uc158\uc744 \uc2e4\ud589\ud558\ub824\uba74 \ucd94\ucc9c \uacb0\uacfc\uac00 \ud544\uc694\ud569\ub2c8\ub2e4."
            else:
                assistant_message = "\uc608\uce21 \uc2dc\ubbac\ub808\uc774\uc158\uc744 \uc2e4\ud589\ud588\uc2b5\ub2c8\ub2e4. \uacb0\uacfc\uac00 UI\uc5d0 \ubc18\uc601\ub418\uc5c8\uc2b5\ub2c8\ub2e4."

        elif toolset == "frontend_trigger":
            trigger_message = node.get("output_format") or _summarize_result(last_result)
            await emit_frontend_trigger(trigger_message, last_result)
            assistant_message = "프론트 트리거를 전송했습니다."

    return WorkflowRun(assistant_message, False, path)


def _find_orchestrator(node_map: dict[str, dict]) -> dict | None:
    for node in node_map.values():
        if node.get("type") == "agent" and node.get("subtype") == "orchestrator":
            return node
    return None


def _plan_route(
    workflow: dict,
    message: str,
) -> tuple[dict[str, dict], dict[str, list[str]], dict, str, list[str]] | None:
    normalized = normalize_workflow(workflow)
    valid, _ = validate_workflow(normalized)
    if not valid:
        return None

    node_map = {node["id"]: node for node in normalized["nodes"] if node.get("id")}
    adjacency = _build_adjacency(normalized["edges"])
    orchestrator = _find_orchestrator(node_map)
    if not orchestrator:
        return None

    candidate_ids = adjacency.get(orchestrator["id"], [])
    if not candidate_ids:
        candidate_ids = [
            node_id
            for node_id, node in node_map.items()
            if node.get("type") == "agent" and node.get("subtype") != "orchestrator"
        ]

    if not candidate_ids:
        return None

    selected_id = _select_candidate(node_map, candidate_ids, message)
    path = _build_linear_path(selected_id, adjacency)
    return node_map, adjacency, orchestrator, selected_id, path


def _select_candidate(node_map: dict[str, dict], candidates: list[str], message: str) -> str:
    normalized = message.lower()
    best_id = candidates[0]
    best_score = -1
    for candidate_id in candidates:
        node = node_map.get(candidate_id)
        if not node:
            continue
        keywords = _normalize_keywords(node.get("keywords"))
        score = sum(1 for keyword in keywords if keyword and keyword in normalized)
        if score > best_score:
            best_score = score
            best_id = candidate_id
    return best_id


def _build_adjacency(edges: list[dict], kind: str | None = None) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if kind and edge.get("kind") != kind:
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        adjacency.setdefault(src, []).append(dst)
    return adjacency


def _build_incoming(edges: list[dict]) -> dict[str, list[str]]:
    incoming: dict[str, list[str]] = {}
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        incoming.setdefault(dst, []).append(src)
    return incoming


def _build_outgoing(edges: list[dict]) -> dict[str, list[dict]]:
    outgoing: dict[str, list[dict]] = {}
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        outgoing.setdefault(src, []).append(edge)
    return outgoing


def _has_agent_ancestor(
    node_id: str,
    node_map: dict[str, dict],
    incoming: dict[str, list[str]],
    visited: set[str] | None = None,
) -> bool:
    if visited is None:
        visited = set()
    if node_id in visited:
        return False
    visited.add(node_id)
    for parent_id in incoming.get(node_id, []):
        parent = node_map.get(parent_id)
        if not parent:
            continue
        if parent.get("type") == "agent":
            return True
        if _has_agent_ancestor(parent_id, node_map, incoming, visited):
            return True
    return False


def _build_linear_path(start_id: str, adjacency: dict[str, list[str]]) -> list[str]:
    path = []
    current = start_id
    visited = set()
    while current and current not in visited:
        path.append(current)
        visited.add(current)
        next_nodes = adjacency.get(current, [])
        if not next_nodes:
            break
        current = next_nodes[0]
    return path


def _workflow_summary(item: dict | None) -> dict:
    if not item:
        return {}
    meta = item.get("meta") or {}
    return {
        "id": item.get("id"),
        "name": meta.get("name") or "워크플로우",
        "description": meta.get("description") or "",
        "updated_at": meta.get("updated_at") or "",
    }


def _new_workflow_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", (name or "").strip().lower()).strip("-")
    if not slug:
        slug = "workflow"
    return f"{slug}-{uuid4().hex[:6]}"


def _find_store_item(items: list[dict], workflow_id: str) -> dict | None:
    for item in items:
        if item.get("id") == workflow_id:
            return item
    return None


def _sync_store_item(items: list[dict], workflow_id: str, workflow: dict) -> None:
    item = _find_store_item(items, workflow_id)
    if not item:
        return
    item["workflow"] = workflow
    item["meta"] = workflow.get("meta") or {}


def _normalize_node_type(value: str | None) -> str | None:
    if not value:
        return value
    return LEGACY_TYPE_MAP.get(value, value)


def _infer_edge_kind(src_type: str | None, dst_type: str | None) -> str:
    if src_type == "agent" and dst_type == "agent":
        return "handoff"
    if src_type == "agent" and dst_type in TOOL_NODE_TYPES:
        return "tool"
    return "flow"


def _normalize_keywords(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        chunks = re.split(r"[,\n]", value)
        return [chunk.strip().lower() for chunk in chunks if chunk.strip()]
    return [str(value).strip().lower()]


def _extract_db_query(message: str) -> str:
    lowered = message.lower()
    lot_match = re.search(r"(lot[-_ ]?[a-z0-9-]+)", lowered, re.IGNORECASE)
    if lot_match:
        raw = lot_match.group(1)
        return raw.upper().replace("_", "-").replace(" ", "")
    query_parts = []

    line_match = re.search(r"(?:line|라인)\s*-?\s*([a-z])", lowered)
    if line_match:
        query_parts.append(f"Line-{line_match.group(1).upper()}")

    if "warn" in lowered or "경고" in lowered:
        query_parts.append("WARN")
    if "ok" in lowered:
        query_parts.append("OK")

    return " ".join(query_parts) if query_parts else message


def _is_affirmative(message: str) -> bool:
    lowered = (message or "").strip().lower()
    if not lowered:
        return False
    affirmative = {
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "yep",
        "\ub124",
        "\uc608",
        "\uc751",
        "\uadf8\ub798",
        "\uc88b\uc544",
        "\uac00\ub2a5",
        "\uc9c4\ud589",
        "\uc2e4\ud589",
        "\ud574\uc918",
    }
    if lowered in affirmative:
        return True
    return any(token in lowered for token in affirmative)


def _has_recommendation(session_id: str) -> bool:
    if not session_id:
        return False
    recommendation = recommendation_store.get(session_id)
    if not isinstance(recommendation, dict):
        return False
    params = recommendation.get("params")
    if not isinstance(params, dict):
        return False
    required = {f"param{idx}" for idx in range(1, 31)}
    return required.issubset(params.keys())


def _should_run_prediction(message: str, session_id: str) -> bool:
    if not message or not _has_recommendation(session_id):
        return False
    recommendation = recommendation_store.get(session_id)
    awaiting_prediction = bool(
        isinstance(recommendation, dict)
        and recommendation.get("awaiting_prediction") is True
    )
    lowered = message.lower()
    triggers = (
        "\uc608\uce21",
        "\uc2dc\ubbac\ub808\uc774\uc158",
        "predict",
        "prediction",
        "simulate",
        "simulation",
    )
    if any(token in lowered for token in triggers):
        return True
    return awaiting_prediction and _is_affirmative(message)


def _to_korean_field(field: str) -> str:
    mapping = {
        "temperature": "온도",
        "voltage": "전압",
        "size": "크기",
        "capacity": "용량",
        "production_mode": "양산/개발",
    }
    return mapping.get(field, field)


def _summarize_result(result: dict | None) -> str:
    if not result:
        return "프론트 트리거가 실행되었습니다."
    if "rows" in result:
        return f"LOT 결과 {len(result.get('rows', []))}건이 표시되었습니다."
    if "recommendation" in result and "result" in result:
        prediction = result.get("result") or {}
        return (
            f"\uc608\uce21 \uacb0\uacfc: \uc6a9\ub7c9 {prediction.get('predicted_capacity', '-')}, "
            f"DC {prediction.get('predicted_dc_capacity', '-')}, "
            f"\uc2e0\ub8b0\uc131 {prediction.get('reliability_pass_prob', '-')}"
        )
    if "result" in result:
        inner = result.get("result", {})
        recommended_model = inner.get("recommended_model")
        rep_lot = inner.get("representative_lot")
        if recommended_model or rep_lot:
            return (
                f"\ucd94\ucc9c \uacb0\uacfc: {recommended_model or '-'}, "
                f"\ub300\ud45c LOT {rep_lot or '-'}"
            )
    return "프론트 트리거가 실행되었습니다."


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
