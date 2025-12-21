from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import Agent, Runner

from .agents import db_agent, simulation_agent, triage_agent
from .config import MODEL_NAME, WORKFLOW_PATH
from .tools import (
    collect_simulation_params,
    emit_frontend_trigger,
    emit_process_data,
    execute_simulation,
    frontend_trigger,
    get_process_data,
    run_simulation,
    update_simulation_params,
)


ALLOWED_TYPES = {"user", "agent", "function"}
ALLOWED_AGENT_SUBTYPES = {"orchestrator", "db_agent", "simulation_agent"}
ALLOWED_FUNCTION_SUBTYPES = {"db_function", "api_function", "frontend_trigger"}
ALLOWED_AGENT_EXECUTION = {"handoff", "as_tool"}


@dataclass
class WorkflowRun:
    assistant_message: str
    used_llm: bool
    path: list[str]


def ensure_workflow() -> dict:
    path = Path(WORKFLOW_PATH)
    if not path.exists():
        save_workflow(default_workflow())
    return load_workflow()


def load_workflow() -> dict:
    path = Path(WORKFLOW_PATH)
    if not path.exists():
        return default_workflow()
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return default_workflow()


def save_workflow(workflow: dict) -> dict:
    normalized = normalize_workflow(workflow)
    normalized.setdefault("meta", {})
    normalized["meta"]["updated_at"] = _now_iso()
    path = Path(WORKFLOW_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2)
    return normalized


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
        subtype = node.get("subtype")
        if not node_id:
            errors.append("id가 없는 노드가 있습니다.")
        else:
            node_ids.append(node_id)
        if node_type not in ALLOWED_TYPES:
            errors.append(f"허용되지 않는 type: {node_type}")
        if node_type == "agent" and subtype not in ALLOWED_AGENT_SUBTYPES:
            errors.append(f"허용되지 않는 agent subtype: {subtype}")
        if node_type == "function" and subtype not in ALLOWED_FUNCTION_SUBTYPES:
            errors.append(f"허용되지 않는 function subtype: {subtype}")

    if len(node_ids) != len(set(node_ids)):
        errors.append("노드 id가 중복되었습니다.")

    node_id_set = set(node_ids)
    for edge in edges:
        if edge.get("from") not in node_id_set or edge.get("to") not in node_id_set:
            errors.append("존재하지 않는 노드로 연결된 edge가 있습니다.")

    node_map = {node.get("id"): node for node in nodes if node.get("id")}
    incoming = _build_incoming(edges)

    user_nodes = [node for node in nodes if node.get("type") == "user"]
    if not user_nodes:
        errors.append("사용자 요청 노드가 필요합니다.")

    orchestrators = [
        node
        for node in nodes
        if node.get("type") == "agent" and node.get("subtype") == "orchestrator"
    ]
    if not orchestrators:
        errors.append("오케스트레이터 에이전트가 필요합니다.")
    if len(orchestrators) > 1:
        errors.append("오케스트레이터는 1개만 허용됩니다.")

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        if node.get("type") == "agent":
            execution_mode = node.get("execution_mode") or "handoff"
            if execution_mode not in ALLOWED_AGENT_EXECUTION:
                errors.append(f"허용되지 않는 execution_mode: {execution_mode}")
            if execution_mode == "as_tool" and not _has_agent_ancestor(
                node_id, node_map, incoming
            ):
                errors.append("as_tool 에이전트는 상위 에이전트에 연결해야 합니다.")
        if node.get("type") == "function":
            if not _has_agent_ancestor(node_id, node_map, incoming):
                errors.append("함수 노드는 에이전트 뒤에 연결해야 합니다.")

    return not errors, errors


def normalize_workflow(workflow: dict) -> dict:
    meta = workflow.get("meta") or {}
    nodes = workflow.get("nodes") or []
    edges = workflow.get("edges") or []

    normalized_nodes = []
    for node in nodes:
        node_type = node.get("type")
        subtype = node.get("subtype") or _default_subtype(node_type)
        execution_mode = (
            node.get("execution_mode") or "handoff" if node_type == "agent" else ""
        )
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
            }
        )

    normalized_edges = [
        {"from": edge.get("from"), "to": edge.get("to")}
        for edge in edges
        if edge.get("from") and edge.get("to")
    ]

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
    plan = _plan_route(workflow, message)
    if not plan:
        return None

    node_map, adjacency, _, selected_id, path = plan

    selected_node = node_map.get(selected_id)
    if not selected_node:
        return None

    if selected_node.get("type") == "function":
        return await _execute_function_path(node_map, path, message)

    if selected_node.get("type") == "agent":
        return await _execute_agent_path(
            node_map,
            adjacency,
            path,
            message,
            session,
            allow_llm,
        )

    return None


def preview_workflow(workflow: dict, message: str) -> dict | None:
    plan = _plan_route(workflow, message)
    if not plan:
        return None

    node_map, _, orchestrator, selected_id, path = plan
    preview_path = path[:]
    if orchestrator and orchestrator.get("id") not in preview_path:
        preview_path.insert(0, orchestrator.get("id"))

    return {
        "selected_id": selected_id,
        "selected_node": _node_summary(node_map.get(selected_id, {})),
        "path": preview_path,
        "path_nodes": [
            _node_summary(node_map[node_id])
            for node_id in preview_path
            if node_id in node_map
        ],
    }


def default_workflow() -> dict:
    now = _now_iso()
    return {
        "meta": {
            "name": "공정 모니터링 기본 워크플로우",
            "description": "오케스트레이터 → DB/시뮬레이션 분기 → 함수 실행 → 프론트 트리거",
            "updated_at": now,
        },
        "nodes": [
            {
                "id": "node-user",
                "type": "user",
                "subtype": "request",
                "label": "사용자 요청",
                "keywords": [],
                "input_format": "자연어 요청",
                "output_format": "텍스트",
                "position": {"x": 80, "y": 160},
            },
            {
                "id": "node-orchestrator",
                "type": "agent",
                "subtype": "orchestrator",
                "execution_mode": "handoff",
                "label": "오케스트레이터",
                "keywords": ["공정", "데이터", "예측", "시뮬레이션"],
                "input_format": "사용자 메시지",
                "output_format": "경로 선택",
                "position": {"x": 300, "y": 160},
            },
            {
                "id": "node-db-agent",
                "type": "agent",
                "subtype": "db_agent",
                "execution_mode": "handoff",
                "label": "DB 에이전트",
                "keywords": ["공정", "라인", "상태", "데이터", "조회"],
                "input_format": "라인/상태 필터",
                "output_format": "공정 레코드",
                "position": {"x": 520, "y": 60},
            },
            {
                "id": "node-sim-agent",
                "type": "agent",
                "subtype": "simulation_agent",
                "execution_mode": "handoff",
                "label": "시뮬레이션 에이전트",
                "keywords": ["예측", "시뮬레이션", "what-if", "forecast"],
                "input_format": "온도/전압/크기/용량",
                "output_format": "예측 결과",
                "position": {"x": 520, "y": 260},
            },
            {
                "id": "node-db-function",
                "type": "function",
                "subtype": "db_function",
                "label": "DB 함수",
                "keywords": [],
                "input_format": "검색 키워드",
                "output_format": "DB 결과",
                "position": {"x": 740, "y": 60},
            },
            {
                "id": "node-api-function",
                "type": "function",
                "subtype": "api_function",
                "label": "API 함수",
                "keywords": [],
                "input_format": "파라미터 JSON",
                "output_format": "시뮬레이션 결과",
                "position": {"x": 740, "y": 260},
            },
            {
                "id": "node-frontend-trigger",
                "type": "function",
                "subtype": "frontend_trigger",
                "label": "프론트 트리거",
                "keywords": [],
                "input_format": "이벤트 메시지",
                "output_format": "UI 표시",
                "position": {"x": 960, "y": 160},
            },
        ],
        "edges": [
            {"from": "node-user", "to": "node-orchestrator"},
            {"from": "node-orchestrator", "to": "node-db-agent"},
            {"from": "node-orchestrator", "to": "node-sim-agent"},
            {"from": "node-db-agent", "to": "node-db-function"},
            {"from": "node-db-function", "to": "node-frontend-trigger"},
            {"from": "node-sim-agent", "to": "node-api-function"},
            {"from": "node-api-function", "to": "node-frontend-trigger"},
        ],
    }


MODEL_KWARGS = {"model": MODEL_NAME} if MODEL_NAME else {}


def _build_agent(**kwargs: object) -> Agent:
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("model", None)
        kwargs.pop("tool_use_behavior", None)
        return Agent(**kwargs)


def _resolve_agent_by_node(node: dict) -> Agent:
    subtype = node.get("subtype")
    if subtype == "db_agent":
        return db_agent
    if subtype == "simulation_agent":
        return simulation_agent
    if subtype == "orchestrator":
        return triage_agent
    return triage_agent


def _build_handoff_description(node: dict) -> str:
    label = node.get("label") or node.get("subtype") or "에이전트"
    input_format = node.get("input_format") or "입력"
    output_format = node.get("output_format") or "출력"
    return f"{label}: {input_format} → {output_format}"


def _build_agent_instance(
    node: dict,
    node_map: dict[str, dict],
    adjacency: dict[str, list[str]],
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

    function_nodes = _collect_function_nodes(node_id, node_map, adjacency)
    sub_agents = _collect_sub_agents(node_id, node_map, adjacency)

    tools = _resolve_tools(function_nodes)
    if not tools:
        tools = list(base_agent.tools or [])

    agent_tools = []
    handoffs = []
    for sub_agent in sub_agents:
        mode = (sub_agent.get("execution_mode") or "handoff").lower()
        if mode == "as_tool":
            agent_tool = _build_agent_tool(
                sub_agent, node_map, adjacency, session, next_visited
            )
            if agent_tool:
                agent_tools.append(agent_tool)
        else:
            handoff_agent = _build_agent_instance(
                sub_agent, node_map, adjacency, session, next_visited
            )
            if handoff_agent:
                handoffs.append(handoff_agent)

    configured = base_agent.clone(
        name=node.get("label") or base_agent.name,
        handoff_description=_build_handoff_description(node),
        tools=_dedupe_tools(tools + agent_tools),
        handoffs=handoffs,
    )
    return configured


def _build_agent_tool(
    node: dict,
    node_map: dict[str, dict],
    adjacency: dict[str, list[str]],
    session: Any,
    visited: set[str],
) -> Any | None:
    agent = _build_agent_instance(node, node_map, adjacency, session, visited)
    tool_name = _safe_tool_name(node.get("label") or node.get("id"))
    tool_description = _build_tool_description(node)
    return agent.as_tool(
        tool_name=tool_name,
        tool_description=tool_description,
        session=session,
    )


async def _execute_agent_path(
    node_map: dict[str, dict],
    adjacency: dict[str, list[str]],
    path: list[str],
    message: str,
    session: Any,
    allow_llm: bool,
) -> WorkflowRun:
    agent_nodes = [
        node_id for node_id in path if node_map[node_id].get("type") == "agent"
    ]
    if not agent_nodes:
        return await _execute_function_path(node_map, path, message)

    agent_node = node_map[agent_nodes[0]]
    has_function_in_path = any(
        node_map[node_id].get("type") == "function" for node_id in path
    )

    if not allow_llm:
        if has_function_in_path:
            return await _execute_function_path(node_map, path, message)
        return WorkflowRun(
            assistant_message="OPENAI_API_KEY가 설정되지 않았습니다. 키를 설정한 뒤 다시 시도해주세요.",
            used_llm=False,
            path=path,
        )

    configured_agent = _build_agent_instance(agent_node, node_map, adjacency, session)
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
        result = await Runner.run(host_agent, input=message, session=session)
        return WorkflowRun(result.final_output, True, path)

    result = await Runner.run(configured_agent, input=message, session=session)
    return WorkflowRun(result.final_output, True, path)


def _collect_sub_agents(
    start_id: str,
    node_map: dict[str, dict],
    adjacency: dict[str, list[str]],
) -> list[dict]:
    sub_agents: list[dict] = []
    visited: set[str] = set()
    stack = list(adjacency.get(start_id, []))
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = node_map.get(node_id)
        if not node:
            continue
        if node.get("type") == "agent":
            sub_agents.append(node)
            continue
        if node.get("type") == "function":
            stack.extend(adjacency.get(node_id, []))
    return sub_agents


def _collect_function_nodes(
    start_id: str,
    node_map: dict[str, dict],
    adjacency: dict[str, list[str]],
) -> list[dict]:
    function_nodes: list[dict] = []
    visited: set[str] = set()
    stack = list(adjacency.get(start_id, []))
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = node_map.get(node_id)
        if not node:
            continue
        if node.get("type") == "function":
            function_nodes.append(node)
            stack.extend(adjacency.get(node_id, []))
            continue
        if node.get("type") == "agent":
            continue
        stack.extend(adjacency.get(node_id, []))
    return function_nodes


def _resolve_tools(function_nodes: list[dict]) -> list:
    tools = []
    for node in function_nodes:
        subtype = node.get("subtype")
        if subtype == "db_function":
            tools.append(get_process_data)
        elif subtype == "api_function":
            tools.extend([update_simulation_params, run_simulation])
        elif subtype == "frontend_trigger":
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
    input_format = node.get("input_format") or "입력"
    output_format = node.get("output_format") or "출력"
    return f"{input_format} → {output_format}"


def _node_summary(node: dict) -> dict:
    return {
        "id": node.get("id"),
        "label": node.get("label"),
        "type": node.get("type"),
        "subtype": node.get("subtype"),
        "execution_mode": node.get("execution_mode"),
    }


async def _execute_function_path(
    node_map: dict[str, dict],
    path: list[str],
    message: str,
) -> WorkflowRun:
    last_result: dict | None = None
    assistant_message = "워크플로우를 실행했습니다."

    for node_id in path:
        node = node_map[node_id]
        if node["type"] != "function":
            continue

        subtype = node.get("subtype")
        if subtype == "db_function":
            query = _extract_db_query(message)
            last_result = await emit_process_data(query, limit=12)
            rows = last_result.get("rows", [])
            assistant_message = f"DB 함수를 실행했습니다. UI에서 {len(rows)}건을 확인하세요."

        elif subtype == "api_function":
            params = _extract_simulation_params(message)
            update_result = collect_simulation_params(**params)
            missing = update_result.get("missing", [])
            if missing:
                missing_text = ", ".join(_to_korean_field(m) for m in missing)
                return WorkflowRun(
                    assistant_message=f"시뮬레이션 입력이 부족합니다: {missing_text}",
                    used_llm=False,
                    path=path,
                )
            last_result = await execute_simulation()
            assistant_message = "시뮬레이션 API 함수를 실행했습니다. 결과가 UI에 반영되었습니다."

        elif subtype == "frontend_trigger":
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


def _build_adjacency(edges: list[dict]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
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


def _default_subtype(node_type: str | None) -> str:
    if node_type == "agent":
        return "orchestrator"
    if node_type == "function":
        return "db_function"
    return "request"


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
    query_parts = []

    line_match = re.search(r"(?:line|라인)\s*-?\s*([a-z])", lowered)
    if line_match:
        query_parts.append(f"Line-{line_match.group(1).upper()}")

    if "warn" in lowered or "경고" in lowered:
        query_parts.append("WARN")
    if "ok" in lowered:
        query_parts.append("OK")

    return " ".join(query_parts) if query_parts else message


def _extract_simulation_params(message: str) -> dict:
    patterns = {
        "temperature": r"(?:온도|temperature|temp)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
        "voltage": r"(?:전압|voltage|volt)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
        "size": r"(?:크기|size)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
        "capacity": r"(?:용량|capacity)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
    }

    params: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            params[key] = float(match.group(1))

    if len(params) < 4:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", message)
        if len(numbers) >= 4 and not params:
            params = {
                "temperature": float(numbers[0]),
                "voltage": float(numbers[1]),
                "size": float(numbers[2]),
                "capacity": float(numbers[3]),
            }

    return params


def _to_korean_field(field: str) -> str:
    mapping = {
        "temperature": "온도",
        "voltage": "전압",
        "size": "크기",
        "capacity": "용량",
    }
    return mapping.get(field, field)


def _summarize_result(result: dict | None) -> str:
    if not result:
        return "프론트 트리거가 실행되었습니다."
    if "rows" in result:
        return f"DB 결과 {len(result.get('rows', []))}건이 표시되었습니다."
    if "result" in result:
        inner = result.get("result", {})
        yield_value = inner.get("predicted_yield")
        if yield_value is not None:
            return f"예측 수율 {yield_value}% 결과가 표시되었습니다."
    return "프론트 트리거가 실행되었습니다."


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
