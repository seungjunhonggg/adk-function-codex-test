from __future__ import annotations

import logging
import time
from typing import Any

from agents import RunHooks

from .events import event_bus

LOG_LABEL_ALLOWLIST = {
    "ROUTE",
    "INTENT",
    "FLOW",
    "PROGRESS",
    "STATE",
    "AGENT",
    "HANDOFF",
    "TOOL",
    "LATENCY",
}

logger = logging.getLogger(__name__)


def _safe_name(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(start: float | None) -> int | None:
    if start is None:
        return None
    return int((time.perf_counter() - start) * 1000)


async def emit_workflow_log(
    label: str,
    detail: str,
    meta: dict | None = None,
) -> dict:
    payload = {"label": label, "detail": detail}
    if meta:
        payload["meta"] = meta
    logger.info("workflow_log %s", payload)
    if label not in LOG_LABEL_ALLOWLIST:
        return payload
    await event_bus.broadcast({"type": "workflow_log", "payload": payload})
    return payload


class WorkflowRunHooks(RunHooks):
    def __init__(self) -> None:
        self._agent_starts: dict[int, list[float]] = {}
        self._tool_starts: dict[int, list[float]] = {}
        self._llm_starts: dict[int, list[float]] = {}

    async def on_agent_start(self, context, agent) -> None:
        agent_name = _safe_name(getattr(agent, "name", None), "Agent")
        self._agent_starts.setdefault(id(agent), []).append(_now())
        await emit_workflow_log("AGENT", f"{agent_name} started")

    async def on_agent_end(self, context, agent, output: Any) -> None:
        agent_name = _safe_name(getattr(agent, "name", None), "Agent")
        starts = self._agent_starts.get(id(agent))
        start = starts.pop() if starts else None
        output_text = str(output or "").strip()
        if output_text:
            preview = output_text[:160]
            await emit_workflow_log("AGENT", f"{agent_name} output", {"preview": preview})
        elapsed = _elapsed_ms(start)
        if elapsed is not None:
            await emit_workflow_log("LATENCY", f"AGENT {agent_name} {elapsed}ms")

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._llm_starts.setdefault(id(agent), []).append(_now())

    async def on_llm_end(self, context, agent, response) -> None:
        agent_name = _safe_name(getattr(agent, "name", None), "Agent")
        starts = self._llm_starts.get(id(agent))
        start = starts.pop() if starts else None
        elapsed = _elapsed_ms(start)
        if elapsed is not None:
            await emit_workflow_log("LATENCY", f"LLM {agent_name} {elapsed}ms")

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        from_name = _safe_name(getattr(from_agent, "name", None), "Agent")
        to_name = _safe_name(getattr(to_agent, "name", None), "Agent")
        await emit_workflow_log("HANDOFF", f"{from_name} -> {to_name}")

    async def on_tool_start(self, context, agent, tool) -> None:
        agent_name = _safe_name(getattr(agent, "name", None), "Agent")
        tool_name = _safe_name(getattr(tool, "name", None), "tool")
        self._tool_starts.setdefault(id(tool), []).append(_now())
        if tool_name in {"db_agent", "simulation_agent"}:
            await emit_workflow_log("ROUTE", f"{agent_name} -> {tool_name}")
        await emit_workflow_log("TOOL", f"{agent_name} -> {tool_name} started")

    async def on_tool_end(self, context, agent, tool, result: str) -> None:
        tool_name = _safe_name(getattr(tool, "name", None), "tool")
        starts = self._tool_starts.get(id(tool))
        start = starts.pop() if starts else None
        elapsed = _elapsed_ms(start)
        await emit_workflow_log("TOOL", f"{tool_name} done")
        if elapsed is not None:
            await emit_workflow_log("LATENCY", f"TOOL {tool_name} {elapsed}ms")
