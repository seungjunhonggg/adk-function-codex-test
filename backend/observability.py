from __future__ import annotations

from typing import Any

from agents import RunHooks

from .events import event_bus


def _safe_name(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return fallback


async def emit_workflow_log(
    label: str,
    detail: str,
    meta: dict | None = None,
) -> dict:
    payload = {"label": label, "detail": detail}
    if meta:
        payload["meta"] = meta
    await event_bus.broadcast({"type": "workflow_log", "payload": payload})
    return payload


class WorkflowRunHooks(RunHooks):
    async def on_agent_start(self, context, agent) -> None:
        agent_name = _safe_name(getattr(agent, "name", None), "Agent")
        await emit_workflow_log("AGENT", f"{agent_name} started")

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        from_name = _safe_name(getattr(from_agent, "name", None), "Agent")
        to_name = _safe_name(getattr(to_agent, "name", None), "Agent")
        await emit_workflow_log("HANDOFF", f"{from_name} -> {to_name}")

    async def on_tool_start(self, context, agent, tool) -> None:
        agent_name = _safe_name(getattr(agent, "name", None), "Agent")
        tool_name = _safe_name(getattr(tool, "name", None), "tool")
        if tool_name in {"db_agent", "simulation_agent"}:
            await emit_workflow_log("ROUTE", f"{agent_name} -> {tool_name}")
        await emit_workflow_log("TOOL", f"{agent_name} -> {tool_name} started")

    async def on_tool_end(self, context, agent, tool, result: str) -> None:
        tool_name = _safe_name(getattr(tool, "name", None), "tool")
        await emit_workflow_log("TOOL", f"{tool_name} done")
