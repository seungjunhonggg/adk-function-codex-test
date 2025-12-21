from agents import function_tool

from .context import current_session_id
from .db import query_process_data
from .events import event_bus
from .simulation import call_simulation_api, simulation_store


async def emit_process_data(query: str = "", limit: int = 12) -> dict:
    rows = query_process_data(query, limit)
    await event_bus.broadcast(
        {
            "type": "db_result",
            "payload": {"query": query, "rows": rows},
        }
    )
    return {"rows": rows}


def collect_simulation_params(
    temperature: float | None = None,
    voltage: float | None = None,
    size: float | None = None,
    capacity: float | None = None,
) -> dict:
    session_id = current_session_id.get()
    params = simulation_store.update(
        session_id,
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
    )
    missing = simulation_store.missing(session_id)
    return {"params": params, "missing": missing}


async def execute_simulation() -> dict:
    session_id = current_session_id.get()
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    if missing:
        return {"status": "missing", "missing": missing}

    result = await call_simulation_api(params)
    await event_bus.broadcast(
        {
            "type": "simulation_result",
            "payload": {"params": params, "result": result},
        }
    )
    return {"status": "ok", "result": result}


async def emit_frontend_trigger(message: str, payload: dict | None = None) -> dict:
    event = {
        "type": "frontend_trigger",
        "payload": {
            "message": message,
            "data": payload or {},
        },
    }
    await event_bus.broadcast(event)
    return event["payload"]


@function_tool
async def get_process_data(query: str = "", limit: int = 12) -> dict:
    return await emit_process_data(query, limit)


@function_tool
def update_simulation_params(
    temperature: float | None = None,
    voltage: float | None = None,
    size: float | None = None,
    capacity: float | None = None,
) -> dict:
    return collect_simulation_params(
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
    )


@function_tool
async def run_simulation() -> dict:
    return await execute_simulation()


@function_tool(strict_mode=False)
async def frontend_trigger(message: str, data: dict | None = None) -> dict:
    return await emit_frontend_trigger(message, data)
