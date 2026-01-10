import re
from datetime import datetime, timedelta

from agents import function_tool


KNOWLEDGE_BASE = [
    {
        "id": "kb-process",
        "title": "로트 모니터링",
        "content": "워크플로우는 LOT 조회와 인접기종 추천/예측 시뮬레이션을 지원합니다.",
    },
    {
        "id": "kb-db",
        "title": "LOT 조회",
        "content": "get_lot_info 도구는 LOT ID로 PostgreSQL/SQLite에서 정보를 조회합니다.",
    },
    {
        "id": "kb-sim",
        "title": "인접기종 추천",
        "content": "run_lot_simulation 도구는 LOT 정보를 추천 입력으로 변환합니다.",
    },
]

STAGE_EVENT_MAP = {
    "recommendation": ["simulation_result"],
    "reference": ["lot_result", "defect_rate_chart"],
    "grid": ["design_candidates"],
    "final": ["final_briefing"],
}

STAGE_ALIASES = {
    "추천": "recommendation",
    "추천결과": "recommendation",
    "레퍼런스": "reference",
    "기준": "reference",
    "기준lot": "reference",
    "reference": "reference",
    "grid": "grid",
    "그리드": "grid",
    "최종": "final",
    "브리핑": "final",
    "final": "final",
}

from .context import current_session_id
from .config import (
    LOT_DB_COLUMNS,
    LOT_DB_CONNECTION_ID,
    LOT_DB_FILTER_OPERATOR,
    LOT_DB_LOT_COLUMN,
    LOT_DB_SCHEMA,
    LOT_DB_TABLE,
    LOT_PARAM_CAPACITY_COLUMN,
    LOT_PARAM_PRODUCTION_COLUMN,
    LOT_PARAM_SIZE_COLUMN,
    LOT_PARAM_TEMPERATURE_COLUMN,
    LOT_PARAM_VOLTAGE_COLUMN,
    LOT_DEFECT_RATE_COLUMN,
)
from .db import query_process_data
from .db_connections import (
    execute_table_query,
    execute_table_query_multi,
    execute_table_query_aggregate,
    get_schema,
)
from .events import event_bus
from .observability import emit_workflow_log
from .pipeline_store import pipeline_store
from .lot_store import lot_store
from .db_view_profile import (
    get_filter_column_names,
    load_column_catalog,
    load_view_profile,
    normalize_view_profile,
    normalize_column_catalog,
    select_catalog_table,
)
from .simulation import (
    call_prediction_api,
    call_simulation_api,
    extract_simulation_params_hybrid,
    recommendation_store,
    simulation_store,
)
from .test_simulation import (
    DEFAULT_CHART_LIMIT,
    DEFAULT_DEFECT_MAX,
    DEFAULT_GRID_LIMIT,
    DEFAULT_TARGET,
    DEFAULT_TOP_K,
    call_grid_search_api,
    filter_lot_candidates,
    grid_search_candidates,
    parse_defect_rate_bounds,
    parse_target_value,
    summarize_defect_rates,
    test_simulation_store,
)


def _parse_columns(value: str | None) -> list[str]:
    if not value:
        return []
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def _format_tool_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) > 120:
        return f"{text[:117]}..."
    return text


def _format_tool_args(**kwargs: object) -> str:
    parts = []
    for key, value in kwargs.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={_format_tool_value(value)}")
    return ", ".join(parts) if parts else "no_args"


def _get_view_profile() -> dict:
    profile = load_view_profile()
    return normalize_view_profile(profile) if isinstance(profile, dict) else {}


_COLUMN_QUERY_HINTS = {
    "불량률": "defect rate",
    "불량": "defect",
    "수율": "yield",
    "용량": "capacity",
    "전압": "voltage",
    "온도": "temperature",
    "크기": "size",
    "사이즈": "size",
    "기종": "chip prod",
    "모델": "model",
    "품번": "part number",
    "로트": "lot",
    "라인": "line",
    "공정": "process",
    "설비": "equipment",
    "날짜": "date",
    "시간": "time",
}
_COLUMN_TOKEN_RE = re.compile(r"[a-z0-9_.]+")


def _normalize_column_query(query: str | None) -> str:
    text = (query or "").lower()
    for raw, replacement in _COLUMN_QUERY_HINTS.items():
        if raw in text:
            text = text.replace(raw, replacement)
    text = re.sub(r"[^a-z0-9_.\s]+", " ", text)
    return " ".join(text.split())


def _tokenize_column_query(query: str) -> list[str]:
    if not query:
        return []
    return _COLUMN_TOKEN_RE.findall(query.lower())


def _normalize_columns_param(columns: object) -> list[str]:
    if not columns:
        return []
    if isinstance(columns, str):
        raw_items = re.split(r"[,\n]", columns)
    elif isinstance(columns, (list, tuple, set)):
        raw_items = list(columns)
    else:
        return []
    cleaned: list[str] = []
    seen = set()
    for item in raw_items:
        if item is None:
            continue
        name = str(item).strip()
        if not name or name in seen:
            continue
        cleaned.append(name)
        seen.add(name)
    return cleaned


def _merge_columns(primary: list[str], extra: list[str]) -> list[str]:
    merged: list[str] = []
    seen = set()
    for name in primary + extra:
        if not name or name in seen:
            continue
        merged.append(name)
        seen.add(name)
    return merged


def _get_column_catalog() -> dict:
    catalog = load_column_catalog()
    return normalize_column_catalog(catalog) if isinstance(catalog, dict) else {"tables": {}}


def _get_schema_columns(profile: dict) -> set[str]:
    connection_id = profile.get("connection_id")
    schema_name = profile.get("schema") or "public"
    table_name = profile.get("table")
    if not connection_id or not table_name:
        return set()
    schema = get_schema(connection_id) or {}
    schema_entry = schema.get("schemas", {}).get(schema_name)
    if not isinstance(schema_entry, dict):
        return set()
    table_entry = schema_entry.get("tables", {}).get(table_name)
    if not isinstance(table_entry, dict):
        return set()
    return {
        col.get("name")
        for col in table_entry.get("columns", [])
        if isinstance(col, dict) and col.get("name")
    }


def _score_catalog_entry(entry: dict, query: str, tokens: list[str]) -> float:
    name = str(entry.get("name") or "").lower()
    if not name:
        return 0.0
    description = str(entry.get("description") or "").lower()
    aliases = [str(alias).lower() for alias in entry.get("aliases") or [] if alias]
    score = 0.0
    if query:
        if query == name:
            score += 6
        if name in query:
            score += 3
        if any(query in alias for alias in aliases):
            score += 3
    for token in tokens:
        if token == name:
            score += 3
        if token in name:
            score += 2
        if any(token in alias for alias in aliases):
            score += 2
        if token in description:
            score += 1
    return score


def _resolve_columns_from_catalog(
    query: str, entries: list[dict], limit: int
) -> list[dict]:
    normalized_query = _normalize_column_query(query)
    tokens = _tokenize_column_query(normalized_query)
    scored: list[tuple[float, dict]] = []
    for entry in entries:
        score = _score_catalog_entry(entry, normalized_query, tokens)
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("name") or "")))
    results: list[dict] = []
    for score, entry in scored[:limit]:
        payload = {
            "column": entry.get("name"),
            "description": entry.get("description") or "",
            "score": round(score, 3),
        }
        aliases = entry.get("aliases") or []
        if aliases:
            payload["aliases"] = aliases[:5]
        unit = entry.get("unit") or ""
        if unit:
            payload["unit"] = unit
        group = entry.get("group") or ""
        if group:
            payload["group"] = group
        results.append(payload)
    return results


def _normalize_metrics_param(metrics: object) -> list[dict]:
    if not metrics:
        return []
    if isinstance(metrics, dict):
        raw_items = [metrics]
    elif isinstance(metrics, (list, tuple)):
        raw_items = list(metrics)
    else:
        return []
    normalized: list[dict] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        agg = item.get("agg") or item.get("aggregate")
        alias = item.get("alias")
        label = item.get("label") or item.get("description")
        unit = item.get("unit")
        normalized.append(
            {
                "column": str(column).strip() if column else "",
                "agg": str(agg).strip() if agg else "",
                "alias": str(alias).strip() if alias else "",
                "label": str(label).strip() if label else "",
                "unit": str(unit).strip() if unit else "",
            }
        )
    return normalized


def _normalize_group_by(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,\n]", value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        return []
    cleaned: list[str] = []
    seen = set()
    for item in raw_items:
        if item is None:
            continue
        name = str(item).strip()
        if not name or name in seen:
            continue
        cleaned.append(name)
        seen.add(name)
    return cleaned


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None


def _compute_recent_range(months: int | None) -> tuple[datetime, datetime] | None:
    if not months or months <= 0:
        return None
    end = datetime.now()
    start = end - timedelta(days=months * 30)
    return start, end


def _summarize_values(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None, "median": None}
    sorted_values = sorted(values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        median = sorted_values[mid]
    return {
        "count": count,
        "min": min(sorted_values),
        "max": max(sorted_values),
        "avg": sum(sorted_values) / count,
        "median": median,
    }


def _sanitize_metric_alias(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "metric"


def _column_meta_map(entries: list[dict]) -> dict:
    meta: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        meta[str(name)] = {
            "description": entry.get("description") or "",
            "unit": entry.get("unit") or "",
        }
    return meta


def _format_filters(filters: list[dict]) -> str:
    parts = []
    for item in filters:
        column = item.get("column")
        operator = item.get("operator")
        value = item.get("value")
        if column and value is not None:
            parts.append(f"{column}{operator}{value}")
    return ", ".join(parts) if parts else "no_filters"


def _normalize_stage(stage: str | None) -> str:
    if not stage:
        return ""
    raw = str(stage).strip().lower()
    if raw in STAGE_EVENT_MAP:
        return raw
    for key, value in STAGE_ALIASES.items():
        if key.lower() in raw:
            return value
    return ""


def _available_stages(events: dict) -> list[str]:
    available = []
    for stage, event_types in STAGE_EVENT_MAP.items():
        if any(event_type in events for event_type in event_types):
            available.append(stage)
    return available


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_defect_values(payload: dict) -> list[float]:
    values = []
    for item in payload.get("lots", []) if isinstance(payload, dict) else []:
        rate = _coerce_float(item.get("defect_rate") if isinstance(item, dict) else None)
        if rate is None:
            continue
        values.append(rate)
    return values


def _build_series_from_lots(lots: list[dict], value_unit: str | None) -> list[dict]:
    series = []
    for idx, item in enumerate(lots, start=1):
        if not isinstance(item, dict):
            continue
        rate = _coerce_float(item.get("defect_rate"))
        if rate is None:
            continue
        if value_unit == "percent":
            rate = rate * 100
        series.append(
            {
                "x": idx,
                "y": rate,
                "lot_id": item.get("lot_id"),
            }
        )
    return series


def _compute_histogram(
    values: list[float],
    bins: int,
    range_min: float | None,
    range_max: float | None,
    normalize: str,
    value_unit: str | None,
) -> dict:
    if not values:
        return {"bins": [], "normalize": normalize, "value_unit": value_unit}

    values_min = min(values)
    values_max = max(values)
    lower = values_min if range_min is None else range_min
    upper = values_max if range_max is None else range_max
    if lower == upper:
        upper = lower + 1e-6
    bins = max(1, int(bins))

    width = (upper - lower) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        if value < lower or value > upper:
            continue
        idx = int((value - lower) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    total = max(1, sum(counts))
    bin_entries = []
    for idx, count in enumerate(counts):
        start = lower + idx * width
        end = start + width
        entry = {"start": start, "end": end, "count": count}
        if normalize == "percent":
            entry["value"] = round((count / total) * 100, 3)
        elif normalize == "ratio":
            entry["value"] = round(count / total, 6)
        bin_entries.append(entry)

    return {
        "bins": bin_entries,
        "normalize": normalize,
        "value_unit": value_unit,
    }


async def _log_route(route: str, detail: str) -> None:
    await emit_workflow_log("STATE", f"{route}: {detail}")


async def _log_tool_call(name: str, **kwargs: object) -> None:
    await emit_workflow_log("TOOL_CALL", f"{name}({_format_tool_args(**kwargs)})")


async def _log_tool_result(name: str, summary: str) -> None:
    await emit_workflow_log("TOOL_RESULT", f"{name}: {summary}")


def _normalize_row(row: dict) -> dict:
    normalized = {}
    for key, value in row.items():
        normalized[key] = value
        normalized[str(key).lower()] = value
    return normalized


def _extract_simulation_params_from_lot(row: dict) -> dict:
    normalized = _normalize_row(row)
    mapping = {
        "temperature": LOT_PARAM_TEMPERATURE_COLUMN,
        "voltage": LOT_PARAM_VOLTAGE_COLUMN,
        "size": LOT_PARAM_SIZE_COLUMN,
        "capacity": LOT_PARAM_CAPACITY_COLUMN,
        "production_mode": LOT_PARAM_PRODUCTION_COLUMN,
    }
    params: dict[str, object] = {}
    for key, column in mapping.items():
        if not column:
            continue
        value = normalized.get(column) or normalized.get(column.lower())
        if value is not None:
            params[key] = value
    return params


def _query_lot_rows(lot_id: str, limit: int) -> dict:
    profile = _get_view_profile()
    connection_id = profile.get("connection_id") or LOT_DB_CONNECTION_ID
    schema_name = profile.get("schema") or LOT_DB_SCHEMA
    table_name = profile.get("table") or LOT_DB_TABLE
    if connection_id and table_name:
        columns = _parse_columns(LOT_DB_COLUMNS)
        result = execute_table_query(
            connection_id=connection_id,
            schema_name=schema_name or "public",
            table_name=table_name,
            columns=columns,
            filter_column=LOT_DB_LOT_COLUMN,
            filter_operator=LOT_DB_FILTER_OPERATOR or "=",
            filter_value=lot_id,
            limit=limit,
        )
        return {
            "lot_id": lot_id,
            "columns": result.get("columns", []),
            "rows": result.get("rows", []),
            "source": "postgresql",
        }
    if LOT_DB_CONNECTION_ID or LOT_DB_TABLE or profile.get("connection_id") or profile.get("table"):
        raise RuntimeError("LOT DB 설정이 불완전합니다. 연결/테이블 정보를 확인하세요.")

    rows = query_process_data(lot_id, limit)
    if rows:
        for row in rows:
            row.setdefault("lot_id", lot_id)
    return {
        "lot_id": lot_id,
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
        "source": "sqlite",
    }


async def emit_lot_info(lot_id: str = "", limit: int = 12) -> dict:
    if not lot_id:
        await emit_frontend_trigger("LOT ID가 필요합니다. 예: LOT123")
        return {"error": "LOT ID가 필요합니다.", "rows": []}
    try:
        payload = _query_lot_rows(lot_id, limit)
    except Exception as error:
        message = f"LOT 조회 실패: {error}"
        await event_bus.broadcast(
            {
                "type": "frontend_trigger",
                "payload": {"message": message, "data": {"lot_id": lot_id}},
            }
        )
        return {"error": str(error), "rows": [], "lot_id": lot_id}

    rows = payload.get("rows", [])
    if rows:
        lot_store.set(
            current_session_id.get(),
            {"lot_id": lot_id, "row": rows[0], "rows": rows},
        )
    await event_bus.broadcast({"type": "lot_result", "payload": payload})
    pipeline_store.set_event(current_session_id.get(), "lot_result", payload)
    if rows:
        message = f"LOT {lot_id} 조회 결과를 표시했습니다."
    else:
        message = f"LOT {lot_id}에 해당하는 데이터가 없습니다."
    await emit_frontend_trigger(message, payload)
    return payload


async def emit_process_data(query: str = "", limit: int = 12) -> dict:
    return await emit_lot_info(query, limit)


def collect_simulation_params(
    temperature: str | None = None,
    voltage: float | None = None,
    size: str | None = None,
    capacity: float | None = None,
    production_mode: str | None = None,
    chip_prod_id: str | None = None,
    clear_keys: list[str] | None = None,
    extra_missing: list[str] | None = None,
) -> dict:
    session_id = current_session_id.get()
    if clear_keys:
        simulation_store.remove_keys(session_id, clear_keys)
    params = simulation_store.update(
        session_id,
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
        production_mode=production_mode,
        chip_prod_id=chip_prod_id,
    )
    missing = simulation_store.missing(session_id)
    if extra_missing:
        for key in extra_missing:
            if key not in missing:
                missing.append(key)
    return {"params": params, "missing": missing}


async def execute_simulation() -> dict:
    session_id = current_session_id.get()
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    if missing:
        return {
            "status": "missing",
            "missing": missing,
            "message": "추천 입력값이 부족합니다.",
        }

    result = await call_simulation_api(params)
    if isinstance(result, dict):
        if (
            "recommended_chip_prod_id" not in result
            and result.get("recommended_model") is not None
        ):
            result["recommended_chip_prod_id"] = result.get("recommended_model")
            result.pop("recommended_model", None)
    recommendation_store.set(
        session_id, {"input_params": params, "awaiting_prediction": True, **result}
    )
    await event_bus.broadcast(
        {
            "type": "simulation_result",
            "payload": {"params": params, "result": result},
        }
    )
    pipeline_store.set_event(
        session_id,
        "simulation_result",
        {"params": params, "result": result},
    )
    await emit_frontend_trigger(
        "인접기종 추천 결과가 도착했습니다.",
        {"params": params, "result": result},
    )
    return {
        "status": "ok",
        "result": result,
        "message": "추천 실행을 완료했습니다.",
    }

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


async def emit_simulation_form(params: dict, missing: list[str]) -> dict:
    session_id = current_session_id.get()
    simulation_store.activate(session_id)
    event = {"type": "simulation_form", "payload": {"params": params, "missing": missing}}
    await event_bus.broadcast(event)
    pipeline_store.set_event(session_id, "simulation_form", event["payload"])
    return event["payload"]


async def show_simulation_stage_impl(stage: str = "") -> dict:
    """Re-display a stored simulation stage UI on the event panel."""
    await _log_tool_call("show_simulation_stage", stage=stage)
    session_id = current_session_id.get()
    events = pipeline_store.get_events(session_id)
    if not events:
        message = "저장된 단계 결과가 없습니다. 먼저 시뮬레이션을 실행해 주세요."
        await _log_tool_result("show_simulation_stage", "status=missing events=0")
        return {"status": "missing", "message": message, "available": []}

    stage_key = _normalize_stage(stage)
    if not stage_key:
        available = _available_stages(events)
        message = "어떤 단계 화면을 보여드릴까요? (추천/레퍼런스/그리드/최종)"
        await _log_tool_result(
            "show_simulation_stage", f"status=missing stage=unknown available={len(available)}"
        )
        return {"status": "missing", "message": message, "available": available}

    event_types = STAGE_EVENT_MAP.get(stage_key, [])
    sent = []
    for event_type in event_types:
        payload = events.get(event_type)
        if payload:
            await event_bus.broadcast({"type": event_type, "payload": payload})
            sent.append(event_type)

    if not sent:
        available = _available_stages(events)
        message = "해당 단계의 저장된 결과가 없습니다."
        await _log_tool_result(
            "show_simulation_stage", f"status=missing stage={stage_key} available={len(available)}"
        )
        return {"status": "missing", "message": message, "available": available}

    await _log_tool_result(
        "show_simulation_stage", f"status=ok stage={stage_key} sent={len(sent)}"
    )
    return {"status": "ok", "stage": stage_key, "sent": sent, "available": _available_stages(events)}


@function_tool
async def show_simulation_stage(stage: str = "") -> dict:
    """Re-display a stored simulation stage UI on the event panel."""
    return await show_simulation_stage_impl(stage)


async def get_simulation_progress_impl() -> dict:
    """Show stored pipeline progress in the event log."""
    await _log_tool_call("get_simulation_progress")
    session_id = current_session_id.get()
    history = pipeline_store.get_status_history(session_id)
    if not history:
        message = "No progress history is available."
        await _log_tool_result("get_simulation_progress", "status=missing history=0")
        return {"status": "missing", "message": message, "history": []}

    for entry in history[-12:]:
        stage = entry.get("stage", "-")
        text = entry.get("message", "")
        await emit_workflow_log("PROGRESS", f"{stage}: {text}")

    await _log_tool_result(
        "get_simulation_progress", f"status=ok history={len(history)}"
    )
    return {"status": "ok", "history": history[-12:]}


@function_tool
async def get_simulation_progress() -> dict:
    """Show stored pipeline progress in the event log."""
    return await get_simulation_progress_impl()


async def reset_simulation_state_impl(reason: str | None = None) -> dict:
    """Reset simulation/session state and clear cached UI events."""
    await _log_tool_call("reset_simulation_state", reason=reason)
    session_id = current_session_id.get()
    simulation_store.clear(session_id)
    recommendation_store.clear(session_id)
    pipeline_store.clear(session_id)
    test_simulation_store.clear(session_id)
    await emit_frontend_trigger("Simulation state reset.", {"reason": reason or ""})
    await _log_tool_result("reset_simulation_state", "status=ok")
    return {"status": "ok", "message": "reset"}


@function_tool
async def reset_simulation_state(reason: str | None = None) -> dict:
    """Reset simulation/session state and clear cached UI events."""
    return await reset_simulation_state_impl(reason=reason)


async def apply_chart_config_impl(
    chart_type: str | None = None,
    bins: int | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    normalize: str | None = None,
    value_unit: str | None = None,
    reset: bool = False,
    note: str | None = None,
) -> dict:
    """Update the defect-rate chart (histogram only)."""
    await _log_route("chart", "apply_chart_config")
    await _log_tool_call(
        "apply_chart_config",
        chart_type=chart_type,
        bins=bins,
        range_min=range_min,
        range_max=range_max,
        normalize=normalize,
        value_unit=value_unit,
        reset=reset,
        note=note,
    )
    session_id = current_session_id.get()
    payload = pipeline_store.get_event(session_id, "defect_rate_chart")
    if not isinstance(payload, dict):
        message = "아직 불량률 차트 데이터가 없습니다. 시뮬레이션을 먼저 실행해 주세요."
        await _log_tool_result("apply_chart_config", "status=missing chart=none")
        return {"status": "missing", "message": message}

    chart_type = (chart_type or "histogram").lower()
    supported_types = {"histogram", "bar", "line", "scatter", "area"}
    if chart_type not in supported_types:
        message = "Unsupported chart type."
        await _log_tool_result("apply_chart_config", f"status=unsupported type={chart_type}")
        return {"status": "unsupported", "message": message, "chart_type": chart_type}

    lots = payload.get("lots", []) if isinstance(payload, dict) else []
    values = _extract_defect_values(payload)
    if not values:
        message = "No defect rate values available."
        await _log_tool_result("apply_chart_config", "status=missing values=0")
        return {"status": "missing", "message": message}

    default_bins = max(6, min(18, int(len(values) ** 0.5)))
    if reset:
        bins = default_bins
        range_min = None
        range_max = None
        normalize = "count"
        value_unit = None

    bins = bins if isinstance(bins, int) and bins > 0 else default_bins
    normalize = (normalize or "count").lower()
    if normalize not in ("count", "percent", "ratio"):
        normalize = "count"
    value_unit = (value_unit or "").lower() or None

    if value_unit == "percent":
        values = [value * 100 for value in values]

    histogram = None
    if chart_type == "histogram":
        histogram = _compute_histogram(values, bins, range_min, range_max, normalize, value_unit)

    series = None
    if chart_type in {"line", "scatter", "area"}:
        series = _build_series_from_lots(lots, value_unit)
    config = {
        "chart_type": chart_type,
        "bins": bins,
        "range_min": range_min,
        "range_max": range_max,
        "normalize": normalize,
        "value_unit": value_unit,
    }
    updated_payload = dict(payload)
    updated_payload["chart_type"] = chart_type
    if histogram is not None:
        updated_payload["histogram"] = histogram
    else:
        updated_payload.pop("histogram", None)
    if series is not None:
        updated_payload["series"] = series
    updated_payload["config"] = config
    await event_bus.broadcast({"type": "defect_rate_chart", "payload": updated_payload})
    pipeline_store.set_event(session_id, "defect_rate_chart", updated_payload)
    pipeline_store.update(session_id, chart_config=config)
    await _log_tool_result("apply_chart_config", f"status=ok bins={bins} normalize={normalize}")
    return {"status": "ok", "chart_type": chart_type, "config": config}


@function_tool(strict_mode=False)
async def apply_chart_config(
    chart_type: str | None = None,
    bins: int | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    normalize: str | None = None,
    value_unit: str | None = None,
    reset: bool = False,
    note: str | None = None,
) -> dict:
    """Update the defect-rate chart (histogram only)."""
    return await apply_chart_config_impl(
        chart_type=chart_type,
        bins=bins,
        range_min=range_min,
        range_max=range_max,
        normalize=normalize,
        value_unit=value_unit,
        reset=reset,
        note=note,
    )


def _search_knowledge_base(query: str, top_k: int) -> list[dict]:
    if not query:
        return KNOWLEDGE_BASE[:top_k]
    lowered = query.lower()
    scored = []
    for item in KNOWLEDGE_BASE:
        content = f"{item['title']} {item['content']}".lower()
        score = content.count(lowered)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


@function_tool
async def get_lot_info(lot_id: str = "", limit: int = 12) -> dict:
    """Fetch LOT/process data by lot_id and emit it to the UI panel."""
    await _log_route("db_lookup", "get_lot_info")
    await _log_tool_call("get_lot_info", lot_id=lot_id, limit=limit)
    result = await emit_lot_info(lot_id, limit)
    if isinstance(result, dict) and result.get("error"):
        await _log_tool_result("get_lot_info", f"error={result.get('error')}")
    else:
        rows = result.get("rows", []) if isinstance(result, dict) else []
        await _log_tool_result("get_lot_info", f"rows={len(rows)}")
    return result


@function_tool
async def get_process_data(query: str = "", limit: int = 12) -> dict:
    """Search process records by query (line/status) and emit results to the UI."""
    await _log_route("db_lookup", "get_process_data")
    await _log_tool_call("get_process_data", query=query, limit=limit)
    result = await emit_process_data(query, limit)
    rows = result.get("rows", []) if isinstance(result, dict) else []
    await _log_tool_result("get_process_data", f"rows={len(rows)}")
    return result


@function_tool
async def resolve_view_columns(query: str = "", limit: int = 5) -> dict:
    """Resolve view column candidates from a natural-language query."""
    await _log_route("db_view", "resolve_view_columns")
    await _log_tool_call("resolve_view_columns", query=query, limit=limit)
    profile = _get_view_profile()
    connection_id = profile.get("connection_id")
    schema_name = profile.get("schema") or "public"
    table_name = profile.get("table")
    if not connection_id or not table_name:
        message = "DB view profile is not configured."
        await _log_tool_result("resolve_view_columns", "status=error missing=profile")
        return {"status": "error", "error": message}

    catalog = _get_column_catalog()
    catalog_table = select_catalog_table(
        catalog, table_name, profile.get("column_catalog_table")
    )
    entries = catalog_table.get("columns", []) if isinstance(catalog_table, dict) else []
    if not entries:
        message = "No column catalog entries are configured."
        await _log_tool_result("resolve_view_columns", "status=missing catalog=0")
        return {"status": "missing", "message": message, "candidates": []}

    allowed_select = set(profile.get("selectable_columns") or [])
    if allowed_select:
        entries = [
            entry for entry in entries if entry.get("name") in allowed_select
        ]

    available_columns = _get_schema_columns(profile)
    if available_columns:
        entries = [
            entry for entry in entries if entry.get("name") in available_columns
        ]

    if not query:
        defaults = catalog_table.get("default_columns") or []
        message = "Provide a column query to resolve."
        await _log_tool_result("resolve_view_columns", "status=missing query=empty")
        return {
            "status": "missing",
            "message": message,
            "defaults": defaults,
            "candidates": [],
        }

    result_limit = limit if isinstance(limit, int) and limit > 0 else 5
    result_limit = max(1, min(int(result_limit), 20))
    candidates = _resolve_columns_from_catalog(query, entries, result_limit)
    if not candidates:
        message = "No columns matched the query."
        await _log_tool_result("resolve_view_columns", "status=missing matches=0")
        return {"status": "missing", "message": message, "candidates": []}

    await _log_tool_result(
        "resolve_view_columns", f"status=ok candidates={len(candidates)}"
    )
    return {
        "status": "ok",
        "table": f"{schema_name}.{table_name}",
        "candidates": candidates,
    }


@function_tool(strict_mode=False)
async def query_view_metrics(
    metrics: list[dict] | dict | None = None,
    filters: list[dict] | None = None,
    time_column: str | None = None,
    recent_months: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    time_bucket: str | None = None,
    group_by: list[str] | None = None,
    chart_type: str | None = None,
    limit: int | None = None,
) -> dict:
    """Query aggregated metrics from the configured DB view."""
    await _log_route("db_view", "query_view_metrics")
    await _log_tool_call(
        "query_view_metrics",
        metrics=metrics,
        filters=_format_filters(filters or []),
        time_column=time_column,
        recent_months=recent_months,
        date_from=date_from,
        date_to=date_to,
        time_bucket=time_bucket,
        group_by=group_by,
        chart_type=chart_type,
        limit=limit,
    )
    profile = _get_view_profile()
    connection_id = profile.get("connection_id")
    schema_name = profile.get("schema") or "public"
    table_name = profile.get("table")
    allowed_filters = set(get_filter_column_names(profile))
    if not connection_id or not table_name:
        message = "DB view profile is not configured."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=error missing=profile")
        return {"status": "error", "error": message, "missing": ["profile"]}
    if not allowed_filters and filters:
        message = "No allowed filter columns are configured."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=error missing=filters")
        return {"status": "error", "error": message, "missing": ["filters"]}

    metric_specs = _normalize_metrics_param(metrics)
    if not metric_specs:
        message = "집계할 컬럼이 필요합니다."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=missing metrics=0")
        return {"status": "missing", "missing": ["metrics"], "message": message}
    for metric in metric_specs:
        if not metric.get("agg"):
            metric["agg"] = "avg"
        else:
            metric["agg"] = metric["agg"].lower()
        if not metric.get("alias"):
            metric["alias"] = _sanitize_metric_alias(
                f"{metric['agg']}_{metric.get('column') or 'all'}"
            )
        else:
            metric["alias"] = _sanitize_metric_alias(metric["alias"])

    catalog = _get_column_catalog()
    catalog_table = select_catalog_table(
        catalog, table_name, profile.get("column_catalog_table")
    )
    catalog_entries = (
        catalog_table.get("columns", []) if isinstance(catalog_table, dict) else []
    )
    meta_map = _column_meta_map(catalog_entries)

    selectable_columns = set(profile.get("selectable_columns") or [])
    catalog_columns = {entry.get("name") for entry in catalog_entries if entry.get("name")}
    available_columns = _get_schema_columns(profile)
    allowed_metrics = set(selectable_columns or catalog_columns)

    invalid_metrics = []
    for metric in metric_specs:
        column = metric.get("column")
        if column:
            if allowed_metrics and column not in allowed_metrics:
                invalid_metrics.append(column)
            if available_columns and column not in available_columns:
                invalid_metrics.append(column)
            meta = meta_map.get(column, {})
            if not metric.get("label") and meta.get("description"):
                metric["label"] = meta["description"]
            if not metric.get("unit") and meta.get("unit"):
                metric["unit"] = meta["unit"]
        else:
            invalid_metrics.append("(empty)")

    if invalid_metrics:
        allowed_hint = (
            f"Allowed: {', '.join(sorted(allowed_metrics))}"
            if allowed_metrics
            else "Allowed: (not configured)"
        )
        message = (
            "Invalid metric columns: "
            f"{', '.join(sorted(set(invalid_metrics)))}. {allowed_hint}"
        )
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=error invalid=metrics")
        return {"status": "error", "error": message, "invalid": sorted(set(invalid_metrics))}

    normalized_filters: list[dict] = []
    invalid_filters = []
    for item in filters or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        value = item.get("value")
        if not column or value is None or value == "":
            continue
        if allowed_filters and column not in allowed_filters:
            invalid_filters.append(column)
            continue
        operator = item.get("operator") or "ilike"
        normalized_filters.append(
            {"column": column, "operator": operator, "value": value}
        )

    if invalid_filters:
        message = (
            "Invalid filter columns: "
            f"{', '.join(sorted(set(invalid_filters)))}. "
            f"Allowed: {', '.join(sorted(allowed_filters))}"
        )
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=error invalid=filters")
        return {"status": "error", "error": message, "invalid": sorted(set(invalid_filters))}

    time_column = (time_column or profile.get("time_column") or "").strip()
    time_bucket = (time_bucket or "").strip().lower()
    time_range_applied = False
    date_from_dt = _parse_iso_datetime(date_from)
    date_to_dt = _parse_iso_datetime(date_to)
    if date_from_dt or date_to_dt:
        if not time_column:
            message = "시간 필터 컬럼이 설정되지 않았습니다."
            await emit_frontend_trigger(message)
            await _log_tool_result("query_view_metrics", "status=error missing=time_column")
            return {"status": "error", "error": message, "missing": ["time_column"]}
        if date_from_dt and date_to_dt:
            normalized_filters.append(
                {"column": time_column, "operator": "between", "value": [date_from_dt, date_to_dt]}
            )
        elif date_from_dt:
            normalized_filters.append(
                {"column": time_column, "operator": ">=", "value": date_from_dt}
            )
        else:
            normalized_filters.append(
                {"column": time_column, "operator": "<=", "value": date_to_dt}
            )
        time_range_applied = True

    recent_months_value = None
    if recent_months is not None:
        try:
            recent_months_value = int(recent_months)
        except (TypeError, ValueError):
            recent_months_value = None

    if not time_range_applied and recent_months_value:
        if not time_column:
            message = "시간 필터 컬럼이 설정되지 않았습니다."
            await emit_frontend_trigger(message)
            await _log_tool_result("query_view_metrics", "status=error missing=time_column")
            return {"status": "error", "error": message, "missing": ["time_column"]}
        recent_range = _compute_recent_range(recent_months_value)
        if recent_range:
            start_dt, end_dt = recent_range
            normalized_filters.append(
                {"column": time_column, "operator": "between", "value": [start_dt, end_dt]}
            )
            time_range_applied = True

    if not normalized_filters:
        message = "필터 또는 기간 조건이 필요합니다."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=missing filters=0")
        return {"status": "missing", "missing": sorted(allowed_filters), "message": message}

    group_columns = _normalize_group_by(group_by)
    invalid_groups = []
    for column in group_columns:
        if allowed_metrics and column not in allowed_metrics:
            invalid_groups.append(column)
        if available_columns and column not in available_columns:
            invalid_groups.append(column)
    if invalid_groups:
        message = "Invalid group columns: " + ", ".join(sorted(set(invalid_groups)))
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=error invalid=group_by")
        return {"status": "error", "error": message, "invalid": sorted(set(invalid_groups))}

    bucket_payload = None
    if time_bucket:
        if not time_column:
            message = "시간 필터 컬럼이 설정되지 않았습니다."
            await emit_frontend_trigger(message)
            await _log_tool_result("query_view_metrics", "status=error missing=time_column")
            return {"status": "error", "error": message, "missing": ["time_column"]}
        bucket_payload = {"column": time_column, "unit": time_bucket, "alias": "bucket"}

    row_limit = limit if isinstance(limit, int) and limit > 0 else 200
    row_limit = max(1, min(int(row_limit), 500))
    try:
        result = execute_table_query_aggregate(
            connection_id=connection_id,
            schema_name=schema_name,
            table_name=table_name,
            metrics=metric_specs,
            filters=normalized_filters,
            group_by=group_columns,
            time_bucket=bucket_payload,
            limit=row_limit,
        )
    except ValueError as exc:
        message = str(exc) or "DB query failed."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=error query=failed")
        return {"status": "error", "error": message}

    rows = result.get("rows", [])
    if not rows:
        message = "조건에 맞는 결과가 없습니다."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_metrics", "status=missing rows=0")
        return {"status": "missing", "message": message, "rows": []}

    primary_metric = metric_specs[0]
    metric_alias = primary_metric.get("alias") or "metric"
    metric_label = primary_metric.get("label") or primary_metric.get("column") or metric_alias
    metric_unit = primary_metric.get("unit") or ""

    chart_type = (chart_type or "").strip().lower()
    if not chart_type and (bucket_payload or group_columns):
        chart_type = "line" if bucket_payload else "bar"
    supported_charts = {"line", "area", "scatter", "bar", "histogram"}
    if chart_type and chart_type not in supported_charts:
        chart_type = "line"

    chart_lots: list[dict] = []
    values: list[float] = []
    if bucket_payload or group_columns:
        for row in rows:
            bucket_value = row.get("bucket") if bucket_payload else None
            labels = []
            if bucket_value is not None:
                labels.append(str(bucket_value))
            for column in group_columns:
                if column in row:
                    labels.append(str(row.get(column)))
            label = " ".join(labels) if labels else "-"
            value = row.get(metric_alias)
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            chart_lots.append(
                {"lot_id": label, "defect_rate": value_float, "label": label}
            )
            values.append(value_float)

    stats = _summarize_values(values)
    stats["value_unit"] = metric_unit

    if chart_type and chart_lots:
        payload = {
            "lots": chart_lots,
            "stats": stats,
            "chart_type": chart_type,
            "metric_label": metric_label,
            "value_unit": metric_unit,
            "filters": normalized_filters,
        }
        await event_bus.broadcast({"type": "defect_rate_chart", "payload": payload})
        pipeline_store.set_event(current_session_id.get(), "defect_rate_chart", payload)

    await _log_tool_result(
        "query_view_metrics",
        f"rows={len(rows)} chart={bool(chart_type and chart_lots)}",
    )
    response = {
        "status": "ok",
        "table": f"{schema_name}.{table_name}",
        "rows": rows,
        "metrics": metric_specs,
        "filters": normalized_filters,
        "time_bucket": time_bucket,
        "group_by": group_columns,
        "stats": stats,
        "chart_type": chart_type or None,
    }
    pipeline_store.set_event(current_session_id.get(), "db_result", response)
    return response


@function_tool(strict_mode=False)
async def query_view_table(
    filters: list[dict] | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Query the configured DB view using allowed filter columns and optional select columns."""
    await _log_route("db_view", "query_view_table")
    await _log_tool_call(
        "query_view_table",
        filters=_format_filters(filters or []),
        columns=_normalize_columns_param(columns),
        limit=limit,
    )
    profile = _get_view_profile()
    connection_id = profile.get("connection_id")
    schema_name = profile.get("schema")
    table_name = profile.get("table")
    allowed = set(get_filter_column_names(profile))
    if not connection_id or not table_name:
        message = "DB view profile is not configured."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_table", "status=error missing=profile")
        return {"status": "error", "error": message, "missing": ["profile"]}
    if not allowed:
        message = "No allowed filter columns are configured."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_table", "status=error missing=filters")
        return {"status": "error", "error": message, "missing": ["filters"]}
    if not filters:
        message = f"Provide filters using: {', '.join(sorted(allowed))}"
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_table", "status=missing missing=filters")
        return {"status": "missing", "missing": sorted(allowed), "message": message}

    normalized_filters: list[dict] = []
    invalid = []
    for item in filters or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        value = item.get("value")
        if not column or value is None or value == "":
            continue
        if column not in allowed:
            invalid.append(column)
            continue
        operator = item.get("operator") or "ilike"
        normalized_filters.append(
            {"column": column, "operator": operator, "value": value}
        )

    if invalid:
        message = (
            "Invalid filter columns: "
            f"{', '.join(sorted(set(invalid)))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_table", "status=error invalid=filters")
        return {"status": "error", "error": message, "invalid": sorted(set(invalid))}
    if not normalized_filters:
        message = f"Provide at least one valid filter from: {', '.join(sorted(allowed))}"
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_table", "status=missing missing=filters")
        return {"status": "missing", "missing": sorted(allowed), "message": message}

    select_columns = _normalize_columns_param(columns)
    default_columns = list(profile.get("result_columns") or [])
    selectable_columns = set(profile.get("selectable_columns") or [])
    catalog = _get_column_catalog()
    catalog_table = select_catalog_table(
        catalog, table_name, profile.get("column_catalog_table")
    )
    catalog_columns = {
        entry.get("name")
        for entry in catalog_table.get("columns", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    available_columns = _get_schema_columns(profile)

    if not default_columns:
        catalog_defaults = catalog_table.get("default_columns") or []
        if catalog_defaults and (
            not available_columns
            or all(col in available_columns for col in catalog_defaults)
        ):
            default_columns = list(catalog_defaults)
    if selectable_columns and default_columns:
        default_columns = [col for col in default_columns if col in selectable_columns]
    if available_columns and default_columns:
        default_columns = [col for col in default_columns if col in available_columns]

    allowed_select = set(selectable_columns or catalog_columns)
    if select_columns:
        invalid = set()
        if allowed_select:
            invalid.update([col for col in select_columns if col not in allowed_select])
        if available_columns:
            invalid.update([col for col in select_columns if col not in available_columns])
        if invalid:
            allowed_hint = (
                f"Allowed: {', '.join(sorted(allowed_select))}"
                if allowed_select
                else "Allowed: (not configured)"
            )
            message = (
                "Invalid select columns: "
                f"{', '.join(sorted(invalid))}. {allowed_hint}"
            )
            await emit_frontend_trigger(message)
            await _log_tool_result("query_view_table", "status=error invalid=columns")
            return {"status": "error", "error": message, "invalid": sorted(invalid)}

    final_columns = (
        _merge_columns(default_columns, select_columns)
        if select_columns
        else list(default_columns)
    )
    if select_columns and not final_columns:
        message = "No valid columns were selected."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_table", "status=missing columns=0")
        return {"status": "missing", "message": message, "missing": ["columns"]}

    row_limit = limit if isinstance(limit, int) and limit > 0 else profile.get("limit", 5)
    row_limit = max(1, min(int(row_limit), 50))
    try:
        result = execute_table_query_multi(
            connection_id=connection_id,
            schema_name=schema_name or "public",
            table_name=table_name,
            columns=final_columns,
            filters=normalized_filters,
            limit=row_limit,
        )
    except ValueError as exc:
        message = str(exc) or "DB query failed."
        await emit_frontend_trigger(message)
        await _log_tool_result("query_view_table", "status=error query=failed")
        return {"status": "error", "error": message}
    payload = {
        "query": _format_filters(normalized_filters),
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "source": "postgresql",
        "limit": row_limit,
    }
    await event_bus.broadcast({"type": "db_result", "payload": payload})
    pipeline_store.set_event(current_session_id.get(), "db_result", payload)
    await _log_tool_result(
        "query_view_table",
        f"rows={len(payload.get('rows', []))} limit={row_limit}",
    )
    return {"status": "ok", **payload}


@function_tool
async def update_simulation_params(
    message: str | None = None,
    temperature: float | None = None,
    voltage: float | None = None,
    size: str | None = None,
    capacity: float | None = None,
    production_mode: str | None = None,
    chip_prod_id: str | None = None,
) -> dict:
    """Update simulation params from user input and emit missing fields to the UI."""
    await _log_tool_call(
        "update_simulation_params",
        message=message,
        temperature=temperature,
        voltage=voltage,
        size=size,
        capacity=capacity,
        production_mode=production_mode,
        chip_prod_id=chip_prod_id,
    )
    parsed, conflicts = await extract_simulation_params_hybrid(message) if message else ({}, [])
    if temperature is not None:
        parsed["temperature"] = temperature
        if "temperature" in conflicts:
            conflicts.remove("temperature")
    if voltage is not None:
        parsed["voltage"] = voltage
        if "voltage" in conflicts:
            conflicts.remove("voltage")
    if size is not None:
        parsed["size"] = size
        if "size" in conflicts:
            conflicts.remove("size")
    if capacity is not None:
        parsed["capacity"] = capacity
        if "capacity" in conflicts:
            conflicts.remove("capacity")
    if production_mode is not None:
        parsed["production_mode"] = production_mode
        if "production_mode" in conflicts:
            conflicts.remove("production_mode")
    if chip_prod_id is not None:
        parsed["chip_prod_id"] = chip_prod_id
        if "chip_prod_id" in conflicts:
            conflicts.remove("chip_prod_id")

    for key in conflicts:
        parsed.pop(key, None)
    result = collect_simulation_params(
        **parsed,
        clear_keys=conflicts,
        extra_missing=conflicts,
    )
    await emit_simulation_form(result["params"], result["missing"])
    missing = result.get("missing", [])
    result["message"] = "추천 입력값을 업데이트했습니다."
    await _log_tool_result(
        "update_simulation_params",
        f"missing={len(missing)} params={len(result.get('params', {}))}",
    )
    return result


@function_tool
async def run_simulation() -> dict:
    """Open the recommendation form, then run if params are complete."""
    await _log_route("recommendation", "run_simulation")
    await _log_tool_call("run_simulation")
    session_id = current_session_id.get()
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    await emit_simulation_form(params, missing)
    if missing:
        await _log_tool_result("run_simulation", f"status=missing missing={len(missing)}")
        return {
            "status": "missing",
            "missing": missing,
            "params": params,
            "message": "추천 입력값이 부족합니다.",
        }
    result = await execute_simulation()
    summary = f"status={result.get('status')}"
    inner = result.get("result") if isinstance(result, dict) else None
    if isinstance(inner, dict) and inner.get("recommended_chip_prod_id"):
        summary = f"{summary}, chip_prod_id={inner.get('recommended_chip_prod_id')}"
    await _log_tool_result("run_simulation", summary)
    return result


@function_tool
async def run_prediction_simulation() -> dict:
    """Run prediction simulation based on the latest recommendation."""
    await _log_route("prediction", "run_prediction_simulation")
    await _log_tool_call("run_prediction_simulation")
    session_id = current_session_id.get()
    recommendation = recommendation_store.get(session_id)
    params = recommendation.get("params") if isinstance(recommendation, dict) else None
    if not params:
        await emit_frontend_trigger("추천 결과가 없어 예측 시뮬레이션을 진행할 수 없습니다.")
        await _log_tool_result(
            "run_prediction_simulation",
            "status=missing missing=recommendation",
        )
        return {
            "status": "missing",
            "missing": ["recommendation"],
            "message": "추천 결과가 없습니다.",
        }

    param_keys = [
        key for key in params.keys() if isinstance(key, str) and key.startswith("param")
    ]
    if not param_keys:
        message = "예측 입력 파라미터가 부족합니다."
        await emit_frontend_trigger(message)
        await _log_tool_result(
            "run_prediction_simulation",
            "status=missing missing_params=0",
        )
        return {
            "status": "missing",
            "missing": [],
            "message": message,
            "recommendation": recommendation,
        }

    result = await call_prediction_api(params)
    payload = {"recommendation": recommendation, "result": result}
    await event_bus.broadcast({"type": "prediction_result", "payload": payload})
    pipeline_store.set_event(current_session_id.get(), "prediction_result", payload)
    await emit_frontend_trigger(
        "예측 시뮬레이션 결과가 도착했습니다.",
        payload,
    )
    recommendation_store.mark_prediction_done(session_id)
    await _log_tool_result("run_prediction_simulation", "status=ok")
    return {
        "status": "ok",
        "message": "예측 실행을 완료했습니다.",
        **payload,
    }


@function_tool
async def filter_lots_by_defect_rate(
    message: str | None = None,
    defect_rate_min: float | None = None,
    defect_rate_max: float | None = None,
    limit: int | None = None,
) -> dict:
    """Filter candidate LOTs by defect rate and emit a chart to the UI."""
    await _log_route("test_simulation", "filter_lots_by_defect_rate")
    await _log_tool_call(
        "filter_lots_by_defect_rate",
        message=message,
        defect_rate_min=defect_rate_min,
        defect_rate_max=defect_rate_max,
        limit=limit,
    )
    session_id = current_session_id.get()

    parsed_min, parsed_max = parse_defect_rate_bounds(message or "")
    min_rate = defect_rate_min if defect_rate_min is not None else parsed_min
    max_rate = defect_rate_max if defect_rate_max is not None else parsed_max
    if min_rate is None and max_rate is None:
        max_rate = DEFAULT_DEFECT_MAX

    row_limit = limit if isinstance(limit, int) and limit > 0 else 50
    row_limit = max(1, min(row_limit, 200))

    candidates = []
    if LOT_DB_CONNECTION_ID and LOT_DB_TABLE and LOT_DEFECT_RATE_COLUMN:
        try:
            filters = []
            if min_rate is not None:
                filters.append(
                    {"column": LOT_DEFECT_RATE_COLUMN, "operator": ">=", "value": min_rate}
                )
            if max_rate is not None:
                filters.append(
                    {"column": LOT_DEFECT_RATE_COLUMN, "operator": "<=", "value": max_rate}
                )
            columns = _parse_columns(LOT_DB_COLUMNS)
            if LOT_DB_LOT_COLUMN and LOT_DB_LOT_COLUMN not in columns:
                columns.append(LOT_DB_LOT_COLUMN)
            if LOT_DEFECT_RATE_COLUMN not in columns:
                columns.append(LOT_DEFECT_RATE_COLUMN)
            result = execute_table_query_multi(
                connection_id=LOT_DB_CONNECTION_ID,
                schema_name=LOT_DB_SCHEMA or "public",
                table_name=LOT_DB_TABLE,
                columns=columns,
                filters=filters,
                limit=row_limit,
            )
            for row in result.get("rows", []):
                lot_id = row.get(LOT_DB_LOT_COLUMN) or row.get("lot_id") or row.get("LOT_ID")
                defect = row.get(LOT_DEFECT_RATE_COLUMN)
                if lot_id and defect is not None:
                    try:
                        defect_value = float(defect)
                    except (TypeError, ValueError):
                        continue
                    candidates.append({"lot_id": lot_id, "defect_rate": defect_value})
        except Exception:
            candidates = []

    if not candidates:
        candidates = test_simulation_store.ensure_candidates(session_id, max(row_limit, 50))
    filtered = filter_lot_candidates(candidates, min_rate, max_rate, row_limit)
    test_simulation_store.set_filtered_lots(
        session_id,
        filtered,
        {"min_rate": min_rate, "max_rate": max_rate},
    )

    stats = summarize_defect_rates(filtered)
    stats["value_unit"] = "ratio"
    chart_lots = filtered[:DEFAULT_CHART_LIMIT]
    await event_bus.broadcast(
        {
            "type": "defect_rate_chart",
            "payload": {
                "lots": chart_lots,
                "filters": {"min_rate": min_rate, "max_rate": max_rate},
                "stats": stats,
                "metric_label": "불량률",
                "value_unit": "ratio",
            },
        }
    )
    pipeline_store.set_event(
        session_id,
        "defect_rate_chart",
        {
            "lots": chart_lots,
            "filters": {"min_rate": min_rate, "max_rate": max_rate},
            "stats": stats,
            "metric_label": "불량률",
            "value_unit": "ratio",
        },
    )

    if not filtered:
        await emit_frontend_trigger("불량률 조건에 맞는 LOT가 없습니다.")
        await _log_tool_result("filter_lots_by_defect_rate", "status=missing rows=0")
        return {
            "status": "missing",
            "missing": ["lot_candidates"],
            "filters": {"min_rate": min_rate, "max_rate": max_rate},
            "message": "불량률 조건에 맞는 LOT가 없습니다.",
        }

    await emit_frontend_trigger(
        f"불량률 필터링 완료: {len(filtered)}건 (그래프 표시)",
        {"count": len(filtered)},
    )
    await _log_tool_result(
        "filter_lots_by_defect_rate",
        f"rows={len(filtered)} min={min_rate} max={max_rate}",
    )
    return {
        "status": "ok",
        "count": len(filtered),
        "filters": {"min_rate": min_rate, "max_rate": max_rate},
        "stats": stats,
        "lots": filtered,
    }


@function_tool
async def run_design_grid_search(
    message: str | None = None,
    target: float | None = None,
    limit: int | None = None,
    top_k: int | None = None,
) -> dict:
    """Run grid search around design values and emit top candidates."""
    await _log_route("test_simulation", "run_design_grid_search")
    await _log_tool_call(
        "run_design_grid_search",
        message=message,
        target=target,
        limit=limit,
        top_k=top_k,
    )
    session_id = current_session_id.get()
    parsed_target = parse_target_value(message or "")
    target_value = target if target is not None else parsed_target
    if target_value is None:
        target_value = test_simulation_store.get_grid_target(session_id)
    if target_value is None:
        target_value = DEFAULT_TARGET

    filtered = test_simulation_store.get_filtered_lots(session_id)
    if not filtered:
        await emit_frontend_trigger("불량률 필터 결과가 없어 설계 탐색을 진행할 수 없습니다.")
        await _log_tool_result("run_design_grid_search", "status=missing filtered_lots=0")
        return {
            "status": "missing",
            "missing": ["filtered_lots"],
            "message": "불량률 필터 결과가 없습니다.",
        }

    grid_limit = limit if isinstance(limit, int) and limit > 0 else DEFAULT_GRID_LIMIT
    grid_limit = max(1, min(grid_limit, 500))
    results = []
    try:
        api_results = await call_grid_search_api(filtered, target_value, grid_limit)
    except Exception:
        api_results = None
    if isinstance(api_results, list) and api_results and isinstance(api_results[0], dict):
        results = api_results
    if not results:
        results = grid_search_candidates(filtered, target_value, grid_limit)
    for idx, item in enumerate(results, start=1):
        item["rank"] = idx
    test_simulation_store.set_grid_results(session_id, results, target_value)

    top_count = top_k if isinstance(top_k, int) and top_k > 0 else DEFAULT_TOP_K
    top_count = max(1, min(top_count, 50))
    top_candidates = results[:top_count]
    await event_bus.broadcast(
        {
            "type": "design_candidates",
            "payload": {
                "candidates": top_candidates,
                "total": len(results),
                "offset": 0,
                "limit": top_count,
                "target": target_value,
            },
        }
    )
    pipeline_store.set_event(
        session_id,
        "design_candidates",
        {
            "candidates": top_candidates,
            "total": len(results),
            "offset": 0,
            "limit": top_count,
            "target": target_value,
        },
    )
    await emit_frontend_trigger(
        f"설계 후보 {len(results)}건 중 상위 {len(top_candidates)}건을 표시했습니다.",
        {"count": len(top_candidates), "total": len(results)},
    )
    await _log_tool_result(
        "run_design_grid_search",
        f"total={len(results)} top={len(top_candidates)} target={target_value}",
    )
    return {
        "status": "ok",
        "target": target_value,
        "total": len(results),
        "candidates": top_candidates,
    }


@function_tool
async def get_design_candidates(
    offset: int = 0,
    limit: int = 10,
    rank: int | None = None,
) -> dict:
    """Fetch stored grid search candidates."""
    await _log_tool_call("get_design_candidates", offset=offset, limit=limit, rank=rank)
    session_id = current_session_id.get()
    if rank is not None and rank > 0:
        offset = max(rank - 1, 0)
        limit = 1

    offset_value = max(0, int(offset))
    limit_value = max(1, min(int(limit), 50))
    candidates, total = test_simulation_store.get_candidates(
        session_id, offset_value, limit_value
    )
    if not candidates:
        await _log_tool_result("get_design_candidates", "status=missing total=0")
        return {
            "status": "missing",
            "missing": ["grid_results"],
            "message": "설계 후보가 없습니다.",
        }

    await event_bus.broadcast(
        {
            "type": "design_candidates",
            "payload": {
                "candidates": candidates,
                "total": total,
                "offset": offset_value,
                "limit": limit_value,
                "target": test_simulation_store.get_grid_target(session_id),
            },
        }
    )
    await _log_tool_result(
        "get_design_candidates",
        f"offset={offset_value} limit={limit_value} total={total}",
    )
    return {
        "status": "ok",
        "total": total,
        "offset": offset_value,
        "limit": limit_value,
        "candidates": candidates,
    }


@function_tool
def get_defect_rate_stats() -> dict:
    """Summarize defect-rate stats from the filtered lots."""
    session_id = current_session_id.get()
    filtered = test_simulation_store.get_filtered_lots(session_id)
    stats = summarize_defect_rates(filtered)
    return {"status": "ok", "stats": stats, "count": stats.get("count", 0)}


@function_tool
def reset_test_simulation() -> dict:
    """Clear the stored test simulation results."""
    session_id = current_session_id.get()
    test_simulation_store.clear(session_id)
    return {"status": "ok", "message": "테스트 시뮬레이션 상태를 초기화했습니다."}


@function_tool
async def open_simulation_form() -> dict:
    """Open the simulation input form with current params and missing fields."""
    await _log_tool_call("open_simulation_form")
    session_id = current_session_id.get()
    params = simulation_store.get(session_id)
    missing = simulation_store.missing(session_id)
    await emit_simulation_form(params, missing)
    pipeline_store.set_event(session_id, "simulation_form", {"params": params, "missing": missing})
    await _log_tool_result(
        "open_simulation_form",
        f"missing={len(missing)} params={len(params)}",
    )
    return {"params": params, "missing": missing}


async def run_lot_simulation_impl(lot_id: str = "") -> dict:
    """Fetch LOT data, derive params, run simulation, and emit the result to the UI."""
    session_id = current_session_id.get()
    lot_payload = None
    await _log_route("simulation", "run_lot_simulation")
    await _log_tool_call("run_lot_simulation", lot_id=lot_id)
    if lot_id:
        lot_payload = await emit_lot_info(lot_id, limit=1)
        if lot_payload.get("error"):
            await _log_tool_result(
                "run_lot_simulation",
                f"status=error error={lot_payload.get('error')}",
            )
            return {"status": "error", "error": lot_payload.get("error")}

    stored = lot_store.get(session_id)
    row = stored.get("row") if stored else None
    if not row:
        await emit_frontend_trigger(
            "LOT 정보가 없어 추천을 진행할 수 없습니다. 먼저 LOT ID를 알려주세요."
        )
        await _log_tool_result("run_lot_simulation", "status=missing missing=lot_id")
        return {
            "status": "missing",
            "missing": ["lot_id"],
            "message": "LOT ID가 필요합니다.",
        }

    params = _extract_simulation_params_from_lot(row)
    update_result = collect_simulation_params(**params)
    await emit_simulation_form(update_result["params"], update_result["missing"])
    result = await execute_simulation()
    payload = {"status": result.get("status"), "lot_id": lot_id or stored.get("lot_id")}
    if result.get("status") == "missing":
        missing = result.get("missing", [])
        payload["missing"] = missing
        if missing:
            missing_text = ", ".join(missing)
            await emit_frontend_trigger(
                f"추천 입력값이 부족합니다: {missing_text}"
            )
            payload["message"] = (
                f"추천 입력값이 부족합니다: {missing_text}"
            )
        await _log_tool_result(
            "run_lot_simulation",
            f"status=missing missing={len(missing)}",
        )
        return payload
    payload["result"] = result.get("result")
    payload["params"] = simulation_store.get(session_id)
    await _log_tool_result("run_lot_simulation", "status=ok")
    return payload


@function_tool
async def run_lot_simulation(lot_id: str = "") -> dict:
    """Fetch LOT data, derive params, run simulation, and emit the result to the UI."""
    return await run_lot_simulation_impl(lot_id=lot_id)


@function_tool(strict_mode=False)
async def frontend_trigger(message: str, data: dict | None = None) -> dict:
    """Send a freeform message to the frontend UI."""
    await _log_tool_call("frontend_trigger", message=message)
    result = await emit_frontend_trigger(message, data)
    await _log_tool_result("frontend_trigger", "sent")
    return result


@function_tool
def file_search(query: str = "", top_k: int = 3) -> dict:
    """Search the internal knowledge base for the query."""
    results = _search_knowledge_base(query, max(1, top_k))
    payload = {"results": results, "query": query, "count": len(results)}
    try:
        import asyncio

        loop = asyncio.get_running_loop()
        loop.create_task(_log_tool_call("file_search", query=query, top_k=top_k))
        loop.create_task(_log_tool_result("file_search", f"count={payload['count']}"))
    except RuntimeError:
        pass
    return payload
