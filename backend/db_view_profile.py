import json
from pathlib import Path

from .config import (
    DB_VIEW_PROFILE_PATH,
    DB_VIEW_COLUMN_CATALOG_PATH,
    LOT_DB_CONNECTION_ID,
    LOT_DB_SCHEMA,
    LOT_DB_TABLE,
)


def load_view_profile() -> dict:
    path = Path(DB_VIEW_PROFILE_PATH)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def load_column_catalog() -> dict:
    path = Path(DB_VIEW_COLUMN_CATALOG_PATH)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def _normalize_column_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
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


def normalize_column_catalog(catalog: dict) -> dict:
    if not isinstance(catalog, dict):
        return {"tables": {}}
    tables = catalog.get("tables")
    if not isinstance(tables, dict):
        tables = {"default": catalog}
    normalized_tables: dict[str, dict] = {}
    for table_name, table_entry in tables.items():
        if not isinstance(table_entry, dict):
            continue
        columns: list[dict] = []
        for item in table_entry.get("columns", []) or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("column")
            if not name:
                continue
            aliases = _normalize_column_list(item.get("aliases") or item.get("alias"))
            columns.append(
                {
                    "name": str(name).strip(),
                    "description": str(item.get("description") or "").strip(),
                    "aliases": aliases,
                    "unit": str(item.get("unit") or "").strip(),
                    "group": str(item.get("group") or "").strip(),
                }
            )
        default_columns = _normalize_column_list(
            table_entry.get("default_columns") or table_entry.get("result_columns")
        )
        normalized_tables[str(table_name)] = {
            "columns": columns,
            "default_columns": default_columns,
        }
    return {"tables": normalized_tables}


def select_catalog_table(
    catalog: dict, table_name: str | None, catalog_table: str | None
) -> dict:
    tables = catalog.get("tables") if isinstance(catalog, dict) else {}
    if not isinstance(tables, dict):
        return {}
    if catalog_table and catalog_table in tables:
        return tables[catalog_table]
    if table_name and table_name in tables:
        return tables[table_name]
    if "default" in tables:
        return tables["default"]
    if len(tables) == 1:
        return next(iter(tables.values()))
    return {}


def normalize_view_profile(profile: dict) -> dict:
    connection_id = profile.get("connection_id") or LOT_DB_CONNECTION_ID
    schema_name = profile.get("schema") or LOT_DB_SCHEMA or "public"
    table_name = profile.get("table") or LOT_DB_TABLE
    limit = profile.get("limit") or 5
    result_columns = _normalize_column_list(
        profile.get("result_columns") or profile.get("default_columns")
    )
    selectable_columns = _normalize_column_list(
        profile.get("selectable_columns") or profile.get("select_columns")
    )
    catalog_table = str(
        profile.get("column_catalog_table") or profile.get("catalog_table") or ""
    ).strip()
    time_column = str(
        profile.get("time_column")
        or profile.get("date_column")
        or profile.get("timestamp_column")
        or ""
    ).strip()

    raw_filters = profile.get("filter_columns") or []
    filters = []
    for item in raw_filters:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("column")
        if not name:
            continue
        filters.append(
            {
                "name": str(name).strip(),
                "description": str(item.get("description") or "").strip(),
            }
        )

    return {
        "connection_id": connection_id,
        "schema": schema_name,
        "table": table_name,
        "limit": int(limit) if isinstance(limit, (int, float)) else 5,
        "filter_columns": filters,
        "result_columns": result_columns,
        "selectable_columns": selectable_columns,
        "column_catalog_table": catalog_table,
        "time_column": time_column,
    }


def get_filter_column_names(profile: dict) -> list[str]:
    return [item["name"] for item in profile.get("filter_columns", [])]


def _format_columns_for_prompt(columns: list[str], max_items: int = 20) -> list[str]:
    if not columns:
        return ["- (none configured)"]
    if len(columns) > max_items:
        sample = ", ".join(columns[:8])
        return [f"- {len(columns)} columns configured (sample: {sample})"]
    return [f"- {name}" for name in columns]


def build_db_agent_prompt(profile: dict) -> str:
    connection_id = profile.get("connection_id") or "unset"
    schema_name = profile.get("schema") or "public"
    table_name = profile.get("table") or "unset_view"
    limit = profile.get("limit") or 5
    result_columns = profile.get("result_columns") or []
    selectable_columns = profile.get("selectable_columns") or []
    time_column = profile.get("time_column") or "not_set"

    filter_lines = []
    for item in profile.get("filter_columns", []):
        name = item.get("name")
        description = item.get("description") or "no description"
        if name:
            filter_lines.append(f"- {name}: {description}")
    if not filter_lines:
        filter_lines = ["- (no filter columns configured)"]

    filters_text = "\n".join(filter_lines)
    result_lines = _format_columns_for_prompt(result_columns)
    result_text = "\n".join(result_lines)
    select_lines = _format_columns_for_prompt(selectable_columns)
    select_text = "\n".join(select_lines)
    return (
        "DB VIEW CONTEXT\n"
        f"- connection_id: {connection_id}\n"
        f"- view_table: {schema_name}.{table_name}\n"
        f"- default_limit: {limit}\n"
        "- allowed_operators: =, >, <, >=, <=, ilike\n"
        "- allowed_filter_columns:\n"
        f"{filters_text}\n"
        "- default_result_columns:\n"
        f"{result_text}\n"
        "- allowed_select_columns:\n"
        f"{select_text}\n"
        f"- time_column: {time_column}\n"
        "Rules:\n"
        "- Only use allowed_filter_columns in filters.\n"
        "- Use ilike for partial text match, = for identifiers.\n"
        "- If the user asks for specific metrics, resolve columns first and pass them in.\n"
        "- For recent N months, prefer query_view_metrics with recent_months=N.\n"
        "- Return up to default_limit rows unless user explicitly requests more."
    )
