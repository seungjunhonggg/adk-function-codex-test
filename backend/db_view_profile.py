import json
from pathlib import Path

from .config import (
    DB_VIEW_PROFILE_PATH,
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


def normalize_view_profile(profile: dict) -> dict:
    connection_id = profile.get("connection_id") or LOT_DB_CONNECTION_ID
    schema_name = profile.get("schema") or LOT_DB_SCHEMA or "public"
    table_name = profile.get("table") or LOT_DB_TABLE
    limit = profile.get("limit") or 5

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
    }


def get_filter_column_names(profile: dict) -> list[str]:
    return [item["name"] for item in profile.get("filter_columns", [])]


def build_db_agent_prompt(profile: dict) -> str:
    connection_id = profile.get("connection_id") or "unset"
    schema_name = profile.get("schema") or "public"
    table_name = profile.get("table") or "unset_view"
    limit = profile.get("limit") or 5

    filter_lines = []
    for item in profile.get("filter_columns", []):
        name = item.get("name")
        description = item.get("description") or "no description"
        if name:
            filter_lines.append(f"- {name}: {description}")
    if not filter_lines:
        filter_lines = ["- (no filter columns configured)"]

    filters_text = "\n".join(filter_lines)
    return (
        "DB VIEW CONTEXT\n"
        f"- connection_id: {connection_id}\n"
        f"- view_table: {schema_name}.{table_name}\n"
        f"- default_limit: {limit}\n"
        "- allowed_operators: =, >, <, >=, <=, ilike\n"
        "- allowed_filter_columns:\n"
        f"{filters_text}\n"
        "Rules:\n"
        "- Only use allowed_filter_columns in filters.\n"
        "- Use ilike for partial text match, = for identifiers.\n"
        "- Return up to default_limit rows unless user explicitly requests more."
    )
