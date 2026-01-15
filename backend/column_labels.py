from __future__ import annotations

import time
from typing import Dict, Tuple

from .config import (
    COLUMN_LABEL_CACHE_TTL_SECONDS,
    COLUMN_LABEL_SCHEMA,
    COLUMN_LABEL_TABLE,
)
from .db_connections import execute_table_query_multi, get_connection


_COLUMN_LABEL_CACHE: Dict[Tuple[str, str, str], dict] = {}
_COLUMN_LABEL_CACHE_TS: Dict[Tuple[str, str, str], float] = {}


def _resolve_label_schema(connection_id: str, schema_name: str | None) -> str:
    if schema_name:
        return schema_name
    if COLUMN_LABEL_SCHEMA:
        return COLUMN_LABEL_SCHEMA
    connection = get_connection(connection_id) or {}
    return str(connection.get("schema") or "public")


def _load_label_map(
    connection_id: str,
    schema: str,
    table: str,
    name_column: str,
    label_column: str,
) -> dict:
    try:
        result = execute_table_query_multi(
            connection_id=connection_id,
            schema_name=schema,
            table_name=table,
            columns=[name_column, label_column],
            filters=[],
            limit=5000,
        )
    except ValueError:
        return {}
    rows = result.get("rows") or []
    mapping: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get(name_column) or "").strip()
        label = str(row.get(label_column) or "").strip()
        if name and label:
            mapping[name] = label
    return mapping


def get_column_label_map(
    connection_id: str,
    schema_name: str | None = None,
    table_name: str | None = None,
    refresh: bool = False,
) -> dict:
    if not connection_id:
        return {}
    table = str(table_name or COLUMN_LABEL_TABLE or "").strip()
    if not table:
        return {}
    schema = _resolve_label_schema(connection_id, schema_name)
    cache_key = (connection_id, schema, table)
    now = time.time()
    if not refresh:
        cached = _COLUMN_LABEL_CACHE.get(cache_key)
        cached_ts = _COLUMN_LABEL_CACHE_TS.get(cache_key, 0.0)
        if cached and (now - cached_ts) < COLUMN_LABEL_CACHE_TTL_SECONDS:
            return cached

    mapping = _load_label_map(connection_id, schema, table, "column_name", "label_ko")
    if not mapping and table != "column_catalog":
        fallback = _load_label_map(
            connection_id, schema, "column_catalog", "column_name", "display_ko"
        )
        if fallback:
            mapping = fallback

    _COLUMN_LABEL_CACHE[cache_key] = mapping
    _COLUMN_LABEL_CACHE_TS[cache_key] = now
    return mapping


def build_column_label_map(columns: list[str], label_map: dict) -> dict:
    if not columns or not label_map:
        return {}
    return {column: label_map[column] for column in columns if column in label_map}
