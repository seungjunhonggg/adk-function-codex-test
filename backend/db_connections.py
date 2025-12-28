from __future__ import annotations

import json
import uuid
from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from .config import DB_CONNECTIONS_PATH

try:
    import psycopg
    from psycopg import sql
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None
    sql = None


logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_connections() -> dict:
    path = Path(DB_CONNECTIONS_PATH)
    if not path.exists():
        return {"connections": []}
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {"connections": []}


def _save_connections(payload: dict) -> None:
    path = Path(DB_CONNECTIONS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def list_connections() -> list[dict]:
    payload = _load_connections()
    results = []
    for item in payload.get("connections", []):
        results.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "db_type": item.get("db_type"),
                "host": item.get("host"),
                "port": item.get("port"),
                "database": item.get("database"),
                "user": item.get("user"),
                "updated_at": item.get("updated_at"),
            }
        )
    return results


def get_connection(connection_id: str) -> dict | None:
    payload = _load_connections()
    for item in payload.get("connections", []):
        if item.get("id") == connection_id:
            return item
    return None


def get_schema(connection_id: str) -> dict | None:
    connection = get_connection(connection_id)
    if not connection:
        return None
    return connection.get("schema") or {}


def preload_schema(connection_id: str, force: bool = False) -> dict | None:
    if not connection_id:
        return None
    connection = get_connection(connection_id)
    if not connection:
        return None
    if not force and connection.get("schema"):
        return connection.get("schema") or {}
    conn = _connect_postgres(connection)
    try:
        schema = _introspect_postgres(conn)
    finally:
        conn.close()

    payload = _load_connections()
    connections = payload.get("connections", [])
    for idx, item in enumerate(connections):
        if item.get("id") == connection_id:
            updated = dict(item)
            updated["schema"] = schema
            updated["updated_at"] = _now_iso()
            connections[idx] = updated
            break
    payload["connections"] = connections
    _save_connections(payload)
    return schema


def _ensure_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError("psycopg 패키지가 필요합니다. requirements.txt를 확인하세요.")


def _connect_postgres(config: dict) -> Any:
    _ensure_psycopg()
    return psycopg.connect(
        host=config.get("host"),
        port=config.get("port") or 5432,
        dbname=config.get("database"),
        user=config.get("user"),
        password=config.get("password"),
        connect_timeout=5,
    )


def _introspect_postgres(conn: Any) -> dict:
    schemas: dict[str, dict] = {}
    query = """
        SELECT table_schema, table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        for row in cur.fetchall():
            schema_name, table_name, column_name, data_type, is_nullable = row
            schema_entry = schemas.setdefault(schema_name, {"tables": {}})
            table_entry = schema_entry["tables"].setdefault(
                table_name, {"columns": []}
            )
            table_entry["columns"].append(
                {
                    "name": column_name,
                    "type": data_type,
                    "nullable": is_nullable == "YES",
                }
            )
    return {"schemas": schemas}


def connect_and_save(config: dict) -> dict:
    db_type = (config.get("db_type") or "").lower()
    if db_type not in {"postgresql", "postgres"}:
        raise ValueError("지원하지 않는 DB 타입입니다. 현재는 PostgreSQL만 지원합니다.")

    conn = _connect_postgres(config)
    try:
        schema = _introspect_postgres(conn)
    finally:
        conn.close()

    payload = _load_connections()
    connection_id = config.get("id") or f"db-{uuid.uuid4().hex[:8]}"
    record = {
        "id": connection_id,
        "name": config.get("name") or connection_id,
        "db_type": db_type,
        "host": config.get("host"),
        "port": config.get("port") or 5432,
        "database": config.get("database"),
        "user": config.get("user"),
        "password": config.get("password"),
        "schema": schema,
        "updated_at": _now_iso(),
    }

    connections = payload.get("connections", [])
    updated = False
    for idx, item in enumerate(connections):
        if item.get("id") == connection_id:
            connections[idx] = record
            updated = True
            break
    if not updated:
        connections.append(record)
    payload["connections"] = connections
    _save_connections(payload)
    return record


def _validate_table_config(
    schema: dict, schema_name: str, table_name: str, columns: list[str]
) -> list[str]:
    schemas = schema.get("schemas") or {}
    schema_entry = schemas.get(schema_name)
    if not schema_entry:
        raise ValueError("스키마를 찾을 수 없습니다.")
    table_entry = schema_entry.get("tables", {}).get(table_name)
    if not table_entry:
        raise ValueError("테이블을 찾을 수 없습니다.")
    available = {col["name"] for col in table_entry.get("columns", [])}
    if not columns:
        return sorted(available)
    invalid = [col for col in columns if col not in available]
    if invalid:
        raise ValueError("존재하지 않는 컬럼이 포함되어 있습니다.")
    return columns


def _get_table_columns(schema: dict, schema_name: str, table_name: str) -> list[str]:
    schemas = schema.get("schemas") or {}
    schema_entry = schemas.get(schema_name)
    if not schema_entry:
        raise ValueError("스키마를 찾을 수 없습니다.")
    table_entry = schema_entry.get("tables", {}).get(table_name)
    if not table_entry:
        raise ValueError("테이블을 찾을 수 없습니다.")
    return [col["name"] for col in table_entry.get("columns", [])]


def _normalize_operator(value: str | None) -> str:
    operator = (value or "ilike").lower()
    allowed = {"=", ">", "<", ">=", "<=", "ilike", "like"}
    if operator not in allowed:
        return "ilike"
    return operator


def execute_table_query(
    connection_id: str,
    schema_name: str,
    table_name: str,
    columns: list[str],
    filter_column: str | None,
    filter_operator: str | None,
    filter_value: str | None,
    limit: int,
) -> dict:
    _ensure_psycopg()
    connection = get_connection(connection_id)
    if not connection:
        raise ValueError("DB 연결을 찾을 수 없습니다.")
    schema = connection.get("schema") or {}
    valid_columns = _validate_table_config(schema, schema_name, table_name, columns)
    operator = _normalize_operator(filter_operator)

    conn = _connect_postgres(connection)
    try:
        query = sql.SQL("SELECT {fields} FROM {schema}.{table}").format(
            fields=sql.SQL(", ").join(sql.Identifier(col) for col in valid_columns),
            schema=sql.Identifier(schema_name),
            table=sql.Identifier(table_name),
        )
        params: list[Any] = []
        if filter_column and filter_value:
            if operator in {"ilike", "like"}:
                query += sql.SQL(" WHERE {col} {op} %s").format(
                    col=sql.Identifier(filter_column),
                    op=sql.SQL(operator.upper()),
                )
                params.append(f"%{filter_value}%")
            else:
                query += sql.SQL(" WHERE {col} {op} %s").format(
                    col=sql.Identifier(filter_column),
                    op=sql.SQL(operator),
                )
                params.append(filter_value)
        if limit:
            query += sql.SQL(" LIMIT %s")
            params.append(limit)

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
    finally:
        conn.close()

    return {"columns": columns, "rows": [dict(zip(columns, row)) for row in rows]}


def execute_table_query_multi(
    connection_id: str,
    schema_name: str,
    table_name: str,
    columns: list[str],
    filters: list[dict],
    limit: int,
) -> dict:
    _ensure_psycopg()
    connection = get_connection(connection_id)
    if not connection:
        raise ValueError("DB 연결을 찾을 수 없습니다.")
    schema = connection.get("schema") or {}
    valid_columns = _validate_table_config(schema, schema_name, table_name, columns)
    available = set(_get_table_columns(schema, schema_name, table_name))

    normalized_filters = []
    for item in filters or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        value = item.get("value")
        if not column or value is None or value == "":
            continue
        if column not in available:
            raise ValueError("존재하지 않는 필터 컬럼이 포함되어 있습니다.")
        operator = _normalize_operator(item.get("operator"))
        normalized_filters.append({"column": column, "operator": operator, "value": value})

    conn = _connect_postgres(connection)
    try:
        query = sql.SQL("SELECT {fields} FROM {schema}.{table}").format(
            fields=sql.SQL(", ").join(sql.Identifier(col) for col in valid_columns),
            schema=sql.Identifier(schema_name),
            table=sql.Identifier(table_name),
        )
        params: list[Any] = []
        if normalized_filters:
            clauses = []
            for item in normalized_filters:
                column = item["column"]
                operator = item["operator"]
                value = item["value"]
                if operator in {"ilike", "like"}:
                    clauses.append(
                        sql.SQL("{col} {op} %s").format(
                            col=sql.Identifier(column),
                            op=sql.SQL(operator.upper()),
                        )
                    )
                    params.append(f"%{value}%")
                else:
                    clauses.append(
                        sql.SQL("{col} {op} %s").format(
                            col=sql.Identifier(column),
                            op=sql.SQL(operator),
                        )
                    )
                    params.append(value)
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
        if limit:
            query += sql.SQL(" LIMIT %s")
            params.append(limit)

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
    finally:
        conn.close()

    return {"columns": columns, "rows": [dict(zip(columns, row)) for row in rows]}
