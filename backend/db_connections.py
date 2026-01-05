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

try:
    import pymysql
except ImportError:  # pragma: no cover - optional dependency
    pymysql = None


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
    db_type = _get_db_type(connection)
    if db_type in {"mariadb", "mysql"}:
        conn = _connect_mariadb(connection)
        try:
            schema = _introspect_mariadb(conn)
        finally:
            conn.close()
    else:
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


def _ensure_pymysql() -> None:
    if pymysql is None:
        raise RuntimeError("pymysql 패키지가 필요합니다. requirements.txt를 확인하세요.")


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


def _connect_mariadb(config: dict) -> Any:
    _ensure_pymysql()
    return pymysql.connect(
        host=config.get("host"),
        port=config.get("port") or 3306,
        user=config.get("user"),
        password=config.get("password"),
        database=config.get("database"),
        connect_timeout=5,
        cursorclass=pymysql.cursors.Cursor,
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


def _introspect_mariadb(conn: Any) -> dict:
    schemas: dict[str, dict] = {}
    query = """
        SELECT table_schema, table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
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


def _get_db_type(connection: dict) -> str:
    return str(connection.get("db_type") or "postgresql").lower()


def connect_and_save(config: dict) -> dict:
    db_type = (config.get("db_type") or "").lower()
    if db_type not in {"postgresql", "postgres", "mariadb", "mysql"}:
        raise ValueError(
            "지원하지 않는 DB 타입입니다. 현재는 PostgreSQL/MariaDB만 지원합니다."
        )

    if db_type in {"mariadb", "mysql"}:
        conn = _connect_mariadb(config)
        try:
            schema = _introspect_mariadb(conn)
        finally:
            conn.close()
    else:
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
        "port": config.get("port") or (3306 if db_type in {"mariadb", "mysql"} else 5432),
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
    allowed = {
        "=",
        ">",
        "<",
        ">=",
        "<=",
        "ilike",
        "like",
        "in",
        "between",
        "is_null",
        "is_not_null",
    }
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
    connection = get_connection(connection_id)
    if not connection:
        raise ValueError("DB 연결을 찾을 수 없습니다.")
    db_type = _get_db_type(connection)
    schema = connection.get("schema") or {}
    valid_columns = _validate_table_config(schema, schema_name, table_name, columns)
    operator = _normalize_operator(filter_operator)

    if db_type in {"mariadb", "mysql"}:
        conn = _connect_mariadb(connection)
        try:
            columns_sql = ", ".join(f"`{col}`" for col in valid_columns)
            query = f"SELECT {columns_sql} FROM `{schema_name}`.`{table_name}`"
            params: list[Any] = []
            if filter_column:
                if operator == "is_not_null":
                    query += f" WHERE `{filter_column}` IS NOT NULL"
                elif operator == "is_null":
                    query += f" WHERE `{filter_column}` IS NULL"
                elif filter_value is not None and filter_value != "":
                    if operator in {"ilike", "like"}:
                        query += f" WHERE `{filter_column}` LIKE %s"
                        params.append(f"%{filter_value}%")
                    elif operator == "in" and isinstance(filter_value, (list, tuple)):
                        placeholders = ", ".join(["%s"] * len(filter_value))
                        query += f" WHERE `{filter_column}` IN ({placeholders})"
                        params.extend(list(filter_value))
                    elif operator == "between" and isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        query += f" WHERE `{filter_column}` BETWEEN %s AND %s"
                        params.extend([filter_value[0], filter_value[1]])
                    else:
                        query += f" WHERE `{filter_column}` {operator} %s"
                        params.append(filter_value)
            if limit:
                query += " LIMIT %s"
                params.append(limit)

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
        finally:
            conn.close()
    else:
        _ensure_psycopg()
        conn = _connect_postgres(connection)
        try:
            query = sql.SQL("SELECT {fields} FROM {schema}.{table}").format(
                fields=sql.SQL(", ").join(sql.Identifier(col) for col in valid_columns),
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name),
            )
            params = []
            if filter_column:
                if operator == "is_not_null":
                    query += sql.SQL(" WHERE {col} IS NOT NULL").format(
                        col=sql.Identifier(filter_column)
                    )
                elif operator == "is_null":
                    query += sql.SQL(" WHERE {col} IS NULL").format(
                        col=sql.Identifier(filter_column)
                    )
                elif filter_value is not None and filter_value != "":
                    if operator in {"ilike", "like"}:
                        query += sql.SQL(" WHERE {col} {op} %s").format(
                            col=sql.Identifier(filter_column),
                            op=sql.SQL(operator.upper()),
                        )
                        params.append(f"%{filter_value}%")
                    elif operator == "in" and isinstance(filter_value, (list, tuple)):
                        placeholders = sql.SQL(", ").join(
                            sql.Placeholder() for _ in range(len(filter_value))
                        )
                        query += sql.SQL(" WHERE {col} IN ({values})").format(
                            col=sql.Identifier(filter_column),
                            values=placeholders,
                        )
                        params.extend(list(filter_value))
                    elif operator == "between" and isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        query += sql.SQL(" WHERE {col} BETWEEN %s AND %s").format(
                            col=sql.Identifier(filter_column)
                        )
                        params.extend([filter_value[0], filter_value[1]])
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
    connection = get_connection(connection_id)
    if not connection:
        raise ValueError("DB 연결을 찾을 수 없습니다.")
    db_type = _get_db_type(connection)
    schema = connection.get("schema") or {}
    valid_columns = _validate_table_config(schema, schema_name, table_name, columns)
    available = set(_get_table_columns(schema, schema_name, table_name))

    normalized_filters = []
    for item in filters or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        if not column:
            continue
        if column not in available:
            raise ValueError("존재하지 않는 필터 컬럼이 포함되어 있습니다.")
        operator = _normalize_operator(item.get("operator"))
        value = item.get("value")
        if operator in {"is_null", "is_not_null"}:
            normalized_filters.append({"column": column, "operator": operator, "value": None})
            continue
        if value is None or value == "":
            continue
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            normalized_filters.append({"column": column, "operator": operator, "value": list(value)})
            continue
        if operator == "in":
            if not isinstance(value, (list, tuple)) or not value:
                continue
            normalized_filters.append({"column": column, "operator": operator, "value": list(value)})
            continue
        normalized_filters.append({"column": column, "operator": operator, "value": value})

    if db_type in {"mariadb", "mysql"}:
        conn = _connect_mariadb(connection)
        try:
            columns_sql = ", ".join(f"`{col}`" for col in valid_columns)
            query = f"SELECT {columns_sql} FROM `{schema_name}`.`{table_name}`"
            params: list[Any] = []
            if normalized_filters:
                clauses = []
                for item in normalized_filters:
                    column = item["column"]
                    operator = item["operator"]
                    value = item["value"]
                    if operator == "is_not_null":
                        clauses.append(f"`{column}` IS NOT NULL")
                    elif operator == "is_null":
                        clauses.append(f"`{column}` IS NULL")
                    elif operator in {"ilike", "like"}:
                        clauses.append(f"`{column}` LIKE %s")
                        params.append(f"%{value}%")
                    elif operator == "in" and isinstance(value, list):
                        placeholders = ", ".join(["%s"] * len(value))
                        clauses.append(f"`{column}` IN ({placeholders})")
                        params.extend(value)
                    elif operator == "between" and isinstance(value, list) and len(value) == 2:
                        clauses.append(f"`{column}` BETWEEN %s AND %s")
                        params.extend([value[0], value[1]])
                    else:
                        clauses.append(f"`{column}` {operator} %s")
                        params.append(value)
                query += " WHERE " + " AND ".join(clauses)
            if limit:
                query += " LIMIT %s"
                params.append(limit)

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
        finally:
            conn.close()
    else:
        _ensure_psycopg()
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
                    if operator == "is_not_null":
                        clauses.append(
                            sql.SQL("{col} IS NOT NULL").format(
                                col=sql.Identifier(column)
                            )
                        )
                    elif operator == "is_null":
                        clauses.append(
                            sql.SQL("{col} IS NULL").format(
                                col=sql.Identifier(column)
                            )
                        )
                    elif operator in {"ilike", "like"}:
                        clauses.append(
                            sql.SQL("{col} {op} %s").format(
                                col=sql.Identifier(column),
                                op=sql.SQL(operator.upper()),
                            )
                        )
                        params.append(f"%{value}%")
                    elif operator == "in" and isinstance(value, list):
                        placeholders = sql.SQL(", ").join(
                            sql.Placeholder() for _ in range(len(value))
                        )
                        clauses.append(
                            sql.SQL("{col} IN ({values})").format(
                                col=sql.Identifier(column),
                                values=placeholders,
                            )
                        )
                        params.extend(value)
                    elif operator == "between" and isinstance(value, list) and len(value) == 2:
                        clauses.append(
                            sql.SQL("{col} BETWEEN %s AND %s").format(
                                col=sql.Identifier(column)
                            )
                        )
                        params.extend([value[0], value[1]])
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
