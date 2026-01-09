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
    tables_query = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_schema, table_name;
    """
    matviews_query = """
        SELECT schemaname, matviewname
        FROM pg_matviews
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY schemaname, matviewname;
    """
    columns_query = """
        SELECT table_schema, table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position;
    """
    matview_columns_query = """
        SELECT n.nspname AS table_schema,
               c.relname AS table_name,
               a.attname AS column_name,
               pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
               CASE WHEN a.attnotnull THEN 'NO' ELSE 'YES' END AS is_nullable
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE c.relkind = 'm'
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY n.nspname, c.relname, a.attnum;
    """
    with conn.cursor() as cur:
        cur.execute(tables_query)
        for schema_name, table_name, _table_type in cur.fetchall():
            schema_entry = schemas.setdefault(schema_name, {"tables": {}})
            schema_entry["tables"].setdefault(table_name, {"columns": []})

        cur.execute(matviews_query)
        for schema_name, table_name in cur.fetchall():
            schema_entry = schemas.setdefault(schema_name, {"tables": {}})
            schema_entry["tables"].setdefault(table_name, {"columns": []})

        table_lookup = {
            (schema_name, table_name)
            for schema_name, schema_entry in schemas.items()
            for table_name in schema_entry.get("tables", {})
        }
        columns_seen: dict[tuple[str, str], set[str]] = {}

        cur.execute(columns_query)
        for row in cur.fetchall():
            schema_name, table_name, column_name, data_type, is_nullable = row
            if (schema_name, table_name) not in table_lookup:
                continue
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
            columns_seen.setdefault(
                (schema_name, table_name), set()
            ).add(column_name)

        cur.execute(matview_columns_query)
        for row in cur.fetchall():
            schema_name, table_name, column_name, data_type, is_nullable = row
            if (schema_name, table_name) not in table_lookup:
                continue
            schema_entry = schemas.setdefault(schema_name, {"tables": {}})
            table_entry = schema_entry["tables"].setdefault(
                table_name, {"columns": []}
            )
            seen = columns_seen.get((schema_name, table_name))
            if seen is None:
                seen = {
                    col.get("name")
                    for col in table_entry.get("columns", [])
                    if isinstance(col, dict)
                }
                columns_seen[(schema_name, table_name)] = seen
            if column_name in seen:
                continue
            table_entry["columns"].append(
                {
                    "name": column_name,
                    "type": data_type,
                    "nullable": is_nullable == "YES",
                }
            )
            seen.add(column_name)
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


def _normalize_aggregate(value: str | None) -> str:
    agg = (value or "").strip().lower()
    aliases = {
        "average": "avg",
        "mean": "avg",
        "minimum": "min",
        "maximum": "max",
        "count_distinct": "count_distinct",
        "distinct_count": "count_distinct",
    }
    agg = aliases.get(agg, agg)
    allowed = {"avg", "min", "max", "sum", "count", "count_distinct"}
    if agg not in allowed:
        raise ValueError("지원하지 않는 집계 함수입니다.")
    return agg


def _sanitize_alias(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "metric"


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


def execute_table_query_aggregate(
    connection_id: str,
    schema_name: str,
    table_name: str,
    metrics: list[dict],
    filters: list[dict] | None = None,
    group_by: list[str] | None = None,
    time_bucket: dict | None = None,
    order_by: list[dict] | None = None,
    limit: int | None = None,
) -> dict:
    connection = get_connection(connection_id)
    if not connection:
        raise ValueError("DB 연결을 찾을 수 없습니다.")
    db_type = _get_db_type(connection)
    schema = connection.get("schema") or {}
    available = set(_get_table_columns(schema, schema_name, table_name))

    group_columns: list[str] = []
    for column in group_by or []:
        if not column:
            continue
        if column not in available:
            raise ValueError("존재하지 않는 그룹 컬럼이 포함되어 있습니다.")
        if column not in group_columns:
            group_columns.append(column)

    bucket_alias = ""
    bucket_column = ""
    bucket_unit = ""
    if time_bucket:
        bucket_column = str(time_bucket.get("column") or "").strip()
        bucket_unit = str(time_bucket.get("unit") or "").strip().lower()
        bucket_alias = _sanitize_alias(str(time_bucket.get("alias") or "bucket"))
        if not bucket_column or bucket_column not in available:
            raise ValueError("존재하지 않는 시간 컬럼이 포함되어 있습니다.")
        if bucket_unit not in {"month", "week", "day", "hour"}:
            raise ValueError("지원하지 않는 시간 버킷입니다.")

    metric_specs: list[dict] = []
    for item in metrics or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        agg = _normalize_aggregate(item.get("agg") or item.get("aggregate"))
        alias = item.get("alias")
        if not alias:
            alias = f"{agg}_{column or 'all'}"
        alias = _sanitize_alias(str(alias))
        if column:
            if column not in available:
                raise ValueError("존재하지 않는 집계 컬럼이 포함되어 있습니다.")
        elif agg != "count":
            raise ValueError("집계 컬럼이 비어 있습니다.")
        metric_specs.append(
            {"column": column, "agg": agg, "alias": alias}
        )

    if not metric_specs:
        raise ValueError("집계 컬럼이 필요합니다.")

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

    order_items: list[dict] = []
    allowed_order = set(group_columns)
    if bucket_alias:
        allowed_order.add(bucket_alias)
    for metric in metric_specs:
        allowed_order.add(metric["alias"])
    for item in order_by or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        if not column or column not in allowed_order:
            continue
        direction = str(item.get("direction") or "asc").lower()
        if direction not in {"asc", "desc"}:
            direction = "asc"
        order_items.append({"column": column, "direction": direction})
    if not order_items and (group_columns or bucket_alias):
        for column in group_columns:
            order_items.append({"column": column, "direction": "asc"})
        if bucket_alias:
            order_items.append({"column": bucket_alias, "direction": "asc"})

    row_limit = max(1, min(int(limit) if isinstance(limit, int) else 200, 500))

    if db_type in {"mariadb", "mysql"}:
        conn = _connect_mariadb(connection)
        try:
            select_parts = []
            group_parts = []
            if bucket_alias:
                if bucket_unit == "month":
                    bucket_expr = f"DATE_FORMAT(`{bucket_column}`, '%Y-%m-01')"
                elif bucket_unit == "week":
                    bucket_expr = f"DATE_FORMAT(`{bucket_column}`, '%x-%v')"
                elif bucket_unit == "day":
                    bucket_expr = f"DATE_FORMAT(`{bucket_column}`, '%Y-%m-%d')"
                else:
                    bucket_expr = f"DATE_FORMAT(`{bucket_column}`, '%Y-%m-%d %H:00:00')"
                select_parts.append(f"{bucket_expr} AS `{bucket_alias}`")
                group_parts.append(bucket_expr)
            for column in group_columns:
                select_parts.append(f"`{column}`")
                group_parts.append(f"`{column}`")
            for metric in metric_specs:
                if metric["agg"] == "count" and not metric["column"]:
                    expr = "COUNT(*)"
                elif metric["agg"] == "count_distinct":
                    expr = f"COUNT(DISTINCT `{metric['column']}`)"
                else:
                    expr = f"{metric['agg'].upper()}(`{metric['column']}`)"
                select_parts.append(f"{expr} AS `{metric['alias']}`")
            query = f"SELECT {', '.join(select_parts)} FROM `{schema_name}`.`{table_name}`"
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
            if group_parts:
                query += " GROUP BY " + ", ".join(group_parts)
            if order_items:
                order_clause = ", ".join(
                    f"`{item['column']}` {item['direction'].upper()}" for item in order_items
                )
                query += f" ORDER BY {order_clause}"
            if row_limit:
                query += " LIMIT %s"
                params.append(row_limit)

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
            select_parts = []
            group_parts = []
            if bucket_alias:
                bucket_expr = sql.SQL("DATE_TRUNC({unit}, {col})").format(
                    unit=sql.Literal(bucket_unit),
                    col=sql.Identifier(bucket_column),
                )
                select_parts.append(
                    sql.SQL("{expr} AS {alias}").format(
                        expr=bucket_expr, alias=sql.Identifier(bucket_alias)
                    )
                )
                group_parts.append(bucket_expr)
            for column in group_columns:
                ident = sql.Identifier(column)
                select_parts.append(ident)
                group_parts.append(ident)
            for metric in metric_specs:
                if metric["agg"] == "count" and not metric["column"]:
                    expr = sql.SQL("COUNT(*)")
                elif metric["agg"] == "count_distinct":
                    expr = sql.SQL("COUNT(DISTINCT {col})").format(
                        col=sql.Identifier(metric["column"])
                    )
                else:
                    expr = sql.SQL("{func}({col})").format(
                        func=sql.SQL(metric["agg"].upper()),
                        col=sql.Identifier(metric["column"]),
                    )
                select_parts.append(
                    sql.SQL("{expr} AS {alias}").format(
                        expr=expr, alias=sql.Identifier(metric["alias"])
                    )
                )
            query = sql.SQL("SELECT {fields} FROM {schema}.{table}").format(
                fields=sql.SQL(", ").join(select_parts),
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
            if group_parts:
                query += sql.SQL(" GROUP BY ") + sql.SQL(", ").join(group_parts)
            if order_items:
                order_parts = []
                for item in order_items:
                    order_parts.append(
                        sql.SQL("{col} {direction}").format(
                            col=sql.Identifier(item["column"]),
                            direction=sql.SQL(item["direction"].upper()),
                        )
                    )
                query += sql.SQL(" ORDER BY ") + sql.SQL(", ").join(order_parts)
            if row_limit:
                query += sql.SQL(" LIMIT %s")
                params.append(row_limit)

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                columns = [desc.name for desc in cur.description]
        finally:
            conn.close()

    return {"columns": columns, "rows": [dict(zip(columns, row)) for row in rows]}
