from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import REFERENCE_RULES_PATH
from .db_connections import execute_table_query_multi


def load_reference_rules() -> dict:
    path = Path(REFERENCE_RULES_PATH)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_reference_rules(rules: dict | None) -> dict:
    normalized = dict(rules or {})
    normalized.setdefault("db", {})
    normalized.setdefault("param_mapping", {})
    normalized.setdefault("param_columns", {})
    normalized.setdefault("param_value_map", {})
    normalized.setdefault("selection", {})
    normalized.setdefault("value1_not_null_columns", [])
    normalized.setdefault("defect_conditions", [])
    normalized.setdefault("detail_columns", [])
    normalized.setdefault("conditions", {})
    normalized.setdefault("grid_search", {})
    normalized.setdefault("defect_rate_aggregate", {})
    normalized["selection"].setdefault("max_candidates", 500)
    param_columns = normalized.get("param_columns")
    if not isinstance(param_columns, dict):
        param_columns = {}
        normalized["param_columns"] = param_columns
    param_columns.setdefault("temperature", "temperature")
    param_columns.setdefault("voltage", "voltage")
    param_columns.setdefault("size", "size")
    param_columns.setdefault("capacity", "capacity")
    param_columns.setdefault("production_mode", "production_mode")
    param_value_map = normalized.get("param_value_map")
    if not isinstance(param_value_map, dict):
        normalized["param_value_map"] = {}
    value1_not_null = normalized.get("value1_not_null_columns")
    if not isinstance(value1_not_null, list):
        value1_not_null = []
        normalized["value1_not_null_columns"] = value1_not_null
    normalized["conditions"].setdefault("required_not_null", [])
    normalized["conditions"].setdefault("design_factor_columns", [])
    normalized["conditions"].setdefault("defect_metrics", [])
    if not value1_not_null and normalized["conditions"].get("required_not_null"):
        normalized["value1_not_null_columns"] = list(
            normalized["conditions"].get("required_not_null", [])
        )
    normalized["grid_search"].setdefault("factors", [])
    normalized["grid_search"].setdefault("max_results", 100)
    normalized["grid_search"].setdefault("top_k", 10)
    normalized["defect_rate_aggregate"].setdefault("columns", [])
    normalized["defect_rate_aggregate"].setdefault("mode", "avg")

    db = normalized.get("db", {})
    tables = db.get("tables")
    if not isinstance(tables, dict):
        tables = {}
        db["tables"] = tables
    tables.setdefault("param_map", "")
    tables.setdefault("chip_prod", "")
    tables.setdefault("lot_search", "")

    param_mapping = normalized.get("param_mapping")
    if not isinstance(param_mapping, dict):
        param_mapping = {}
        normalized["param_mapping"] = param_mapping
    param_mapping.setdefault("schema", "")
    param_mapping.setdefault("table", "")
    param_mapping.setdefault("name_column", "")
    mappings = param_mapping.get("mappings")
    if not isinstance(mappings, dict):
        mappings = {}
        param_mapping["mappings"] = mappings
    return normalized


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_condition(value: Any, rule: dict) -> bool:
    if value is None:
        return False
    text = str(value)
    min_length = rule.get("min_length")
    if isinstance(min_length, int) and len(text) < min_length:
        return False
    tokens = rule.get("contains_any") or []
    if not tokens:
        return True
    return any(token in text for token in tokens if isinstance(token, str))


def _all_not_null(row: dict, columns: list[str]) -> bool:
    for column in columns:
        value = row.get(column)
        if value is None or value == "":
            return False
    return True


def _defect_thresholds_ok(row: dict, metrics: list[dict]) -> bool:
    for metric in metrics:
        column = metric.get("column")
        minimum = metric.get("min")
        if not column:
            continue
        value = _coerce_float(row.get(column))
        if value is None:
            return False
        if minimum is not None and value < float(minimum):
            return False
    return True


def _score_row(row: dict, rules: dict) -> tuple[int, dict]:
    conditions = rules.get("conditions", {})
    screen_rule = conditions.get("screen_durable_spec") or {}
    cond1 = _string_condition(row.get(screen_rule.get("column")), screen_rule)
    cond2 = _all_not_null(row, conditions.get("required_not_null", []))
    cond3 = _all_not_null(row, conditions.get("design_factor_columns", []))
    cond4 = _defect_thresholds_ok(row, conditions.get("defect_metrics", []))
    score = sum([cond1, cond2, cond3, cond4])
    return score, {"cond1": cond1, "cond2": cond2, "cond3": cond3, "cond4": cond4}


def _aggregate_defect_rate(row: dict, columns: list[str]) -> float | None:
    values = []
    for column in columns:
        value = _coerce_float(row.get(column))
        if value is None:
            continue
        values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _get_value(row: dict, column: str | None) -> Any:
    if not column:
        return None
    if column in row:
        return row.get(column)
    lowered = column.lower()
    if lowered in row:
        return row.get(lowered)
    return None


def _normalize_defect_conditions(rules: dict) -> list[dict]:
    conditions = rules.get("defect_conditions") or []
    normalized: list[dict] = []
    for item in conditions:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        if not column:
            continue
        operator = item.get("operator") or "="
        value = item.get("value")
        if value is None and operator not in {"is_null", "is_not_null"}:
            continue
        key = str(item.get("key") or column)
        label = str(item.get("label") or key)
        normalized.append(
            {
                "key": key,
                "label": label,
                "column": column,
                "operator": operator,
                "value": value,
            }
        )
    return normalized


def _build_param_filters(params: dict, rules: dict) -> list[dict]:
    param_columns = rules.get("param_columns") or {}
    value_map = rules.get("param_value_map") or {}
    filters: list[dict] = []
    for param_key, column in param_columns.items():
        if not column:
            continue
        value = params.get(param_key)
        if value is None or value == "":
            continue
        mapped = value
        mapping = value_map.get(param_key)
        if isinstance(mapping, dict):
            if value in mapping:
                mapped = mapping[value]
            else:
                mapped = mapping.get(str(value), value)
        filters.append({"column": column, "operator": "=", "value": mapped})
    return filters


def _resolve_table_name(rules: dict, key: str) -> str:
    db = rules.get("db", {})
    tables = db.get("tables") or {}
    table = tables.get(key) if isinstance(tables, dict) else None
    if table:
        return str(table)
    return str(db.get("table") or "")


def _map_params_from_table(params: dict, rules: dict) -> dict:
    mapping = rules.get("param_mapping") or {}
    mappings = mapping.get("mappings") or {}
    if not isinstance(mappings, dict) or not mappings:
        return params
    db = rules.get("db", {})
    connection_id = db.get("connection_id")
    if not connection_id:
        return params
    table_name = mapping.get("table") or _resolve_table_name(rules, "param_map")
    if not table_name:
        return params
    schema_name = mapping.get("schema") or db.get("schema") or "public"
    name_column = mapping.get("name_column")
    updated = dict(params)

    for param_key, config in mappings.items():
        if param_key not in params:
            continue
        if not isinstance(config, dict):
            continue
        input_column = config.get("input_column")
        output_column = config.get("output_column")
        if not input_column or not output_column:
            continue
        filters = []
        if name_column:
            filters.append({"column": name_column, "operator": "=", "value": param_key})
        filters.append(
            {"column": input_column, "operator": "=", "value": params.get(param_key)}
        )
        try:
            result = execute_table_query_multi(
                connection_id=connection_id,
                schema_name=schema_name,
                table_name=table_name,
                columns=[output_column],
                filters=filters,
                limit=1,
            )
        except Exception:
            continue
        rows = result.get("rows", []) if isinstance(result, dict) else []
        if not rows:
            continue
        mapped_value = rows[0].get(output_column)
        if mapped_value is not None:
            updated[param_key] = mapped_value
    return updated


def _build_not_null_filters(columns: list[str]) -> list[dict]:
    filters = []
    for column in columns:
        if column:
            filters.append({"column": column, "operator": "is_not_null"})
    return filters


def _build_defect_filters(conditions: list[dict]) -> list[dict]:
    filters = []
    for condition in conditions:
        column = condition.get("column")
        if not column:
            continue
        filters.append(
            {
                "column": column,
                "operator": condition.get("operator") or "=",
                "value": condition.get("value"),
            }
        )
    return filters


def _count_by_key(rows: list[dict], key_column: str, id_column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = _get_value(row, key_column)
        if key is None:
            continue
        key_text = str(key)
        lot_id = _get_value(row, id_column)
        if lot_id is not None:
            token = (key_text, str(lot_id))
            if token in seen:
                continue
            seen.add(token)
        counts[key_text] = counts.get(key_text, 0) + 1
    return counts


def _select_latest_row(rows: list[dict], date_column: str) -> dict | None:
    latest = None
    latest_dt = None
    for row in rows:
        dt = _parse_datetime(_get_value(row, date_column))
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest = row
            latest_dt = dt
    return latest


def _build_columns(rules: dict) -> list[str]:
    columns = set()
    db = rules.get("db", {})
    columns.update(
        {
            db.get("lot_id_column") or "lot_id",
            db.get("chip_prod_id_column") or "chip_prod_id",
            db.get("input_date_column") or "input_date",
        }
    )
    columns.update(rules.get("value1_not_null_columns", []) or [])
    columns.update(rules.get("detail_columns", []) or [])
    conditions = rules.get("conditions", {})
    screen = conditions.get("screen_durable_spec") or {}
    if screen.get("column"):
        columns.add(screen["column"])
    columns.update(conditions.get("required_not_null", []))
    columns.update(conditions.get("design_factor_columns", []))
    for metric in conditions.get("defect_metrics", []):
        column = metric.get("column")
        if column:
            columns.add(column)
    grid = rules.get("grid_search", {})
    for factor in grid.get("factors", []):
        column = factor.get("column")
        if column:
            columns.add(column)
    return [column for column in columns if column]


def _matches_filter(row: dict, item: dict) -> bool:
    column = item.get("column")
    operator = (item.get("operator") or "=").lower()
    value = item.get("value")
    row_value = _get_value(row, column)
    if operator == "is_null":
        return row_value is None or row_value == ""
    if operator == "is_not_null":
        return row_value is not None and row_value != ""
    if row_value is None:
        return False
    if operator in {"like", "ilike"}:
        haystack = str(row_value)
        needle = str(value or "")
        if operator == "ilike":
            return needle.lower() in haystack.lower()
        return needle in haystack
    if operator == "in":
        if not isinstance(value, list):
            return False
        return row_value in value or str(row_value) in [str(v) for v in value]
    if operator == "between":
        if not isinstance(value, list) or len(value) != 2:
            return False
        low = _coerce_float(value[0])
        high = _coerce_float(value[1])
        current = _coerce_float(row_value)
        if low is None or high is None or current is None:
            return False
        return low <= current <= high
    if operator in {">", ">=", "<", "<="}:
        current = _coerce_float(row_value)
        target = _coerce_float(value)
        if current is None or target is None:
            return False
        if operator == ">":
            return current > target
        if operator == ">=":
            return current >= target
        if operator == "<":
            return current < target
        if operator == "<=":
            return current <= target
    if operator == "=":
        if value is None:
            return False
        current = _coerce_float(row_value)
        target = _coerce_float(value)
        if current is not None and target is not None:
            return current == target
        return str(row_value) == str(value)
    return False


def _apply_demo_filters(rows: list[dict], filters: list[dict]) -> None:
    if not rows:
        return
    row = rows[0]
    for item in filters:
        column = item.get("column")
        operator = (item.get("operator") or "=").lower()
        value = item.get("value")
        if not column:
            continue
        if operator == "is_null":
            row[column] = None
            continue
        if operator == "is_not_null":
            if row.get(column) in (None, ""):
                row[column] = 1
            continue
        if operator in {"like", "ilike"}:
            row[column] = value or "demo"
            continue
        if operator == "in":
            if isinstance(value, list) and value:
                row[column] = value[0]
            continue
        if operator == "between":
            if isinstance(value, list) and len(value) == 2:
                low = _coerce_float(value[0])
                high = _coerce_float(value[1])
                if low is not None and high is not None:
                    row[column] = (low + high) / 2
            continue
        if value is None:
            continue
        row[column] = value


def _filter_demo_rows(rows: list[dict], filters: list[dict]) -> list[dict]:
    if not filters:
        return rows
    filtered = []
    for row in rows:
        if all(_matches_filter(row, item) for item in filters):
            filtered.append(row)
    return filtered


def _query_rows_for_filters(
    rules: dict,
    filters: list[dict],
    columns: list[str],
    limit: int,
    table_name: str | None = None,
) -> tuple[str, list[dict]]:
    db = rules.get("db", {})
    connection_id = db.get("connection_id")
    resolved_table = table_name or db.get("table")
    if not connection_id or not resolved_table:
        chip_prod_column = db.get("chip_prod_id_column") or "chip_prod_id"
        lot_id_column = db.get("lot_id_column") or "lot_id"
        filter_column = chip_prod_column
        filter_value = "DEMO-CHIP"
        for item in filters:
            if item.get("column") == lot_id_column and item.get("value"):
                filter_column = lot_id_column
                filter_value = str(item.get("value"))
                break
        if filter_column != lot_id_column:
            for item in filters:
                if item.get("column") == chip_prod_column and item.get("value"):
                    filter_column = chip_prod_column
                    filter_value = str(item.get("value"))
                    break
        demo_columns = list(
            dict.fromkeys(
                columns
                + [
                    item.get("column")
                    for item in filters
                    if isinstance(item, dict) and item.get("column")
                ]
            )
        )
        rows = [dict(row) for row in _build_demo_rows(rules, demo_columns, filter_column, filter_value)]
        desired_chip = None
        for item in filters:
            if item.get("column") == chip_prod_column and item.get("value"):
                desired_chip = str(item.get("value"))
                break
        if not desired_chip:
            desired_chip = filter_value if filter_column == chip_prod_column else "DEMO-CHIP"
        for row in rows:
            if chip_prod_column and desired_chip:
                row[chip_prod_column] = desired_chip
        _apply_demo_filters(rows, filters)
        filtered = _filter_demo_rows(rows, filters)
        if not filtered:
            filtered = rows
        row_limit = limit if isinstance(limit, int) and limit > 0 else 12
        return "demo", filtered[:row_limit]
    result = execute_table_query_multi(
        connection_id=connection_id,
        schema_name=db.get("schema") or "public",
        table_name=resolved_table,
        columns=columns,
        filters=filters,
        limit=limit,
    )
    rows = result.get("rows", []) if isinstance(result, dict) else []
    return "postgresql", rows


def select_reference_from_params(params: dict, model_override: str | None = None) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    db = rules.get("db", {})
    chip_prod_column = db.get("chip_prod_id_column") or "chip_prod_id"
    lot_id_column = db.get("lot_id_column") or "lot_id"
    input_date_column = db.get("input_date_column") or "input_date"
    chip_prod_table = _resolve_table_name(rules, "chip_prod")
    lot_search_table = _resolve_table_name(rules, "lot_search")
    max_candidates = rules.get("selection", {}).get("max_candidates", 500)
    limit = max_candidates if isinstance(max_candidates, int) and max_candidates > 0 else 0

    mapped_params = _map_params_from_table(params, rules)
    param_filters = _build_param_filters(mapped_params, rules)
    if model_override:
        param_filters.append(
            {"column": chip_prod_column, "operator": "=", "value": model_override}
        )

    base_columns = [chip_prod_column, lot_id_column, input_date_column]
    source, chip_prod_rows = _query_rows_for_filters(
        rules,
        param_filters,
        [chip_prod_column],
        limit,
        table_name=chip_prod_table,
    )
    chip_prod_ids = sorted(
        {
            str(_get_value(row, chip_prod_column))
            for row in chip_prod_rows
            if _get_value(row, chip_prod_column) is not None
        }
    )
    if not chip_prod_ids:
        return {
            "status": "missing",
            "error": "조건에 맞는 chip_prod_id가 없습니다.",
            "chip_prod_ids": [],
            "source": source,
        }

    value1_filters = param_filters + _build_not_null_filters(
        rules.get("value1_not_null_columns", []) or []
    )
    _, value1_rows = _query_rows_for_filters(
        rules,
        value1_filters,
        base_columns,
        limit,
        table_name=lot_search_table,
    )
    value1_counts = _count_by_key(value1_rows, chip_prod_column, lot_id_column)

    defect_conditions = _normalize_defect_conditions(rules)
    defect_filters_all = _build_defect_filters(defect_conditions)
    value2_all_filters = value1_filters + defect_filters_all
    _, value2_all_rows = _query_rows_for_filters(
        rules,
        value2_all_filters,
        base_columns,
        limit,
        table_name=lot_search_table,
    )

    if defect_conditions:
        latest_row = _select_latest_row(value2_all_rows, input_date_column)
    else:
        latest_row = _select_latest_row(value1_rows, input_date_column)
    if not latest_row:
        return {
            "status": "missing",
            "error": "조건에 맞는 LOT가 없습니다.",
            "chip_prod_ids": chip_prod_ids,
            "value1_counts": value1_counts,
            "source": source,
        }

    selected_chip_prod = _get_value(latest_row, chip_prod_column)
    selected_chip_prod_id = (
        str(selected_chip_prod) if selected_chip_prod is not None else None
    )
    selected_lot_id_value = _get_value(latest_row, lot_id_column)
    selected_lot_id = str(selected_lot_id_value) if selected_lot_id_value else ""
    if not selected_chip_prod_id:
        selected_chip_prod_id = model_override or (chip_prod_ids[0] if chip_prod_ids else "")

    value2_all_counts = _count_by_key(value2_all_rows, chip_prod_column, lot_id_column)
    defect_rates = []
    value2_counts_by_condition: dict[str, int] = {}
    for condition in defect_conditions:
        condition_filters = value1_filters + _build_defect_filters([condition])
        _, condition_rows = _query_rows_for_filters(
            rules,
            condition_filters,
            base_columns,
            limit,
            table_name=lot_search_table,
        )
        counts = _count_by_key(condition_rows, chip_prod_column, lot_id_column)
        value2_count = counts.get(selected_chip_prod_id or "", 0)
        value2_counts_by_condition[condition["key"]] = value2_count
        value1_count = value1_counts.get(selected_chip_prod_id or "", 0)
        rate = None
        if value1_count > 0:
            rate = (value1_count - value2_count) / value1_count
        defect_rates.append(
            {
                "key": condition["key"],
                "label": condition.get("label") or condition["key"],
                "column": condition.get("column"),
                "value1_count": value1_count,
                "value2_count": value2_count,
                "defect_rate": rate,
            }
        )

    detail_columns = rules.get("detail_columns", []) or []
    grid_columns = [
        factor.get("column")
        for factor in rules.get("grid_search", {}).get("factors", [])
        if factor.get("column")
    ]
    detail_query_columns = list(
        dict.fromkeys(base_columns + grid_columns + list(detail_columns))
    )
    detail_row = None
    if selected_lot_id:
        _, detail_rows = _query_rows_for_filters(
            rules,
            [{"column": lot_id_column, "operator": "=", "value": selected_lot_id}],
            detail_query_columns,
            1,
            table_name=lot_search_table,
        )
        detail_row = detail_rows[0] if detail_rows else None

    return {
        "status": "ok",
        "chip_prod_ids": chip_prod_ids,
        "selected_chip_prod_id": selected_chip_prod_id,
        "selected_lot_id": selected_lot_id,
        "selected_row": detail_row or latest_row,
        "value1_counts": value1_counts,
        "value2_all_counts": value2_all_counts,
        "value2_counts_by_condition": value2_counts_by_condition,
        "defect_rates": defect_rates,
        "source": source,
        "columns": detail_query_columns,
    }


def get_lot_detail_by_id(lot_id: str) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    db = rules.get("db", {})
    lot_id_column = db.get("lot_id_column") or "lot_id"
    chip_prod_column = db.get("chip_prod_id_column") or "chip_prod_id"
    input_date_column = db.get("input_date_column") or "input_date"
    detail_columns = rules.get("detail_columns", []) or []
    grid_columns = [
        factor.get("column")
        for factor in rules.get("grid_search", {}).get("factors", [])
        if factor.get("column")
    ]
    lot_search_table = _resolve_table_name(rules, "lot_search")
    query_columns = list(
        dict.fromkeys([chip_prod_column, lot_id_column, input_date_column] + grid_columns + list(detail_columns))
    )
    source, rows = _query_rows_for_filters(
        rules,
        [{"column": lot_id_column, "operator": "=", "value": lot_id}],
        query_columns,
        1,
        table_name=lot_search_table,
    )
    if not rows:
        return {"status": "missing", "error": "LOT 정보를 찾지 못했습니다.", "source": source}
    return {"status": "ok", "row": rows[0], "columns": query_columns, "source": source}


_DEMO_CACHE: dict[str, list[dict]] = {}


def _demo_base_value(seed: int, step: float, precision: int = 4) -> float:
    return round(seed * step, precision)


def _build_demo_rows(
    rules: dict, columns: list[str], filter_column: str, filter_value: str
) -> list[dict]:
    cache_key = f"{filter_column}:{filter_value}"
    cached = _DEMO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    db = rules.get("db", {})
    conditions = rules.get("conditions", {})
    grid = rules.get("grid_search", {})
    lot_id_column = db.get("lot_id_column") or "lot_id"
    chip_prod_column = db.get("chip_prod_id_column") or "chip_prod_id"
    input_date_column = db.get("input_date_column") or "input_date"
    screen_column = (conditions.get("screen_durable_spec") or {}).get("column")
    value1_required = rules.get("value1_not_null_columns", []) or []
    required = list(dict.fromkeys(list(conditions.get("required_not_null", [])) + list(value1_required)))
    design_columns = list(conditions.get("design_factor_columns", []))
    defect_columns = [
        metric.get("column")
        for metric in conditions.get("defect_metrics", [])
        if metric.get("column")
    ]
    defect_condition_columns = [
        item.get("column") or item.get("name")
        for item in (rules.get("defect_conditions") or [])
        if isinstance(item, dict) and (item.get("column") or item.get("name"))
    ]
    defect_columns = list(dict.fromkeys(defect_columns + defect_condition_columns))
    grid_columns = [
        factor.get("column")
        for factor in grid.get("factors", [])
        if factor.get("column")
    ]
    param_columns = rules.get("param_columns") or {}
    param_defaults = {
        "temperature": 120,
        "voltage": 3.7,
        "size": 12,
        "capacity": 6,
        "production_mode": "양산",
    }

    model_name = filter_value or "DEMO-MODEL"
    total_rows = 1 if filter_column == lot_id_column else 8
    now = datetime.utcnow()
    rows: list[dict] = []

    for idx in range(total_rows):
        row = {column: None for column in columns}
        row[chip_prod_column] = model_name
        row[lot_id_column] = f"DEMO-{model_name[:8]}-{idx + 1:03d}"
        row[input_date_column] = (now - timedelta(days=idx)).isoformat() + "Z"
        if screen_column:
            row[screen_column] = "Normal/MF"

        for col in required:
            row[col] = _demo_base_value(idx + 10, 0.15)
        for idx_col, col in enumerate(design_columns, start=1):
            row[col] = _demo_base_value(idx + idx_col, 0.2)
        for idx_col, col in enumerate(defect_columns, start=1):
            row[col] = _demo_base_value(idx + idx_col, 0.01)
        for param_key, column in param_columns.items():
            if not column:
                continue
            if column in columns:
                default_value = param_defaults.get(param_key)
                if default_value is not None:
                    row[column] = default_value

        for col in grid_columns:
            if col == "sheet_t":
                base = 12.0
            elif col == "laydown":
                base = 4.5
            elif col == "active_layer":
                base = 1.2
            else:
                base = 2.0
            row[col] = round(base + idx * 0.1, 4)

        if filter_column:
            row[filter_column] = filter_value
        rows.append(row)

    if filter_column == chip_prod_column and rows:
        rows[0][lot_id_column] = "ALA7K1K"

    if len(rows) > 3 and screen_column:
        rows[-1][screen_column] = "OTHER"
    if len(rows) > 4 and required:
        rows[-2][required[0]] = None
    if len(rows) > 5 and design_columns:
        rows[-3][design_columns[0]] = None
    if len(rows) > 6 and defect_columns:
        rows[-4][defect_columns[0]] = -1

    _DEMO_CACHE[cache_key] = rows
    return rows


def _query_rows(
    filter_column: str, filter_value: str, table_name: str | None = None
) -> tuple[dict, list[dict]]:
    rules = normalize_reference_rules(load_reference_rules())
    db = rules.get("db", {})
    connection_id = db.get("connection_id")
    resolved_table = table_name or db.get("table")
    columns = _build_columns(rules)
    if not connection_id or not resolved_table:
        rows = _build_demo_rows(rules, columns, filter_column, filter_value)
        return {"rules": rules, "columns": columns, "source": "demo"}, rows

    filters = [
        {
            "column": filter_column,
            "operator": "=",
            "value": filter_value,
        }
    ]
    max_candidates = rules.get("selection", {}).get("max_candidates", 500)
    try:
        result = execute_table_query_multi(
            connection_id=connection_id,
            schema_name=db.get("schema") or "public",
            table_name=resolved_table,
            columns=columns,
            filters=filters,
            limit=max(1, int(max_candidates)),
        )
        rows = result.get("rows", []) if isinstance(result, dict) else []
        return {"rules": rules, "columns": columns, "source": "postgresql"}, rows
    except Exception:
        rows = _build_demo_rows(rules, columns, filter_column, filter_value)
        return {"rules": rules, "columns": columns, "source": "demo"}, rows


def _build_reference_payload(rows: list[dict], meta: dict) -> dict:
    rules = meta.get("rules", {})
    db = rules.get("db", {})
    lot_id_column = db.get("lot_id_column") or "lot_id"
    input_date_column = db.get("input_date_column") or "input_date"
    aggregate_columns = rules.get("defect_rate_aggregate", {}).get("columns", [])

    scored_rows = []
    for row in rows:
        score, condition_state = _score_row(row, rules)
        scored_rows.append(
            {
                "row": row,
                "score": score,
                "conditions": condition_state,
                "input_date": _parse_datetime(row.get(input_date_column)),
            }
        )

    all_match = [item for item in scored_rows if item["score"] == 4]
    pool = all_match or scored_rows
    pool.sort(
        key=lambda item: (
            item["score"],
            item["input_date"] or datetime.min,
        ),
        reverse=True,
    )
    selected = pool[0]
    selected_row = selected["row"]

    defect_rates = []
    for item in scored_rows:
        rate = _aggregate_defect_rate(item["row"], aggregate_columns)
        if rate is None:
            continue
        lot_id = item["row"].get(lot_id_column)
        if lot_id:
            defect_rates.append({"lot_id": lot_id, "defect_rate": rate})

    return {
        "status": "ok",
        "lot_id": selected_row.get(lot_id_column),
        "row": selected_row,
        "score": selected["score"],
        "conditions": selected["conditions"],
        "columns": meta.get("columns", []),
        "rows": rows,
        "source": meta.get("source"),
        "defect_rates": defect_rates,
        "candidate_count": len(rows),
    }


def select_reference_lot(chip_prod_id: str) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    lot_search_table = _resolve_table_name(rules, "lot_search")
    meta, rows = _query_rows(
        rules.get("db", {}).get("chip_prod_id_column") or "chip_prod_id",
        chip_prod_id,
        table_name=lot_search_table,
    )
    if not rows:
        if meta.get("status") == "error":
            return meta
        return {"status": "missing", "error": "reference lot 후보가 없습니다.", "rows": []}
    return _build_reference_payload(rows, meta)


def select_reference_lot_by_id(lot_id: str) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    lot_search_table = _resolve_table_name(rules, "lot_search")
    column = rules.get("db", {}).get("lot_id_column") or "lot_id"
    meta, rows = _query_rows(column, lot_id, table_name=lot_search_table)
    if not rows:
        if meta.get("status") == "error":
            return meta
        return {"status": "missing", "error": "reference lot 후보가 없습니다.", "rows": []}
    return _build_reference_payload(rows, meta)


def build_grid_values(row: dict, rules: dict, overrides: dict | None = None) -> dict:
    grid = {}
    overrides = overrides or {}
    for factor in rules.get("grid_search", {}).get("factors", []):
        name = factor.get("name")
        column = factor.get("column")
        if not name:
            continue
        if name in overrides:
            base = overrides[name]
        else:
            base = row.get(column) if column else None
        base_value = _coerce_float(base)
        if base_value is None:
            grid[name] = [base] if base is not None else []
            continue
        percent = float(factor.get("range_percent", 10))
        points = max(2, int(factor.get("points", 5)))
        low = base_value * (1 - percent / 100)
        high = base_value * (1 + percent / 100)
        step = (high - low) / (points - 1)
        grid[name] = [round(low + step * idx, 6) for idx in range(points)]
    return grid
