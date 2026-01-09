from __future__ import annotations

import json
from functools import cmp_to_key
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import REFERENCE_RULES_PATH
from .db_connections import execute_table_query_multi, get_schema


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
    normalized.setdefault("defect_conditions", [])
    normalized.setdefault("detail_columns", [])
    normalized.setdefault("conditions", {})
    normalized.setdefault("grid_search", {})
    normalized.setdefault("defect_metric_catalog", {})
    normalized.setdefault("defect_rate_source", {})
    normalized.setdefault("final_briefing", {})
    normalized.setdefault(
        "grid_match_fields",
        [
            "active_powder_base",
            "active_powder_additives",
            "ldn_avr_value",
            "cast_dsgn_thk",
        ],
    )
    normalized.setdefault("grid_match_field_aliases", {"ldn_avr_value": ["ldn_avg_value"]})
    normalized.setdefault("grid_match_recent_months", 6)
    normalized.setdefault("grid_match_row_limit", 50)
    normalized.setdefault("grid_defect_columns", [])
    normalized.setdefault("grid_defect_value_unit", "")
    selection = normalized.get("selection")
    if not isinstance(selection, dict):
        selection = {}
        normalized["selection"] = selection
    selection.setdefault("max_candidates", 500)
    selection.setdefault("lot_search_sort", [])
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
    defect_metric_catalog = normalized.get("defect_metric_catalog")
    if not isinstance(defect_metric_catalog, dict):
        normalized["defect_metric_catalog"] = {}
    conditions = normalized.get("conditions")
    if not isinstance(conditions, dict):
        conditions = {}
        normalized["conditions"] = conditions
    conditions.setdefault("required_not_null", [])
    conditions.setdefault("design_factor_columns", [])
    legacy_not_null = normalized.pop("value1_not_null_columns", None)
    if isinstance(legacy_not_null, list) and legacy_not_null:
        if not conditions.get("required_not_null"):
            conditions["required_not_null"] = list(legacy_not_null)
    normalized["grid_search"].setdefault("factors", [])
    normalized["grid_search"].setdefault("max_results", 100)
    normalized["grid_search"].setdefault("top_k", 10)
    defect_rate_source = normalized.get("defect_rate_source")
    if not isinstance(defect_rate_source, dict):
        defect_rate_source = {}
        normalized["defect_rate_source"] = defect_rate_source
    defect_rate_source.setdefault("connection_id", "")
    defect_rate_source.setdefault("schema", "")
    defect_rate_source.setdefault("table", "")
    defect_rate_source.setdefault("lot_id_column", "lot_id")
    defect_rate_source.setdefault("update_date_column", "")
    defect_rate_source.setdefault("rate_columns", [])
    defect_rate_source.setdefault("value_unit", "")

    final_briefing = normalized.get("final_briefing")
    if not isinstance(final_briefing, dict):
        final_briefing = {}
        normalized["final_briefing"] = final_briefing
    final_briefing.setdefault("max_blocks", 3)
    final_briefing.setdefault("design_fields", [])
    final_briefing.setdefault("design_labels", {})
    final_briefing.setdefault("chart_bins", 8)

    match_fields = normalized.get("grid_match_fields")
    if not isinstance(match_fields, list):
        normalized["grid_match_fields"] = []
    match_aliases = normalized.get("grid_match_field_aliases")
    if not isinstance(match_aliases, dict):
        normalized["grid_match_field_aliases"] = {}
    grid_defect_columns = normalized.get("grid_defect_columns")
    if not isinstance(grid_defect_columns, list):
        normalized["grid_defect_columns"] = []
    try:
        grid_recent = int(normalized.get("grid_match_recent_months", 6))
    except (TypeError, ValueError):
        grid_recent = 6
    normalized["grid_match_recent_months"] = max(0, grid_recent)
    try:
        grid_row_limit = int(normalized.get("grid_match_row_limit", 50))
    except (TypeError, ValueError):
        grid_row_limit = 50
    normalized["grid_match_row_limit"] = max(1, grid_row_limit)

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


def _normalize_defect_rate_value(value: Any, value_unit: str) -> float | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    if value_unit == "percent":
        return numeric / 100.0
    return numeric


def _infer_rate_columns(row: dict, exclude: set[str]) -> list[str]:
    columns: list[str] = []
    for key, value in row.items():
        if key in exclude:
            continue
        if _coerce_float(value) is None:
            continue
        columns.append(key)
    return columns


def _looks_like_defect_column(column: str) -> bool:
    lowered = column.lower()
    hints = ("defect", "fail", "reject", "ng", "rate")
    return any(token in lowered for token in hints)


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


def _score_row(row: dict, rules: dict) -> tuple[int, dict]:
    conditions = rules.get("conditions", {})
    screen_rule = conditions.get("screen_durable_spec") or {}
    cond1 = _string_condition(row.get(screen_rule.get("column")), screen_rule)
    cond2 = _all_not_null(row, conditions.get("required_not_null", []))
    cond3 = _all_not_null(row, conditions.get("design_factor_columns", []))
    score = sum([cond1, cond2, cond3])
    return score, {"cond1": cond1, "cond2": cond2, "cond3": cond3}


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


def _normalize_lot_search_sort(rules: dict) -> list[dict]:
    selection = rules.get("selection") or {}
    sort_specs = selection.get("lot_search_sort") or []
    if not isinstance(sort_specs, list):
        return []
    normalized: list[dict] = []
    for item in sort_specs:
        if not isinstance(item, dict):
            continue
        columns = item.get("columns")
        normalized_columns = [
            str(column) for column in columns if column
        ] if isinstance(columns, list) else []
        if not normalized_columns:
            column = item.get("column") or item.get("name")
            if not column:
                continue
            normalized_columns = [str(column)]
        order = str(item.get("order") or item.get("direction") or "asc").lower()
        if order not in {"asc", "desc"}:
            order = "asc"
        priority = item.get("value_priority")
        value_priority: dict[str, int] | None = None
        if isinstance(priority, dict):
            value_priority = {}
            for key, value in priority.items():
                try:
                    value_priority[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            if not value_priority:
                value_priority = None
        combine = str(item.get("combine") or "").lower()
        if combine not in {"min", "max", "sum"}:
            combine = "min" if len(normalized_columns) > 1 else ""
        normalized.append(
            {
                "columns": normalized_columns,
                "order": order,
                "value_priority": value_priority,
                "combine": combine,
            }
        )
    return normalized


def _lot_search_sort_columns(rules: dict) -> list[str]:
    columns: list[str] = []
    for item in _normalize_lot_search_sort(rules):
        for column in item.get("columns") or []:
            if column:
                columns.append(column)
    return columns


def _resolve_grid_match_fields(rules: dict) -> list[str]:
    fields = rules.get("grid_match_fields") or []
    if not isinstance(fields, list):
        return []
    return [str(field) for field in fields if field]


def _resolve_grid_match_aliases(rules: dict) -> dict[str, list[str]]:
    aliases = rules.get("grid_match_field_aliases") or {}
    if not isinstance(aliases, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, value in aliases.items():
        if not key or not isinstance(value, list):
            continue
        items = [str(item) for item in value if item]
        if items:
            normalized[str(key)] = items
    return normalized


def _resolve_defect_metric_groups(rules: dict) -> list[dict]:
    catalog = rules.get("defect_metric_catalog")
    if not isinstance(catalog, dict):
        return []
    groups = catalog.get("groups")
    if not isinstance(groups, list):
        return []
    return [group for group in groups if isinstance(group, dict)]


def _resolve_defect_metric_columns(
    rules: dict, include_children: bool = True
) -> list[str]:
    columns: list[str] = []
    for group in _resolve_defect_metric_groups(rules):
        metric = group.get("metric")
        if metric:
            columns.append(str(metric))
        if include_children:
            children = group.get("children") or []
            if isinstance(children, list):
                for child in children:
                    if child:
                        columns.append(str(child))
    return [column for column in dict.fromkeys(columns) if column]


def _resolve_defect_metric_main_columns(rules: dict) -> list[str]:
    columns: list[str] = []
    for group in _resolve_defect_metric_groups(rules):
        metric = group.get("metric")
        if metric:
            columns.append(str(metric))
    return [column for column in dict.fromkeys(columns) if column]


def _resolve_grid_defect_columns(
    rules: dict, available: list[str] | None = None
) -> list[str]:
    columns = [column for column in (rules.get("grid_defect_columns") or []) if column]
    if columns:
        return columns
    columns = [column for column in (rules.get("post_grid_defect_columns") or []) if column]
    if columns:
        return columns
    if available:
        return [column for column in available if _looks_like_defect_column(column)]
    return []


def _resolve_grid_defect_value_unit(rules: dict) -> str:
    value_unit = str(rules.get("grid_defect_value_unit") or "").strip().lower()
    if value_unit:
        return value_unit
    return str(rules.get("post_grid_defect_value_unit") or "").strip().lower()


def _extract_match_values(
    design: dict, fields: list[str], aliases: dict[str, list[str]]
) -> dict[str, object]:
    if not isinstance(design, dict) or not fields:
        return {}
    extracted: dict[str, object] = {}
    for field in fields:
        if not field:
            continue
        value = design.get(field)
        if value in (None, ""):
            for alias in aliases.get(field, []):
                value = design.get(alias)
                if value not in (None, ""):
                    break
        if value in (None, ""):
            continue
        extracted[field] = value
    return extracted


def _summarize_defect_column_avgs(
    rows: list[dict], columns: list[str]
) -> list[dict]:
    summary = []
    for column in columns:
        values: list[float] = []
        for row in rows:
            value = _coerce_float(_get_value(row, column))
            if value is None:
                continue
            values.append(value)
        avg = sum(values) / len(values) if values else None
        summary.append(
            {"column": column, "avg": avg, "count": len(values)}
        )
    return summary


def _get_param_columns(rules: dict, table_key: str | None = None) -> dict:
    param_columns_by_table = rules.get("param_columns_by_table") or {}
    if table_key and isinstance(param_columns_by_table, dict):
        if table_key in param_columns_by_table:
            table_columns = param_columns_by_table.get(table_key)
            if isinstance(table_columns, dict):
                return table_columns
            if table_columns in {None, "", False}:
                return {}
    param_columns = rules.get("param_columns") or {}
    return param_columns if isinstance(param_columns, dict) else {}


def _build_param_filters(params: dict, rules: dict, table_key: str | None = None) -> list[dict]:
    param_columns = _get_param_columns(rules, table_key)
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


def _get_grid_payload_columns(rules: dict) -> list[str]:
    grid = rules.get("grid_search", {}) if isinstance(rules, dict) else {}
    payload = grid.get("payload_columns")
    columns: list[str] = []
    if isinstance(payload, list):
        columns = payload
    elif isinstance(payload, dict):
        for key in ("ref", "sim"):
            subset = payload.get(key)
            if isinstance(subset, list):
                columns.extend(subset)
    if not columns:
        return []
    return list(dict.fromkeys([str(column) for column in columns if column]))


def _get_available_columns(
    connection_id: str, schema_name: str, table_name: str
) -> list[str] | None:
    schema = get_schema(connection_id)
    if not schema or not isinstance(schema, dict):
        return None
    schemas = schema.get("schemas") or {}
    if not isinstance(schemas, dict):
        return None
    schema_entry = schemas.get(schema_name)
    if not isinstance(schema_entry, dict):
        return None
    table_entry = schema_entry.get("tables", {}).get(table_name)
    if not isinstance(table_entry, dict):
        return None
    columns = table_entry.get("columns", [])
    if not isinstance(columns, list):
        return None
    available = [
        col.get("name")
        for col in columns
        if isinstance(col, dict) and col.get("name")
    ]
    return available or None


def _filter_filters_by_schema(filters: list[dict], available: list[str] | None) -> list[dict]:
    if not available:
        return filters
    available_set = set(available)
    filtered: list[dict] = []
    for item in filters or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column") or item.get("name")
        if not column or column not in available_set:
            continue
        filtered.append(item)
    return filtered


def _filter_columns_by_schema(columns: list[str], available: list[str] | None) -> list[str]:
    if not available:
        return columns
    available_set = set(available)
    filtered = [column for column in columns if column in available_set]
    if filtered:
        return filtered
    return list(available)


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


def _build_design_filters(rules: dict) -> list[dict]:
    conditions = rules.get("conditions", {})
    required_not_null = conditions.get("required_not_null", []) or []
    design_columns = conditions.get("design_factor_columns", []) or []
    columns = list(
        dict.fromkeys(list(required_not_null) + list(design_columns))
    )
    return _build_not_null_filters(columns)


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


def _compare_sort_values(a_value: Any, b_value: Any, order: str) -> int:
    if a_value is None and b_value is None:
        return 0
    if a_value is None:
        return 1
    if b_value is None:
        return -1
    if isinstance(a_value, datetime) or isinstance(b_value, datetime):
        a_dt = a_value if isinstance(a_value, datetime) else _parse_datetime(a_value)
        b_dt = b_value if isinstance(b_value, datetime) else _parse_datetime(b_value)
        if a_dt is None and b_dt is None:
            return 0
        if a_dt is None:
            return 1
        if b_dt is None:
            return -1
        if a_dt == b_dt:
            return 0
        if order == "desc":
            return -1 if a_dt > b_dt else 1
        return -1 if a_dt < b_dt else 1
    a_num = _coerce_float(a_value)
    b_num = _coerce_float(b_value)
    if a_num is not None and b_num is not None:
        if a_num == b_num:
            return 0
        if order == "desc":
            return -1 if a_num > b_num else 1
        return -1 if a_num < b_num else 1
    a_text = str(a_value)
    b_text = str(b_value)
    if a_text == b_text:
        return 0
    if order == "desc":
        return -1 if a_text > b_text else 1
    return -1 if a_text < b_text else 1


def _priority_rank(value: Any, priority: dict[str, int], default_rank: int) -> int:
    if value is None:
        return default_rank
    return priority.get(str(value), default_rank)


def _combine_priority_ranks(ranks: list[int], combine: str) -> int:
    if not ranks:
        return 0
    if combine == "max":
        return max(ranks)
    if combine == "sum":
        return sum(ranks)
    return min(ranks)


def _select_lot_search_row(
    rows: list[dict], rules: dict, date_column: str
) -> dict | None:
    if not rows:
        return None
    sort_specs = _normalize_lot_search_sort(rules)
    if not sort_specs:
        return _select_latest_row(rows, date_column)

    def compare(left: dict, right: dict) -> int:
        for spec in sort_specs:
            columns = spec.get("columns") or []
            if not columns:
                continue
            order = spec.get("order") or "asc"
            priority = spec.get("value_priority")
            if isinstance(priority, dict) and priority:
                min_rank = min(priority.values())
                max_rank = max(priority.values())
                default_rank = (
                    min_rank - 1 if order == "desc" else max_rank + 1
                )
                left_ranks = [
                    _priority_rank(_get_value(left, column), priority, default_rank)
                    for column in columns
                ]
                right_ranks = [
                    _priority_rank(_get_value(right, column), priority, default_rank)
                    for column in columns
                ]
                combine = spec.get("combine") or "min"
                result = _compare_sort_values(
                    _combine_priority_ranks(left_ranks, combine),
                    _combine_priority_ranks(right_ranks, combine),
                    order,
                )
                if result != 0:
                    return result
                continue
            for column in columns:
                result = _compare_sort_values(
                    _get_value(left, column),
                    _get_value(right, column),
                    order,
                )
                if result != 0:
                    return result
        if date_column:
            return _compare_sort_values(
                _parse_datetime(_get_value(left, date_column)),
                _parse_datetime(_get_value(right, date_column)),
                "desc",
            )
        return 0

    return sorted(rows, key=cmp_to_key(compare))[0]


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
    columns.update(rules.get("detail_columns", []) or [])
    conditions = rules.get("conditions", {})
    screen = conditions.get("screen_durable_spec") or {}
    if screen.get("column"):
        columns.add(screen["column"])
    columns.update(conditions.get("required_not_null", []))
    columns.update(conditions.get("design_factor_columns", []))
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
    if operator == "is_distinct_from":
        if row_value is None and value is None:
            return False
        if row_value is None or value is None:
            return True
        current = _coerce_float(row_value)
        target = _coerce_float(value)
        if current is not None and target is not None:
            return current != target
        return str(row_value) != str(value)
    if operator == "eq_or_null":
        if value is None:
            return row_value is None or row_value == ""
        if row_value is None or row_value == "":
            return True
        current = _coerce_float(row_value)
        target = _coerce_float(value)
        if current is not None and target is not None:
            return current == target
        return str(row_value) == str(value)
    if row_value is None:
        return False
    if operator in {"like", "ilike"}:
        haystack = str(row_value)
        needle = str(value or "").replace("%", "")
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
    schema_name = db.get("schema") or "public"
    available = _get_available_columns(connection_id, schema_name, resolved_table)
    filters = _filter_filters_by_schema(filters, available)
    columns = _filter_columns_by_schema(columns, available)
    result = execute_table_query_multi(
        connection_id=connection_id,
        schema_name=schema_name,
        table_name=resolved_table,
        columns=columns,
        filters=filters,
        limit=limit,
    )
    rows = result.get("rows", []) if isinstance(result, dict) else []
    return "postgresql", rows


def select_reference_from_params(
    params: dict, chip_prod_override: str | None = None
) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    db = rules.get("db", {})
    connection_id = db.get("connection_id") or ""
    schema_name = db.get("schema") or "public"
    chip_prod_column = db.get("chip_prod_id_column") or "chip_prod_id"
    lot_id_column = db.get("lot_id_column") or "lot_id"
    input_date_column = db.get("input_date_column") or "input_date"
    chip_prod_table = _resolve_table_name(rules, "chip_prod")
    lot_search_table = _resolve_table_name(rules, "lot_search")
    max_candidates = rules.get("selection", {}).get("max_candidates", 500)
    limit = max_candidates if isinstance(max_candidates, int) and max_candidates > 0 else 0

    mapped_params = _map_params_from_table(params, rules)
    chip_prod_filters = _build_param_filters(mapped_params, rules, table_key="chip_prod")
    lot_search_filters = _build_param_filters(mapped_params, rules, table_key="lot_search")
    user_chip_prod = params.get("chip_prod_id")
    user_chip_prod_text = str(user_chip_prod).strip() if user_chip_prod else ""
    chip_prod_override_text = (
        str(chip_prod_override).strip() if chip_prod_override else ""
    )
    use_chip_prod_like = bool(user_chip_prod_text) and (
        not chip_prod_override_text or chip_prod_override_text == user_chip_prod_text
    )
    if use_chip_prod_like:
        lot_search_filters.append(
            {
                "column": chip_prod_column,
                "operator": "like",
                "value": f"%{user_chip_prod_text}%",
            }
        )
    elif chip_prod_override_text:
        chip_prod_filters.append(
            {
                "column": chip_prod_column,
                "operator": "=",
                "value": chip_prod_override_text,
            }
        )
        lot_search_filters.append(
            {
                "column": chip_prod_column,
                "operator": "=",
                "value": chip_prod_override_text,
            }
        )

    defect_conditions = _normalize_defect_conditions(rules)
    defect_columns = [
        condition.get("column")
        for condition in defect_conditions
        if condition.get("column")
    ]
    sort_columns = _lot_search_sort_columns(rules)
    base_columns = list(
        dict.fromkeys(
            [chip_prod_column, lot_id_column, input_date_column]
            + sort_columns
            + defect_columns
        )
    )
    source = ""
    chip_prod_ids: list[str] = []
    if not use_chip_prod_like:
        source, chip_prod_rows = _query_rows_for_filters(
            rules,
            chip_prod_filters,
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
        if chip_prod_ids:
            lot_search_filters.append(
                {
                    "column": chip_prod_column,
                    "operator": "in",
                    "value": chip_prod_ids,
                }
            )

    base_filters = lot_search_filters + _build_design_filters(rules)
    defect_filters_all = _build_defect_filters(defect_conditions)
    ref_lot_filters = base_filters + defect_filters_all
    ref_source, ref_rows = _query_rows_for_filters(
        rules,
        ref_lot_filters,
        base_columns,
        limit,
        table_name=lot_search_table,
    )
    if not source:
        source = ref_source

    if use_chip_prod_like:
        chip_prod_ids = sorted(
            {
                str(_get_value(row, chip_prod_column))
                for row in ref_rows
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
    latest_row = _select_lot_search_row(ref_rows, rules, input_date_column)
    if not latest_row:
        return {
            "status": "missing",
            "error": "조건에 맞는 LOT가 없습니다.",
            "chip_prod_ids": chip_prod_ids,
            "source": source,
        }

    selected_chip_prod = _get_value(latest_row, chip_prod_column)
    selected_chip_prod_id = (
        str(selected_chip_prod) if selected_chip_prod is not None else None
    )
    selected_lot_id_value = _get_value(latest_row, lot_id_column)
    selected_lot_id = str(selected_lot_id_value) if selected_lot_id_value else ""
    if not selected_chip_prod_id:
        selected_chip_prod_id = chip_prod_override or (
            chip_prod_ids[0] if chip_prod_ids else ""
        )

    defect_rates: list[dict] = []

    detail_columns = rules.get("detail_columns", []) or []
    grid_columns = [
        factor.get("column")
        for factor in rules.get("grid_search", {}).get("factors", [])
        if factor.get("column")
    ]
    payload_columns = _get_grid_payload_columns(rules)
    payload_db_columns: list[str] = []
    payload_missing_columns: list[str] = []
    if payload_columns:
        available = (
            _get_available_columns(connection_id, schema_name, lot_search_table)
            if connection_id and lot_search_table
            else None
        )
        if available:
            available_set = set(available)
            payload_db_columns = [
                column for column in payload_columns if column in available_set
            ]
            payload_missing_columns = [
                column for column in payload_columns if column not in available_set
            ]
        else:
            payload_db_columns = payload_columns
    detail_query_columns = list(
        dict.fromkeys(base_columns + grid_columns + list(detail_columns) + payload_db_columns)
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

    reference_columns = list(
        dict.fromkeys([lot_id_column] + defect_columns)
    )
    reference_rows = []
    if reference_columns:
        for row in ref_rows:
            if selected_chip_prod_id:
                chip_value = _get_value(row, chip_prod_column)
                if str(chip_value or "") != selected_chip_prod_id:
                    continue
            reference_rows.append(
                {column: _get_value(row, column) for column in reference_columns}
            )

    return {
        "status": "ok",
        "chip_prod_ids": chip_prod_ids,
        "selected_chip_prod_id": selected_chip_prod_id,
        "selected_lot_id": selected_lot_id,
        "selected_row": detail_row or latest_row,
        "defect_rates": defect_rates,
        "source": source,
        "columns": detail_query_columns,
        "reference_columns": reference_columns,
        "reference_rows": reference_rows,
        "grid_payload_columns": payload_columns,
        "grid_payload_db_columns": payload_db_columns,
        "grid_payload_missing_columns": payload_missing_columns,
    }


def _resolve_reference_detail_columns(rules: dict) -> list[str]:
    config = rules.get("final_briefing") or {}
    columns = config.get("reference_detail_columns") if isinstance(config, dict) else []
    if not isinstance(columns, list) or not columns:
        columns = rules.get("reference_detail_columns") or []
    if not isinstance(columns, list):
        return []
    return [str(column) for column in columns if column]


def get_lot_detail_by_id(
    lot_id: str,
    columns: list[str] | None = None,
    use_available_columns: bool = False,
) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    db = rules.get("db", {})
    connection_id = db.get("connection_id") or ""
    schema_name = db.get("schema") or "public"
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
    query_columns = []
    if isinstance(columns, list) and columns:
        query_columns = list(columns)
    else:
        override_columns = _resolve_reference_detail_columns(rules)
        if override_columns:
            query_columns = override_columns
        elif use_available_columns and connection_id and lot_search_table:
            available = _get_available_columns(connection_id, schema_name, lot_search_table)
            if available:
                query_columns = list(available)
        if not query_columns:
            query_columns = list(
                dict.fromkeys(
                    [chip_prod_column, lot_id_column, input_date_column]
                    + grid_columns
                    + list(detail_columns)
                )
            )
    if lot_id_column and lot_id_column not in query_columns:
        query_columns.insert(0, lot_id_column)
    if not query_columns:
        query_columns = [lot_id_column]
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


def get_defect_rates_by_lot_id(lot_id: str) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    defect_source = rules.get("defect_rate_source", {})
    if not isinstance(defect_source, dict):
        defect_source = {}
    table_name = defect_source.get("table") or ""
    if not table_name:
        return {
            "status": "missing",
            "defect_rates": [],
            "source": "none",
            "lot_id": lot_id,
        }

    db = rules.get("db", {})
    connection_id = defect_source.get("connection_id") or db.get("connection_id") or ""
    schema_name = defect_source.get("schema") or db.get("schema") or "public"
    lot_id_column = defect_source.get("lot_id_column") or "lot_id"
    update_date_column = defect_source.get("update_date_column") or ""
    rate_columns = defect_source.get("rate_columns") or []
    if not isinstance(rate_columns, list):
        rate_columns = []
    rate_columns = [column for column in rate_columns if column]
    value_unit = str(defect_source.get("value_unit") or "").strip().lower()

    columns = []
    if rate_columns:
        columns = [lot_id_column]
        if update_date_column:
            columns.append(update_date_column)
        columns.extend(rate_columns)
        columns = list(dict.fromkeys(columns))

    filters = [{"column": lot_id_column, "operator": "=", "value": lot_id}]
    rows: list[dict] = []
    source = "demo"
    if connection_id and table_name:
        try:
            result = execute_table_query_multi(
                connection_id=connection_id,
                schema_name=schema_name or "public",
                table_name=table_name,
                columns=columns,
                filters=filters,
                limit=20,
            )
            rows = result.get("rows", []) if isinstance(result, dict) else []
            source = "postgresql"
        except Exception:
            rows = _build_demo_rows(rules, columns, lot_id_column, lot_id)
            source = "demo"
    else:
        rows = _build_demo_rows(rules, columns, lot_id_column, lot_id)
        source = "demo"

    if not rows:
        return {
            "status": "missing",
            "defect_rates": [],
            "source": source,
            "lot_id": lot_id,
        }

    selected_row = None
    if update_date_column:
        selected_row = _select_latest_row(rows, update_date_column)
    if not selected_row:
        selected_row = rows[0]

    if not rate_columns:
        conditions = rules.get("conditions", {})
        aggregate_columns = _resolve_defect_metric_columns(
            rules, include_children=True
        )
        defect_condition_columns = [
            item.get("column") or item.get("name")
            for item in (rules.get("defect_conditions") or [])
            if isinstance(item, dict) and (item.get("column") or item.get("name"))
        ]
        candidate_columns = [
            col
            for col in dict.fromkeys(aggregate_columns + defect_condition_columns)
            if col and col in selected_row
        ]
        if candidate_columns:
            rate_columns = candidate_columns
        else:
            exclude = {lot_id_column}
            if update_date_column:
                exclude.add(update_date_column)
            exclude.update(conditions.get("required_not_null", []) or [])
            exclude.update(conditions.get("design_factor_columns", []) or [])
            exclude.update(rules.get("detail_columns", []) or [])
            grid = rules.get("grid_search", {}) or {}
            for factor in grid.get("factors", []):
                if not isinstance(factor, dict):
                    continue
                column = factor.get("column") or factor.get("name")
                if column:
                    exclude.add(column)
            param_columns = rules.get("param_columns") or {}
            if isinstance(param_columns, dict):
                for column in param_columns.values():
                    if column:
                        exclude.add(column)
            inferred = _infer_rate_columns(selected_row, exclude)
            rate_columns = [col for col in inferred if _looks_like_defect_column(col)]

    defect_rates = []
    for column in rate_columns:
        rate = _normalize_defect_rate_value(selected_row.get(column), value_unit)
        if rate is None:
            continue
        defect_rates.append(
            {
                "label": column,
                "column": column,
                "defect_rate": rate,
            }
        )

    return {
        "status": "ok",
        "defect_rates": defect_rates,
        "source": source,
        "lot_id": lot_id,
        "value_unit": value_unit,
    }


_DEMO_CACHE: dict[str, list[dict]] = {}


def _demo_base_value(seed: int, step: float, precision: int = 4) -> float:
    return round(seed * step, precision)


def _demo_fallback_value(column: str, row_index: int, col_index: int) -> object:
    name = column.lower()
    if "date" in name or "time" in name:
        return (datetime.utcnow() - timedelta(days=row_index)).isoformat() + "Z"
    if "flag" in name or name.endswith("_yn") or "pass" in name or "ispass" in name:
        return "OK"
    if "id" in name:
        return f"DEMO-{column.upper()[:12]}"
    if any(
        key in name
        for key in (
            "name",
            "type",
            "mode",
            "div",
            "powder",
            "binder",
            "dispersant",
            "paste",
        )
    ):
        return "DEMO"
    if any(
        key in name
        for key in ("rate", "ratio", "percent", "percentage", "cv", "sigma")
    ):
        return _demo_base_value(row_index + col_index, 0.01)
    if any(
        key in name
        for key in (
            "avg",
            "thk",
            "size",
            "length",
            "width",
            "voltage",
            "temp",
            "pressure",
            "shrinkage",
            "overlap",
            "layer",
            "area",
            "optical",
            "electrode",
            "dielectric",
            "firing",
            "casting",
            "corr",
            "amount",
            "value",
            "cap",
            "capa",
        )
    ):
        return _demo_base_value(row_index + col_index, 0.1)
    return _demo_base_value(row_index + col_index, 0.1)


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
    required = list(dict.fromkeys(list(conditions.get("required_not_null", []))))
    design_columns = list(conditions.get("design_factor_columns", []))
    defect_condition_columns = [
        item.get("column") or item.get("name")
        for item in (rules.get("defect_conditions") or [])
        if isinstance(item, dict) and (item.get("column") or item.get("name"))
    ]
    defect_columns = list(dict.fromkeys(defect_condition_columns))
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

    chip_prod_id = filter_value or "DEMO-MODEL"
    total_rows = 1 if filter_column == lot_id_column else 8
    now = datetime.utcnow()
    rows: list[dict] = []

    for idx in range(total_rows):
        row = {column: None for column in columns}
        row[chip_prod_column] = chip_prod_id
        row[lot_id_column] = f"DEMO-{chip_prod_id[:8]}-{idx + 1:03d}"
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

        for idx_col, col in enumerate(columns, start=1):
            if row.get(col) is None:
                row[col] = _demo_fallback_value(col, idx, idx_col)

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
    aggregate_columns = _resolve_defect_metric_main_columns(rules)

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
        if base is None and "default" in factor:
            base = factor.get("default")
        base_value = _coerce_float(base)
        if base_value is None:
            grid[name] = [base] if base is not None else []
            continue
        percent = float(factor.get("range_percent", 10))
        points_raw = int(factor.get("points", 5))
        if points_raw <= 1 or percent == 0:
            grid[name] = [round(base_value, 6)]
            continue
        points = max(2, points_raw)
        low = base_value * (1 - percent / 100)
        high = base_value * (1 + percent / 100)
        step = (high - low) / (points - 1)
        grid[name] = [round(low + step * idx, 6) for idx in range(points)]
    return grid


def _grid_factor_column_map(rules: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for factor in rules.get("grid_search", {}).get("factors", []):
        name = factor.get("name")
        column = factor.get("column") or name
        if name and column:
            mapping[name] = column
    return mapping


def _filter_rows_by_recent_months(
    rows: list[dict], date_column: str, recent_months: int
) -> list[dict]:
    if not rows or not date_column or recent_months <= 0:
        return rows
    cutoff = datetime.now() - timedelta(days=recent_months * 30)
    filtered = []
    for row in rows:
        dt = _parse_datetime(_get_value(row, date_column))
        if dt is None or dt < cutoff:
            continue
        filtered.append(row)
    return filtered


def _summarize_post_grid_defects(
    rows: list[dict], columns: list[str], value_unit: str
) -> dict:
    summary: dict[str, dict] = {}
    for column in columns:
        values: list[float] = []
        for row in rows:
            value = _normalize_defect_rate_value(_get_value(row, column), value_unit)
            if value is None:
                continue
            values.append(value)
        if not values:
            summary[column] = {"count": 0, "avg": None, "min": None, "max": None}
            continue
        summary[column] = {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _resolve_post_grid_defect_columns(rules: dict) -> list[str]:
    columns = [column for column in (rules.get("post_grid_defect_columns") or []) if column]
    if columns:
        return columns
    aggregate_columns = _resolve_defect_metric_columns(rules, include_children=True)
    conditions = rules.get("conditions", {})
    defect_condition_columns = [
        item.get("column") or item.get("name")
        for item in (rules.get("defect_conditions") or [])
        if isinstance(item, dict) and (item.get("column") or item.get("name"))
    ]
    return [
        column
        for column in dict.fromkeys(aggregate_columns + defect_condition_columns)
        if column
    ]


def _aggregate_defect_rate_for_row(
    row: dict, columns: list[str], value_unit: str
) -> float | None:
    values: list[float] = []
    for column in columns:
        value = _normalize_defect_rate_value(_get_value(row, column), value_unit)
        if value is None:
            continue
        values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def collect_post_grid_defects(
    design_candidates: list[dict],
    rules: dict,
    chip_prod_id: str | None = None,
) -> dict:
    normalized = normalize_reference_rules(rules)
    defect_columns = _resolve_post_grid_defect_columns(normalized)
    recent_months = normalized.get("post_grid_recent_months")
    if not isinstance(recent_months, int) or recent_months < 0:
        recent_months = 0
    input_unit = str(normalized.get("post_grid_defect_value_unit") or "").strip().lower()
    output_unit = "ratio" if input_unit == "percent" else input_unit
    if not defect_columns or not isinstance(design_candidates, list):
        return {
            "columns": defect_columns,
            "items": [],
            "recent_months": recent_months,
            "value_unit": output_unit,
        }

    db = normalized.get("db", {})
    lot_search_table = _resolve_table_name(normalized, "lot_search")
    lot_id_column = db.get("lot_id_column") or "lot_id"
    date_column = db.get("input_date_column") or ""
    if not date_column:
        defect_source = normalized.get("defect_rate_source", {})
        if isinstance(defect_source, dict):
            date_column = defect_source.get("update_date_column") or ""

    factor_map = _grid_factor_column_map(normalized)
    allowed_design_keys = set(factor_map.keys())
    max_candidates = normalized.get("selection", {}).get("max_candidates", 0)
    row_limit = max_candidates if isinstance(max_candidates, int) and max_candidates > 0 else 0

    items = []
    for candidate in design_candidates[:3]:
        if not isinstance(candidate, dict):
            continue
        design = candidate.get("design")
        if not isinstance(design, dict) or not design:
            continue
        filters = []
        if chip_prod_id:
            filters.append(
                {
                    "column": db.get("chip_prod_id_column") or "chip_prod_id",
                    "operator": "=",
                    "value": chip_prod_id,
                }
            )
        design_columns = []
        for key, value in design.items():
            if allowed_design_keys and key not in allowed_design_keys:
                continue
            column = factor_map.get(key) or key
            if not column:
                continue
            design_columns.append(column)
            if value is None or value == "":
                continue
            filters.append({"column": column, "operator": "=", "value": value})
        if not filters:
            continue

        query_columns = list(
            dict.fromkeys(
                [lot_id_column]
                + design_columns
                + ([date_column] if date_column else [])
                + defect_columns
            )
        )
        source, rows = _query_rows_for_filters(
            normalized,
            filters,
            query_columns,
            row_limit,
            table_name=lot_search_table,
        )
        rows = _filter_rows_by_recent_months(rows, date_column, recent_months)
        stats = _summarize_post_grid_defects(rows, defect_columns, input_unit)
        lot_rates: dict[str, list[float]] = {}
        for row in rows:
            lot_id = _get_value(row, lot_id_column)
            if lot_id is None:
                continue
            rate = _aggregate_defect_rate_for_row(row, defect_columns, input_unit)
            if rate is None:
                continue
            lot_key = str(lot_id)
            lot_rates.setdefault(lot_key, []).append(rate)
        lot_defect_rates = []
        for lot_id, rates in lot_rates.items():
            if not rates:
                continue
            lot_defect_rates.append(
                {
                    "lot_id": lot_id,
                    "defect_rate": sum(rates) / len(rates),
                }
            )
        lot_defect_rates.sort(
            key=lambda item: item.get("defect_rate", 0), reverse=True
        )
        items.append(
            {
                "rank": candidate.get("rank"),
                "predicted_target": candidate.get("predicted_target"),
                "design": design,
                "lot_count": len(lot_defect_rates),
                "defect_stats": stats,
                "lot_defect_rates": lot_defect_rates,
                "sample_lots": [item.get("lot_id") for item in lot_defect_rates[:5]],
                "source": source,
            }
        )

    return {
        "columns": defect_columns,
        "items": items,
        "recent_months": recent_months,
        "value_unit": output_unit,
    }


def collect_grid_candidate_matches(design_candidates: list[dict], rules: dict) -> dict:
    normalized = normalize_reference_rules(rules)
    if not isinstance(design_candidates, list) or not design_candidates:
        return {
            "items": [],
            "match_fields": [],
            "defect_columns": [],
            "recent_months": normalized.get("grid_match_recent_months", 6),
            "value_unit": _resolve_grid_defect_value_unit(normalized),
            "row_limit": normalized.get("grid_match_row_limit", 50),
        }

    db = normalized.get("db", {})
    connection_id = db.get("connection_id") or ""
    schema_name = db.get("schema") or "public"
    lot_search_table = _resolve_table_name(normalized, "lot_search")
    lot_id_column = db.get("lot_id_column") or "lot_id"
    chip_prod_column = db.get("chip_prod_id_column") or "chip_prod_id"
    date_column = db.get("input_date_column") or "design_input_date"

    available = None
    if connection_id and lot_search_table:
        available = _get_available_columns(connection_id, schema_name, lot_search_table)

    match_fields = _resolve_grid_match_fields(normalized)
    match_aliases = _resolve_grid_match_aliases(normalized)
    defect_columns = _resolve_grid_defect_columns(normalized, available)
    recent_months = normalized.get("grid_match_recent_months", 6)
    row_limit = normalized.get("grid_match_row_limit", 50)
    value_unit = _resolve_grid_defect_value_unit(normalized)

    items: list[dict] = []
    max_blocks = normalized.get("final_briefing", {}).get("max_blocks", 3)
    try:
        max_blocks = int(max_blocks)
    except (TypeError, ValueError):
        max_blocks = 3
    max_blocks = max(1, max_blocks)

    for index, candidate in enumerate(design_candidates[:max_blocks], start=1):
        if not isinstance(candidate, dict):
            continue
        design = candidate.get("design")
        if not isinstance(design, dict):
            design = {}
        match_values = _extract_match_values(design, match_fields, match_aliases)
        missing_fields = [
            field for field in match_fields if field not in match_values
        ]
        filters = []
        if match_values:
            for field, value in match_values.items():
                if available and field not in available:
                    continue
                filters.append({"column": field, "operator": "=", "value": value})

        if not filters:
            items.append(
                {
                    "rank": candidate.get("rank") or index,
                    "design": design,
                    "match_fields": match_fields,
                    "missing_fields": missing_fields,
                    "matched_values": match_values,
                    "rows": [],
                    "columns": [],
                    "row_count": 0,
                    "defect_column_avgs": [],
                    "source": "none",
                }
            )
            continue

        if date_column and isinstance(recent_months, int) and recent_months > 0:
            cutoff = datetime.now() - timedelta(days=recent_months * 30)
            if connection_id:
                filters.append(
                    {
                        "column": date_column,
                        "operator": "between",
                        "value": [cutoff.isoformat(), datetime.now().isoformat()],
                    }
                )

        query_columns = [
            column
            for column in dict.fromkeys(
                [lot_id_column, chip_prod_column, date_column]
                + list(match_values.keys())
                + defect_columns
            )
            if column
        ]

        source = "demo"
        rows: list[dict] = []
        error = None
        try:
            source, rows = _query_rows_for_filters(
                normalized,
                filters,
                query_columns,
                row_limit,
                table_name=lot_search_table,
            )
        except Exception as exc:
            error = str(exc)
            rows = []
            source = "error"

        if not connection_id and date_column and isinstance(recent_months, int):
            rows = _filter_rows_by_recent_months(rows, date_column, recent_months)

        averages = _summarize_defect_column_avgs(rows, defect_columns)
        item_payload = {
            "rank": candidate.get("rank") or index,
            "design": design,
            "match_fields": match_fields,
            "missing_fields": missing_fields,
            "matched_values": match_values,
            "rows": rows,
            "columns": query_columns,
            "row_count": len(rows),
            "defect_column_avgs": averages,
            "source": source,
        }
        if error:
            item_payload["error"] = error
        items.append(item_payload)

    return {
        "items": items,
        "match_fields": match_fields,
        "defect_columns": defect_columns,
        "recent_months": recent_months,
        "value_unit": value_unit,
        "row_limit": row_limit,
    }
