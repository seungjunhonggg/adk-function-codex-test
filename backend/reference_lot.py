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
    normalized.setdefault("selection", {})
    normalized.setdefault("conditions", {})
    normalized.setdefault("grid_search", {})
    normalized.setdefault("defect_rate_aggregate", {})
    normalized["selection"].setdefault("max_candidates", 500)
    normalized["conditions"].setdefault("required_not_null", [])
    normalized["conditions"].setdefault("design_factor_columns", [])
    normalized["conditions"].setdefault("defect_metrics", [])
    normalized["grid_search"].setdefault("factors", [])
    normalized["grid_search"].setdefault("max_results", 100)
    normalized["grid_search"].setdefault("top_k", 10)
    normalized["defect_rate_aggregate"].setdefault("columns", [])
    normalized["defect_rate_aggregate"].setdefault("mode", "avg")
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
    required = list(conditions.get("required_not_null", []))
    design_columns = list(conditions.get("design_factor_columns", []))
    defect_columns = [
        metric.get("column")
        for metric in conditions.get("defect_metrics", [])
        if metric.get("column")
    ]
    grid_columns = [
        factor.get("column")
        for factor in grid.get("factors", [])
        if factor.get("column")
    ]

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


def _query_rows(filter_column: str, filter_value: str) -> tuple[dict, list[dict]]:
    rules = normalize_reference_rules(load_reference_rules())
    db = rules.get("db", {})
    connection_id = db.get("connection_id")
    table_name = db.get("table")
    columns = _build_columns(rules)
    if not connection_id or not table_name:
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
            table_name=table_name,
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
    meta, rows = _query_rows(
        normalize_reference_rules(load_reference_rules())
        .get("db", {})
        .get("chip_prod_id_column")
        or "chip_prod_id",
        chip_prod_id,
    )
    if not rows:
        if meta.get("status") == "error":
            return meta
        return {"status": "missing", "error": "reference lot 후보가 없습니다.", "rows": []}
    return _build_reference_payload(rows, meta)


def select_reference_lot_by_id(lot_id: str) -> dict:
    rules = normalize_reference_rules(load_reference_rules())
    column = rules.get("db", {}).get("lot_id_column") or "lot_id"
    meta, rows = _query_rows(column, lot_id)
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
