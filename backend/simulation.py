import re
from typing import Dict, List

import httpx
from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

from .config import GRID_SEARCH_API_URL, MODEL_NAME, PREDICT_API_URL, SIM_API_URL


REQUIRED_KEYS = (
    "temperature",
    "voltage",
    "size",
    "capacity",
    "production_mode",
)
CHIP_PROD_ID_MIN_LEN = 6
CHIP_PROD_ID_TOKEN_RE = re.compile(
    rf"\b[a-z0-9][a-z0-9-_]{{{CHIP_PROD_ID_MIN_LEN - 1},}}\b",
    re.IGNORECASE,
)
CHIP_PROD_ID_EXCLUDE_PREFIXES = ("param", "lot", "lotid", "lot_id")
LLM_SIM_PARSE_CONFIDENCE_MIN = 0.5
CAPACITY_UNIT_MULTIPLIERS = {
    "pf": 1.0,
    "p": 1.0,
    "nf": 1e3,
    "n": 1e3,
    "uf": 1e6,
    "u": 1e6,
    "mf": 1e9,
    "m": 1e9,
    "f": 1e12,
}
CAPACITY_VALUE_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*([munp]f?|f)?", re.IGNORECASE
)
MLCC_GLOSSARY_HINTS = (
    "Glossary (KR/EN): "
    "chip_prod_id=기종/모델/품번/제품명/part number; "
    "capacity=용량/정전용량/capacitance; "
    "voltage=전압/정격전압/rated voltage; "
    "size=사이즈/치수/L/W/LxW/length/width; "
    "temperature=온도/공정온도/sintering temp; "
    "production_mode=양산/개발."
)


def _build_agent(**kwargs: object) -> Agent:
    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("model", None)
        return Agent(**kwargs)


MODEL_KWARGS = {"model": MODEL_NAME} if MODEL_NAME else {}


class SimulationParamDecision(BaseModel):
    temperature: str | None = None
    voltage: float | None = None
    size: str | None = None
    capacity: float | None = None
    production_mode: str | None = None
    chip_prod_id: str | None = None
    confidence: float = 0.0
    notes: str = ""


simulation_param_agent = _build_agent(
    name="Simulation Param Parser",
    instructions=(
        "Extract MLCC simulation parameters from the user message. "
        "Return JSON with keys: temperature (string), voltage (float), size (string), "
        "capacity (float), production_mode ('양산' or '개발' or null), chip_prod_id (string), "
        "confidence (0-1), notes. "
        "Capacity must be returned in pF. Convert units if specified (uF/nF/pF/F). "
        "Only include values explicitly mentioned. "
        "chip_prod_id may be partial; keep it exactly as written and do not expand. "
        "Examples: CL32Y106KCYBNB -> chip_prod_id=CL32Y106KCYBNB, "
        "CL32Y106KC -> chip_prod_id=CL32Y106KC, "
        "32Y106KC -> chip_prod_id=32Y106KC. "
        "If chip_prod_id appears without a keyword, treat any token with letters+digits "
        "and length >= 6 as chip_prod_id (unless it is clearly a param name like param1). "
        "If conflicting values are present for a field, set that field to null and note it. "
        "If unsure, set fields to null. "
        f"{MLCC_GLOSSARY_HINTS}"
    ),
    output_type=AgentOutputSchema(SimulationParamDecision, strict_json_schema=False),
    **MODEL_KWARGS,
)


class SimulationStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, float | str]] = {}
        self._active: Dict[str, bool] = {}

    def update(self, session_id: str, **kwargs: object) -> Dict[str, float | str]:
        record: Dict[str, float | str] = self._data.get(session_id, {})
        for key, value in kwargs.items():
            if key == "chip_prod_id":
                text = _normalize_chip_prod_id(value) or ""
                if text:
                    record[key] = text
                continue
            if key == "production_mode":
                mode = _normalize_production_mode(value)
                if mode is not None:
                    record[key] = mode
                continue
            if key == "temperature":
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    record[key] = text
                continue
            if key == "size":
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    record[key] = text
                continue
            if key == "capacity":
                normalized = _normalize_capacity_to_pf(value)
                if normalized is not None:
                    record[key] = normalized
                continue
            coerced = _to_float(value)
            if coerced is not None:
                record[key] = coerced
        self._data[session_id] = record
        return record

    def get(self, session_id: str) -> Dict[str, float | str]:
        return dict(self._data.get(session_id, {}))

    def missing(self, session_id: str) -> List[str]:
        record = self._data.get(session_id, {})
        return [key for key in REQUIRED_KEYS if key not in record]

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)
        self._active.pop(session_id, None)

    def remove_keys(self, session_id: str, keys: list[str]) -> Dict[str, float | str]:
        record = self._data.get(session_id, {})
        if not record or not keys:
            return dict(record)
        for key in keys:
            record.pop(key, None)
        self._data[session_id] = record
        return dict(record)

    def activate(self, session_id: str) -> None:
        if session_id:
            self._active[session_id] = True

    def deactivate(self, session_id: str) -> None:
        self._active.pop(session_id, None)

    def is_active(self, session_id: str) -> bool:
        return bool(self._active.get(session_id))


simulation_store = SimulationStore()


async def call_simulation_api(params: Dict[str, float | str]) -> Dict[str, float | str]:
    if SIM_API_URL:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(SIM_API_URL, json=params)
            response.raise_for_status()
            return response.json()
    return _simulate_locally(params)


def _simulate_locally(params: Dict[str, float | str]) -> Dict[str, float | str]:
    temperature = _parse_numeric(params["temperature"], "temperature")
    voltage = _parse_numeric(params["voltage"], "voltage")
    size = _parse_numeric(params["size"], "size")
    capacity = _parse_numeric(params["capacity"], "capacity")
    chip_prod_id = "CL32Y106KC5ER9B"
    representative_lot = "ALA7K1K"

    base = (temperature * 0.02) + (voltage * 0.8) + (size * 0.15) + (capacity * 0.5)
    params_30 = {}
    for idx in range(1, 31):
        params_30[f"param{idx}"] = round(base + idx * 0.37, 3)

    return {
        "recommended_chip_prod_id": chip_prod_id,
        "representative_lot": representative_lot,
        "params": params_30,
    }


async def call_prediction_api(params: Dict[str, float | str]) -> Dict[str, float | str]:
    if PREDICT_API_URL:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(PREDICT_API_URL, json=params)
            response.raise_for_status()
            return response.json()
    return _predict_locally(params)


async def call_grid_search_api(payload: dict) -> list[dict]:
    if GRID_SEARCH_API_URL:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(GRID_SEARCH_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return _normalize_grid_search_results(data)
    return _simulate_grid_search(payload)


def _normalize_grid_search_results(data: object) -> list[dict]:
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return _normalize_grid_candidates(data.get("results"))
        datas = data.get("datas") or data.get("data")
        if isinstance(datas, dict) and isinstance(datas.get("sim"), list):
            return _normalize_grid_candidates(datas.get("sim"))
    if isinstance(data, list):
        return _normalize_grid_candidates(data)
    return []


def _normalize_grid_candidates(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    results: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        design = item.get("design") if isinstance(item.get("design"), dict) else dict(item)
        entry = {"design": design}
        if "predicted_target" in item:
            entry["predicted_target"] = item.get("predicted_target")
        results.append(entry)
    return results


def _predict_locally(params: Dict[str, float | str]) -> Dict[str, float | str]:
    values = []
    for value in params.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    avg = sum(values) / len(values) if values else 0.0
    predicted_capacity = round(5.8 + avg * 0.02, 3)
    predicted_dc_capacity = round(predicted_capacity * 0.92, 3)
    reliability_pass_prob = round(min(0.99, max(0.75, 0.78 + avg * 0.003)), 3)
    return {
        "predicted_capacity": predicted_capacity,
        "predicted_dc_capacity": predicted_dc_capacity,
        "reliability_pass_prob": reliability_pass_prob,
    }


def extract_simulation_params(message: str) -> dict:
    patterns = {
        "temperature": r"(?:온도|temperature|temp)[^\d+-]*([-+]?\d+(?:\.\d+)?)",
        "voltage": r"(?:전압|voltage|volt)[^\d+-]*([-+]?\d+(?:\.\d+)?)",
        "size": r"(?:크기|size)[^\d+-]*([-+]?\d+(?:\.\d+)?)",
        "capacity": r"(?:용량|capacity)[^\d+-]*([-+]?\d+(?:\.\d+)?\s*(?:[munp]f?|f)?)",
    }

    params: dict[str, float | str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key in {"temperature", "size"}:
                params[key] = value
            elif key == "capacity":
                normalized = _normalize_capacity_to_pf(value)
                if normalized is not None:
                    params[key] = normalized
            else:
                params[key] = float(value)

    lowered = message.lower()
    if re.search(r"(양산|생산|mass|production|prod)", lowered):
        params["production_mode"] = "양산"
    elif re.search(r"(개발|dev|development)", lowered):
        params["production_mode"] = "개발"

    chip_prod_match = re.search(
        r"(?:chip_prod_id|model|모델|기종|제품)\s*[:=]?\s*([a-z0-9-_]+)",
        message,
        re.IGNORECASE,
    )
    chip_prod_id = None
    if chip_prod_match:
        chip_prod_id = _normalize_chip_prod_id(chip_prod_match.group(1))
    else:
        candidate = _extract_chip_prod_id_candidate(message)
        if candidate:
            chip_prod_id = _normalize_chip_prod_id(candidate)
    if chip_prod_id:
        params["chip_prod_id"] = chip_prod_id
    if "capacity" not in params:
        unit_match = re.search(
            r"\b([-+]?\d+(?:\.\d+)?\s*(?:[munp]f?|f))\b", message, re.IGNORECASE
        )
        if unit_match:
            normalized = _normalize_capacity_to_pf(unit_match.group(1))
            if normalized is not None:
                params["capacity"] = normalized

    numeric_keys = ("temperature", "voltage", "size", "capacity")
    numeric_count = sum(1 for key in numeric_keys if key in params)
    if numeric_count < 4:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", message)
        if len(numbers) >= 4 and numeric_count == 0:
            params.update(
                {
                    "temperature": numbers[0],
                    "voltage": float(numbers[1]),
                    "size": numbers[2],
                    "capacity": float(numbers[3]),
                }
            )

    return params


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).strip().lower()


def _normalize_chip_prod_id(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value)).strip()
    if not text:
        return None
    return text.upper()


def _numeric_token(value: object) -> str | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return match.group(0) if match else None


def _extract_chip_prod_id_candidate(message: str) -> str | None:
    if not message:
        return None
    tokens = CHIP_PROD_ID_TOKEN_RE.findall(message)
    if not tokens:
        return None
    candidates = []
    for token in tokens:
        if not re.search(r"[a-z]", token, re.IGNORECASE):
            continue
        if not re.search(r"\d", token):
            continue
        lowered = token.lower()
        if lowered.startswith(CHIP_PROD_ID_EXCLUDE_PREFIXES):
            continue
        candidates.append(token)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _values_equivalent(key: str, left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    if key == "chip_prod_id":
        return _normalize_chip_prod_id(left) == _normalize_chip_prod_id(right)
    if key == "capacity":
        left_value = _normalize_capacity_to_pf(left)
        right_value = _normalize_capacity_to_pf(right)
        if left_value is None or right_value is None:
            return False
        return abs(left_value - right_value) <= 1e-6
    if key == "voltage":
        left_value = _to_float(left)
        right_value = _to_float(right)
        if left_value is None or right_value is None:
            return False
        return abs(left_value - right_value) <= 1e-6
    if key in {"temperature", "size"}:
        left_token = _numeric_token(left)
        right_token = _numeric_token(right)
        if left_token and right_token:
            return left_token == right_token
        return _normalize_text(left) == _normalize_text(right)
    if key == "production_mode":
        return _normalize_production_mode(left) == _normalize_production_mode(right)
    return _normalize_text(left) == _normalize_text(right)


def _normalize_llm_params(payload: dict) -> tuple[dict[str, float | str], float]:
    params: dict[str, float | str] = {}
    confidence = _to_float(payload.get("confidence")) or 0.0
    for key in (
        "temperature",
        "voltage",
        "size",
        "capacity",
        "production_mode",
        "chip_prod_id",
    ):
        value = payload.get(key)
        if value is None:
            continue
        if key in {"temperature", "size"}:
            text = str(value).strip()
            if text:
                params[key] = text
            continue
        if key == "chip_prod_id":
            text = _normalize_chip_prod_id(value)
            if text:
                params[key] = text
            continue
        if key == "production_mode":
            normalized = _normalize_production_mode(value)
            if normalized is not None:
                params[key] = normalized
            continue
        if key == "capacity":
            coerced = _normalize_capacity_to_pf(value)
        else:
            coerced = _to_float(value)
        if coerced is not None:
            params[key] = coerced
    return params, confidence


async def _extract_simulation_params_llm(message: str) -> tuple[dict[str, float | str], float]:
    if not message:
        return {}, 0.0
    try:
        result = await Runner.run(simulation_param_agent, input=message)
    except Exception:
        return {}, 0.0
    output = result.final_output
    if output is None:
        return {}, 0.0
    if isinstance(output, BaseModel):
        payload = output.model_dump() if hasattr(output, "model_dump") else output.dict()
    elif isinstance(output, dict):
        payload = output
    else:
        return {}, 0.0
    return _normalize_llm_params(payload)


async def extract_simulation_params_hybrid(
    message: str,
) -> tuple[dict[str, float | str], list[str]]:
    if not message:
        return {}, []
    rule_params = extract_simulation_params(message)
    llm_params, llm_confidence = await _extract_simulation_params_llm(message)
    if llm_confidence < LLM_SIM_PARSE_CONFIDENCE_MIN:
        llm_params = {}

    merged: dict[str, float | str] = {}
    conflicts: list[str] = []
    keys = set(rule_params) | set(llm_params)
    for key in keys:
        rule_value = rule_params.get(key)
        llm_value = llm_params.get(key)
        if rule_value is not None and llm_value is not None:
            if _values_equivalent(key, rule_value, llm_value):
                merged[key] = rule_value
            else:
                conflicts.append(key)
            continue
        if rule_value is not None:
            merged[key] = rule_value
            continue
        if llm_value is not None:
            merged[key] = llm_value
    conflicts.sort()
    return merged, conflicts


def _normalize_capacity_to_pf(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\u03bc", "u").replace("\u00b5", "u")
    match = CAPACITY_VALUE_RE.search(text)
    if not match:
        return None
    number = _to_float(match.group(1))
    if number is None:
        return None
    unit = (match.group(2) or "").strip().lower()
    multiplier = CAPACITY_UNIT_MULTIPLIERS.get(unit, None)
    if multiplier is None:
        if unit == "":
            multiplier = 1.0
        else:
            return None
    return number * multiplier


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_numeric(value: object, label: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"{label} must include a numeric value.")
    return float(match.group(0))


def _normalize_production_mode(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "양산" if value else "개발"
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"mass", "production", "prod", "양산", "생산", "true", "1"}:
        return "양산"
    if text in {"dev", "development", "개발", "false", "0"}:
        return "개발"
    return None


def _simulate_grid_search(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    grid = payload.get("grid")
    if not isinstance(grid, dict):
        grid = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if not isinstance(grid, dict) or not grid:
        return []

    grid_items: list[tuple[str, list[object]]] = []
    for key, raw_values in grid.items():
        values = raw_values
        if not isinstance(values, list):
            values = [values]
        cleaned = [value for value in values if value not in (None, "")]
        if not cleaned:
            continue
        grid_items.append((key, cleaned))
    if not grid_items:
        return []

    results = []
    numeric_samples: list[float] = []
    for _, values in grid_items:
        for value in values:
            numeric = _to_float(value)
            if numeric is not None:
                numeric_samples.append(numeric)
    base = sum(numeric_samples) / len(numeric_samples) if numeric_samples else 0.0

    def _recurse(index: int, current: dict, score_sum: float) -> None:
        if index == len(grid_items):
            score = base + score_sum * 0.01
            results.append(
                {
                    "design": dict(current),
                    "predicted_target": round(score, 4),
                }
            )
            return
        key, values = grid_items[index]
        for value in values:
            numeric = _to_float(value)
            current[key] = value
            _recurse(index + 1, current, score_sum + (numeric or 0.0))
        current.pop(key, None)

    _recurse(0, {}, 0.0)
    results.sort(key=lambda item: item.get("predicted_target", 0), reverse=True)
    max_results = payload.get("max_results")
    limit = int(max_results) if isinstance(max_results, int) and max_results > 0 else 100
    return results[:limit]



class RecommendationStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, object]] = {}

    def set(self, session_id: str, payload: Dict[str, object]) -> None:
        if not session_id:
            return
        record = dict(payload)
        if "awaiting_prediction" not in record:
            record["awaiting_prediction"] = isinstance(record.get("params"), dict)
        self._data[session_id] = record

    def get(self, session_id: str) -> Dict[str, object]:
        return dict(self._data.get(session_id, {}))

    def update_params(self, session_id: str, params: Dict[str, object]) -> Dict[str, object]:
        if not session_id:
            return {}
        record = dict(self._data.get(session_id, {}))
        existing = record.get("params")
        if not isinstance(existing, dict):
            existing = {}
        for key, value in params.items():
            if not isinstance(key, str) or not key.startswith("param"):
                continue
            coerced = _to_float(value)
            existing[key] = coerced if coerced is not None else value
        record["params"] = existing
        record["awaiting_prediction"] = True
        self._data[session_id] = record
        return dict(record)

    def remove_params(self, session_id: str, keys: list[str]) -> Dict[str, object]:
        if not session_id:
            return {}
        record = dict(self._data.get(session_id, {}))
        existing = record.get("params")
        if not isinstance(existing, dict):
            existing = {}
        for key in keys:
            if not isinstance(key, str) or not key.startswith("param"):
                continue
            existing.pop(key, None)
        record["params"] = existing
        record["awaiting_prediction"] = True
        self._data[session_id] = record
        return dict(record)

    def mark_prediction_done(self, session_id: str) -> None:
        if not session_id:
            return
        record = self._data.get(session_id)
        if not isinstance(record, dict):
            return
        record["awaiting_prediction"] = False
        self._data[session_id] = record

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)


recommendation_store = RecommendationStore()
