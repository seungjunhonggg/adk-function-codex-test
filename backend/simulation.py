import re
from typing import Dict, List

import httpx

from .config import PREDICT_API_URL, SIM_API_URL


REQUIRED_KEYS = (
    "temperature",
    "voltage",
    "size",
    "capacity",
    "production_mode",
)


class SimulationStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, float | str]] = {}
        self._active: Dict[str, bool] = {}

    def update(self, session_id: str, **kwargs: object) -> Dict[str, float | str]:
        record: Dict[str, float | str] = self._data.get(session_id, {})
        for key, value in kwargs.items():
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
    production_mode = str(params.get("production_mode") or "").lower()

    mode_tag = "M" if production_mode in {"mass", "production"} else "D"
    model_name = f"ADJ-{int(temperature)}-{int(size)}-{mode_tag}"
    lot_seed = int(abs(voltage * 100)) + int(abs(capacity * 10))
    representative_lot = f"LOT-{lot_seed:04d}"

    base = (temperature * 0.02) + (voltage * 0.8) + (size * 0.15) + (capacity * 0.5)
    params_30 = {}
    for idx in range(1, 31):
        params_30[f"param{idx}"] = round(base + idx * 0.37, 3)

    return {
        "recommended_model": model_name,
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
        "capacity": r"(?:용량|capacity)[^\d+-]*([-+]?\d+(?:\.\d+)?)",
    }

    params: dict[str, float | str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            value = match.group(1)
            params[key] = value if key == "temperature" else float(value)

    lowered = message.lower()
    if re.search(r"(양산|생산|mass|production|prod)", lowered):
        params["production_mode"] = "mass"
    elif re.search(r"(개발|dev|development)", lowered):
        params["production_mode"] = "dev"

    numeric_keys = ("temperature", "voltage", "size", "capacity")
    numeric_count = sum(1 for key in numeric_keys if key in params)
    if numeric_count < 4:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", message)
        if len(numbers) >= 4 and numeric_count == 0:
            params.update(
                {
                    "temperature": numbers[0],
                    "voltage": float(numbers[1]),
                    "size": float(numbers[2]),
                    "capacity": float(numbers[3]),
                }
            )

    return params


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
        return "mass" if value else "dev"
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"mass", "production", "prod", "양산", "생산", "true", "1"}:
        return "mass"
    if text in {"dev", "development", "개발", "false", "0"}:
        return "dev"
    return None



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
