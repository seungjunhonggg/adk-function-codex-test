import re
from typing import Dict, List

import httpx

from .config import SIM_API_URL


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
    temperature = float(params["temperature"])
    voltage = float(params["voltage"])
    size = float(params["size"])
    capacity = float(params["capacity"])
    production_mode = str(params.get("production_mode") or "").lower()

    temp_penalty = abs(temperature - 120.0) * 0.15
    volt_penalty = abs(voltage - 3.7) * 12.0
    size_penalty = abs(size - 12.0) * 0.8
    cap_penalty = abs(capacity - 6.0) * 1.5

    score = 98.0 - temp_penalty - volt_penalty - size_penalty - cap_penalty
    if production_mode in {"dev", "development", "개발"}:
        score -= 2.0
    elif production_mode in {"mass", "production", "양산"}:
        score += 1.0
    score = max(50.0, min(99.0, score))

    if score >= 90.0:
        risk = "낮음"
    elif score >= 75.0:
        risk = "중간"
    else:
        risk = "높음"

    throughput = 980.0 + (score - 70.0) * 4.2

    return {
        "predicted_yield": round(score, 2),
        "risk_band": risk,
        "estimated_throughput": round(throughput, 1),
        "notes": (
            "양산 기준 안정 구간"
            if production_mode in {"mass", "production", "양산"} and risk == "낮음"
            else "설정 검토 필요"
        ),
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
            params[key] = float(match.group(1))

    lowered = message.lower()
    if re.search(r"(양산|mass|production|prod)", lowered):
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
                    "temperature": float(numbers[0]),
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


def _normalize_production_mode(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "mass" if value else "dev"
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"mass", "production", "prod", "양산", "true", "1"}:
        return "mass"
    if text in {"dev", "development", "개발", "false", "0"}:
        return "dev"
    return None
