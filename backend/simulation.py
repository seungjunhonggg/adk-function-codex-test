from typing import Dict, List

import httpx

from .config import SIM_API_URL


REQUIRED_KEYS = ("temperature", "voltage", "size", "capacity")


class SimulationStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, float]] = {}

    def update(self, session_id: str, **kwargs: object) -> Dict[str, float]:
        record = self._data.get(session_id, {})
        for key, value in kwargs.items():
            coerced = _to_float(value)
            if coerced is not None:
                record[key] = coerced
        self._data[session_id] = record
        return record

    def get(self, session_id: str) -> Dict[str, float]:
        return dict(self._data.get(session_id, {}))

    def missing(self, session_id: str) -> List[str]:
        record = self._data.get(session_id, {})
        return [key for key in REQUIRED_KEYS if key not in record]

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)


simulation_store = SimulationStore()


async def call_simulation_api(params: Dict[str, float]) -> Dict[str, float | str]:
    if SIM_API_URL:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(SIM_API_URL, json=params)
            response.raise_for_status()
            return response.json()
    return _simulate_locally(params)


def _simulate_locally(params: Dict[str, float]) -> Dict[str, float | str]:
    temperature = float(params["temperature"])
    voltage = float(params["voltage"])
    size = float(params["size"])
    capacity = float(params["capacity"])

    temp_penalty = abs(temperature - 120.0) * 0.15
    volt_penalty = abs(voltage - 3.7) * 12.0
    size_penalty = abs(size - 12.0) * 0.8
    cap_penalty = abs(capacity - 6.0) * 1.5

    score = 98.0 - temp_penalty - volt_penalty - size_penalty - cap_penalty
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
        "notes": "안정 구간" if risk == "낮음" else "설정 검토 필요",
    }


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
