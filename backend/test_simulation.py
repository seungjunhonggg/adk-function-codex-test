from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Tuple

import httpx

from .config import GRID_SEARCH_API_URL

DEFAULT_DEFECT_MAX = 0.03
DEFAULT_TARGET = 0.95
DEFAULT_GRID_LIMIT = 100
DEFAULT_TOP_K = 10
DEFAULT_CHART_LIMIT = 20


def _seed_from(*parts: str) -> int:
    raw = "|".join(parts)
    return abs(hash(raw)) % (2**32)


def _normalize_rate(value: float) -> float:
    if value > 1:
        return value / 100.0
    return value


def parse_defect_rate_bounds(message: str | None) -> Tuple[float | None, float | None]:
    if not message:
        return None, None
    if not re.search(r"(불량|defect)", message, re.IGNORECASE):
        return None, None

    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%?\s*(?:~|-|to)\s*(\d+(?:\.\d+)?)\s*%?",
        message,
        re.IGNORECASE,
    )
    if range_match:
        low = _normalize_rate(float(range_match.group(1)))
        high = _normalize_rate(float(range_match.group(2)))
        return min(low, high), max(low, high)

    value_match = re.search(r"(\d+(?:\.\d+)?)\s*(%|퍼|percent)?", message, re.IGNORECASE)
    if not value_match:
        return None, None
    value = _normalize_rate(float(value_match.group(1)))

    if re.search(r"(이상|초과|greater|over|above)", message, re.IGNORECASE):
        return value, None
    if re.search(r"(이하|미만|less|under|below)", message, re.IGNORECASE):
        return None, value
    return None, value


def parse_target_value(message: str | None) -> float | None:
    if not message:
        return None
    if not re.search(r"(목표|타겟|target)", message, re.IGNORECASE):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(%|퍼|percent)?", message, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        return value / 100.0
    return value / 100.0 if value > 1 else value


def _build_candidate_lots(session_id: str, count: int) -> List[dict]:
    rng = random.Random(_seed_from(session_id, "lot_candidates"))
    base = abs(hash(session_id)) % 9000 + 1000
    lots = []
    for idx in range(count):
        lot_id = f"LOT-{(base + idx) % 10000:04d}"
        defect_rate = round(rng.uniform(0.005, 0.08), 4)
        lots.append({"lot_id": lot_id, "defect_rate": defect_rate})
    return lots


def filter_lot_candidates(
    lots: List[dict],
    min_rate: float | None,
    max_rate: float | None,
    limit: int,
) -> List[dict]:
    filtered = []
    for lot in lots:
        rate = lot.get("defect_rate")
        if rate is None:
            continue
        if min_rate is not None and rate < min_rate:
            continue
        if max_rate is not None and rate > max_rate:
            continue
        filtered.append(lot)
    filtered.sort(key=lambda item: item.get("defect_rate") or 0)
    return filtered[:limit]


def summarize_defect_rates(lots: List[dict]) -> dict:
    rates = [item.get("defect_rate") for item in lots if item.get("defect_rate") is not None]
    if not rates:
        return {"count": 0, "min": None, "max": None, "avg": None, "median": None}
    return {
        "count": len(rates),
        "min": min(rates),
        "max": max(rates),
        "avg": round(mean(rates), 4),
        "median": round(median(rates), 4),
    }


def _base_design_for_lot(lot_id: str) -> Dict[str, float]:
    rng = random.Random(_seed_from(lot_id, "design_base"))
    return {
        "design_a": round(rng.uniform(80, 120), 2),
        "design_b": round(rng.uniform(0.8, 1.2), 3),
        "design_c": round(rng.uniform(10, 20), 2),
    }


def _jitter(value: float, rng: random.Random, pct: float) -> float:
    return value * (1 + rng.uniform(-pct, pct))


def _predict_target(design: Dict[str, float]) -> float:
    base = (
        design.get("design_a", 0) * 0.002
        + design.get("design_b", 0) * 0.06
        + design.get("design_c", 0) * 0.006
    )
    return max(0.75, min(0.995, 0.72 + base))


def grid_search_candidates(
    lots: List[dict],
    target: float | None,
    limit: int,
) -> List[dict]:
    rng = random.Random(_seed_from(str(lots[0].get("lot_id", "")), "grid_search"))
    candidates = []
    target_value = target if target is not None else DEFAULT_TARGET
    attempts_per_lot = max(6, int(limit / max(1, len(lots))) * 2)

    for lot in lots:
        base = _base_design_for_lot(str(lot.get("lot_id", "")))
        for _ in range(attempts_per_lot):
            design = {
                key: round(_jitter(value, rng, 0.06), 4) for key, value in base.items()
            }
            predicted = _predict_target(design)
            score = abs(predicted - target_value)
            candidates.append(
                {
                    "lot_id": lot.get("lot_id"),
                    "defect_rate": lot.get("defect_rate"),
                    "design": design,
                    "predicted_target": round(predicted, 4),
                    "score": round(score, 6),
                }
            )

    candidates.sort(key=lambda item: (item["score"], -item["predicted_target"]))
    return candidates[:limit]


async def call_grid_search_api(
    lots: List[dict],
    target: float | None,
    limit: int,
) -> List[dict] | None:
    if not GRID_SEARCH_API_URL:
        return None
    payload = {"lots": lots, "target": target, "limit": limit}
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(GRID_SEARCH_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
    if isinstance(data, dict):
        candidates = data.get("candidates")
        return candidates if isinstance(candidates, list) else None
    if isinstance(data, list):
        return data
    return None


@dataclass
class TestSimulationState:
    candidate_lots: List[dict] = field(default_factory=list)
    filtered_lots: List[dict] = field(default_factory=list)
    filters: Dict[str, float | None] = field(default_factory=dict)
    grid_results: List[dict] = field(default_factory=list)
    grid_target: float | None = None


class TestSimulationStore:
    def __init__(self) -> None:
        self._data: Dict[str, TestSimulationState] = {}

    def _state(self, session_id: str) -> TestSimulationState:
        return self._data.setdefault(session_id, TestSimulationState())

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def ensure_candidates(self, session_id: str, count: int = 50) -> List[dict]:
        state = self._state(session_id)
        if not state.candidate_lots:
            state.candidate_lots = _build_candidate_lots(session_id, count)
        return list(state.candidate_lots)

    def set_filtered_lots(
        self,
        session_id: str,
        lots: List[dict],
        filters: Dict[str, float | None],
    ) -> None:
        state = self._state(session_id)
        state.filtered_lots = list(lots)
        state.filters = dict(filters)

    def get_filtered_lots(self, session_id: str) -> List[dict]:
        return list(self._state(session_id).filtered_lots)

    def get_filters(self, session_id: str) -> Dict[str, float | None]:
        return dict(self._state(session_id).filters)

    def set_grid_results(
        self,
        session_id: str,
        results: List[dict],
        target: float | None,
    ) -> None:
        state = self._state(session_id)
        state.grid_results = list(results)
        state.grid_target = target

    def get_grid_results(self, session_id: str) -> List[dict]:
        return list(self._state(session_id).grid_results)

    def get_grid_target(self, session_id: str) -> float | None:
        return self._state(session_id).grid_target

    def get_candidates(
        self,
        session_id: str,
        offset: int,
        limit: int,
    ) -> Tuple[List[dict], int]:
        results = self.get_grid_results(session_id)
        total = len(results)
        sliced = results[offset : offset + limit]
        return sliced, total


test_simulation_store = TestSimulationStore()
