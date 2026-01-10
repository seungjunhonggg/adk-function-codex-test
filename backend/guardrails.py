from __future__ import annotations

import re
from typing import Iterable

from agents.guardrail import GuardrailFunctionOutput, input_guardrail, output_guardrail

from .context import current_session_id
from .pipeline_store import pipeline_store


MLCC_REQUEST_TOKENS = (
    "mlcc",
    "다층",
    "세라믹",
    "콘덴서",
    "공정",
    "lot",
    "로트",
    "불량",
    "불량률",
    "결함",
    "차트",
    "그래프",
    "히스토그램",
    "추천",
    "시뮬",
    "시뮬레이션",
    "그리드",
    "레퍼",
    "레퍼런스",
    "chip",
    "칩",
    "기종",
    "품번",
    "용량",
    "전압",
    "사이즈",
    "temperature",
    "voltage",
    "capacity",
    "size",
)
MLCC_OUTPUT_SKIP_TOKENS = (
    "알려",
    "필요",
    "부족",
    "확인",
    "질문",
    "요청",
    "입력",
)
LOT_ID_RE = re.compile(r"\bLOT[-_ ]?[A-Z0-9]{2,}\b", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _looks_like_question(text: str) -> bool:
    trimmed = text.strip()
    if not trimmed:
        return False
    if trimmed.endswith("?"):
        return True
    return _contains_any(trimmed, MLCC_OUTPUT_SKIP_TOKENS)


def _has_lot_id(text: str) -> bool:
    return bool(LOT_ID_RE.search(text))


def _collect_event_keys(session_id: str) -> set[str]:
    events = pipeline_store.get_events(session_id)
    return set(events.keys()) if isinstance(events, dict) else set()


def _missing_evidence_events(text: str, event_keys: set[str]) -> list[str]:
    missing: set[str] = set()
    lowered = text.lower()

    if _contains_any(lowered, ("불량", "불량률", "defect", "히스토그램", "차트", "그래프")):
        if not ({"defect_rate_chart", "db_result"} & event_keys):
            missing.update({"defect_rate_chart", "db_result"})

    if _contains_any(lowered, ("lot", "로트", "lot_id")) or _has_lot_id(text):
        if not ({"lot_result", "db_result"} & event_keys):
            missing.update({"lot_result", "db_result"})

    if _contains_any(
        lowered, ("추천", "시뮬", "시뮬레이션", "기종", "칩", "chip", "part")
    ):
        if not ({"simulation_result", "final_briefing"} & event_keys):
            missing.update({"simulation_result", "final_briefing"})

    if _contains_any(lowered, ("브리핑", "최종", "요약", "final")):
        if "final_briefing" not in event_keys:
            missing.add("final_briefing")

    return sorted(missing)


def guardrail_fallback_message(missing_events: Iterable[str] | None) -> str:
    missing_set = set(missing_events or [])
    if {"simulation_result", "final_briefing"} & missing_set:
        return (
            "추천 결과가 없어 브리핑을 만들 수 없습니다. 먼저 시뮬레이션/추천을 진행해 주세요."
        )
    if {"db_result", "lot_result"} & missing_set:
        return "LOT/DB 근거가 없습니다. 먼저 DB 조회를 실행해 주세요."
    if "defect_rate_chart" in missing_set:
        return "불량률 차트 근거가 없습니다. 먼저 추천이나 DB 조회를 진행해 주세요."
    return "근거 데이터가 부족합니다. 먼저 추천 또는 DB 조회를 진행해 주세요."


@input_guardrail(name="mlcc_input_guardrail", run_in_parallel=False)
def mlcc_input_guardrail(_ctx, _agent, input_value):
    if not isinstance(input_value, str):
        return GuardrailFunctionOutput(output_info={}, tripwire_triggered=False)
    text = _normalize_text(input_value)
    if not _contains_any(text, MLCC_REQUEST_TOKENS):
        return GuardrailFunctionOutput(output_info={}, tripwire_triggered=False)
    return GuardrailFunctionOutput(
        output_info={"reason": "mlcc_request"},
        tripwire_triggered=True,
    )


@output_guardrail(name="mlcc_output_guardrail")
def mlcc_output_guardrail(_ctx, _agent, output_value):
    text = str(output_value or "")
    if not text.strip() or _looks_like_question(text):
        return GuardrailFunctionOutput(output_info={}, tripwire_triggered=False)
    if not _contains_any(text, MLCC_REQUEST_TOKENS) and not _has_lot_id(text):
        return GuardrailFunctionOutput(output_info={}, tripwire_triggered=False)
    session_id = current_session_id.get()
    event_keys = _collect_event_keys(session_id)
    missing = _missing_evidence_events(text, event_keys)
    if missing:
        return GuardrailFunctionOutput(
            output_info={"missing_events": missing},
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(output_info={}, tripwire_triggered=False)
