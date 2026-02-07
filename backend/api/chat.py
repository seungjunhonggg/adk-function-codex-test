import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from google.genai import types

from ..core import db_production, schemas
from ..core.adk_runtime import get_runner

router = APIRouter()


def _format_sse(event: str, payload: dict[str, Any]) -> str:
    # SSE 응답 문자열을 만든다.
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_client_ip(raw_request: Request) -> str | None:
    # 프록시 환경을 고려해 클라이언트 IP를 추출한다.
    forwarded = raw_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if raw_request.client:
        return raw_request.client.host
    return None


def _event_text(event) -> str | None:
    # ADK 이벤트에서 텍스트 파트만 추출한다.
    content = getattr(event, "content", None)
    if not content or not getattr(content, "parts", None):
        return None
    text = "".join(part.text or "" for part in content.parts)
    return text or None


def _field_spec(key: str, value: Any) -> dict[str, Any]:
    # 폼 렌더링에 필요한 필드 스펙을 만든다.
    label_map = {
        "temperature": "온도",
        "voltage": "전압",
        "size": "사이즈",
        "capacity": "용량",
        "chip_prod_id": "CHIP 기종",
        "reference_lot_id": "기준 LOT ID",
        "top_k": "Top-K",
        "months": "조회 개월 수",
    }
    number_fields = {"top_k", "months"}
    return {
        "key": key,
        "label": label_map.get(key, key),
        "type": "number" if key in number_fields else "text",
        "value": "" if value is None else str(value),
    }


def _lookup_state_value(state: dict[str, Any], key: str) -> Any:
    # 상태에서 키 값을 우선순위에 따라 조회한다.
    input_params = state.get("input_params", {})
    if key in input_params:
        return input_params.get(key)
    selections = state.get("selections", {})
    if key in selections:
        return selections.get(key)
    configs = state.get("configs", {})
    if key in configs:
        return configs.get(key)
    return None


def _build_ui_action(state: dict[str, Any]) -> dict[str, Any] | None:
    # pending_action을 프론트 액션으로 변환한다.
    pending = state.get("pending_action")
    if not isinstance(pending, dict):
        return None

    # 추가 입력 요청을 폼 액션으로 만든다.
    if pending.get("type") == "need_input":
        fields = pending.get("fields") or []
        specs = [_field_spec(key, _lookup_state_value(state, key)) for key in fields]
        return {
            "type": "open_form",
            "stage": pending.get("stage"),
            "form_id": f"form-{pending.get('stage', 'stage')}",
            "title": pending.get("message", "입력값을 보완해 주세요."),
            "submit_action": "apply_user_patch",
            "fields": specs,
        }

    # 단계 확인 요청을 확인 액션으로 만든다.
    if pending.get("type") == "confirm_stage":
        return {
            "type": "confirm_stage",
            "stage": pending.get("stage"),
            "title": "다음 단계 진행 확인",
            "message": pending.get("summary", ""),
            "actions": [
                {"id": "approve", "label": "진행"},
                {"id": "reject", "label": "수정"},
            ],
        }

    return None


def _build_workflow_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    # 진행 상태 스냅샷을 만든다.
    return {
        "sim_step": state.get("sim_step"),
        "stage_status": state.get("stage_status", {}),
        "pending_action": state.get("pending_action"),
    }


def _build_response_envelope(answer_text: str, state: dict[str, Any]) -> dict[str, Any]:
    # 정규 응답 포맷을 만든다.
    return {
        "route": "simulation",
        "assistant": {
            "text": answer_text,
        },
        "ui_action": _build_ui_action(state),
        "workflow": _build_workflow_snapshot(state),
    }


@router.post("/api/chat/stream")
async def api_chat_stream(
    request: schemas.ChatRequest,
    raw_request: Request,
) -> StreamingResponse:
    # 모드에 따라 DB 저장 여부를 결정한다.
    use_db = not request.demo
    client_ip = _extract_client_ip(raw_request)
    if use_db:
        db_production.upsert_session_ip(request.session_id, client_ip)

    # 러너를 선택한다.
    runner = get_runner(request.demo)

    async def event_stream():
        # 시작 진행 로그를 먼저 보낸다.
        yield _format_sse(
            "progress",
            {"logs": [{"text": "요청을 분석하고 있어요", "status": "in_progress"}]},
        )

        answer_chunks: list[str] = []
        try:
            # 사용자 입력을 ADK 메시지로 만든다.
            content = types.Content(
                role="user",
                parts=[types.Part(text=request.message)],
            )

            # ADK 이벤트를 순회하며 텍스트 응답을 누적한다.
            async for event in runner.run_async(
                user_id=request.session_id,
                session_id=request.session_id,
                new_message=content,
            ):
                text = _event_text(event)
                if text:
                    answer_chunks.append(text)

            # 최신 상태를 조회한다.
            state = await runner.get_state_async(
                user_id=request.session_id,
                session_id=request.session_id,
            )
            answer_text = "".join(answer_chunks).strip()

            # 정규 응답을 구성한다.
            envelope = _build_response_envelope(answer_text, state)

            # 최종 이벤트를 보낸다.
            yield _format_sse("final", envelope)
        except Exception as exc:
            # 예외 발생 시 오류 응답을 보낸다.
            payload = {
                "route": "error",
                "assistant": {"text": f"처리 중 오류가 발생했어요: {exc}"},
                "ui_action": None,
                "workflow": {},
            }
            yield _format_sse("final", payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
