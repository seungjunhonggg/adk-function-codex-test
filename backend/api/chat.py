import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from google.genai import types

from ..core import db_production, schemas
from ..core.adk_runtime import get_runner

router = APIRouter()


def _format_sse(event: str, payload: dict) -> str:
    # SSE 포맷 문자열을 만든다.
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_client_ip(raw_request: Request) -> str | None:
    # 프록시 헤더에서 IP를 우선 확인한다.
    forwarded = raw_request.headers.get("x-forwarded-for")
    if forwarded:
        # 첫 번째 IP를 사용한다.
        return forwarded.split(",")[0].strip()
    # 직접 연결 IP를 사용한다.
    client = raw_request.client
    if client:
        return client.host
    return None


def _event_payload(event) -> dict | None:
    # 이벤트 콘텐츠를 JSON으로 파싱한다.
    content = getattr(event, "content", None)
    if not content or not getattr(content, "parts", None):
        return None
    text = "".join(part.text or "" for part in content.parts)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@router.post("/api/chat/stream")
async def api_chat_stream(
    request: schemas.ChatRequest,
    raw_request: Request,
) -> StreamingResponse:
    # 데모 여부에 따라 세션 스토어를 선택한다.
    use_db = not request.demo
    # 클라이언트 IP를 추출한다.
    client_ip = _extract_client_ip(raw_request)
    # 프로덕션이면 IP를 저장한다.
    if use_db:
        db_production.upsert_session_ip(request.session_id, client_ip)
    # 러너를 선택한다.
    runner = get_runner(request.demo)

    async def event_stream():
        try:
            # 사용자 메시지를 콘텐츠로 만든다.
            content = types.Content(role="user", parts=[types.Part(text=request.message)])
            # ADK 이벤트를 스트리밍한다.
            async for event in runner.run_async(
                user_id=request.session_id,
                session_id=request.session_id,
                new_message=content,
            ):
                payload = _event_payload(event)
                if not payload:
                    continue
                if payload.get("type") == "progress":
                    yield _format_sse("progress", {"logs": payload.get("logs", [])})
                elif payload.get("type") == "final":
                    final_payload = payload.get("payload", {})
                    yield _format_sse("final", final_payload)
        except Exception as exc:
            # 에러 응답을 만든다.
            error_payload = {
                "route": "simulation",
                "blocks": [
                    {
                        "type": "text",
                        "section": "summary",
                        "value": f"내부 오류가 발생했습니다: {exc}",
                    }
                ],
                "tables": {},
                "charts": [],
            }
            yield _format_sse("final", error_payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
