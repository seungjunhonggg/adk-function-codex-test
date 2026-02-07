# /api/chat 워크플로우 (v5, Google ADK)

## 목적
- MLCC 시뮬레이션 단계를 순차로 진행한다.
- 단계 완료 후 사용자 확인(컨펌)을 필수로 받는다.
- 중간 변경 요청 시 dirty 단계부터 재실행한다.
- 입력 부족 상태에서는 다음 단계로 진행하지 않는다.

## 요청 스키마
```json
{
  "session_id": "string",
  "message": "string",
  "overrides": {},
  "demo": false
}
```

## 내부 상태 핵심 필드
- `sim_step`: 현재 진행 가능한 단계 번호
- `input_params`: 사용자 입력 인자
- `selections`: 사용자 선택값
- `configs`: 실행 설정값 (`top_k`, `months`)
- `stage_status`: 단계별 상태 (`idle|done|dirty|gap`)
- `stage_outputs`: 단계별 산출물 요약
- `pending_action`: 다음 UI 액션 유도 정보
- `last_gap`: 최근 입력 부족/불일치 정보

## 단계 진행 규칙
- 각 단계 도구는 `sim_step` gate를 통과할 때만 실행한다.
- `pending_action.type=confirm_stage` 상태에서는 다음 단계 실행을 차단한다.
- `pending_action.type=need_input` 상태에서는 후단계 실행을 차단한다.
- `apply_user_patch`는 변경 필드 기준으로 dirty 시작 단계를 계산하고 이후 단계 산출물을 무효화한다.

## SSE 이벤트
- `event: progress`
  - `{ "logs": [...] }` 형태의 진행 상태 전달
- `event: final`
  - 최종 응답 payload 전달

## Final Payload 계약
`/api/chat/stream`의 `final`은 아래 **정식 envelope만** 보낸다.

### 1) 정식 Envelope (신규)
```json
{
  "route": "simulation",
  "assistant": {
    "text": "단계 결과 요약"
  },
  "ui_action": {
    "type": "open_form",
    "stage": "1-2",
    "form_id": "form-1-2",
    "title": "입력값을 보완해 주세요.",
    "submit_action": "apply_user_patch",
    "fields": []
  },
  "workflow": {
    "sim_step": 2,
    "stage_status": {},
    "pending_action": {}
  }
}
```

## 프론트 처리 원칙
- 프론트는 `assistant.text`와 `ui_action`을 직접 소비해 화면 컴포넌트를 구성한다.
- `ui_action.type=open_form`이면 입력 폼을 렌더하고, 제출 시 `apply_user_patch` 액션 메시지를 보낸다.
- `ui_action.type=confirm_stage`이면 진행/수정 버튼을 렌더하고 클릭값을 사용자 메시지로 전송한다.
