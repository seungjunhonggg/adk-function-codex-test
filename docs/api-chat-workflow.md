# /api/chat 워크플로우 (v1, Google ADK)

## 목적
- 캐주얼 대화와 시뮬레이션 요청을 분기한다.
- 시뮬레이션 결과는 표/차트(결정적) + 서술(LLM)로 제공한다.
- 브리핑은 한글 라벨만 사용한다.

## 입력
- session_id
- user_message
- optional: overrides (중간 단계 변경 값)
- optional: demo (데모 모드 여부)

## 라우팅 (RootAgent)
1. RootAgent가 세션 상태를 로드하고 ADK session.state에 저장한다.
2. RouteAgent(LlmAgent, output_schema)로 의도 분류
   - casual → 일반 대화 응답
   - simulation → CommandAgent 실행
   - select_candidates 액션 payload면 바로 simulation으로 진입
3. CommandAgent(LlmAgent, output_schema)로 action 결정
   - run / reset / explain_stage
4. explain_stage면 ExplainAgent(LlmAgent)로 근거 설명 생성
5. run이면 입력/변경 파싱 후 SequentialAgent 실행
6. reset이면 세션 상태 초기화 후 안내 블록 반환

## SequentialAgent 구성 (시뮬레이션 단계)
- RootAgent 하위에 SequentialAgent를 배치하고, 각 단계별 StageAgent가 순차 실행된다.
- StageAgent는 before_agent_callback으로 **dirty 단계가 아니면 스킵**한다.
- StageAgent에서 gap이 발생하면 temp:halt 플래그를 세우고 이후 단계는 callback으로 스킵된다.

### 단계 목록
- 1-1 입력 수집
- 1-2 칩기종 조회
- 1-3 레퍼런스 LOT 선정
- 1-4 최적화 API payload
- 1-5 top-k 선정
- 1-6 최근 6개월 유사 설계 조회
- 1-7 불량률 집계 + 차트
- 1-8 브리핑 서술 생성

## 입력 파싱 (LLM)
- simulation + action이 run일 때 InputAgent로 1-1 입력을 추출한다.
- simulation + action이 run일 때 UpdateAgent가 필요한 입력/변경 값을 추출한다.
- ADK output_schema(Pydantic) 기반 구조화 출력 사용.
- chip_prod_id가 있으면 나머지 4개 입력이 없어도 누락으로 보지 않는다.
- chip_prod_id가 없으면 4개 입력을 모두 받아야 한다.
- 누락 필드가 있으면 다음 질문으로 안내한다.

## 변경 요청 처리 (DAG dirty)
- UpdateAgent로 변경 값을 추출한다.
- 값이 없으면 pending_action으로 보류하고 질문만 반환한다.
- 값이 있으면 상태를 갱신하고 dirty_stages를 계산한다.
- dirty_stages 중 가장 앞 단계부터 브리핑 범위를 제한한다.

## 데이터 공백 처리 (gap)
- 각 단계에서 strict 조회 결과가 0건이면 gap을 반환한다.
- gap이 발생하면 temp:halt 플래그를 세우고 이후 단계는 callback으로 스킵한다.
- GapAgent가 확인 질문을 생성한다.
- 후보 선택이 필요한 경우 table_select 블록을 반환한다.
- 사용자가 선택을 제출하면 해당 단계부터 재실행한다.
- JSON 선택이 없으면 SelectionAgent가 텍스트 선택을 추출한다.

## 요청/응답 스키마 (v1)
### 요청
```json
{
  "session_id": "string",
  "message": "string",
  "overrides": {},
  "demo": false
}
```

### 응답
```json
{
  "route": "casual | simulation",
  "blocks": [],
  "tables": {},
  "charts": []
}
```

### table_select 블록 (선택 UI)
```json
{
  "type": "table_select",
  "table_key": "chip_type_candidates_table",
  "id_field": "chip_type_id",
  "selection_field": "chip_type_ids",
  "allow_multi": true,
  "action": "select_candidates",
  "submit_label": "해당 기종으로 진행"
}
```

### 선택 제출 payload
```json
{
  "action": "select_candidates",
  "selection": {
    "chip_type_ids": ["CT-001", "CT-003"]
  }
}
```

## 스트리밍 응답 (SSE)
- `/api/chat/stream`은 text/event-stream으로 progress + final 이벤트를 보낸다.
- progress 이벤트는 StageAgent가 단계 시작 시 JSON으로 전송한다.
- final 이벤트는 RootAgent가 최종 응답 페이로드를 전송한다.

## 메모리/상태
- 세션 상태는 기존 state 스키마(input_params, stage_outputs, stage_notes 등)를 유지한다.
- ADK session.state에는 앱 상태(app:mlcc_state)와 런타임 상태(temp:*)가 분리되어 저장된다.
- 프로덕션은 Postgres(data_portal 스키마)에 상태를 영속화한다.
- 데모 모드는 인메모리 세션 스토어로 상태를 유지한다.

## 브리핑 생성 (LLM)
- 1-8 단계에서 BriefingAgent가 단계별 텍스트를 생성한다.
- 표/차트 참조(table_ref/chart_ref)는 코드에서 삽입한다.
- briefing_hint는 dirty 시작 단계가 있을 때 첫 문장에 반영한다.

## 컨텍스트 예산
- LLM 입력 4k 토큰 이내 유지.
- raw 배열(top-k/불량률 상세)은 컨텍스트에 넣지 않는다.
- 표/차트는 LLM 전용 요약 투영본만 주입하고 하드캡을 적용한다.

## 주의
- ADK output_schema 사용 시 tools는 함께 사용하지 않는다.
- StageAgent는 before_agent_callback으로 dirty/halt를 판단해 스킵한다.

상태 스키마 상세는 `docs/state-schema.md`를 따른다.
