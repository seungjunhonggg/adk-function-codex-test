# /api/chat 워크플로우 (v0)

## 목적
- 캐주얼 대화와 시뮬레이션 요청을 분기한다.
- 시뮬레이션 결과는 표/차트(결정적) + 서술(LLM)로 제공한다.
- 브리핑은 한글 라벨만 사용한다.

## 입력
- session_id
- user_message
- optional: overrides (중간 단계 변경 값)
- optional: demo (데모 모드 여부)

## 라우팅
1. RouteAgent로 의도 분류 (OpenAI ADK structured output)
   - casual → 일반 대화 응답
   - simulation → CommandAgent 실행
2. CommandAgent로 action 결정
   - run → 시뮬레이션 시작/진행
   - explain_stage → 특정 단계 근거 설명
   - 상태 힌트(브리핑 완료 여부/보류 액션)를 참고해 run 우선
3. explain_stage면 ExplainAgent로 근거 설명을 생성(필요한 표/차트만 포함)
4. run이면 1-1~1-8 실행 (변경 요청도 run에서 처리)
5. 1-1 입력에서 chip_type이 포함되면 1-2 생략
6. chip_type이 부분 입력이면 1-3에서 LIKE %chip_type% 조건으로 조회

## 입력 파싱 (LLM)
- simulation + action이 run일 때 InputAgent로 1-1 입력을 추출한다.
- simulation + action이 run일 때 UpdateAgent가 필요한 입력/변경 값을 추출한다.
- output_type으로 구조화하여 필드가 없으면 null로 둔다.
- 누락 필드가 있으면 다음 질문으로 안내한다.
- 누락이 있으면 브리핑 생성은 생략한다.

## 변경 요청 처리 (run 내부)
- UpdateAgent로 변경 값을 추출한다.
- 값이 없으면 pending_action으로 보류하고 질문만 반환한다.
- 값이 있으면 상태를 갱신하고 시뮬레이션을 재실행한다.
- 변경된 dirty 단계 중 가장 앞 단계부터 브리핑 범위를 제한한다(예: 1-4 변경 → 1-4~1-8).
- briefing_hint를 전달해 첫 문장에 변경 반영 문구를 포함한다.

## 요청/응답 스키마 (v0)
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

## 스트리밍 응답 (SSE)
- `/api/chat/stream`은 text/event-stream으로 진행 로그와 최종 응답을 보낸다.
- progress 이벤트는 `{"logs": [{"text": "...", "status": "..."}]}` 형식이다.
- final 이벤트는 `/api/chat`과 동일한 응답 스키마를 사용한다.

### 진행 로그 블록
- blocks 첫 줄에 progress_log를 추가할 수 있다.
- logs는 text와 status(in_progress/done/pending/error)를 가진다.
- UI에서 status는 진행중/완료/대기/오류로 표시한다.

## 시뮬레이션 단계
- 1-1 입력 수집
- 1-2 칩기종 조회(옵션)
- 1-3 레퍼런스 LOT 선정
- 1-4 최적화 API 호출
- 1-5 top-k 선정
- 1-6 최근 6개월 유사 설계 조회
- 1-7 불량률 집계(평균값) + 공정불량률 차트 생성
- 1-8 브리핑 서술 생성

상세 데이터 계약은 `docs/data-contracts.md`를 따른다.

## 브리핑 생성 (LLM)
- LLM은 서술만 작성하고, 표/차트 값은 결정적 처리.
- 서술은 표/차트에 있는 값만 인용.
- children 지표는 기본 숨김, 요청 시 확장.
- structured output은 OpenAI ADK structured output을 사용한다.
- 응답 직전에 report_column_labels 기반 한글 라벨 매핑을 적용한다.
- stage_sequence(단계 순서/근거/표/차트 키)를 함께 전달해 단계별 순서로 블록을 만든다.
- stage_notes는 근거 라인만 추려 stage_sequence.note로 전달한다.
- 1-3 단계는 reference_lot_candidates_table만 사용하고 선택 행을 강조한다.

### LLM 출력 형식 예시
```json
{
  "blocks": [
    {"type": "text", "section": "summary", "value": "..." },
    {"type": "table_ref", "table_key": "reference_lot_candidates_table"},
    {"type": "table_ref", "table_key": "top_k_table"},
    {"type": "chart_ref", "chart_id": "defect_rate_summary"},
    {"type": "text", "section": "conclusion", "value": "..."}
  ]
}
```

## 컨텍스트 예산
- LLM 입력 4k 토큰 이내 유지.
- raw 배열(top-k/불량률 상세)은 컨텍스트에 넣지 않는다.
- 표/차트는 LLM 전용 요약 투영본만 주입하고 하드캡을 적용한다.
- 원본 표/차트는 파일로 저장하고 raw_refs에 경로만 보관한다.

## 메모리/상태
- 세션 메모리: input_params, chip_type, stage_outputs(요약본), stage_notes, last_explain_stage
- 중간 변경 시 무효화:
  - 1-1 변경 → 1-2~1-8 재계산
  - 1-3 변경 → 1-4~1-8 재계산
  - 1-5 변경(k 변경) → 1-6~1-8 재계산
- 값 없는 변경 요청은 pending_action으로 보류하고 재질문한다.
- 입력 파싱 결과는 기존 input_params와 병합한다(새 값만 덮어씀).
- 데모 단계는 인메모리 세션 스토어로 상태를 유지한다.
- 원본 표/차트는 `data/raw_outputs/<session_id>/*.json`에 저장한다.

상태 스키마 상세는 `docs/state-schema.md`를 따른다.

## 후속 질문 처리
- 사용자가 특정 단계 근거를 요청하면 explain_stage로 stage_notes + 증거를 LLM에 전달해 설명한다.
- 필요 시 해당 단계 표/차트만 함께 전달한다.
- children 지표 요청 시 report_defect_children 기반으로 확장 표 생성.

## 에러 처리 (간단)
- 필수 입력 누락: 즉시 안내 후 재질문.
- API 실패: ref 기반 요약만 제공 + 재시도 안내.

## Casual route (LLM)
- route=casual uses CasualAgent to generate text blocks.
- tables/charts are empty for casual replies.

## Stage catalog mapping
- CommandAgent/UpdateAgent use a stage catalog to map user phrases (e.g., REF LOT) to internal stage IDs.
