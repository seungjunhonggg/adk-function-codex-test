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
   - update_input → 입력/선택 변경
   - explain_stage → 특정 단계 근거 설명
3. explain_stage면 ExplainAgent로 근거 설명을 생성(필요한 표/차트만 포함)
4. run/update_input이면 1-1~1-8 실행
5. 1-1 입력에서 chip_type이 포함되면 1-2 생략

## 입력 파싱 (LLM)
- simulation + action이 run일 때 InputAgent로 1-1 입력을 추출한다.
- simulation + action이 update_input일 때 UpdateAgent가 필요한 입력/변경 값을 추출한다.
- output_type으로 구조화하여 필드가 없으면 null로 둔다.
- 누락 필드가 있으면 다음 질문으로 안내한다.
- 누락이 있으면 브리핑 생성은 생략한다.

## update_input 처리
- action이 update_input이면 UpdateAgent로 변경 값을 추출한다.
- 값이 없으면 pending_action으로 보류하고 질문만 반환한다.
- 값이 있으면 상태를 갱신하고 시뮬레이션을 재실행한다.

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

## 시뮬레이션 단계
- 1-1 입력 수집
- 1-2 칩기종 조회(옵션)
- 1-3 레퍼런스 LOT 선정
- 1-4 최적화 API 호출
- 1-5 top-k 선정
- 1-6 최근 6개월 유사 설계 조회
- 1-7 불량률 집계 + 차트 데이터 생성
- 1-8 브리핑 서술 생성

상세 데이터 계약은 `docs/data-contracts.md`를 따른다.

## 브리핑 생성 (LLM)
- LLM은 서술만 작성하고, 표/차트 값은 결정적 처리.
- 서술은 표/차트에 있는 값만 인용.
- children 지표는 기본 숨김, 요청 시 확장.
- structured output은 OpenAI ADK structured output을 사용한다.
- 응답 직전에 report_column_labels 기반 한글 라벨 매핑을 적용한다.

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
- 표 요약/차트 데이터만 주입한다.

## 메모리/상태
- 세션 메모리: input_params, chip_type, stage_outputs, stage_notes, last_explain_stage
- 중간 변경 시 무효화:
  - 1-1 변경 → 1-2~1-7 재계산
  - 1-3 변경 → 1-4~1-7 재계산
  - 1-5 변경(k 변경) → 1-6~1-7 재계산
- 값 없는 변경 요청은 pending_action으로 보류하고 재질문한다.
- 입력 파싱 결과는 기존 input_params와 병합한다(새 값만 덮어씀).
- 데모 단계는 인메모리 세션 스토어로 상태를 유지한다.

상태 스키마 상세는 `docs/state-schema.md`를 따른다.

## 후속 질문 처리
- 사용자가 특정 단계 근거를 요청하면 explain_stage로 stage_notes + 증거를 LLM에 전달해 설명한다.
- 필요 시 해당 단계 표/차트만 함께 전달한다.
- children 지표 요청 시 report_defect_children 기반으로 확장 표 생성.

## 에러 처리 (간단)
- 필수 입력 누락: 즉시 안내 후 재질문.
- API 실패: ref 기반 요약만 제공 + 재시도 안내.
