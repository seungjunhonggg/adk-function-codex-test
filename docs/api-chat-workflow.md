# /api/chat 워크플로우 가이드

## 목적
MLCC 개발자용 플랫폼에서 `/api/chat`이 처리하는 전체 흐름을 설명합니다.  
특히 에이전트 라우팅, 시뮬레이션/DB 조회 분기, 이벤트 스트리밍, 메모리 구조를 중점으로 정리합니다.

## 전체 흐름 (모식도)
```mermaid
flowchart TD
  U[User] -->|POST /api/chat| API[FastAPI /api/chat]
  API --> Route[route_agent]
  Route -->|primary_intent| Handler{workflow handler}

  Handler -->|simulation_run/edit| Sim[_maybe_handle_simulation_message]
  Handler -->|db_query| DB[db_agent + query tools]
  Handler -->|chart_edit| Chart[chart_agent + apply_chart_config]
  Handler -->|stage_view| Stage[show_simulation_stage]
  Handler -->|chat/unknown| Chat[conversation_agent]

  Sim --> Ref[_run_reference_pipeline]
  Ref --> Events[event_bus]
  DB --> Events
  Chart --> Events
  Stage --> Events

  Handler -->|assistant message| Session[SQLiteSession (sessions.db)]
  Events --> UI[Frontend event panel]
```

## 라우팅 로직 요약
1) `/api/chat`는 `route_agent`를 통해 intent를 결정합니다.  
2) intent에 따라 `WORKFLOW_HANDLERS`가 분기됩니다.  
3) LLM 비활성화 시(`OPENAI_API_KEY` 없음)에는 휴리스틱 라우팅으로 fallback됩니다.

주요 intent:
- `simulation_run` / `simulation_edit`
- `db_query`
- `chart_edit`
- `stage_view`
- `chat` / `unknown`

핵심 파일:
- `backend/app.py`: `/api/chat`, 라우팅, 핸들러
- `backend/agents.py`: `route_agent`, `chart_agent`, `db_agent`, `conversation_agent`

## 시뮬레이션 플로우 (simulation_run/edit)
핸들러: `backend/app.py::_maybe_handle_simulation_message`

1) 입력 파싱/보정  
   - `extract_simulation_params_hybrid` → `simulation_store` 업데이트  
   - `capacity`는 pF로 정규화됨 (`backend/simulation.py`)

2) 입력 패널 업데이트  
   - `emit_simulation_form` 이벤트로 UI 패널 갱신

3) 레퍼런스 파이프라인 실행  
   - `backend/app.py::_run_reference_pipeline`  
   - 내부 주요 단계:
     - 레퍼런스 LOT 선택 (`reference_lot.py`)
     - LOT 불량률 히스토그램(1차) 이벤트 송신
     - grid_search 후보 생성
     - TOP3 설계안의 최근 3개월 LOT 불량률 통계 조회
     - 최종 브리핑 이벤트/메시지 송신

4) 이벤트 스트리밍  
   - `event_bus.broadcast`로 프론트 카드 업데이트  
   - 주요 이벤트: `simulation_form`, `lot_result`, `defect_rate_chart`,  
     `design_candidates`, `final_defect_chart`, `final_briefing`

핵심 파일:
- `backend/reference_lot.py`: 레퍼런스 LOT, post-grid 불량 통계
- `backend/reference_lot_rules.json`: 설계/불량 컬럼 설정
- `backend/tools.py`: 이벤트 및 차트 관련 유틸

## DB 조회 플로우 (db_query)
핸들러: `backend/app.py::_handle_db_query_workflow`

1) `db_agent`가 사용자의 요청을 DB 조회로 변환  
2) 필요 시 `resolve_view_columns`로 컬럼 후보를 정규화  
3) 조회/집계는 `query_view_table` 또는 `query_view_metrics` 실행  
4) 결과는 `db_result` 또는 `defect_rate_chart` 이벤트로 전달

관련 문서:
- `docs/db-agent-workflow.md`

## 차트 수정 플로우 (chart_edit)
핸들러: `backend/app.py::_handle_chart_request`

- `chart_agent`가 요청을 해석 → `apply_chart_config_impl` 적용  
- 기존 `defect_rate_chart` 이벤트를 업데이트 (히스토그램/라인/바 등)

## 스테이지 화면 재생 (stage_view)
핸들러: `backend/app.py::_handle_stage_view_request`

- 사용자가 “추천/레퍼런스/그리드/최종” 화면 요청 시  
  `stage_focus` 이벤트와 함께 해당 카드로 포커스 이동

## 메모리 구조 (중요)
### 1) 장기 대화 메모리 (SQLite)
- `SQLiteSession`이 `sessions.db`에 대화 메시지를 저장  
- `/api/chat` 실행마다 세션 단위로 누적

### 2) 파이프라인 상태 메모리 (프로세스 메모리)
- `pipeline_store`: 단계 상태/이벤트/워크플로우 ID/요약 보관
- 이벤트 패널 재현에 필요한 payload를 저장

### 3) 시뮬레이션 파라미터 메모리
- `simulation_store`: 온도/전압/사이즈/용량/양산 여부 + chip_prod_id
- `recommendation_store`: 추천 결과와 예측 시뮬레이션 대기 상태

### 4) LOT/그리드 결과 메모리
- `lot_store`: 마지막 LOT 조회 결과
- `test_simulation_store`: grid 후보/결과 캐시 (테스트용)

### 5) 스키마/설정 캐시
- `db_connections.json` 내 `schema` 필드 (앱 시작 시 preload)
- `db_view_profile.json`, `db_view_columns.json`은 조회/매핑 기준

### 메모리 요약 정책
- 긴 메시지는 `_build_message_memory_summary`로 요약  
- `pipeline_store`에 `pending_memory_summary`로 저장  
- 다음 `_append_assistant_message`에서 요약본을 `sessions.db`에 기록  
  → 컨텍스트 용량을 절약하면서 핵심만 유지

## 이벤트 패널 업데이트 규칙
UI는 `event_bus` 이벤트만 보고 갱신합니다.
- 예: `defect_rate_chart`, `final_defect_chart`, `final_briefing`
- `pipeline_store`는 이벤트 재생/복구용 캐시 역할
  
추가: 새 시뮬레이션 요청에 파라미터가 부족한 경우, `/api/chat` 응답에
`ui_event=simulation_form`을 함께 넣어 WebSocket 미연결 상태에서도 입력 패널이 즉시 표시됩니다.

## 확장 포인트
1) 새로운 워크플로우 추가  
   - `backend/agents.py`에 intent 정의  
   - `backend/app.py`에 핸들러 등록

2) DB 질의 확장  
   - `db_view_profile.json` 및 `db_view_columns.json` 확장  
   - `db_agent` 프롬프트 갱신

3) 메모리 전략 변경  
   - `pipeline_store` 요약 정책 조정  
   - `sessions.db` 저장 로직 변경

## 파일 맵 요약
- `/api/chat`: `backend/app.py`
- 라우팅/에이전트: `backend/agents.py`
- 시뮬레이션/파라미터: `backend/simulation.py`
- 레퍼런스 LOT/불량: `backend/reference_lot.py`, `backend/reference_lot_rules.json`
- 이벤트/스트림: `backend/events.py`
- 상태 저장: `backend/pipeline_store.py`, `backend/lot_store.py`
