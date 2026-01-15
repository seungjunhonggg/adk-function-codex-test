# /api/chat 워크플로우 가이드

## 목적
MLCC 개발자용 플랫폼에서 `/api/chat`이 처리하는 전체 흐름을 설명합니다.  
특히 에이전트 라우팅, 시뮬레이션/DB 조회 분기, 이벤트 스트리밍, 메모리 구조를 중점으로 정리합니다.

## 전체 흐름 (모식도)
```mermaid
flowchart TD
  U[User] -->|POST /api/chat| API[FastAPI /api/chat]
  API --> Policy[pending_action policy]
  Policy -->|confirmed| Handler{workflow handler}
  Policy --> Route[route_agent]
  Route -->|primary_intent=chat/unknown| Chat[conversation_agent / discussion_agent]
  Route -->|primary_intent!=chat| Planner[planner_agent]
  Planner -->|run_step| Handler{workflow handler}
  Planner -->|no plan/chat-only| Handler{workflow handler}

  Handler -->|simulation_run/edit| Sim[_maybe_handle_simulation_message]
  Handler -->|db_query| DB[db_agent + query tools]
  Handler -->|chart_edit| Chart[chart_agent + apply_chart_config]
  Handler -->|fallback| Chat[conversation_agent]

  Sim --> Ref[_run_reference_pipeline]
  Ref --> Events[event_bus]
  DB --> Events
  Chart --> Events

  Handler -->|assistant message| Session[SQLiteSession (sessions.db)]
  Events --> UI[Frontend event panel]
```

## 라우팅 로직 요약
1) `/api/chat`는 먼저 `pending_action` 정책 레이어를 확인합니다.  
2) pending_action이 확인되면 해당 워크플로우를 즉시 실행하고 응답합니다.  
3) pending_action이 없으면 먼저 `route_agent`로 의도를 분기합니다.  
4) primary_intent가 `chat/unknown`이면 바로 대화 흐름으로 처리합니다.  
   - `conversation_mode=discussion`이고 `final_briefing` 이벤트가 있으면 `discussion_agent`로 전환합니다.
5) 그 외 intent면 `planner_agent`로 단계화를 시도합니다.  
   - planner context JSON에는 `memory_keys`, `missing_sim_fields`, `has_chip_prod_id`, `events`에 더해  
     `keyword_hints`, `simulation_active`, `route_hint`가 포함됩니다.
   - planner가 plan을 반환하지 않거나 chat-only plan이면 라우터의 primary_intent 핸들러로 fallback합니다.
   - 필수 입력이 누락된 경우에는 confirmation 대신 입력 요청(예: `simulation_form`)을 우선합니다.
6) planner는 한 턴에 여러 step을 연속 실행하며, 누락/에러가 발생하면 해당 step에서 멈춥니다.  
7) planner 경로에서는 `briefing` step 직전에 상세/간단 브리핑 선택을 확인합니다.  
   - 확인 질문은 모델이 직접 생성하며, `pending_action=briefing_choice`로 다음 입력을 받습니다.  
   - 사용자가 다른 의도로 응답하면 `pending_action`을 해제하고 일반 라우팅으로 진행합니다.  
8) 선택 결과가 `간단`이면 `briefing_agent` 요약, `상세`면 최종 브리핑 스트리밍을 출력합니다.  
   - planner 루프 동안 `planner_batch=true`로 표시하고, `_run_reference_pipeline`은 브리핑 스트리밍을 생략한 채 이벤트만 저장합니다. 선택 이후에 브리핑을 출력합니다.  
추가: planner가 `next_action=confirm`을 반환하면 `pending_action`에 저장하고 확인 질문을 출력합니다.

추가: `/api/chat`는 `intent=simulation_run` + `params`를 받으면,
라우터/플래너를 거치지 않고 백엔드에서 파라미터를 확정해 바로 파이프라인을 실행합니다.  
이때 백엔드는 `collect_simulation_params`로 입력값을 갱신하고 `emit_simulation_form`을 송신한 뒤,
`_run_reference_pipeline(emit_briefing=false)`로 이벤트를 저장하고, 이후 브리핑 선택 질문만 출력합니다.
추가: `emit_simulation_form`은 `conversation_mode=execution`으로 전환합니다.  
최종 브리핑이 출력되면 `conversation_mode=discussion`으로 전환합니다.

주요 intent:
- `simulation_run` / `simulation_edit`
- `db_query`
- `chart_edit`
- `chat` / `unknown`

참고: `simulation_run` 의도는 "추천/인접/시뮬/예측"뿐 아니라 "시작/실행" 키워드도 포함하며,
파라미터가 없어도 입력 패널(`simulation_form`)을 바로 열어 줍니다.

핵심 파일:
- `backend/app.py`: `/api/chat`, 라우팅, 핸들러
- `backend/agents.py`: `planner_agent`, `route_agent`, `chart_agent`, `db_agent`, `conversation_agent`, `briefing_agent`
- `backend/guardrails.py`: MLCC 입력/출력 가드레일

## 시뮬레이션 플로우 (simulation_run/edit)
핸들러: `backend/app.py::_maybe_handle_simulation_message`

1) 입력 파싱/보정  
   - `extract_simulation_params_hybrid` → `simulation_store` 업데이트  
   - `capacity`는 pF로 정규화됨 (`backend/simulation.py`)
   - 삭제/제외 요청은 `clear_fields`로 전달되어 해당 파라미터를 제거

2) 입력 패널 업데이트  
   - `emit_simulation_form` 이벤트로 UI 패널 갱신

3) 레퍼런스 파이프라인 실행  
   - `backend/app.py::_run_reference_pipeline`  
   - 내부 주요 단계:
    - 레퍼런스 LOT 선택 (`reference_lot.py`)
      - ref_lot_search_filter: 파라미터 필터 + conditions.required_not_null +
        defect_conditions + (파라미터 기반으로 찾은 chip_prod_id IN 조건)을 단일 조회로 적용
      - lot_search 불량 필터: cutting_defect/measure_defect는 S 등급/A 등급/B 등급,
        contact_defect=0 또는 NULL,
        x_fr_ispass/pass_halt/pass_8585/pass_burn_in/x_df_ispass/x_odb_pass_yn은
        IS DISTINCT FROM 'NG'
      - 정렬 우선순위: cutting_defect+measure_defect 등급(S > A > B) → bdv_avg 내림차순
        → x_tr_short_defect_rate 오름차순 → 최신 input_date
      - 필터링 결과(LOT_ID + defect_conditions 컬럼)는 최종 브리핑 메시지에
        마크다운 표로 포함
      - rerun 시 이전 grid_overrides가 있으면 재활용
    - LOT 불량 조건 요약 표 이벤트 송신(조건/연산/값 표시용 table)
    - grid_search 후보 생성
      - payload 형식: sim_type=ver4, data(ref/sim), targets(electrode_c_avg*1.05 등),
        params(screen_* / active_layer / cover_sheet_thk / total_cover_layer_num /
        gap_sheet_thk / ldn_avr_value / cast_dsgn_thk)
      - data.ref/data.sim은 `grid_search.payload_columns` 리스트를 기준으로 구성
        - DB에 존재하는 컬럼만 ref_lot 조회 단계에서 함께 가져옴
        - DB에 없는 컬럼은 `payload_fill_value`(기본 -1)로 채워 전송
        - sim 값이 없으면 `payload_fallback_to_ref=true`일 때 ref 값을 우선 사용
      - 응답 형식: result.datas.sim 순서대로 TOP 후보 설계값 (0=1순위)
      - 후보 설계값에서 active_powder_base/active_powder_additives/ldn_avr_value/cast_dsgn_thk 추출
      - mdh_base_view_total에서 설계값 동일 + design_input_date 최근 6개월 조건으로 매칭 LOT 조회
        - 조회는 전체 컬럼(SELECT *) 기준으로 수행
        - 브리핑 표는 `column_briefing_table`의 `post_grid_lot_search` 컬럼만 표시
        - 매칭 진단 정보(`post_grid_defects.diagnostics`)를 함께 남김
      - 후보별 불량 인자 평균( `grid_defect_columns` 또는 `post_grid_defect_columns` 설정값 )을
        막대그래프 이벤트(`defect_rate_chart`, bar_orientation=vertical)로 송신
      - 데모 환경에서도 포스트그리드 불량 인자를 보이려면 해당 컬럼 리스트를
        `grid_defect_columns`/`post_grid_defect_columns`에 명시해야 함
    - TOP3 설계안의 최근 6개월 LOT 불량률 통계 조회
    - 최종 브리핑 이벤트/메시지 송신
      - 순차 브리핑 구성: 1) 레퍼 LOT 후보 요약+표(상위 10개) →
        2) 선택 Ref LOT 상세 표(컬럼 다수는 분할 표) →
        3) 그리드 서치 요약 표(rank/electrode_c_avg/grinding_t_avg/active_layer/cast_dsgn_thk/ldn_avr_value)
      - `design_blocks` 포함
      - OPENAI_API_KEY가 있으면 서술부 문장만 LLM으로 다듬고(표/숫자 고정),
        검증 실패 시 템플릿을 그대로 사용
    - WebSocket이 연결된 경우 브리핑 텍스트를 `chat_stream_*`로 델타 스트리밍하고,
      표는 `briefing_table`로 분리 전송
      (응답에는 `streamed=true`가 포함되어 HTTP 응답의 중복 출력 방지)
    - 섹션 간 딜레이는 `BRIEFING_STREAM_DELAY_SECONDS`로 조정 가능 (기본 0.03s)
    - 가짜 스트리밍 모드: 텍스트는 `chat_stream_*` 델타로 흘리고,
      표는 `briefing_table` 이벤트로 분리 전송해 애니메이션 표시
    - planner 경로에서는 브리핑 전에 상세/간단 선택을 확인한 뒤 출력
    - 브리핑 완료 후 post-grid LOT 기준 공정불량률 차트를 계산해 `defect_rate_chart`로 송신하고,
      검사불량률 차트는 사용자 확인 후 추가로 송신

4) 이벤트 스트리밍  
   - `event_bus.broadcast`로 프론트 카드 업데이트  
   - 주요 이벤트: `simulation_form`, `lot_result`, `defect_rate_chart`,  
     `design_candidates`, `final_briefing`
   - 시뮬레이션 관련 이벤트는 `run_id`를 포함해 최신 실행만 필터링되도록 구성
   - 진행 로그 이벤트:
     - `pipeline_status`: 단계별 진행 메시지 (프론트에서 누적 로그로 표시)
       - planner 루프는 `stage=planner`로 step 진행 로그를 남김
     - `pipeline_stage_tables`: 단계별 표(마크다운) 목록 전송,
       진행 로그 클릭 시 해당 표를 펼쳐서 확인 가능
   - 브리핑 본문은 아래 이벤트로 스트리밍 가능
     - `chat_stream_start` → `chat_stream_block_start` → `chat_stream_delta` →
       `chat_stream_block_end` → `chat_stream_end`
     - 표는 `briefing_table`로 별도 전송 (프론트에서 애니메이션 처리)

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
   (`db_result`는 `pipeline_store.events`에도 저장)

관련 문서:
- `docs/db-agent-workflow.md`

## 차트 수정 플로우 (chart_edit)
핸들러: `backend/app.py::_handle_chart_request`

- `chart_agent`가 요청을 해석 → `apply_chart_config_impl` 적용  
- 기존 `defect_rate_chart` 이벤트를 업데이트 (히스토그램/라인/바 등)

## 메모리 구조 (중요)
### 1) 장기 대화 메모리 (SQLite)
- `SQLiteSession`이 `sessions.db`에 대화 메시지를 저장  
- `/api/chat` 실행마다 세션 단위로 누적

### 2) 파이프라인 상태 메모리 (프로세스 메모리)
- `pipeline_store`: 단계 상태/이벤트/워크플로우 ID/요약 보관
- `planner_batch`: planner 루프 동안 true로 표시해 브리핑 스트리밍을 억제
- `planner_state`/`planner_goal`: planner 단계/의존성/다음 액션 스냅샷 저장
- `briefing_text`/`briefing_summary`: 최종 브리핑 출력과 요약 캐시
- `briefing_text_mode`/`briefing_text_run_id`: 브리핑 출력 종류(간단/상세)와 run_id 캐시
- `briefing_mode`/`briefing_mode_run_id`: 사용자가 선택한 브리핑 유형과 run_id
- `conversation_mode`/`briefing_preference`: 실행/토론 모드 및 브리핑 선호도 기록
- `pending_action`/`pending_plan`/`pending_inputs`/`dialogue_state`: 제안→확인→실행 상태 관리 (briefing_choice는 미응답 시 해제)
- 이벤트 패널 재현에 필요한 payload를 저장
- 단계별 진행 로그용 `stage_tables`(reference/grid 표 목록)도 이벤트로 저장
- `stage_inputs`: 단계별 입력 스냅샷(recommendation/reference/grid/selection/chart)과 `run_id`를 저장해
  특정 단계 수정 시 재실행 기준으로 사용
- `PIPELINE_STATE_DB_PATH`(기본 `sessions.db`)의 `pipeline_state` 테이블에 스냅샷을 저장해
  서버 재시작 시에도 세션별 이벤트 복원이 가능하도록 함
- `reference` 페이로드에는 최종 브리핑 표 생성을 위한
  `reference_columns`/`reference_rows`(LOT_ID + defect_conditions) 포함
- `discussion_agent` 입력 컨텍스트는 grid 관련 payload를 축약해 전달
  - TOP3 후보만 유지하고 `grid_search.factors`의 설계조건 키/컬럼만 포함
  - `final_briefing`/`grid`/`design_candidates`에서 불필요한 상세 컬럼은 제외

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

## 가드레일
- `conversation_agent`: 입력/출력 가드레일로 MLCC 요청을 일반 대화로 처리하지 않도록 차단
- `discussion_agent`: 출력 가드레일로 브리핑 이후 답변이 근거 없는 수치/LOT를 포함하지 않도록 제어
- `briefing_agent`: 출력 가드레일로 근거 없는 LOT/불량률/파라미터 언급을 방지
- 출력 가드레일은 `pipeline_store` 이벤트 유무를 근거로 검사하고, 부족 시 fallback 메시지를 반환

## 이벤트 패널 업데이트 규칙
UI는 `event_bus` 이벤트만 보고 갱신합니다.
- 예: `defect_rate_chart`, `final_briefing`
- `pipeline_store`는 이벤트 재생/복구용 캐시 역할
  
추가: 새 시뮬레이션 요청에 파라미터가 부족한 경우, `/api/chat` 응답에
`ui_event=simulation_form`을 함께 넣어 WebSocket 미연결 상태에서도 입력 패널이 즉시 표시됩니다.

추가: 최종 브리핑은 `design_blocks`를 받아 설계값 표 + 평균 불량률 히스토그램을
후보별로 묶어서 렌더링합니다. (Top3 기준)

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
- 가드레일: `backend/guardrails.py`
