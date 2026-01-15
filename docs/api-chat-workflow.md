# /api/chat 워크플로우 가이드

## 목적
MLCC 개발자용 플랫폼에서 `/api/chat`이 처리하는 전체 흐름을 설명합니다.  
특히 에이전트 라우팅, 시뮬레이션/DB 조회 분기, 이벤트 스트리밍, 메모리 구조를 중점으로 정리합니다.

## 전체 흐름 (모식도)
```mermaid
flowchart TD
  U[User] -->|POST /api/chat| API[FastAPI /api/chat]
  API -->|intent=simulation_run or simulation_active| SimAgent[시뮬레이션 에이전트]
  API -->|else| Orchestrator[오케스트레이터]
  Orchestrator -->|handoff| SimAgent

  SimAgent --> SimTool[run_simulation_workflow]
  SimAgent --> BriefTool[run_detailed_briefing]
  SimAgent --> ChartTool[chart_agent + apply_chart_config]
  SimTool --> Ref[_run_reference_pipeline]

  Ref --> Events[event_bus]
  BriefTool --> Events
  ChartTool --> Events

  SimAgent --> Session[SQLiteSession (sessions.db)]
  Orchestrator --> Session
  Events --> UI[Frontend event panel]
```

## 라우팅 로직 요약
1) `/api/chat`는 메시지 수신 후 `request.params`가 있으면 `PARAMS_JSON: {...}` 라인을 붙입니다.  
2) `intent=simulation_run` 또는 `simulation_store.is_active`면 시뮬레이션 에이전트를 바로 실행합니다.  
3) 그 외는 오케스트레이터가 일반 대화를 처리합니다.  
4) 오케스트레이터는 시뮬레이션/차트/브리핑 요청 시 handoff로 시뮬레이션 에이전트를 호출합니다.  
5) 시뮬레이션 에이전트는 `run_simulation_workflow`로 입력 수집과 파이프라인 실행을 처리합니다.  
6) 상세 브리핑 요청 시 `run_detailed_briefing`으로 UI 스트리밍만 수행하고, 응답은 짧게 반환합니다.  
7) 차트 변경 요청은 `chart_agent` → `apply_chart_config` 순으로 처리합니다.

주요 intent:
- `simulation_run`
- `chart_edit`
- `chat`

핵심 파일:
- `backend/app.py`: `/api/chat` 엔트리 포인트
- `backend/agents.py`: `orchestrator_agent`, `simulation_flow_agent`, `chart_agent`
- `backend/tools.py`: `run_simulation_workflow`, `run_detailed_briefing`, `apply_chart_config`

## 시뮬레이션 플로우 (simulation_run/edit)
핸들러: `backend/tools.py::run_simulation_workflow`

1) 입력 파싱/보정  
   - `PARAMS_JSON`이 있으면 먼저 반영  
   - `extract_simulation_params_hybrid` → `simulation_store` 업데이트  
   - `capacity`는 pF로 정규화됨 (`backend/simulation.py`)

2) 입력 패널 업데이트  
   - `emit_simulation_form` 이벤트로 UI 패널 갱신
   - 파라미터 변경/누락이 있을 때만 패널을 갱신

3) 레퍼런스 파이프라인 실행  
   - `backend/app.py::_run_reference_pipeline(emit_briefing=false)`  
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
      - 후보 설계값에서 `grid_match_fields` 컬럼값만 추출
      - `mdh_base_view_total_4`에서 동일 설계 LOT 조회
        - 매칭 기준: `grid_match_fields` + design_input_date 최근 6개월
        - 조회는 전체 컬럼(SELECT *) 기준으로 수행
        - 브리핑 표는 `column_briefing_table`의 `post_grid_lot_search` 컬럼만 표시
        - 매칭 진단은 `candidate_matches` 기반으로 표시
      - 그리드 결과 표 컬럼은 `column_briefing_table`의 `grid_search` 컬럼만 사용
      - 후보별 불량 인자 평균( `grid_defect_columns` 또는 `post_grid_defect_columns` 설정값 )을
        막대그래프 이벤트(`defect_rate_chart`, bar_orientation=vertical)로 송신
      - 데모 환경에서도 포스트그리드 불량 인자를 보이려면 해당 컬럼 리스트를
        `grid_defect_columns`/`post_grid_defect_columns`에 명시해야 함
    - TOP3 설계안의 최근 6개월 LOT 불량률 통계 조회
    - 최종 브리핑 요약 payload 저장
      - `candidate_matches` 포함 (post_grid_lot_search 표/진단용)
      - `design_blocks` 포함
      - 브리핑 본문은 `run_detailed_briefing` 호출 시에만 생성/스트리밍
    - 상세 브리핑은 `chat_stream_*` + `briefing_table` 이벤트로 스트리밍
      - 응답 텍스트는 짧게 유지해 컨텍스트를 줄임
    - 상위 N/블록 수 변경 요청은 grid 재실행으로 처리

4) 이벤트 스트리밍  
   - `event_bus.broadcast`로 프론트 카드 업데이트  
   - 주요 이벤트: `simulation_form`, `lot_result`, `defect_rate_chart`,  
     `design_candidates`, `final_briefing`
   - 시뮬레이션 관련 이벤트는 `run_id`를 포함해 최신 실행만 필터링되도록 구성
   - 진행 로그 이벤트:
     - `pipeline_status`: 단계별 진행 메시지 (프론트에서 누적 로그로 표시)
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
현재 `/api/chat` 플로우에서는 DB 조회 에이전트를 사용하지 않습니다.

## 차트 수정 플로우 (chart_edit)
핸들러: `backend/agents.py::simulation_flow_agent`

- `chart_agent`가 요청을 해석 → `apply_chart_config` 적용  
- 기존 `defect_rate_chart` 이벤트를 업데이트 (히스토그램/라인/바 등)

## 메모리 구조 (중요)
### 1) 장기 대화 메모리 (SQLite)
- `SQLiteSession`이 `sessions.db`에 대화 메시지를 저장  
- `/api/chat` 실행마다 세션 단위로 누적

### 2) 파이프라인 상태 메모리 (프로세스 메모리)
- `pipeline_store`: 단계 상태/이벤트/워크플로우 ID/요약 보관
- `briefing_text`/`briefing_summary`: 상세 브리핑 본문과 요약 캐시  
  - 에이전트 컨텍스트에는 요약만 유지해 길이를 제한
- `chart_config`: 최신 차트 스펙 저장 (chart_type/bins/range_min/range_max/normalize/value_unit)
- 이벤트 패널 재현에 필요한 payload를 저장
- 단계별 진행 로그용 `stage_tables`(reference/grid 표 목록)도 이벤트로 저장
- `stage_inputs`: 단계별 입력 스냅샷(recommendation/reference/grid/selection/chart)과 `run_id`를 저장해
  특정 단계 수정 시 재실행 기준으로 사용
- `PIPELINE_STATE_DB_PATH`(기본 `sessions.db`)의 `pipeline_state` 테이블에 스냅샷을 저장해
  서버 재시작 시에도 세션별 이벤트 복원이 가능하도록 함
- `reference` 페이로드에는 최종 브리핑 표 생성을 위한
  `reference_columns`/`reference_rows`(LOT_ID + defect_conditions) 포함
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
- 상세 브리핑 본문은 `pipeline_store.briefing_text`에 저장하고 UI로만 스트리밍  
- 에이전트 응답은 짧은 요약으로 유지해 `sessions.db` 컨텍스트 길이를 제한

## 가드레일
현재 `/api/chat` 플로우에서는 가드레일을 사용하지 않습니다.  
필요 시 각 에이전트에 input/output guardrail을 추가해 확장할 수 있습니다.

## 이벤트 패널 업데이트 규칙
UI는 `event_bus` 이벤트만 보고 갱신합니다.
- 예: `defect_rate_chart`, `final_briefing`
- `pipeline_store`는 이벤트 재생/복구용 캐시 역할
  
추가: 시뮬레이션 입력 패널은 `emit_simulation_form` 이벤트로 갱신됩니다.

추가: 최종 브리핑은 `design_blocks`를 받아 설계값 표 + 평균 불량률 히스토그램을
후보별로 묶어서 렌더링합니다. (Top3 기준)

## 확장 포인트
1) 새로운 워크플로우 추가  
   - `backend/agents.py`에 에이전트/툴 추가  
   - `backend/tools.py`에 워크플로우 툴 정의

2) 메모리 전략 변경  
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
