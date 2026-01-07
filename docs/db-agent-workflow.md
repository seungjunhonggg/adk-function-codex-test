# DB 에이전트 워크플로우 가이드

## 목적
DB 에이전트는 **1개의 PostgreSQL DB의 1개 view 테이블**을 대상으로, 사용자가 요청한 조건(최대 10개 필터 컬럼)을 이용해 조회하고 결과를 브리핑하는 역할을 합니다.  
테이블은 약 300만 행/200 컬럼 규모이며, 기본적으로 **필터 조건 + 제한된 행 수(기본 5)**로만 조회합니다.

## 사전 준비 (설정 파일)
1) `db_connections.json`  
- 연결 정보 저장 파일입니다. `/api/db/connect`로 등록하거나 직접 입력합니다.
- 스키마는 앱 시작 시 로딩되어 `schema` 필드에 저장됩니다.

2) `db_view_profile.json`  
- **조회 대상 view 테이블**과 **WHERE에 사용할 컬럼 목록(최대 10개)**을 정의합니다.  
- 예시 키:
  - `connection_id`, `schema`, `table`, `limit`
  - `filter_columns` (name + description)
  - `result_columns` (결과 기본 반환 컬럼)
  - `selectable_columns` (선택 가능한 반환 컬럼)
  - `column_catalog_table` (컬럼 카탈로그에서 사용할 테이블 키)
  - `time_column` (최근 N개월/트렌드에 사용할 시간 컬럼)

3) `db_view_columns.json`  
- 사용자 요청을 컬럼명으로 매핑하기 위한 **컬럼 카탈로그**입니다.
- 각 컬럼에 `name`, `description`, `aliases`를 정의합니다.

## 전체 흐름 (요약)
1) 사용자 입력 → `/api/chat`  
2) 오케스트레이터 → `db_agent` (as_tool)  
3) `db_agent` → `resolve_view_columns` (optional)  
4) `db_agent` → `query_view_table(filters, columns, limit)` 또는 `query_view_metrics(...)`  
5) `query_view_table`/`query_view_metrics` → `execute_table_query_multi`/`execute_table_query_aggregate`  
6) 결과 → 이벤트 스트림 `db_result`/`defect_rate_chart` → UI 카드 표시  
7) 오케스트레이터가 한국어로 결과 브리핑/추가 질문

## 내부 동작 상세
### 1) 에이전트 라우팅
- 오케스트레이터가 “DB 조회 의도”로 판단하면 `db_agent`를 호출합니다.
- `db_agent`는 `db_view_profile.json` 내용을 프롬프트에 포함하여:
  - 어떤 테이블을 조회할지
  - 어떤 컬럼만 필터로 쓸 수 있는지
  - 결과로 반환할 기본/선택 컬럼
  - 기본 제한이 몇 개인지를 알고 있습니다.

### 2) 필터/컬럼 결정
- `db_agent`는 사용자 요청에서 조건을 추출해 **필터 배열**을 구성합니다.  
  예: `[{column: "lot_id", operator: "=", value: "LOT-1234"}]`
- 허용된 필터 컬럼(`filter_columns`) 외의 컬럼은 사용하지 않습니다.
- 사용자가 특정 수치를 요청하면 `resolve_view_columns`로 컬럼 후보를 찾고,
  `query_view_table(columns=[...])`로 반환 컬럼을 제한합니다.
- 평균/최소/최대/트렌드 요청은 `query_view_metrics`를 사용합니다.

### 3) 쿼리 실행 (`query_view_table`)
- 프로파일(뷰 테이블/허용 필터/기본 LIMIT)을 읽습니다.
- 필터 컬럼 검증:
  - 허용 목록에 없는 컬럼 → 에러 반환
  - 유효한 필터가 하나도 없으면 → “필터 필요” 메시지 반환
- 필터 연산자:
  - 기본 `ilike`(부분 문자열), 식별자에는 `=` 권장
- 결과 컬럼은 `result_columns` 또는 요청 `columns`로 제한합니다.
- 결과는 기본 LIMIT 5, 최대 50까지 제한합니다.

### 4) 집계/트렌드 실행 (`query_view_metrics`)
- `metrics`(avg/min/max/count 등), `recent_months`, `time_bucket`을 사용해 집계합니다.
- `time_column`이 필요합니다. (예: `design_input_date`)
- 트렌드 요청은 `defect_rate_chart` 이벤트로 라인 차트를 표시합니다.

### 5) 실제 SQL 구성 (`execute_table_query_multi`/`execute_table_query_aggregate`)
- `db_connections.json`의 스키마 정보를 사용하여 테이블/컬럼을 검증합니다.
- 모든 필터는 **AND 조건**으로 결합합니다.
- 반환 컬럼은 **요청 columns 또는 기본 result_columns** 기준입니다.

### 5) UI 이벤트와 브리핑
- 조회 결과는 `db_result` 이벤트로 이벤트 패널에 표시됩니다.
- 오케스트레이터는 결과 요약을 자연스러운 한국어로 브리핑합니다.

## 실패/예외 처리
1) 프로파일 미설정  
→ “DB view profile is not configured.”  
2) 허용 컬럼 없음  
→ “No allowed filter columns are configured.”  
3) 필터 누락  
→ “Provide filters using: …”  
4) 잘못된 컬럼  
→ “Invalid filter columns … Allowed: …”

## 커스터마이징 포인트
- `db_view_profile.json`의 `filter_columns`만 바꿔도 에이전트가 자동으로 갱신된 컬럼을 사용합니다.
- 기본 LIMIT은 `limit`에서 변경 가능합니다.
- 조회 컬럼을 줄이고 싶다면 `result_columns` 또는 `columns` 인자를 사용하세요.
- 컬럼 매핑 정확도를 높이려면 `db_view_columns.json`에 alias/description을 추가하세요.
- 트렌드용 시간 컬럼은 `time_column`으로 지정하세요.
