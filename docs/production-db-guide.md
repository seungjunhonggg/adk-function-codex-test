# 정식 DB 연동 가이드 (v0)

## 목적
- 데모 스펙과 DB 결과 형식을 맞춘다.
- 표/차트 키와 컬럼명을 통일한다.
- 브리핑/하이라이트/요약이 정상 동작하도록 기준을 제공한다.

## 적용 위치
- 라벨 매핑: `backend/core/db_production.py::fetch_column_label_map`
- 시뮬레이션 데이터: `backend/core/db_production.py::build_simulation_from_db`
- 적용 시점: `backend/core/demo.py`에서 label_map을 읽어 테이블/차트에 적용

## 1) 라벨 매핑 테이블
- 테이블명: `column_label_map`
- 권장 컬럼:
  - `column_key` (영문 컬럼명)
  - `korean_label` (한글 컬럼명)

### 구현 방법
`backend/core/db_production.py`의 `_query_column_label_map`에 DB 조회 코드를 작성한다.

예시 SQL:
```sql
SELECT column_key, korean_label
FROM column_label_map
WHERE column_key IS NOT NULL;
```

반환 형식(파이썬):
```py
[
  {"column_key": "rank", "korean_label": "순위"},
  {"column_key": "lot_id", "korean_label": "LOT ID"}
]
```

## 2) 시뮬레이션 결과 구조
`build_simulation_from_db(...)`에서 아래 키로 결과를 구성한다.

### 필수 tables 키
- `input_params_table`
- `chip_type_candidates_table`
- `reference_lot_candidates_table`
- `reference_lot_table`
- `top_k_table`
- `recent_similar_table`
- `defect_rate_table`

### 필수 charts
- `chart_id: "defect_rate_summary"`

### 각 테이블 최소 컬럼
- `input_params_table`: 항목/값 형태(예: item, value 또는 항목, 값)
- `chip_type_candidates_table`: `chip_type_id`, `chip_type_name`, `match_count`
- `reference_lot_candidates_table`: `lot_id`, `chip_type_id`, `defect_score`
- `reference_lot_table`: `lot_id`, `chip_type_id`, `defect_score`
- `top_k_table`: `rank`, `predicted_capacity`, `total_layer` (+ 설계 파라미터)
- `recent_similar_table`: `rank`, `match_count`, `date_range_start`, `date_range_end`, `representative_lot_id`
- `defect_rate_table`: `rank` + defect 지표 컬럼들

### 차트 points 포맷
```json
{
  "chart_id": "defect_rate_summary",
  "type": "bar",
  "series": [
    { "name": "rank 1", "points": [ { "x": "ci_def_rate", "y": 0.12 } ] }
  ]
}
```

## 3) 하이라이트 기준
다음 키가 있어야 자동 강조가 작동한다.
- `reference_lot_candidates_table`: `lot_id`
- `top_k_table`: `rank`
- `recent_similar_table`: `rank`
- `defect_rate_table`: `rank`

## 4) 연결 방법 (권장)
1. `db_production.py`에 실제 DB 연결/쿼리를 구현한다.
2. `build_simulation_from_db`가 위 표/차트 구조를 반환하도록 맞춘다.
3. `backend/api/chat.py`에서 `demo._build_simulation_stub` 대신 DB 함수를 호출하도록 연결한다.

## 5) 최소 변경 체크리스트
- 테이블 키 이름이 위 목록과 동일한지
- rank/lot_id 같은 하이라이트 키가 존재하는지
- `column_label_map`이 실제 DB에 있는지
