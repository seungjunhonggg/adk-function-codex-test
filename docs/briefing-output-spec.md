# 브리핑 산출물 스펙 (v0)

## 진행 계획 (현재)
1. 브리핑 산출물 구조, 표, 차트 스키마 확정
2. 단계별 데이터 계약(1-1~1-8)과 컬럼 설정 방식 정의
3. 산출물 스펙에 맞춘 에이전트/툴/메모리/컨텍스트 전략 정의

## 목표
- 목표 길이: 2k~3k 토큰(LLM 서술 텍스트만).
- 모든 표 데이터는 결정적 처리(LLM은 표 계산 금지).
- 차트는 데이터만 생성하고 렌더링은 UI에서 처리.
- 불량률 차트 기본은 막대, 사용자 요청 시 선/스캐터로 변경.
- 브리핑에 노출되는 모든 컬럼 라벨은 한글 매핑 테이블을 사용.

## 출력 구조
### 0) 요약
- 핵심 결과 3개 + 추천 후보 1개.
- 아래 표에 있는 값만 인용.

### 1) 입력 요약 (1-1)
- 표: `input_params_table`
- 필드: temperature, voltage, size, capacity, dev_flag, powder_size.

### 2) 칩기종 후보 (1-2)
- 표: `chip_type_candidates_table`
- 필드: chip_type_id, chip_type_name, match_count, notes.

### 3) 레퍼런스 LOT 선정 (1-3)
- 표: `reference_lot_candidates_table` (후보 10개)
- 표: `reference_lot_table`
- 필드: lot_id, chip_type_id, defect_score, defect_metrics_summary.
- 서술: 선정 이유 2~3문장(불량률 최소 기준).

### 4) 최적화 탐색 요약 (1-4)
- 서술 전용: 탐색 범위, 탐색 방식, API 호출 시각.

### 5) Top-K 후보 (1-5)
- 표: `top_k_table`
- 필드:
  - rank
  - design_param_1..design_param_10 (8~10개, 설정 기반)
  - predicted_capacity (API 필드 `electrode_c_avg`)
- 주의: 표에는 설계값 + 예상용량만 노출.
- rank는 API 결과 `result['datas']['sim']`의 `rank` 값(낮을수록 우수).

### 6) 최근 6개월 유사 설계 (1-6)
- 표: `recent_similar_table`
- 필드: candidate_rank, match_count, date_range_start, date_range_end, representative_lot_id.

### 7) 불량률 요약 (1-7)
- 표: `defect_rate_table`
- 필드: candidate_rank, defect_metric, defect_avg, defect_min, defect_max.
- 차트: 불량률 비교(기본 막대).
- 불량률 지표 목록은 DB 설정으로 관리.
- children 지표는 기본 브리핑에서 숨기고 요청 시만 확장 표로 제공.

### 8) 결론 및 다음 질문
- 추천 후보 + 리스크 2개 + 후속 질문 2개.

## 차트 스키마 (데이터 전용)
UI는 아래 스키마를 받아 차트를 렌더링한다.

```json
{
  "chart_id": "defect_rate_summary",
  "type": "bar | line | scatter",
  "title": "Defect rate comparison",
  "x_label": "Candidate",
  "y_label": "Defect rate",
  "series": [
    {
      "name": "metric_name",
      "points": [
        { "x": "rank_1", "y": 0.12 },
        { "x": "rank_2", "y": 0.18 }
      ]
    }
  ],
  "notes": "Optional short note"
}
```

### 차트 타입 선택 규칙
- 기본: 불량률 비교는 막대.
- 사용자 요청이 추이/시간축이면 선.
- 두 수치간 상관관계 요청이면 스캐터.
- 사용자가 명시한 타입이 있으면 우선 적용.

## 컨텍스트 예산
- 최종 출력: 2k~3k 토큰.
- LLM 입력은 4k 토큰 이내 권장.
- raw 배열(top-k/불량률 상세)은 컨텍스트에 넣지 않음.
- 표 요약과 차트 데이터만 주입.

## 컬럼 설정 (선택)
컬럼 변경을 코드 수정 없이 반영하기 위해 설정 테이블을 사용한다.

### 옵션 A: DB 설정
테이블: `report_table_columns`
- table_key (예: "top_k_table")
- column_key (예: "design_param_1")
- label
- order_index
- visible
- source_field (DB/API 필드명)
- format_hint (예: "float_3", "percent")

### 옵션 B: JSON 설정
파일: `config/report_tables.json`
- 구조는 `report_table_columns`와 동일

### 컬럼 라벨 매핑
영문 컬럼명을 한글 라벨로 매핑한다.
- 테이블: `report_column_labels`
- 필드: source_field, korean_label
- 브리핑에는 `korean_label`만 사용

## 남은 확인 사항
- design_param_1..design_param_10 실제 컬럼명(DB 설정).
- 불량률 지표 목록(DB 설정).
