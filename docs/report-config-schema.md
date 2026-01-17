# 리포트 설정 스키마 (v0)

## 1) report_table_columns
브리핑 표 컬럼을 동적으로 관리한다.

필드:
- table_key (text)
- column_key (text)
- label (text)
- order_index (int)
- visible (bool)
- source_field (text)
- format_hint (text)

## 2) report_column_labels
영문 컬럼명을 한글 라벨로 변환한다.

필드:
- source_field (text)
- korean_label (text)

## 3) report_defect_groups
불량률 중분류 지표를 관리한다.

필드:
- group_key (text)
- metric (text)
- unit (text)  # percent, ppm 등
- order_index (int)
- visible (bool)

## 4) report_defect_children
중분류에 속하는 하위 지표(children)를 관리한다.

필드:
- group_key (text)
- child_metric (text)
- order_index (int)
- visible (bool)
