from typing import Any

from .schemas import InputParams


# 라벨 매핑 테이블/컬럼명을 정의한다.
_COLUMN_LABEL_TABLE = "column_label_map"
_COLUMN_KEY_FIELD = "column_key"
_COLUMN_LABEL_FIELD = "korean_label"


def fetch_column_label_map() -> dict[str, str]:
    # column_label_map 테이블에서 라벨 매핑을 조회한다.
    rows = _query_column_label_map()
    # 조회 결과를 매핑 dict로 변환한다.
    label_map: dict[str, str] = {}
    for row in rows:
        # 영문 컬럼명과 한글 라벨을 꺼낸다.
        column_key = row.get(_COLUMN_KEY_FIELD)
        column_label = row.get(_COLUMN_LABEL_FIELD)
        # 필수 값이 없으면 건너뛴다.
        if not column_key or not column_label:
            continue
        # 매핑 결과에 추가한다.
        label_map[column_key] = column_label
    return label_map


def _query_column_label_map() -> list[dict[str, Any]]:
    # 실제 DB 조회 로직을 구현한다.
    # 예시 SQL:
    # SELECT column_key, korean_label
    # FROM column_label_map
    # WHERE column_key IS NOT NULL;
    return []


def build_simulation_from_db(
    input_params: InputParams,
    configs: dict[str, Any],
    selections: dict[str, Any],
    user_prefs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    # 테이블 컨테이너를 준비한다.
    tables: dict[str, Any] = {}
    # 차트 컨테이너를 준비한다.
    charts: list[dict[str, Any]] = []
    # 단계 노트 컨테이너를 준비한다.
    stage_notes: dict[str, str] = {}
    # TODO: 아래 순서로 테이블/차트를 채운다.
    # 1-1: input_params_table
    # 1-2: chip_type_candidates_table
    # 1-3: reference_lot_candidates_table, reference_lot_table
    # 1-5: top_k_table
    # 1-6: recent_similar_table
    # 1-7: defect_rate_table, defect_rate_summary
    # 1-4/1-8: 필요 시 stage_notes만 채운다.
    return tables, charts, stage_notes
