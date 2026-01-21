from typing import Any, Callable

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
    dirty_stages: list[str] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    # 테이블 컨테이너를 준비한다.
    tables: dict[str, Any] = {}
    # 차트 컨테이너를 준비한다.
    charts: list[dict[str, Any]] = []
    # 단계 노트 컨테이너를 준비한다.
    stage_notes: dict[str, str] = {}
    # dirty 스테이지 집합을 준비한다.
    dirty_set = set(dirty_stages or [])

    def _should_run(stage: str) -> bool:
        # dirty 정보가 없으면 전체 실행한다.
        if not dirty_set:
            return True
        # dirty에 포함된 단계만 실행한다.
        return stage in dirty_set

    def _emit_progress(stage: str) -> None:
        # 프론트 로깅 콜백이 있으면 단계 진행을 알린다.
        if not progress_cb:
            return
        # 콜백은 SSE에서 stage별 progress 이벤트를 보내도록 구현한다.
        progress_cb(stage)

    # 1-1: input_params_table
    if _should_run("1-1"):
        # 1-1 진행 로그를 보낸다.
        _emit_progress("1-1")
        # TODO: 입력값을 테이블로 정규화한다.
        # tables["input_params_table"] = [...]
        # stage_notes["1-1"] = "..."

    # 1-2: chip_type_candidates_table
    if _should_run("1-2"):
        # 1-2 진행 로그를 보낸다.
        _emit_progress("1-2")
        # TODO: 칩기종 후보를 조회해 테이블을 만든다.
        # tables["chip_type_candidates_table"] = [...]
        # stage_notes["1-2"] = "..."

    # 1-3: reference_lot_candidates_table, reference_lot_table
    if _should_run("1-3"):
        # 1-3 진행 로그를 보낸다.
        _emit_progress("1-3")
        # TODO: 레퍼런스 LOT 후보와 선택 테이블을 만든다.
        # tables["reference_lot_candidates_table"] = [...]
        # tables["reference_lot_table"] = [...]
        # stage_notes["1-3"] = "..."

    # 1-4: payload 구성 근거(표는 reference_lot_table 사용)
    if _should_run("1-4"):
        # 1-4 진행 로그를 보낸다.
        _emit_progress("1-4")
        # TODO: stage_notes만 필요하면 여기서 작성한다.
        # stage_notes["1-4"] = "..."

    # 1-5: top_k_table
    if _should_run("1-5"):
        # 1-5 진행 로그를 보낸다.
        _emit_progress("1-5")
        # TODO: top-k 결과 테이블을 만든다.
        # tables["top_k_table"] = [...]
        # stage_notes["1-5"] = "..."

    # 1-6: recent_similar_table
    if _should_run("1-6"):
        # 1-6 진행 로그를 보낸다.
        _emit_progress("1-6")
        # TODO: 최근 유사 설계 테이블을 만든다.
        # tables["recent_similar_table"] = [...]
        # stage_notes["1-6"] = "..."

    # 1-7: defect_rate_table, defect_rate_summary
    if _should_run("1-7"):
        # 1-7 진행 로그를 보낸다.
        _emit_progress("1-7")
        # TODO: 불량률 테이블과 차트를 만든다.
        # tables["defect_rate_table"] = [...]
        # charts.append({...})
        # stage_notes["1-7"] = "..."

    # 1-8: 브리핑 근거 노트
    if _should_run("1-8"):
        # 1-8 진행 로그를 보낸다.
        _emit_progress("1-8")
        # TODO: 브리핑 근거 노트를 만든다.
        # stage_notes["1-8"] = "..."
    return tables, charts, stage_notes
