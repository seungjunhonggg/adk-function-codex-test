from typing import Any

from .db_production import fetch_column_label_map
from .schemas import InputParams
from .state import INPUT_LABEL_MAP, _format_missing_summary, _get_missing_fields


def _get_demo_label_mapping() -> dict[str, str]:
    # 데모용 한글 라벨 매핑을 준비한다.
    return {
        "chip_type_id": "칩기종 ID",
        "chip_type_name": "칩기종명",
        "match_count": "매칭수",
        "notes": "비고",
        "lot_id": "LOT ID",
        "defect_score": "불량률 점수",
        "defect_metrics_summary": "불량률 요약",
        "rank": "순위",
        "active_powder_base": "활성파우더베이스",
        "active_powder_additives": "활성파우더첨가제",
        "ldn_avr_value": "LDN 평균값",
        "cast_dsgn_thk": "캐스팅 설계 두께",
        "grinding_l_avg": "연마 L 평균",
        "grinding_w_avg": "연마 W 평균",
        "grinding_t_avg": "연마 T 평균",
        "total_layer": "총 레이어",
        "predicted_capacity": "예상 용량",
        "candidate_rank": "후보 순위",
        "date_range_start": "기간 시작",
        "date_range_end": "기간 종료",
        "representative_lot_id": "대표 LOT",
    }


def _get_db_label_mapping() -> dict[str, str]:
    # DB 연결 시 한글 라벨 매핑을 조회한다.
    return fetch_column_label_map()


def _get_label_mapping(demo: bool) -> dict[str, str]:
    # 데모 여부에 따라 한글 라벨 매핑을 결정한다.
    if demo:
        return _get_demo_label_mapping()
    return _get_db_label_mapping()


def _map_row_labels(row: dict[str, Any], label_map: dict[str, str]) -> dict[str, Any]:
    # 행의 컬럼명을 한글 라벨로 변환한다.
    return {label_map.get(key, key): value for key, value in row.items()}


def _map_table_labels(
    tables: dict[str, Any], label_map: dict[str, str]
) -> dict[str, Any]:
    # 모든 테이블 컬럼명을 한글 라벨로 변환한다.
    mapped: dict[str, Any] = {}
    for table_key, rows in tables.items():
        if isinstance(rows, list):
            mapped_rows = []
            for row in rows:
                if isinstance(row, dict):
                    mapped_rows.append(_map_row_labels(row, label_map))
                else:
                    mapped_rows.append(row)
            mapped[table_key] = mapped_rows
        else:
            mapped[table_key] = rows
    return mapped


def _map_chart_labels(
    charts: list[dict[str, Any]], label_map: dict[str, str]
) -> list[dict[str, Any]]:
    # 차트의 x 라벨을 한글로 변환한다.
    mapped: list[dict[str, Any]] = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        next_chart = dict(chart)
        series_list = next_chart.get("series", [])
        if isinstance(series_list, list):
            mapped_series: list[dict[str, Any]] = []
            for series in series_list:
                if not isinstance(series, dict):
                    continue
                next_series = dict(series)
                points = next_series.get("points", [])
                if isinstance(points, list):
                    mapped_points: list[dict[str, Any]] = []
                    for point in points:
                        if not isinstance(point, dict):
                            continue
                        next_point = dict(point)
                        raw_x = next_point.get("x")
                        if isinstance(raw_x, str):
                            next_point["x"] = label_map.get(raw_x, raw_x)
                        mapped_points.append(next_point)
                    next_series["points"] = mapped_points
                mapped_series.append(next_series)
            next_chart["series"] = mapped_series
        mapped.append(next_chart)
    return mapped


def _build_stage_notes(
    input_params: InputParams,
    tables: dict[str, Any],
    charts: list[dict[str, Any]],
    configs: dict[str, Any],
    selections: dict[str, Any],
    user_prefs: dict[str, Any],
) -> dict[str, str]:
    # 단계 근거 요약을 만든다.
    notes: dict[str, str] = {}
    # 1-1 입력 근거를 만든다.
    filled_keys = [key for key, value in input_params.dict().items() if value]
    filled_labels = [INPUT_LABEL_MAP.get(key, key) for key in filled_keys]
    missing_fields = _get_missing_fields(input_params)
    if missing_fields:
        line1 = "근거: 필수 입력이 누락됨"
        line3 = "출력: 입력 보완 필요"
    else:
        line1 = "근거: 입력값 확보 완료"
        line3 = "출력: 입력 확정"
    line2 = f"입력: {', '.join(filled_labels) if filled_labels else '-'}"
    notes["1-1"] = "\n".join([line1, line2, line3])
    # 1-2 칩기종 후보 근거를 만든다.
    chip_rows = tables.get("chip_type_candidates_table", [])
    chip_count = len(chip_rows) if isinstance(chip_rows, list) else 0
    notes["1-2"] = (
        "사용자가 준 인자값에 맞는 3개월 이내 하이러너 기종을 검색하였음. "
        f"총 {chip_count}개의 기종이 검색되었음."
    )
    # 1-3 레퍼런스 LOT 근거를 만든다.
    ref_selected = tables.get("reference_lot_table", [])
    ref_top = ref_selected[0] if isinstance(ref_selected, list) and ref_selected else {}
    ref_id = ref_top.get("lot_id", "-")
    notes["1-3"] = (
        "해당 기종들중 신뢰성 결과 및 불량률 검사 등급을 기준으로 상위 LOT들을 "
        "선별하였으며, 그중 불량률이 제일 낮은 LOT를 reference로 선정하였습니다."
    )
    # 1-4 API payload 근거를 만든다.
    ref_lot_id = selections.get("reference_lot_id") or ref_id or "-"
    notes["1-4"] = (
        f"선정된 ref_lot의 ID는 {ref_lot_id}이며, 설계값은 하기 표와 같습니다."
    )
    # 1-5 top-k 근거를 만든다.
    notes["1-5"] = (
        "REF LOT을 기준으로 용량을 5%수준 높이고, 액티브 층, Sheet T, "
        "Laydown을 +-5% 수준을 만족하는 설계값을 grid search 한 결과를 "
        "상위 sorting 하였습니다."
    )
    # 1-6 최근 유사 설계 근거를 만든다.
    notes["1-6"] = (
        "상위 추천설계를 모재/첨가제, S/T, L/D 동일 설계 조건으로 "
        "최근 6개월 이내로 검색한 결과는 하기와 같습니다."
    )
    # 1-7 불량률 집계 근거를 만든다.
    notes["1-7"] = (
        "각 추천설계별 LOT들의 6개월 평균 공정불량률은 하기와 같습니다. "
        "그중 공정불량률은 하기 차트에 나타내었습니다."
    )
    return notes


def _build_casual_stub() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # 캐주얼 응답 블록을 만든다.
    blocks = [
        {
            "type": "text",
            "section": "casual",
            "value": "안녕하세요. 시뮬레이션이 필요하면 '시뮬레이션'이라고 말해 주세요.",
        }
    ]
    return blocks, {}, []


def _build_simulation_stub(
    request: Any,
    input_params: InputParams,
    configs: dict[str, Any],
    selections: dict[str, Any],
    user_prefs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    # 시뮬레이션 더미 응답을 만든다.
    summary = "시뮬레이션을 시작합니다. 필요한 입력을 확인 중입니다."
    # 누락 입력을 먼저 확인한다.
    missing = _get_missing_fields(input_params)
    if request.demo and not missing:
        summary = "데모 모드로 시뮬레이션을 시작합니다. 필요한 입력을 확인 중입니다."
    # 누락된 입력을 확인한다.
    if missing:
        summary = _format_missing_summary(missing)
    blocks = [
        {"type": "text", "section": "summary", "value": summary},
        {"type": "table_ref", "table_key": "input_params_table"},
    ]
    # 입력 요약 표를 만든다.
    input_rows = [
        {"항목": INPUT_LABEL_MAP[key], "값": value or ""}
        for key, value in input_params.dict().items()
    ]
    tables = {"input_params_table": input_rows}
    charts: list[dict[str, Any]] = []
    # 데모 모드일 때만 간단한 표/차트를 채운다.
    # 데모 모드도 입력 완료 후에만 결과를 채운다.
    if request.demo and not missing:
        blocks.append({"type": "table_ref", "table_key": "chip_type_candidates_table"})
        blocks.append({"type": "table_ref", "table_key": "reference_lot_candidates_table"})
        blocks.append({"type": "table_ref", "table_key": "reference_lot_table"})
        blocks.append({"type": "table_ref", "table_key": "top_k_table"})
        blocks.append({"type": "table_ref", "table_key": "recent_similar_table"})
        blocks.append({"type": "table_ref", "table_key": "defect_rate_table"})
        blocks.append({"type": "chart_ref", "chart_id": "defect_rate_summary"})
        # 칩기종 후보 표를 만든다.
        tables["chip_type_candidates_table"] = [
            {"chip_type_id": "CT-001", "chip_type_name": "MLCC-A", "match_count": 12, "notes": "고온/고전압"},
            {"chip_type_id": "CT-002", "chip_type_name": "MLCC-B", "match_count": 9, "notes": "용량 우선"},
            {"chip_type_id": "CT-003", "chip_type_name": "MLCC-C", "match_count": 7, "notes": "소형화"},
            {"chip_type_id": "CT-004", "chip_type_name": "MLCC-D", "match_count": 5, "notes": "개발품"},
            {"chip_type_id": "CT-005", "chip_type_name": "MLCC-E", "match_count": 4, "notes": "표준형"},
        ]
        # 레퍼런스 LOT 후보 표(10개)를 만든다.
        tables["reference_lot_candidates_table"] = [
            {"lot_id": "LOT-CAND-001", "chip_type_id": "CT-001", "defect_score": 0.08, "defect_metrics_summary": "ci_def_rate 0.08%, fr_defect_rate 80ppm"},
            {"lot_id": "LOT-CAND-002", "chip_type_id": "CT-001", "defect_score": 0.10, "defect_metrics_summary": "ci_def_rate 0.10%, fr_defect_rate 95ppm"},
            {"lot_id": "LOT-CAND-003", "chip_type_id": "CT-001", "defect_score": 0.12, "defect_metrics_summary": "ci_def_rate 0.12%, fr_defect_rate 110ppm"},
            {"lot_id": "LOT-CAND-004", "chip_type_id": "CT-001", "defect_score": 0.14, "defect_metrics_summary": "ci_def_rate 0.14%, fr_defect_rate 125ppm"},
            {"lot_id": "LOT-CAND-005", "chip_type_id": "CT-001", "defect_score": 0.16, "defect_metrics_summary": "ci_def_rate 0.16%, fr_defect_rate 140ppm"},
            {"lot_id": "LOT-CAND-006", "chip_type_id": "CT-001", "defect_score": 0.18, "defect_metrics_summary": "ci_def_rate 0.18%, fr_defect_rate 155ppm"},
            {"lot_id": "LOT-CAND-007", "chip_type_id": "CT-001", "defect_score": 0.20, "defect_metrics_summary": "ci_def_rate 0.20%, fr_defect_rate 170ppm"},
            {"lot_id": "LOT-CAND-008", "chip_type_id": "CT-001", "defect_score": 0.22, "defect_metrics_summary": "ci_def_rate 0.22%, fr_defect_rate 185ppm"},
            {"lot_id": "LOT-CAND-009", "chip_type_id": "CT-001", "defect_score": 0.24, "defect_metrics_summary": "ci_def_rate 0.24%, fr_defect_rate 200ppm"},
            {"lot_id": "LOT-CAND-010", "chip_type_id": "CT-001", "defect_score": 0.26, "defect_metrics_summary": "ci_def_rate 0.26%, fr_defect_rate 215ppm"},
        ]
        # 선택된 칩기종을 후보 표에 반영한다.
        selected_chip_type_ids = selections.get("chip_type_ids") or []
        selected_chip_type_id = (
            selected_chip_type_ids[0] if selected_chip_type_ids else None
        )
        if selected_chip_type_id and len(selected_chip_type_ids) == 1:
            for row in tables["reference_lot_candidates_table"]:
                row["chip_type_id"] = selected_chip_type_id
        # 선택된 레퍼런스 LOT를 최종 표에 반영한다.
        selected_ref_id = selections.get("reference_lot_id")
        selected_ref_row = None
        if selected_ref_id:
            for row in tables["reference_lot_candidates_table"]:
                if row.get("lot_id") == selected_ref_id:
                    selected_ref_row = dict(row)
                    break
        if not selected_ref_row:
            default_row = (
                tables["reference_lot_candidates_table"][0]
                if tables["reference_lot_candidates_table"]
                else {}
            )
            selected_ref_row = dict(default_row) if default_row else {}
            if selected_ref_row and selected_ref_id:
                selected_ref_row["lot_id"] = selected_ref_id
            if selected_ref_row and selected_chip_type_id and len(selected_chip_type_ids) == 1:
                selected_ref_row["chip_type_id"] = selected_chip_type_id
        # 선택된 레퍼런스 LOT를 후보 표에서 강조 표시한다.
        if not selected_ref_id and selected_ref_row:
            selected_ref_id = selected_ref_row.get("lot_id")
        if selected_ref_id:
            for row in tables["reference_lot_candidates_table"]:
                if row.get("lot_id") == selected_ref_id:
                    row["__row_state"] = "selected"
        tables["reference_lot_table"] = [selected_ref_row] if selected_ref_row else []
        # top-k 값을 적용해 표를 만든다.
        top_k_value = configs.get("top_k") or 5
        base_top_k_rows = [
            {
                "rank": 1,
                "active_powder_base": "A",
                "active_powder_additives": "X1",
                "ldn_avr_value": 1.1,
                "cast_dsgn_thk": 2.2,
                "grinding_l_avg": 0.31,
                "grinding_w_avg": 0.29,
                "grinding_t_avg": 0.28,
                "total_layer": 320,
                "predicted_capacity": 10.5,
            },
            {
                "rank": 2,
                "active_powder_base": "B",
                "active_powder_additives": "X2",
                "ldn_avr_value": 1.0,
                "cast_dsgn_thk": 2.0,
                "grinding_l_avg": 0.30,
                "grinding_w_avg": 0.28,
                "grinding_t_avg": 0.27,
                "total_layer": 310,
                "predicted_capacity": 10.1,
            },
            {
                "rank": 3,
                "active_powder_base": "C",
                "active_powder_additives": "X3",
                "ldn_avr_value": 1.05,
                "cast_dsgn_thk": 2.1,
                "grinding_l_avg": 0.29,
                "grinding_w_avg": 0.27,
                "grinding_t_avg": 0.26,
                "total_layer": 300,
                "predicted_capacity": 9.9,
            },
            {
                "rank": 4,
                "active_powder_base": "D",
                "active_powder_additives": "X4",
                "ldn_avr_value": 1.2,
                "cast_dsgn_thk": 2.3,
                "grinding_l_avg": 0.32,
                "grinding_w_avg": 0.30,
                "grinding_t_avg": 0.29,
                "total_layer": 330,
                "predicted_capacity": 10.8,
            },
            {
                "rank": 5,
                "active_powder_base": "E",
                "active_powder_additives": "X5",
                "ldn_avr_value": 0.98,
                "cast_dsgn_thk": 1.95,
                "grinding_l_avg": 0.28,
                "grinding_w_avg": 0.26,
                "grinding_t_avg": 0.25,
                "total_layer": 295,
                "predicted_capacity": 9.7,
            },
        ]
        top_k_rows = base_top_k_rows[:top_k_value]
        if top_k_value > len(base_top_k_rows):
            last_row = top_k_rows[-1] if top_k_rows else {}
            for _ in range(top_k_value - len(base_top_k_rows)):
                # 부족한 랭크 데이터를 뒤에 이어서 만든다.
                rank = len(top_k_rows) + 1
                prev = last_row or {
                    "ldn_avr_value": 1.0,
                    "cast_dsgn_thk": 2.0,
                    "grinding_l_avg": 0.30,
                    "grinding_w_avg": 0.28,
                    "grinding_t_avg": 0.27,
                    "total_layer": 300,
                    "predicted_capacity": 10.0,
                }
                next_row = {
                    "rank": rank,
                    "active_powder_base": chr(ord("A") + (rank - 1) % 26),
                    "active_powder_additives": f"X{rank}",
                    "ldn_avr_value": round(prev["ldn_avr_value"] - 0.02, 2),
                    "cast_dsgn_thk": round(prev["cast_dsgn_thk"] - 0.05, 2),
                    "grinding_l_avg": round(prev["grinding_l_avg"] - 0.01, 2),
                    "grinding_w_avg": round(prev["grinding_w_avg"] - 0.01, 2),
                    "grinding_t_avg": round(prev["grinding_t_avg"] - 0.01, 2),
                    "total_layer": max(prev["total_layer"] - 5, 1),
                    "predicted_capacity": round(prev["predicted_capacity"] - 0.2, 2),
                }
                top_k_rows.append(next_row)
                last_row = next_row
        # rank 1 행을 강조 표시한다.
        for row in top_k_rows:
            if row.get("rank") == 1:
                row["__row_state"] = "selected"
        tables["top_k_table"] = top_k_rows
        # 최근 6개월 유사 설계 표를 만든다.
        tables["recent_similar_table"] = [
            {
                "rank": 1,
                "match_count": 8,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-071",
            },
            {
                "rank": 2,
                "match_count": 6,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-088",
            },
            {
                "rank": 3,
                "match_count": 5,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-103",
            },
            {
                "rank": 4,
                "match_count": 4,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-120",
            },
            {
                "rank": 5,
                "match_count": 3,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-134",
            },
        ]
        # rank 1 행을 강조 표시한다.
        for row in tables["recent_similar_table"]:
            if row.get("rank") == 1:
                row["__row_state"] = "selected"
        # 불량률 요약 표를 만든다(모든 metric 포함).
        # ??? ?? rank ??? wide ??? ???.
        metric_specs = [
            {"metric": "ci_def_rate", "base": 0.12, "step": 0.02},
            {"metric": "fr_defect_rate", "base": 120, "step": 15},
            {"metric": "gr_short_defect_rate", "base": 0.08, "step": 0.01},
            {"metric": "tvi_defect_rate_f", "base": 0.06, "step": 0.01},
            {"metric": "tr_short_defect_rate", "base": 0.05, "step": 0.01},
            {"metric": "df_def_rate", "base": 0.09, "step": 0.01},
            {"metric": "soul_defect_rate_f", "base": 0.04, "step": 0.01},
            {"metric": "gm_defect_rate_f", "base": 0.03, "step": 0.01},
            {"metric": "pi_def_rate", "base": 0.11, "step": 0.02},
            {"metric": "mf_def_rate", "base": 0.07, "step": 0.01},
            {"metric": "ttm_defect_rate_f", "base": 0.05, "step": 0.01},
            {"metric": "sum_burn_ppm", "base": 90, "step": 12},
            {"metric": "sum_8585_ppm", "base": 110, "step": 14},
            {"metric": "fail_halt_ppm", "base": 70, "step": 10},
        ]
        defect_rows = []
        for rank in range(1, top_k_value + 1):
            row = {"rank": rank}
            for spec in metric_specs:
                row[spec["metric"]] = spec["base"] + spec["step"] * (rank - 1)
            defect_rows.append(row)
        tables["defect_rate_table"] = defect_rows
        # ????? ??? ???(??? 6? ???).
        chart_type = user_prefs.get("chart_type", "bar")
        chart_metrics = [
            "ci_def_rate",
            "tvi_defect_rate_f",
            "df_def_rate",
            "pi_def_rate",
            "mf_def_rate",
            "ttm_defect_rate_f",
        ]
        chart_series = []
        for row in defect_rows:
            rank = row.get("rank")
            points = []
            for metric in chart_metrics:
                points.append(
                    {"x": metric, "y": row.get(metric, 0), "rank": rank}
                )
            chart_series.append({"name": f"rank {rank}", "points": points})
        charts = [
            {
                "chart_id": "defect_rate_summary",
                "type": chart_type,
                "title": "공정불량률",
                "subtitle": f"rank 1~{top_k_value} 기준",
                "x_label": "불량종류",
                "y_label": "불량률",
                "unit": "%",
                "series": chart_series,
                "notes": "",
            }
        ]
    # 단계 근거 요약을 만든다.
    stage_notes = _build_stage_notes(
        input_params, tables, charts, configs, selections, user_prefs
    )
    # 한글 라벨 매핑을 적용한다.
    label_map = _get_label_mapping(request.demo)
    tables = _map_table_labels(tables, label_map)
    charts = _map_chart_labels(charts, label_map)
    return blocks, tables, charts, stage_notes
