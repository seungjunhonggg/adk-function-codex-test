from typing import Any

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
        "defect_metric": "불량률 지표",
        "defect_avg": "평균",
        "defect_min": "최소",
        "defect_max": "최대",
    }


def _get_db_label_mapping() -> dict[str, str]:
    # DB 연결 시 한글 라벨 매핑을 조회한다.
    return {}


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
    if input_params.chip_type:
        line1 = "근거: chip_type 입력으로 1-2 생략"
        line2 = f"입력: chip_type={input_params.chip_type}"
        line3 = "출력: 1-2 생략"
        notes["1-2"] = "\n".join([line1, line2, line3])
    else:
        chip_rows = tables.get("chip_type_candidates_table", [])
        chip_count = len(chip_rows) if isinstance(chip_rows, list) else 0
        chip_top = chip_rows[0] if chip_count else {}
        chip_id = chip_top.get("chip_type_id", "-")
        chip_match = chip_top.get("match_count", "-")
        line1 = (
            "근거: match_count 높은 순 정렬" if chip_count else "근거: 후보 데이터 없음"
        )
        line2 = f"입력: 후보 {chip_count}개"
        line3 = (
            f"출력: {chip_id} (match_count={chip_match})"
            if chip_count
            else "출력: -"
        )
        notes["1-2"] = "\n".join([line1, line2, line3])
    # 1-3 레퍼런스 LOT 근거를 만든다.
    ref_rows = tables.get("reference_lot_candidates_table", [])
    ref_count = len(ref_rows) if isinstance(ref_rows, list) else 0
    ref_selected = tables.get("reference_lot_table", [])
    ref_top = ref_selected[0] if isinstance(ref_selected, list) and ref_selected else {}
    ref_id = ref_top.get("lot_id", "-")
    ref_score = ref_top.get("defect_score", "-")
    line1 = "근거: defect_score 낮은 LOT 선정" if ref_top else "근거: 후보 데이터 없음"
    line2 = f"입력: 후보 {ref_count}개"
    line3 = f"출력: {ref_id} (defect_score={ref_score})" if ref_top else "출력: -"
    notes["1-3"] = "\n".join([line1, line2, line3])
    # 1-4 API payload 근거를 만든다.
    ref_lot_id = selections.get("reference_lot_id") or ref_id or "-"
    chip_type_value = selections.get("chip_type_id") or input_params.chip_type or "-"
    line1 = "근거: 입력값 + ref LOT로 payload 구성"
    line2 = f"입력: ref_lot={ref_lot_id}, chip_type={chip_type_value}"
    line3 = "출력: API payload 구성"
    notes["1-4"] = "\n".join([line1, line2, line3])
    # 1-5 top-k 근거를 만든다.
    top_k_rows = tables.get("top_k_table", [])
    top_k_count = len(top_k_rows) if isinstance(top_k_rows, list) else 0
    top_k_row = top_k_rows[0] if top_k_count else {}
    top_k_value = configs.get("top_k")
    top_k_sort = configs.get("top_k_sort", "rank")
    top_rank = top_k_row.get("rank", "-")
    top_capacity = top_k_row.get("predicted_capacity", "-")
    line1 = f"근거: {top_k_sort} 오름차순 정렬 + top_k={top_k_value}"
    line2 = f"입력: top_k={top_k_value}"
    line3 = (
        f"출력: rank1={top_rank}, predicted_capacity={top_capacity}"
        if top_k_count
        else "출력: -"
    )
    notes["1-5"] = "\n".join([line1, line2, line3])
    # 1-6 최근 유사 설계 근거를 만든다.
    recent_rows = tables.get("recent_similar_table", [])
    recent_count = len(recent_rows) if isinstance(recent_rows, list) else 0
    recent_top = recent_rows[0] if recent_count else {}
    date_start = recent_top.get("date_range_start", "-")
    date_end = recent_top.get("date_range_end", "-")
    rep_lot = recent_top.get("representative_lot_id", "-")
    match_count = recent_top.get("match_count", "-")
    core_params = configs.get("core_match_params", [])
    core_text = ", ".join(core_params) if core_params else "-"
    line1 = "근거: 최근 6개월 + 핵심 파라미터 매칭"
    line2 = f"입력: 기간={date_start}~{date_end}, core_params={core_text}"
    line3 = (
        f"출력: 대표 LOT={rep_lot}, match_count={match_count}"
        if recent_count
        else "출력: -"
    )
    notes["1-6"] = "\n".join([line1, line2, line3])
    # 1-7 불량률 집계 근거를 만든다.
    defect_rows = tables.get("defect_rate_table", [])
    metric_set = {
        row.get("defect_metric")
        for row in defect_rows
        if isinstance(row, dict) and row.get("defect_metric")
    }
    metric_count = len(metric_set)
    chart_type = user_prefs.get("chart_type", "bar")
    chart = next(
        (item for item in charts if item.get("chart_id") == "defect_rate_summary"),
        None,
    )
    series_name = "-"
    first_value = "-"
    if chart and chart.get("series"):
        series = chart["series"][0]
        series_name = series.get("name", "-")
        points = series.get("points", [])
        if points:
            first_value = points[0].get("y", "-")
    line1 = f"근거: metric {metric_count}개 집계 + chart_type={chart_type}"
    line2 = f"입력: chart_type={chart_type}"
    line3 = f"출력: {series_name} rank1={first_value}"
    notes["1-7"] = "\n".join([line1, line2, line3])
    # 1-8 브리핑 근거를 만든다.
    line1 = "근거: 표/차트 요약 기반 브리핑 생성"
    line2 = "입력: stage_outputs 표/차트"
    line3 = "출력: briefing_blocks"
    notes["1-8"] = "\n".join([line1, line2, line3])
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
        skip_chip_type = bool(input_params.chip_type)
        if not skip_chip_type:
            blocks.append(
                {"type": "table_ref", "table_key": "chip_type_candidates_table"}
            )
        blocks.append({"type": "table_ref", "table_key": "reference_lot_candidates_table"})
        blocks.append({"type": "table_ref", "table_key": "reference_lot_table"})
        blocks.append({"type": "table_ref", "table_key": "top_k_table"})
        blocks.append({"type": "table_ref", "table_key": "recent_similar_table"})
        blocks.append({"type": "table_ref", "table_key": "defect_rate_table"})
        blocks.append({"type": "chart_ref", "chart_id": "defect_rate_summary"})
        # 칩기종 후보 표를 만든다.
        if not skip_chip_type:
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
        # 레퍼런스 LOT 표(최종 선정 1개)를 만든다.
        tables["reference_lot_table"] = [
            {
                "lot_id": "LOT-CAND-001",
                "chip_type_id": "CT-001",
                "defect_score": 0.08,
                "defect_metrics_summary": "ci_def_rate 0.08%, fr_defect_rate 80ppm",
            }
        ]
        # top-k 표를 만든다.
        tables["top_k_table"] = [
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
        # 최근 6개월 유사 설계 표를 만든다.
        tables["recent_similar_table"] = [
            {
                "candidate_rank": 1,
                "match_count": 8,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-071",
            },
            {
                "candidate_rank": 2,
                "match_count": 6,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-088",
            },
            {
                "candidate_rank": 3,
                "match_count": 5,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-103",
            },
            {
                "candidate_rank": 4,
                "match_count": 4,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-120",
            },
            {
                "candidate_rank": 5,
                "match_count": 3,
                "date_range_start": "2025-07-01",
                "date_range_end": "2025-12-31",
                "representative_lot_id": "LOT-2025-134",
            },
        ]
        # 불량률 요약 표를 만든다(모든 metric 포함).
        metric_specs = [
            {"metric": "ci_def_rate", "base": 0.12, "step": 0.02, "delta": 0.02},
            {"metric": "fr_defect_rate", "base": 120, "step": 15, "delta": 10},
            {"metric": "gr_short_defect_rate", "base": 0.08, "step": 0.01, "delta": 0.01},
            {"metric": "tvi_defect_rate_f", "base": 0.06, "step": 0.01, "delta": 0.01},
            {"metric": "tr_short_defect_rate", "base": 0.05, "step": 0.01, "delta": 0.01},
            {"metric": "df_def_rate", "base": 0.09, "step": 0.01, "delta": 0.01},
            {"metric": "soul_defect_rate_f", "base": 0.04, "step": 0.01, "delta": 0.01},
            {"metric": "gm_defect_rate_f", "base": 0.03, "step": 0.01, "delta": 0.01},
            {"metric": "pi_def_rate", "base": 0.11, "step": 0.02, "delta": 0.02},
            {"metric": "mf_def_rate", "base": 0.07, "step": 0.01, "delta": 0.01},
            {"metric": "ttm_defect_rate_f", "base": 0.05, "step": 0.01, "delta": 0.01},
            {"metric": "sum_burn_ppm", "base": 90, "step": 12, "delta": 8},
            {"metric": "sum_8585_ppm", "base": 110, "step": 14, "delta": 9},
            {"metric": "fail_halt_ppm", "base": 70, "step": 10, "delta": 7},
        ]
        defect_rows = []
        for rank in range(1, 6):
            for spec in metric_specs:
                avg = spec["base"] + spec["step"] * (rank - 1)
                min_value = avg - spec["delta"]
                max_value = avg + spec["delta"]
                if min_value < 0:
                    min_value = 0
                defect_rows.append(
                    {
                        "candidate_rank": rank,
                        "defect_metric": spec["metric"],
                        "defect_avg": avg,
                        "defect_min": min_value,
                        "defect_max": max_value,
                    }
                )
        tables["defect_rate_table"] = defect_rows
        # 불량률 차트를 만든다.
        chart_type = user_prefs.get("chart_type", "bar")
        charts = [
            {
                "chart_id": "defect_rate_summary",
                "type": chart_type,
                "title": "불량률 비교",
                "x_label": "후보",
                "y_label": "불량률",
                "series": [
                    {
                        "name": "ci_def_rate",
                        "points": [
                            {"x": "rank_1", "y": 0.12},
                            {"x": "rank_2", "y": 0.18},
                            {"x": "rank_3", "y": 0.16},
                            {"x": "rank_4", "y": 0.20},
                            {"x": "rank_5", "y": 0.14},
                        ],
                    }
                ],
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
    return blocks, tables, charts, stage_notes
