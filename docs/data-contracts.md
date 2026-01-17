# 데이터 계약 (v0)

## 공통 규칙
- 내부 필드명은 영문을 사용한다.
- 브리핑 출력은 한글 라벨만 사용한다(매핑 테이블 필수).
- 표 데이터는 결정적 처리, LLM은 서술만 담당한다.
- raw 배열(top-k/불량률 상세)은 저장만 하고 컨텍스트에 넣지 않는다.

## 설정 테이블 (DB 기반)
### report_table_columns
- table_key, column_key, label, order_index, visible, source_field, format_hint
- 브리핑 표 컬럼을 동적으로 관리한다.

### report_column_labels
- source_field, korean_label
- 모든 영문 컬럼명을 한글 라벨로 변환한다.

### report_defect_groups
- group_key, metric, unit, order_index, visible
- 불량률 중분류(지표) 목록과 표시 규칙을 관리한다.

### report_defect_children
- group_key, child_metric, order_index, visible
- 중분류에 속한 하위 지표(children) 목록을 관리한다.

## Step 0: 라우팅
입력:
- user_message
출력:
- route: casual | simulation
- chip_type이 입력되면 1-2를 생략하고 1-3으로 진행.

## Step 1-1: 입력 수집
입력(사용자):
- temperature, voltage, size, capacity, dev_flag, powder_size
- 선택: chip_type_name 또는 chip_type_id
출력:
- input_params (정규화)
- has_chip_type 플래그

## Step 1-2: 칩기종 조회 (has_chip_type이면 생략)
소스:
- 칩기종 테이블(이름 미정)
입력:
- input_params
출력:
- chip_type_candidates: {chip_type_id, chip_type_name, match_score} 리스트

## Step 1-3: 레퍼런스 LOT 선정
소스:
- 칩 LOT 마스터: mdh_base_view_total_4
입력:
- chip_type_id 리스트
규칙:
- 핵심 설계 인자 NotNull + 불량률 최소 조건(DB 규칙 관리)
출력:
- reference_lot_candidates: {lot_id, chip_type_id, defect_score, defect_metrics_summary} 10개
- reference_lot: {lot_id, chip_type_id, defect_score, defect_metrics_summary}

## Step 1-4: 최적화 API 호출
소스:
- 외부 최적화 API
입력:
- reference_lot 설계값
출력:
- api_payload
- api_response (raw)

### payload 구조 요약
- data.ref와 data.sim은 동일 컬럼 리스트를 사용한다.
- 값은 ref_row와 sim_params(+grid_overrides) 기반으로 채운다.
- targets/params는 코드 로직에 따라 계산한다.
- api_response.result['datas']에 ref/sim이 함께 온다.
  - ref: 단일 객체
  - sim: rank 포함 리스트
- 응답에서 사용하는 값은 datas.ref, datas.sim만이다(그 외 필드는 무시).

### payload 컬럼 표준화 규칙
- 중복 컬럼은 제거한다.
- `active_powder_addtives` → `active_powder_additives`로 표준화한다.
- `active_powder_additives`가 정식 필드명이다.
- payload에는 표준화된 컬럼만 전송한다.

### payload 컬럼(ref/sim 공통, 표준화)

```json
{
  "ref": [
    "app_type",
    "chip_prod_id",
    "electrode_c_avg",
    "volume_percentage",
    "capacity_predict",
    "target_capa_normal_dist",
    "target_capa_cv",
    "grinding_l_avg",
    "grinding_w_avg",
    "grinding_t_avg",
    "dc_bias_lfhv",
    "dc_bias_hfhv",
    "active_layer",
    "total_layer",
    "dielectric_constant",
    "dielectric_constanc_calc",
    "cast_dsgn_thk",
    "ldn_avr_value",
    "chip_type",
    "screen_chip_size_leng",
    "screen_mrgn_leng",
    "screen_chip_size_widh",
    "screen_mrgn_widh",
    "cover_sheet_thk",
    "total_cover_layer_num",
    "top_cover_layer_num",
    "bot_cover_layer_num",
    "gap_sheet_thk",
    "dc_lfhv_spec_step_voltage",
    "dc_lfhv_spec_capa_change",
    "dc_hfhv_spec_step_voltage",
    "dc_hfhv_spec_step_capa_voltage",
    "dc_3sigma",
    "firing_prfl_name",
    "active_binder",
    "cover_binder",
    "gap_sheet_binder",
    "warranty_time",
    "aging_rate",
    "capa_cv",
    "upper_equal_dirtn_value",
    "lower_equal_dirtn_value",
    "equal_dirtn_value",
    "active_part_bfr_layer_num",
    "cent_brf_layer_num",
    "cent_upper_equal_dirtn_value",
    "cent_lower_equal_dirtn_value",
    "bfr_layer_num",
    "bfr_sheet_thk",
    "screen_durable_spec_name",
    "screen_ovrl_area",
    "optical_l",
    "optical_l_margin",
    "optical_w",
    "optical_w_margin",
    "optical_thk",
    "optical_cover_thk",
    "optical_overlap",
    "electrode_thk",
    "electrode_shrinkage",
    "total_thk",
    "shrinkage_l",
    "shrinkage_l_margin",
    "shrinkage_w",
    "shrinkage_w_margin",
    "shrinkage_cover",
    "optical_connectivity",
    "dielectric_thk",
    "dielectric_shrinkage",
    "firing_overlap",
    "cur_site_div",
    "lot_id",
    "pressure_avg",
    "final_l_avg",
    "final_w_avg",
    "final_t_avg",
    "dc_dielectric_shrinkage",
    "dc_lfhv_flag",
    "dc_hflv_flag",
    "active_powder_base",
    "active_powder_additives",
    "ni_paste_powder_xrf",
    "cutting_t_avg",
    "fire_acti_t",
    "fire_paste_t",
    "pre_dielectric_thk",
    "connectivity",
    "shrinkage_t",
    "cover_batch_binder_amount_used",
    "gap_sheet_batch_binder_amount_user",
    "firing_paste_corr_coeff",
    "active_dispersant",
    "active_batch_powder_amount_used",
    "cutting_l_avg",
    "active_pasting_w",
    "calc_paste",
    "prntg_inr_paste_lot_id",
    "firing_print_smear",
    "cutting_l_corr_coeff",
    "ni_paste_metal_xrf",
    "cutting_w_avg",
    "gap_sheet_batch_powder_amount_used",
    "gapsheet_dispersant",
    "gap_sheet_casting_w",
    "firing_t_corr_coeff",
    "cover_casting_w",
    "pre_elctrode_thk",
    "cover_dispersant",
    "firing_w_corr_coeff",
    "cover_batch_powder_amount_used"
  ],
  "sim": [
    "app_type",
    "chip_prod_id",
    "electrode_c_avg",
    "volume_percentage",
    "capacity_predict",
    "target_capa_normal_dist",
    "target_capa_cv",
    "grinding_l_avg",
    "grinding_w_avg",
    "grinding_t_avg",
    "dc_bias_lfhv",
    "dc_bias_hfhv",
    "active_layer",
    "total_layer",
    "dielectric_constant",
    "dielectric_constanc_calc",
    "cast_dsgn_thk",
    "ldn_avr_value",
    "chip_type",
    "screen_chip_size_leng",
    "screen_mrgn_leng",
    "screen_chip_size_widh",
    "screen_mrgn_widh",
    "cover_sheet_thk",
    "total_cover_layer_num",
    "top_cover_layer_num",
    "bot_cover_layer_num",
    "gap_sheet_thk",
    "dc_lfhv_spec_step_voltage",
    "dc_lfhv_spec_capa_change",
    "dc_hfhv_spec_step_voltage",
    "dc_hfhv_spec_step_capa_voltage",
    "dc_3sigma",
    "firing_prfl_name",
    "active_binder",
    "cover_binder",
    "gap_sheet_binder",
    "warranty_time",
    "aging_rate",
    "capa_cv",
    "upper_equal_dirtn_value",
    "lower_equal_dirtn_value",
    "equal_dirtn_value",
    "active_part_bfr_layer_num",
    "cent_brf_layer_num",
    "cent_upper_equal_dirtn_value",
    "cent_lower_equal_dirtn_value",
    "bfr_layer_num",
    "bfr_sheet_thk",
    "screen_durable_spec_name",
    "screen_ovrl_area",
    "optical_l",
    "optical_l_margin",
    "optical_w",
    "optical_w_margin",
    "optical_thk",
    "optical_cover_thk",
    "optical_overlap",
    "electrode_thk",
    "electrode_shrinkage",
    "total_thk",
    "shrinkage_l",
    "shrinkage_l_margin",
    "shrinkage_w",
    "shrinkage_w_margin",
    "shrinkage_cover",
    "optical_connectivity",
    "dielectric_thk",
    "dielectric_shrinkage",
    "firing_overlap",
    "cur_site_div",
    "lot_id",
    "pressure_avg",
    "final_l_avg",
    "final_w_avg",
    "final_t_avg",
    "dc_dielectric_shrinkage",
    "dc_lfhv_flag",
    "dc_hflv_flag",
    "active_powder_base",
    "active_powder_additives",
    "ni_paste_powder_xrf",
    "cutting_t_avg",
    "fire_acti_t",
    "fire_paste_t",
    "pre_dielectric_thk",
    "connectivity",
    "shrinkage_t",
    "cover_batch_binder_amount_used",
    "gap_sheet_batch_binder_amount_user",
    "firing_paste_corr_coeff",
    "active_dispersant",
    "active_batch_powder_amount_used",
    "cutting_l_avg",
    "active_pasting_w",
    "calc_paste",
    "prntg_inr_paste_lot_id",
    "firing_print_smear",
    "cutting_l_corr_coeff",
    "ni_paste_metal_xrf",
    "cutting_w_avg",
    "gap_sheet_batch_powder_amount_used",
    "gapsheet_dispersant",
    "gap_sheet_casting_w",
    "firing_t_corr_coeff",
    "cover_casting_w",
    "pre_elctrode_thk",
    "cover_dispersant",
    "firing_w_corr_coeff",
    "cover_batch_powder_amount_used"
  ]
}
```

## Step 1-5: Top-K 선정
소스:
- api_response.result['datas']['sim']
규칙:
- `rank` 값이 낮을수록 우수.
- rank 기준으로 top-k 추출(k는 설정값).
출력:
- top_k_raw (저장만, 컨텍스트 제외)
- top_k_table (rank + 설계 파라미터 + predicted_capacity)
비고:
- predicted_capacity = API 필드 `electrode_c_avg`.
- 설계 파라미터 8~10개는 `report_table_columns`로 정의.

## Step 1-6: 최근 6개월 유사 설계
소스:
- 칩 LOT 마스터: mdh_base_view_total_4
입력:
 - top-k 설계 파라미터(핵심 4개)
   - active_powder_base
   - active_powder_additives
   - ldn_avr_value
   - cast_dsgn_thk
 - design_input_date 최근 6개월 조건
출력:
- recent_similar_table:
  {candidate_rank, match_count, date_range_start, date_range_end, representative_lot_id}

## Step 1-7: 불량률 집계 (최근 6개월)
소스:
- 칩 LOT 마스터: mdh_base_view_total_4
입력:
 - top-k 설계 파라미터(핵심 4개)
   - active_powder_base
   - active_powder_additives
   - ldn_avr_value
   - cast_dsgn_thk
 - design_input_date 최근 6개월 조건
출력:
- defect_rate_table:
  {candidate_rank, defect_metric, defect_avg, defect_min, defect_max}
- defect_chart_data (브리핑 스펙의 차트 스키마)
비고:
- 지표 목록은 `report_defect_groups`, `report_defect_children`에서 관리.
- 브리핑 기본 표에는 metric만 노출하고 children은 요청 시 확장한다.

## 불량률 지표 시드 (초안)
아래 JSON을 DB 초기값으로 등록한다.

```json
[
  {
    "key": "cutting_chip_defect",
    "metric": "ci_def_rate",
    "children": ["ci_def_01", "ci_def_02", "ci_def_03", "ci_def_04", "ci_def_05", "ci_def_99"],
    "unit": "percent"
  },
  {
    "key": "firing_chip_defect",
    "metric": "fr_defect_rate",
    "children": ["fr_def_01", "fr_def_02", "fr_def_03", "fr_def_04", "fr_def_04_1", "fr_def_05", "fr_def_06", "fr_def_07", "fr_def_08", "fr_def_09", "fr_def_10", "fr_def_11", "fr_def_99"],
    "unit": "ppm"
  },
  {
    "key": "grinding_chip_defect",
    "metric": "gr_short_defect_rate",
    "children": [],
    "unit": "percent"
  },
  {
    "key": "grinding_visual_defect",
    "metric": "tvi_defect_rate_f",
    "children": [],
    "unit": "percent"
  },
  {
    "key": "electrode_chip_defect",
    "metric": "tr_short_defect_rate",
    "children": [],
    "unit": "percent"
  },
  {
    "key": "measurement_defect",
    "metric": "df_def_rate",
    "children": ["dr_def_01", "dr_def_02", "dr_def_03"],
    "unit": "percent"
  },
  {
    "key": "measurement_ultrasound_defect",
    "metric": "soul_defect_rate_f",
    "children": [],
    "unit": "percent"
  },
  {
    "key": "measurement_good_mold_defect",
    "metric": "gm_defect_rate_f",
    "children": [],
    "unit": "percent"
  },
  {
    "key": "appearance_defect",
    "metric": "pi_def_rate",
    "children": ["pi_def_01", "pi_def_02", "pi_def_03", "pi_def_04", "pi_def_99"],
    "unit": "percent"
  },
  {
    "key": "mf_defect",
    "metric": "mf_def_rate",
    "children": [],
    "unit": "percent"
  },
  {
    "key": "electrode_grinding_visual_defect",
    "metric": "ttm_defect_rate_f",
    "children": [],
    "unit": "percent"
  },
  {
    "key": "burn_in",
    "metric": "sum_burn_ppm",
    "children": [],
    "unit": "ppm"
  },
  {
    "key": "reliability_8585",
    "metric": "sum_8585_ppm",
    "children": [],
    "unit": "ppm"
  },
  {
    "key": "halt",
    "metric": "fail_halt_ppm",
    "children": [],
    "unit": "ppm"
  }
]
```

## Step 1-8: 브리핑 렌더
입력:
- 표 + 차트 데이터 + 요약 메모
출력:
- 브리핑 텍스트(2k~3k 토큰)
규칙:
- 영문 컬럼명은 반드시 한글 라벨로 변환.
- 서술은 표/차트에 있는 값만 인용.

## 데모 모드 (DB 미연결)
- 데모 데이터로 표를 생성한다.
- 실제 스키마/format_hint 규칙과 일치해야 한다.
- 시드 고정 랜덤을 사용해 재현 가능하게 만든다.

## 남은 확인 사항
- 칩기종 테이블명 및 컬럼명.
- API 응답 예시.
