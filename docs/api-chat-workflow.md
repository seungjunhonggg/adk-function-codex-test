# /api/chat Workflow

## 목표
- MLCC 시뮬레이션을 `root_agent` 1개로 단계형 실행한다.
- 단계 실행 순서는 `before_tool_callback` 계약으로 강제한다.
- 사용자 컨펌/수정 요청은 세션 state 기반으로 처리한다.

## 상태 구조
- `sim_step`: 현재 실행 단계 (`2~5`)
- `sim_phase`: 현재 페이즈 (`collect_input | run_stage | wait_confirm | done`)
- `sim_inputs`: 사용자 입력 (`temperature`, `voltage`, `size`, `capacity`, `chip_prod_id`)
- `sim_stage_outputs`: 단계별 결과 (`1-2`, `1-3`, `1-4`, `1-5`)
- `sim_stage_status`: 단계별 상태 (`done`)
- `sim_last_brief`: 마지막 브리핑 문장
- `sim_last_gap`: 마지막 부족 정보/실패 원인
- `sim_pending_next_step`: 컨펌 후 이동할 다음 단계
- `sim_allowed_tools`: 현재 호출 가능한 tool 목록

## 툴 목록
- `set_sim_input`
- `run_stage_2_find_chip`
- `run_stage_3_find_ref_lot`
- `run_stage_4_optimize`
- `run_stage_5_select_topk`
- `confirm_next_step`
- `revise_from_stage`
- `brief_current_state`

## 실행 규칙
1. `collect_input`에서는 입력/브리핑 툴만 허용한다.
2. `run_stage`에서는 현재 `sim_step`에 해당하는 실행 툴만 허용한다.
3. 각 단계 실행이 끝나면 `wait_confirm`으로 전환한다.
4. `wait_confirm`에서는 `confirm_next_step` 또는 `revise_from_stage`를 사용한다.
5. `revise_from_stage`가 호출되면 해당 단계 이후 결과를 모두 무효화한다.

## before_tool_callback 계약
- 호출 전 `sim_allowed_tools`를 계산해 state에 저장한다.
- 허용되지 않은 tool 호출은 차단하고 차단 사유를 반환한다.
- 단계 선행조건이 없으면 실행을 차단한다.
  - `run_stage_2_find_chip`: 최소 입력 필요
  - `run_stage_3_find_ref_lot`: `1-2` 결과 필요
  - `run_stage_4_optimize`: `1-3` 결과 필요

## Instruction state 템플릿
- `{sim_step?}`
- `{sim_phase?}`
- `{sim_inputs?}`
- `{sim_last_brief?}`
- `{sim_pending_next_step?}`
- `{sim_last_gap?}`
- `{sim_allowed_tools?}`
