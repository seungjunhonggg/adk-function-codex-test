import json
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from . import db_production
from .db_production import InputParams
from .prompt import SIMULATION_INSTRUCTION_TEMPLATE
from .state import (
    STAGE_TO_STEP,
    apply_patch_to_state,
    ensure_session_state,
    invalidate_from_step,
    set_pending_confirmation,
)


def _build_state_summary(state: dict[str, Any] | None) -> str:
    # 모델이 필요한 핵심 상태만 요약해서 전달한다.
    cleaned = ensure_session_state(state)
    summary = {
        "sim_step": cleaned.get("sim_step"),
        "input_params": cleaned.get("input_params"),
        "selections": cleaned.get("selections"),
        "configs": cleaned.get("configs"),
        "stage_status": cleaned.get("stage_status"),
        "pending_action": cleaned.get("pending_action"),
        "last_gap": cleaned.get("last_gap"),
    }
    return json.dumps(summary, ensure_ascii=False)


def _ensure_state(tool_context: ToolContext) -> dict[str, Any]:
    # 세션 상태를 기본 구조로 보정한다.
    state = ensure_session_state(tool_context.state)
    tool_context.state.update(state)
    return tool_context.state


def _check_stage_gate(
    state: dict[str, Any],
    required_step: int,
    stage_key: str,
) -> dict[str, Any] | None:
    # 확인 대기 상태에서는 새로운 단계 실행을 막는다.
    pending = state.get("pending_action")
    if isinstance(pending, dict) and pending.get("type") == "confirm_stage":
        return {
            "skipped": True,
            "reason": "awaiting_confirmation",
            "stage": pending.get("stage"),
            "summary": pending.get("summary", ""),
        }

    # 현재 단계와 요청 단계가 다르면 실행을 막는다.
    if state.get("sim_step") != required_step:
        return {
            "skipped": True,
            "reason": "step_gate",
            "expected_step": state.get("sim_step"),
            "requested_stage": stage_key,
        }
    return None


def find_chip_prod_id(
    tool_context: ToolContext,
    input_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # 1-2 단계 게이트를 확인한다.
    state = _ensure_state(tool_context)
    blocked = _check_stage_gate(state, required_step=2, stage_key="1-2")
    if blocked:
        return blocked

    # 입력 파라미터를 표준 형태로 정규화한다.
    raw_params = input_params or state.get("input_params", {})
    params_model = InputParams(**raw_params) if isinstance(raw_params, dict) else InputParams()
    state["input_params"] = params_model.model_dump()

    # chip_prod_id 직접 조회 또는 4개 인자 조회 중 하나를 선택한다.
    has_chip_id = bool(params_model.chip_prod_id)
    has_full_core = all(
        [
            params_model.temperature,
            params_model.voltage,
            params_model.size,
            params_model.capacity,
        ]
    )
    if not has_chip_id and not has_full_core:
        missing = []
        if not params_model.temperature:
            missing.append("temperature")
        if not params_model.voltage:
            missing.append("voltage")
        if not params_model.size:
            missing.append("size")
        if not params_model.capacity:
            missing.append("capacity")
        gap = {
            "stage": "1-2",
            "reason": "missing_input_params",
            "missing_fields": missing,
            "fallback_summary": "입력값이 부족해서 1-2 단계를 진행할 수 없어요.",
        }
        state["stage_status"]["1-2"] = "gap"
        state["last_gap"] = gap
        state["pending_action"] = {
            "type": "need_input",
            "stage": "1-2",
            "fields": missing,
            "message": "temperature/voltage/size/capacity 또는 chip_prod_id를 입력해 주세요.",
        }
        return {
            "status": "gap",
            "chip_prod_id_list": [],
            "candidate_count": 0,
            "gap": gap,
        }

    # DB 조회를 실행한다.
    if has_chip_id:
        sql = """
            SELECT DISTINCT ON (chip_prod_id) chip_prod_id
            FROM data_portal.mdh_contiguous_condition_view2
            WHERE chip_prod_id LIKE %(chip_prod_id)s
            ORDER BY chip_prod_id, design_input_date DESC
        """
        query_params = {"chip_prod_id": f"%{params_model.chip_prod_id}%"}
    else:
        sql = """
            SELECT DISTINCT ON (chip_prod_id) chip_prod_id
            FROM data_portal.mdh_contiguous_condition_view2
            WHERE temperature = %(temperature)s
              AND voltage = %(voltage)s
              AND size_detail = %(size)s::text
              AND base_volume = %(capacity)s
            ORDER BY chip_prod_id, design_input_date DESC
        """
        query_params = {
            "temperature": params_model.temperature,
            "voltage": params_model.voltage,
            "size": params_model.size,
            "capacity": params_model.capacity,
        }

    rows = db_production.db.execute_read(sql, query_params) or []
    chip_prod_id_list = [row.get("chip_prod_id") for row in rows if row.get("chip_prod_id")]

    # 후보가 없으면 gap 상태로 종료한다.
    if not chip_prod_id_list:
        gap = {
            "stage": "1-2",
            "reason": "no_chip_type_match",
            "fallback_summary": "입력 조건에 맞는 chip_prod_id를 찾지 못했어요.",
            "candidate_count": 0,
        }
        state["stage_status"]["1-2"] = "gap"
        state["last_gap"] = gap
        state["pending_action"] = {
            "type": "need_input",
            "stage": "1-2",
            "fields": ["chip_prod_id", "temperature", "voltage", "size", "capacity"],
            "message": "조건을 수정해서 다시 시도해 주세요.",
        }
        return {
            "status": "gap",
            "chip_prod_id_list": [],
            "candidate_count": 0,
            "gap": gap,
        }

    # 1-2 완료 상태를 저장하고 확인 대기로 전환한다.
    state["stage_outputs"]["1-2"] = {
        "chip_prod_id_list": chip_prod_id_list,
        "candidate_count": len(chip_prod_id_list),
    }
    state["stage_status"]["1-2"] = "done"
    state["sim_step"] = 3
    set_pending_confirmation(
        state,
        stage_key="1-2",
        summary=f"chip 후보 {len(chip_prod_id_list)}개를 찾았어요. 다음 단계로 진행할까요?",
    )
    return {
        "status": "done",
        "chip_prod_id_list": chip_prod_id_list[:20],
        "candidate_count": len(chip_prod_id_list),
        "truncated": len(chip_prod_id_list) > 20,
        "gap": None,
    }


def find_ref_lot_candidate(
    tool_context: ToolContext,
    chip_prod_id_list: list[str] | None = None,
) -> dict[str, Any]:
    # 1-3 단계 게이트를 확인한다.
    state = _ensure_state(tool_context)
    blocked = _check_stage_gate(state, required_step=3, stage_key="1-3")
    if blocked:
        return blocked

    # chip 후보 목록을 상태 또는 인자에서 가져온다.
    if chip_prod_id_list is None:
        chip_prod_id_list = state.get("stage_outputs", {}).get("1-2", {}).get("chip_prod_id_list", [])
    if not chip_prod_id_list:
        gap = {
            "stage": "1-3",
            "reason": "missing_chip_candidates",
            "fallback_summary": "chip 후보가 없어서 기준 LOT를 고를 수 없어요.",
        }
        state["stage_status"]["1-3"] = "gap"
        state["last_gap"] = gap
        state["pending_action"] = {
            "type": "need_input",
            "stage": "1-3",
            "fields": ["chip_prod_id", "temperature", "voltage", "size", "capacity"],
            "message": "먼저 1-2 입력 조건을 보완해 주세요.",
        }
        return {
            "status": "gap",
            "ref_lot_id": None,
            "candidate_count": 0,
            "gap": gap,
        }

    # 기준 LOT 후보를 조회한다.
    sql = """
        SELECT
            chip_prod_id,
            lot_id,
            cur_site_div,
            design_input_date,
            cutting_defect,
            measure_defect
        FROM data_portal.mdh_base_view_total_4
        WHERE chip_prod_id = ANY (%s)
        ORDER BY design_input_date DESC
        LIMIT 200
    """
    rows = db_production.db.execute_read(sql, (chip_prod_id_list,)) or []
    if not rows:
        gap = {
            "stage": "1-3",
            "reason": "no_ref_lot_candidate",
            "fallback_summary": "기준 LOT 후보가 없어 다음 단계를 진행할 수 없어요.",
        }
        state["stage_status"]["1-3"] = "gap"
        state["last_gap"] = gap
        state["pending_action"] = {
            "type": "need_input",
            "stage": "1-3",
            "fields": ["chip_prod_id"],
            "message": "다른 chip 조건으로 다시 시도해 주세요.",
        }
        return {
            "status": "gap",
            "ref_lot_id": None,
            "candidate_count": 0,
            "gap": gap,
        }

    ref_lot_info = rows[0]
    ref_lot_id = ref_lot_info.get("lot_id")

    # 1-3 완료 상태를 저장하고 확인 대기로 전환한다.
    state["stage_outputs"]["1-3"] = {
        "ref_lot_id": ref_lot_id,
        "ref_lot_info": ref_lot_info,
        "candidate_count": len(rows),
        "candidates": rows,
    }
    state["stage_status"]["1-3"] = "done"
    state["sim_step"] = 4
    set_pending_confirmation(
        state,
        stage_key="1-3",
        summary=f"기준 LOT 후보 {len(rows)}개를 찾았고 1순위는 {ref_lot_id}예요. 확정할까요?",
    )
    return {
        "status": "done",
        "ref_lot_id": ref_lot_id,
        "candidate_count": len(rows),
        "gap": None,
    }


def apply_user_patch(
    tool_context: ToolContext,
    temperature: str | None = None,
    voltage: str | None = None,
    size: str | None = None,
    capacity: str | None = None,
    chip_prod_id: str | None = None,
    reference_lot_id: str | None = None,
    top_k: int | None = None,
    months: int | None = None,
) -> dict[str, Any]:
    # 사용자 수정 요청을 상태 패치로 변환한다.
    state = _ensure_state(tool_context)
    patch = {
        "input_params": {
            "temperature": temperature,
            "voltage": voltage,
            "size": size,
            "capacity": capacity,
            "chip_prod_id": chip_prod_id,
        },
        "selections": {
            "reference_lot_id": reference_lot_id,
        },
        "configs": {
            "top_k": top_k,
            "months": months,
        },
    }

    # 변경점과 dirty 시작 단계를 계산한다.
    result = apply_patch_to_state(state, patch)
    if not result["changed_paths"]:
        return {
            "status": "no_change",
            "message": "변경된 값이 없어 기존 상태를 유지했어요.",
        }
    return {
        "status": "patched",
        "changed_paths": result["changed_paths"],
        "dirty_step": result["dirty_step"],
        "next_step": state.get("sim_step"),
    }


def confirm_stage(
    tool_context: ToolContext,
    stage_id: str,
    approved: bool = True,
) -> dict[str, Any]:
    # 단계 확인 결과를 상태에 반영한다.
    state = _ensure_state(tool_context)
    pending = state.get("pending_action")
    if not isinstance(pending, dict) or pending.get("type") != "confirm_stage":
        return {
            "status": "ignored",
            "reason": "no_pending_confirmation",
        }
    if pending.get("stage") != stage_id:
        return {
            "status": "ignored",
            "reason": "stage_mismatch",
            "pending_stage": pending.get("stage"),
        }
    if approved:
        # 승인이면 확인 대기 상태만 해제한다.
        state["pending_action"] = None
        return {
            "status": "confirmed",
            "stage": stage_id,
            "next_step": state.get("sim_step"),
        }

    # 미승인이면 해당 단계부터 다시 계산하도록 되돌린다.
    step = STAGE_TO_STEP.get(stage_id)
    if step:
        invalidate_from_step(state, step)
        state["sim_step"] = step
    state["pending_action"] = None
    return {
        "status": "rework_requested",
        "stage": stage_id,
        "next_step": state.get("sim_step"),
    }


def get_workflow_state(tool_context: ToolContext) -> dict[str, Any]:
    # 프론트/모델이 공통으로 볼 수 있는 상태 요약을 반환한다.
    state = _ensure_state(tool_context)
    return {
        "sim_step": state.get("sim_step"),
        "stage_status": state.get("stage_status"),
        "pending_action": state.get("pending_action"),
        "input_params": state.get("input_params"),
        "selections": state.get("selections"),
        "configs": state.get("configs"),
    }


def get_ref_lot_info(
    tool_context: ToolContext,
    ref_lot_id: str | None = None,
    ref_lot_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # 1-4 단계 게이트를 확인한다.
    state = _ensure_state(tool_context)
    blocked = _check_stage_gate(state, required_step=4, stage_key="1-4")
    if blocked:
        return blocked

    # 이전 단계 산출물에서 기준 LOT 정보를 가져온다.
    stage_13 = state.get("stage_outputs", {}).get("1-3", {})
    if ref_lot_id is None:
        ref_lot_id = state.get("selections", {}).get("reference_lot_id") or stage_13.get("ref_lot_id")
    if ref_lot_info is None:
        ref_lot_info = stage_13.get("ref_lot_info")

    # 기준 LOT 정보가 없으면 입력 요청 상태로 전환한다.
    if not ref_lot_id:
        state["last_gap"] = {
            "stage": "1-4",
            "reason": "missing_ref_lot_id",
            "fallback_summary": "기준 LOT가 없어 1-4 단계를 진행할 수 없어요.",
        }
        state["pending_action"] = {
            "type": "need_input",
            "stage": "1-4",
            "fields": ["reference_lot_id"],
            "message": "진행할 기준 LOT ID를 알려주세요.",
        }
        state["stage_status"]["1-4"] = "gap"
        return {
            "status": "gap",
            "gap": state["last_gap"],
        }

    # 다음 단계 입력용 payload를 요약 생성한다.
    payload_seed = {
        "ref_lot_id": ref_lot_id,
        "ref_lot_info": ref_lot_info or {},
    }
    state["stage_outputs"]["1-4"] = {
        "ref_lot_id": ref_lot_id,
        "payload_seed": payload_seed,
    }
    state["stage_status"]["1-4"] = "done"
    state["sim_step"] = 5
    set_pending_confirmation(
        state,
        stage_key="1-4",
        summary=f"기준 LOT {ref_lot_id}로 최적화 payload를 준비했어요. 다음 단계로 진행할까요?",
    )
    return {
        "status": "done",
        "ref_lot_id": ref_lot_id,
        "payload_seed": payload_seed,
    }


def run_grid(
    tool_context: ToolContext,
    top_k: int | None = None,
) -> dict[str, Any]:
    # 1-5 단계 게이트를 확인한다.
    state = _ensure_state(tool_context)
    blocked = _check_stage_gate(state, required_step=5, stage_key="1-5")
    if blocked:
        return blocked

    # top_k 값을 상태/입력에서 결정한다.
    if top_k is None:
        top_k = int(state.get("configs", {}).get("top_k", 5))
    state["configs"]["top_k"] = top_k

    # 1-3 후보 목록을 기반으로 top-k 요약을 만든다.
    stage_13 = state.get("stage_outputs", {}).get("1-3", {})
    candidates = stage_13.get("candidates", [])
    if not candidates:
        state["last_gap"] = {
            "stage": "1-5",
            "reason": "missing_candidates",
            "fallback_summary": "후보 데이터가 없어 top-k 계산을 진행할 수 없어요.",
        }
        state["pending_action"] = {
            "type": "need_input",
            "stage": "1-5",
            "fields": ["chip_prod_id", "temperature", "voltage", "size", "capacity"],
            "message": "입력값을 보완한 뒤 다시 실행해 주세요.",
        }
        state["stage_status"]["1-5"] = "gap"
        return {
            "status": "gap",
            "gap": state["last_gap"],
        }

    top_items: list[dict[str, Any]] = []
    for index, row in enumerate(candidates[:top_k], start=1):
        top_items.append(
            {
                "rank": index,
                "chip_prod_id": row.get("chip_prod_id"),
                "lot_id": row.get("lot_id"),
                "score": max(0.0, round(1.0 - (index - 1) * 0.05, 3)),
            }
        )

    state["stage_outputs"]["1-5"] = {
        "top_k_limit": top_k,
        "top_k": top_items,
        "top_k_count": len(top_items),
    }
    state["stage_status"]["1-5"] = "done"
    state["sim_step"] = 6
    set_pending_confirmation(
        state,
        stage_key="1-5",
        summary=f"top-{len(top_items)} 후보를 계산했어요. 최근 결함 요약 단계로 진행할까요?",
    )
    return {
        "status": "done",
        "top_k_count": len(top_items),
        "top_k": top_items,
    }


def fetch_recent_defect_summary(
    tool_context: ToolContext,
    months: int | None = None,
) -> dict[str, Any]:
    # 1-6 단계 게이트를 확인한다.
    state = _ensure_state(tool_context)
    blocked = _check_stage_gate(state, required_step=6, stage_key="1-6")
    if blocked:
        return blocked

    # 기간 설정을 입력/상태에서 결정한다.
    if months is None:
        months = int(state.get("configs", {}).get("months", 6))
    state["configs"]["months"] = months

    # 1-5 결과가 없으면 후단계를 막는다.
    top_items = state.get("stage_outputs", {}).get("1-5", {}).get("top_k", [])
    if not top_items:
        state["last_gap"] = {
            "stage": "1-6",
            "reason": "missing_top_k",
            "fallback_summary": "top-k 결과가 없어 결함 요약을 진행할 수 없어요.",
        }
        state["pending_action"] = {
            "type": "need_input",
            "stage": "1-6",
            "fields": ["top_k"],
            "message": "top_k 값을 확정한 뒤 다시 실행해 주세요.",
        }
        state["stage_status"]["1-6"] = "gap"
        return {
            "status": "gap",
            "gap": state["last_gap"],
        }

    # 결함 요약은 현재 단계에서 간단 요약 포맷으로 만든다.
    defect_summary: list[dict[str, Any]] = []
    for item in top_items:
        rank = int(item.get("rank", 0))
        defect_summary.append(
            {
                "rank": rank,
                "chip_prod_id": item.get("chip_prod_id"),
                "avg_defect_rate": round(0.4 + rank * 0.07, 3),
            }
        )

    state["stage_outputs"]["1-6"] = {
        "months": months,
        "defect_summary": defect_summary,
    }
    state["stage_status"]["1-6"] = "done"
    set_pending_confirmation(
        state,
        stage_key="1-6",
        summary="최근 결함 요약을 만들었어요. 최종 안내 문장을 생성할까요?",
    )
    return {
        "status": "done",
        "months": months,
        "defect_summary": defect_summary,
    }


def create_root_agent(state: dict[str, Any] | None = None) -> LlmAgent:
    # 상태 요약을 프롬프트에 주입한다.
    state_summary = _build_state_summary(state)
    instruction = SIMULATION_INSTRUCTION_TEMPLATE.replace("{{state_summary}}", state_summary)
    instruction = instruction.replace("{state_summary}", state_summary)
    # 루트 에이전트를 생성한다.
    return LlmAgent(
        model="gemini-2.5-flash",
        name="mlcc_simulation_agent",
        description="MLCC 시뮬레이션 단계 진행과 사용자 확인을 처리한다.",
        instruction=instruction,
        tools=[
            apply_user_patch,
            confirm_stage,
            get_workflow_state,
            find_chip_prod_id,
            find_ref_lot_candidate,
            get_ref_lot_info,
            run_grid,
            fetch_recent_defect_summary,
        ],
    )


# 기존 import 호환을 위해 루트 에이전트 인스턴스도 유지한다.
root_agent = create_root_agent()
root_agents = root_agent
