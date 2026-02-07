import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .db_production import find_chip_prod_id, find_ref_lot_candidate
from .schemas import InputParams

# 상태 키: 단계 실행 상태를 한 곳에서 관리한다.
PHASE_COLLECT_INPUT = "collect_input"
PHASE_RUN_STAGE = "run_stage"
PHASE_WAIT_CONFIRM = "wait_confirm"
PHASE_DONE = "done"

# 상태 키: 단계 번호를 한 곳에서 관리한다.
STEP_STAGE_2 = 2
STEP_STAGE_3 = 3
STEP_STAGE_4 = 4
STEP_STAGE_5 = 5
FINAL_STEP = STEP_STAGE_5


def _ensure_state_defaults(state: dict[str, Any]) -> None:
    # 상태 초기화: 필수 상태 키가 없으면 기본값을 채운다.
    if "sim_step" not in state:
        state["sim_step"] = STEP_STAGE_2
    # 상태 초기화: phase 기본값을 collect_input으로 둔다.
    if "sim_phase" not in state:
        state["sim_phase"] = PHASE_COLLECT_INPUT
    # 상태 초기화: 입력값 저장 구조를 만든다.
    if "sim_inputs" not in state:
        state["sim_inputs"] = {
            "temperature": None,
            "voltage": None,
            "size": None,
            "capacity": None,
            "chip_prod_id": None,
        }
    # 상태 초기화: 단계별 결과 저장 구조를 만든다.
    if "sim_stage_outputs" not in state:
        state["sim_stage_outputs"] = {}
    # 상태 초기화: 단계별 완료 상태를 저장한다.
    if "sim_stage_status" not in state:
        state["sim_stage_status"] = {}
    # 상태 초기화: 마지막 요약 문장을 저장한다.
    if "sim_last_brief" not in state:
        state["sim_last_brief"] = ""
    # 상태 초기화: 마지막 갭 정보를 저장한다.
    if "sim_last_gap" not in state:
        state["sim_last_gap"] = None
    # 상태 초기화: 다음 단계 컨펌 대기 정보를 저장한다.
    if "sim_pending_next_step" not in state:
        state["sim_pending_next_step"] = None
    # 상태 초기화: 허용된 tool 목록을 저장한다.
    if "sim_allowed_tools" not in state:
        state["sim_allowed_tools"] = []


def _invalidate_from_stage(state: dict[str, Any], from_step: int) -> None:
    # 단계 무효화: 지정 단계 이후 결과를 삭제한다.
    stage_outputs = state["sim_stage_outputs"]
    stage_status = state["sim_stage_status"]
    for step in range(from_step, FINAL_STEP + 1):
        stage_key = f"1-{step}"
        stage_outputs.pop(stage_key, None)
        stage_status.pop(stage_key, None)
    # 단계 무효화: 진행 단계를 되돌리고 phase를 run_stage로 맞춘다.
    state["sim_step"] = from_step
    state["sim_phase"] = PHASE_RUN_STAGE
    state["sim_pending_next_step"] = None
    state["sim_last_gap"] = None


def _has_minimum_input(sim_inputs: dict[str, Any]) -> bool:
    # 입력 검증: chip_prod_id가 있으면 입력 완료로 본다.
    if sim_inputs.get("chip_prod_id"):
        return True
    # 입력 검증: 4개 인자가 모두 있으면 입력 완료로 본다.
    required = ("temperature", "voltage", "size", "capacity")
    return all(sim_inputs.get(key) for key in required)


def _compute_allowed_tools(state: dict[str, Any]) -> list[str]:
    # 허용 툴 계산: 현재 phase에 맞는 기본 툴 목록을 만든다.
    phase = state["sim_phase"]
    step = int(state["sim_step"])
    if phase == PHASE_COLLECT_INPUT:
        return ["set_sim_input", "brief_current_state", "revise_from_stage"]
    if phase == PHASE_WAIT_CONFIRM:
        return [
            "confirm_next_step",
            "brief_current_state",
            "revise_from_stage",
            "set_sim_input",
        ]
    if phase == PHASE_DONE:
        return ["set_sim_input", "brief_current_state", "revise_from_stage"]
    if step == STEP_STAGE_2:
        return ["run_stage_2_find_chip", "brief_current_state"]
    if step == STEP_STAGE_3:
        return ["run_stage_3_find_ref_lot", "brief_current_state"]
    if step == STEP_STAGE_4:
        return ["run_stage_4_optimize", "brief_current_state"]
    if step == STEP_STAGE_5:
        return ["run_stage_5_select_topk", "brief_current_state"]
    return ["brief_current_state"]


def _parse_user_decision(text: str) -> str:
    # 결정 파싱: 텍스트를 소문자로 정규화한다.
    normalized = text.strip().lower()
    # 결정 파싱: 수정 의도를 가장 우선으로 판단한다.
    if re.search(r"(수정|변경|바꿔|다시|되돌)", normalized):
        return "REVISE"
    # 결정 파싱: 진행 의도를 두 번째로 판단한다.
    if re.search(r"(진행|계속|다음|확인|ok|yes|네|응|ㅇㅇ)", normalized):
        return "CONTINUE"
    # 결정 파싱: 그 외는 질문으로 처리한다.
    return "QUESTION"


def set_sim_input(
    tool_context: ToolContext,
    temperature: str | None = None,
    voltage: str | None = None,
    size: str | None = None,
    capacity: str | None = None,
    chip_prod_id: str | None = None,
) -> dict[str, Any]:
    # 입력 저장: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 입력 저장: 전달된 값만 덮어쓴다.
    updates = {
        "temperature": temperature,
        "voltage": voltage,
        "size": size,
        "capacity": capacity,
        "chip_prod_id": chip_prod_id,
    }
    for key, value in updates.items():
        if value is not None and value != "":
            state["sim_inputs"][key] = value
    # 입력 저장: 최소 입력이 채워지면 1-2 단계 실행 준비로 전환한다.
    if _has_minimum_input(state["sim_inputs"]):
        state["sim_phase"] = PHASE_RUN_STAGE
        state["sim_step"] = STEP_STAGE_2
        state["sim_last_gap"] = None
    else:
        # 입력 저장: 아직 입력이 부족하면 입력 수집 phase를 유지한다.
        state["sim_phase"] = PHASE_COLLECT_INPUT
    # 입력 저장: 마지막 요약 문장을 갱신한다.
    state["sim_last_brief"] = "입력값을 갱신했습니다."
    return {
        "ok": True,
        "sim_phase": state["sim_phase"],
        "sim_step": state["sim_step"],
        "sim_inputs": state["sim_inputs"],
    }


def run_stage_2_find_chip(tool_context: ToolContext) -> dict[str, Any]:
    # 1-2 단계: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 1-2 단계: 현재 입력을 InputParams로 변환한다.
    params = InputParams(**state["sim_inputs"])
    # 1-2 단계: 기존 DB 툴을 호출해 chip 후보를 조회한다.
    results, chip_prod_id_list, gap = find_chip_prod_id(params)
    # 1-2 단계: 결과를 상태에 저장한다.
    stage_key = "1-2"
    state["sim_stage_outputs"][stage_key] = {
        "candidate_count": len(chip_prod_id_list),
        "chip_prod_id_list": chip_prod_id_list,
        "preview": results[:5],
        "gap": gap,
    }
    state["sim_stage_status"][stage_key] = "done"
    # 1-2 단계: 다음 단계 컨펌 대기 상태로 전환한다.
    state["sim_phase"] = PHASE_WAIT_CONFIRM
    if chip_prod_id_list:
        state["sim_pending_next_step"] = STEP_STAGE_3
        state["sim_last_brief"] = f"1-2 단계 완료: 칩기종 후보 {len(chip_prod_id_list)}개를 찾았습니다."
        state["sim_last_gap"] = None
    else:
        state["sim_pending_next_step"] = STEP_STAGE_2
        state["sim_last_brief"] = "1-2 단계 완료: 조건에 맞는 칩기종 후보를 찾지 못했습니다."
        state["sim_last_gap"] = gap
    return {
        "ok": True,
        "stage": stage_key,
        "candidate_count": len(chip_prod_id_list),
        "next_step": state["sim_pending_next_step"],
        "gap": gap,
    }


def run_stage_3_find_ref_lot(tool_context: ToolContext) -> dict[str, Any]:
    # 1-3 단계: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 1-3 단계: 1-2 단계 결과에서 chip 목록을 가져온다.
    chip_prod_id_list = (
        state.get("sim_stage_outputs", {})
        .get("1-2", {})
        .get("chip_prod_id_list", [])
    )
    # 1-3 단계: 기존 DB 툴을 호출해 레퍼런스 LOT 후보를 조회한다.
    stage_result = find_ref_lot_candidate(
        tool_context=tool_context,
        chip_prod_id_list=chip_prod_id_list,
    )
    # 1-3 단계: 하위 툴이 게이트 응답(dict)을 반환하면 그대로 전달한다.
    if isinstance(stage_result, dict):
        state["sim_phase"] = PHASE_WAIT_CONFIRM
        state["sim_pending_next_step"] = STEP_STAGE_3
        state["sim_last_brief"] = "1-3 단계 실행이 게이트 조건으로 보류되었습니다."
        return {"ok": False, "stage": "1-3", "detail": stage_result}
    # 1-3 단계: 정상 응답(tuple)일 때만 값을 언패킹한다.
    lot_candidates, ref_lot_info, ref_lot_id = stage_result
    # 1-3 단계: 결과를 상태에 저장한다.
    stage_key = "1-3"
    state["sim_stage_outputs"][stage_key] = {
        "candidate_count": len(lot_candidates),
        "ref_lot_id": ref_lot_id,
        "ref_lot_info": ref_lot_info,
        "preview": lot_candidates[:5],
    }
    state["sim_stage_status"][stage_key] = "done"
    # 1-3 단계: 다음 단계 컨펌 대기 상태로 전환한다.
    state["sim_phase"] = PHASE_WAIT_CONFIRM
    if ref_lot_id:
        state["sim_pending_next_step"] = STEP_STAGE_4
        state["sim_last_brief"] = f"1-3 단계 완료: 레퍼런스 LOT를 {ref_lot_id}로 선정했습니다."
        state["sim_last_gap"] = None
    else:
        state["sim_pending_next_step"] = STEP_STAGE_3
        state["sim_last_brief"] = "1-3 단계 완료: 레퍼런스 LOT를 선정하지 못했습니다."
        state["sim_last_gap"] = {
            "stage": "1-3",
            "reason": "no_ref_lot_candidate",
        }
    return {
        "ok": True,
        "stage": stage_key,
        "ref_lot_id": ref_lot_id,
        "candidate_count": len(lot_candidates),
        "next_step": state["sim_pending_next_step"],
    }


def run_stage_4_optimize(tool_context: ToolContext) -> dict[str, Any]:
    # 1-4 단계: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 1-4 단계: 1-3 결과에서 ref_lot_id를 읽는다.
    ref_lot_id = state.get("sim_stage_outputs", {}).get("1-3", {}).get("ref_lot_id")
    # 1-4 단계: 실제 최적화 API 연결 전에는 최소 payload만 저장한다.
    stage_key = "1-4"
    state["sim_stage_outputs"][stage_key] = {
        "ref_lot_id": ref_lot_id,
        "status": "prepared",
    }
    state["sim_stage_status"][stage_key] = "done"
    # 1-4 단계: 다음 단계 컨펌 대기로 전환한다.
    state["sim_phase"] = PHASE_WAIT_CONFIRM
    state["sim_pending_next_step"] = STEP_STAGE_5
    state["sim_last_brief"] = "1-4 단계 완료: 최적화 요청 payload를 준비했습니다."
    return {
        "ok": True,
        "stage": stage_key,
        "ref_lot_id": ref_lot_id,
        "next_step": state["sim_pending_next_step"],
    }


def run_stage_5_select_topk(tool_context: ToolContext, top_k: int = 5) -> dict[str, Any]:
    # 1-5 단계: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 1-5 단계: 현재는 결과 슬롯만 만들고 종료 상태로 전환한다.
    stage_key = "1-5"
    state["sim_stage_outputs"][stage_key] = {
        "top_k": top_k,
        "items": [],
    }
    state["sim_stage_status"][stage_key] = "done"
    # 1-5 단계: 마지막 단계 완료로 처리한다.
    state["sim_phase"] = PHASE_DONE
    state["sim_pending_next_step"] = None
    state["sim_last_brief"] = f"1-5 단계 완료: 상위 {top_k}개 후보를 정리했습니다."
    return {
        "ok": True,
        "stage": stage_key,
        "top_k": top_k,
        "done": True,
    }


def confirm_next_step(tool_context: ToolContext, user_decision: str) -> dict[str, Any]:
    # 컨펌 처리: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 컨펌 처리: 사용자 결정을 규칙 기반으로 파싱한다.
    action = _parse_user_decision(user_decision)
    # 컨펌 처리: 수정 요청이면 대기 상태를 유지한다.
    if action == "REVISE":
        state["sim_phase"] = PHASE_WAIT_CONFIRM
        state["sim_last_brief"] = "수정할 항목을 알려주세요. 필요한 단계부터 다시 계산합니다."
        return {"ok": True, "action": action, "sim_phase": state["sim_phase"]}
    # 컨펌 처리: 질문이면 대기 상태를 유지한다.
    if action == "QUESTION":
        state["sim_phase"] = PHASE_WAIT_CONFIRM
        state["sim_last_brief"] = "질문에 답변한 뒤 진행 여부를 다시 확인하겠습니다."
        return {"ok": True, "action": action, "sim_phase": state["sim_phase"]}
    # 컨펌 처리: 진행이면 다음 단계를 실행 상태로 전환한다.
    next_step = state.get("sim_pending_next_step")
    if next_step and int(next_step) <= FINAL_STEP:
        state["sim_step"] = int(next_step)
        state["sim_phase"] = PHASE_RUN_STAGE
        state["sim_pending_next_step"] = None
        state["sim_last_brief"] = f"{state['sim_step']} 단계 실행을 시작합니다."
    else:
        state["sim_phase"] = PHASE_DONE
        state["sim_pending_next_step"] = None
        state["sim_last_brief"] = "모든 단계를 완료했습니다."
    return {
        "ok": True,
        "action": action,
        "sim_phase": state["sim_phase"],
        "sim_step": state["sim_step"],
    }


def revise_from_stage(
    tool_context: ToolContext,
    from_step: int,
    note: str | None = None,
) -> dict[str, Any]:
    # 수정 처리: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 수정 처리: 단계 범위를 2~5로 고정한다.
    clamped_step = max(STEP_STAGE_2, min(FINAL_STEP, int(from_step)))
    # 수정 처리: 해당 단계 이후 결과를 무효화한다.
    _invalidate_from_stage(state, clamped_step)
    # 수정 처리: 사용자에게 수정 재실행 상태를 알린다.
    state["sim_last_brief"] = f"{clamped_step} 단계부터 다시 계산합니다."
    if note:
        state["sim_last_gap"] = {"reason": "user_revision", "note": note}
    return {
        "ok": True,
        "sim_step": state["sim_step"],
        "sim_phase": state["sim_phase"],
        "invalidated_from": clamped_step,
    }


def brief_current_state(tool_context: ToolContext) -> dict[str, Any]:
    # 브리핑: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 브리핑: 사용자에게 보여줄 핵심 상태만 요약한다.
    return {
        "ok": True,
        "sim_step": state["sim_step"],
        "sim_phase": state["sim_phase"],
        "sim_last_brief": state["sim_last_brief"],
        "sim_pending_next_step": state["sim_pending_next_step"],
        "sim_last_gap": state["sim_last_gap"],
        "sim_allowed_tools": state["sim_allowed_tools"],
    }


def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict | None:
    # 툴 가드: 상태 기본값을 먼저 보장한다.
    state = tool_context.state
    _ensure_state_defaults(state)
    # 툴 가드: 현재 phase/step 기준 허용 툴을 계산한다.
    allowed_tools = _compute_allowed_tools(state)
    state["sim_allowed_tools"] = allowed_tools
    # 툴 가드: 허용 목록에 없는 툴은 즉시 차단한다.
    if tool.name not in allowed_tools:
        state["sim_last_brief"] = f"{tool.name} 호출을 차단했습니다. 허용된 툴만 사용하세요."
        return {
            "ok": False,
            "blocked": True,
            "reason": "tool_not_allowed",
            "allowed_tools": allowed_tools,
            "sim_phase": state["sim_phase"],
            "sim_step": state["sim_step"],
        }
    # 툴 가드: 1-2 단계 실행 전 필수 입력을 검사한다.
    if tool.name == "run_stage_2_find_chip" and not _has_minimum_input(state["sim_inputs"]):
        state["sim_phase"] = PHASE_COLLECT_INPUT
        state["sim_last_brief"] = "입력값이 부족합니다. 4개 인자 또는 chip_prod_id를 먼저 입력하세요."
        return {
            "ok": False,
            "blocked": True,
            "reason": "insufficient_input",
        }
    # 툴 가드: 1-3 단계 실행 전 1-2 결과를 검사한다.
    if tool.name == "run_stage_3_find_ref_lot":
        stage_2_output = state.get("sim_stage_outputs", {}).get("1-2", {})
        if not stage_2_output.get("chip_prod_id_list"):
            state["sim_step"] = STEP_STAGE_2
            state["sim_phase"] = PHASE_RUN_STAGE
            state["sim_last_brief"] = "1-2 단계 결과가 없어 1-3 단계를 차단했습니다."
            return {
                "ok": False,
                "blocked": True,
                "reason": "missing_stage_2_output",
            }
    # 툴 가드: 1-4 단계 실행 전 1-3 결과를 검사한다.
    if tool.name == "run_stage_4_optimize":
        stage_3_output = state.get("sim_stage_outputs", {}).get("1-3", {})
        if not stage_3_output.get("ref_lot_id"):
            state["sim_step"] = STEP_STAGE_3
            state["sim_phase"] = PHASE_RUN_STAGE
            state["sim_last_brief"] = "1-3 단계 결과가 없어 1-4 단계를 차단했습니다."
            return {
                "ok": False,
                "blocked": True,
                "reason": "missing_stage_3_output",
            }
    # 툴 가드: 차단 사유가 없으면 호출을 허용한다.
    return None


ROOT_AGENT_INSTRUCTION = """
당신은 MLCC 시뮬레이션 전용 루트 에이전트다.
항상 현재 상태를 확인하고 허용된 tool만 호출한다.

[현재 상태]
- sim_step: {sim_step?}
- sim_phase: {sim_phase?}
- sim_inputs: {sim_inputs?}
- sim_last_brief: {sim_last_brief?}
- sim_pending_next_step: {sim_pending_next_step?}
- sim_last_gap: {sim_last_gap?}
- sim_allowed_tools: {sim_allowed_tools?}

[행동 규칙]
1. sim_phase가 collect_input이면 set_sim_input 또는 brief_current_state만 사용한다.
2. sim_phase가 run_stage면 현재 sim_step 단계의 실행 tool만 사용한다.
3. sim_phase가 wait_confirm이면 confirm_next_step 또는 revise_from_stage를 우선 사용한다.
4. 사용자 요청이 수정이면 revise_from_stage를 호출한 뒤 다시 실행한다.
5. 단계 결과 설명은 짧고 명확하게 작성한다.
"""


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="mlcc_simulation_agent",
    description="MLCC 시뮬레이션 단계를 순차 진행하고 사용자 컨펌을 처리한다.",
    instruction=ROOT_AGENT_INSTRUCTION,
    before_tool_callback=before_tool_callback,
    tools=[
        set_sim_input,
        run_stage_2_find_chip,
        run_stage_3_find_ref_lot,
        run_stage_4_optimize,
        run_stage_5_select_topk,
        confirm_next_step,
        revise_from_stage,
        brief_current_state,
    ],
)
