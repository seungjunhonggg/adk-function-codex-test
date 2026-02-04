import json

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from .db_production import find_chip_prod_id
from .db_production import find_ref_lot_candidate
from .prompt import SIMULATION_INSTRUCTION_TEMPLATE

_SIM_STEP_STAGE_MAP = {
    4: "1-4",
    5: "1-5",
    6: "1-6",
}


def _build_state_summary(state):
    # 프롬프트에 넣을 상태 요약을 만든다.
    if not isinstance(state, dict):
        return "{}"
    # 입력/선택/설정/상태만 추린다.
    summary = {
        "input_params": state.get("input_params"),
        "selections": state.get("selections"),
        "configs": state.get("configs"),
        "stage_status": state.get("stage_status"),
        "pending_action": state.get("pending_action"),
        "last_gap": state.get("last_gap"),
    }
    # JSON 문자열로 변환한다.
    return json.dumps(summary, ensure_ascii=False)

def _ensure_state(tool_context):
    # 상태 객체를 가져온다.
    state = tool_context.state
    # stage_status 기본값을 보장한다.
    if "stage_status" not in state:
        state["stage_status"] = {}
    # stage_outputs 기본값을 보장한다.
    if "stage_outputs" not in state:
        state["stage_outputs"] = {}
    return state


def _ensure_sim_step(state, default_step):
    # sim_step 기본값을 보장한다.
    if "sim_step" not in state:
        state["sim_step"] = default_step


def _gate_sim_step(state, required_step):
    # sim_step 게이트를 확인한다.
    _ensure_sim_step(state, required_step)
    return state.get("sim_step") == required_step


def _invalidate_from_step(state, start_step):
    # 시작 단계 이후 결과를 무효화한다.
    stage_outputs = state.get("stage_outputs", {})
    stage_status = state.get("stage_status", {})
    for step in range(start_step, 7):
        stage_key = _SIM_STEP_STAGE_MAP.get(step)
        if not stage_key:
            continue
        stage_outputs.pop(stage_key, None)
        stage_status[stage_key] = "dirty"
    state["stage_outputs"] = stage_outputs
    state["stage_status"] = stage_status


def get_ref_lot_info(tool_context: ToolContext, ref_lot_id=None, ref_lot_info=None):
    """
    1-4 단계 툴.
    사용 시점: sim_step=4일 때.
    입력: ref_lot_id 또는 ref_lot_info
    출력: payload_seed(요약)
    예시: {"ref_lot_id":"LOT-1"} → {"ref_lot_id":"LOT-1","ref_lot_info":{...}}
    """
    # 1-4 단계: 상태를 준비한다.
    state = _ensure_state(tool_context)
    # 1-4 단계: ref LOT 입력을 정리한다.
    if ref_lot_info is None:
        stage_output = state.get("stage_outputs", {}).get("1-3", {})
        ref_lot_info = stage_output.get("ref_lot_info")
    info = ref_lot_info or {}
    # 1-4 단계: ref_lot_id가 바뀌면 이후 결과를 무효화한다.
    prev_ref_lot_id = state.get("stage_outputs", {}).get("1-3", {}).get("ref_lot_id")
    next_ref_lot_id = ref_lot_id or prev_ref_lot_id
    if prev_ref_lot_id and next_ref_lot_id and prev_ref_lot_id != next_ref_lot_id:
        _invalidate_from_step(state, 4)
        state["sim_step"] = 4
    # 1-4 단계: sim_step 게이트를 확인한다.
    if not _gate_sim_step(state, 4):
        state["pending_action"] = {
            "action": "wait_step",
            "target_step": 4,
            "current_step": state.get("sim_step"),
        }
        return {
            "skipped": True,
            "reason": "step_gate",
            "expected_step": state.get("sim_step"),
        }
    # 1-4 단계: 최적화 API payload 씨앗을 만든다.
    payload_seed = {
        "ref_lot_id": ref_lot_id,
        "ref_lot_info": info,
    }
    # 1-4 단계: 상태에 요약을 저장한다.
    state["stage_outputs"]["1-4"] = {"ref_lot_id": ref_lot_id, "payload_seed": payload_seed}
    # 1-4 단계: 상태 완료 마크를 남긴다.
    state["stage_status"]["1-4"] = "done"
    state["sim_step"] = 5
    # 1-4 단계: 요약 결과만 반환한다.
    return payload_seed


def run_grid(tool_context: ToolContext, payload_seed=None, top_k=5):
    """
    1-5 단계 툴.
    사용 시점: sim_step=5일 때.
    입력: payload_seed, top_k
    출력: top_k 요약
    예시: {"top_k":5} → {"top_k":[...], "top_k_count":5}
    """
    # 1-5 단계: 상태를 준비한다.
    state = _ensure_state(tool_context)
    # 1-5 단계: payload_seed를 기준으로 top-k 결과를 만든다.
    if payload_seed is None:
        payload_seed = state.get("stage_outputs", {}).get("1-4", {}).get("payload_seed")
    # 1-5 단계: top_k가 바뀌면 이후 결과를 무효화한다.
    prev_top_k = state.get("stage_outputs", {}).get("1-5", {}).get("top_k_limit")
    if prev_top_k is not None and prev_top_k != top_k:
        _invalidate_from_step(state, 5)
        state["sim_step"] = 5
    # 1-5 단계: sim_step 게이트를 확인한다.
    if not _gate_sim_step(state, 5):
        state["pending_action"] = {
            "action": "wait_step",
            "target_step": 5,
            "current_step": state.get("sim_step"),
        }
        return {
            "skipped": True,
            "reason": "step_gate",
            "expected_step": state.get("sim_step"),
        }
    top_k_candidates = []
    # 1-5 단계: 결과 요약을 만든다.
    summary = {
        "top_k": top_k_candidates,
        "top_k_count": len(top_k_candidates),
        "top_k_limit": top_k,
    }
    # 1-5 단계: 상태에 요약을 저장한다.
    state["stage_outputs"]["1-5"] = summary
    # 1-5 단계: 상태 완료 마크를 남긴다.
    state["stage_status"]["1-5"] = "done"
    state["sim_step"] = 6
    # 1-5 단계: 요약만 반환한다.
    return summary


def fetch_recent_defect_summary(tool_context: ToolContext, top_k=None, months=6):
    """
    1-6 단계 툴.
    사용 시점: sim_step=6일 때.
    입력: top_k, months
    출력: 최근 6개월 요약
    예시: {"months":6} → {"defect_summary":[...]}
    """
    # 1-6 단계: 상태를 준비한다.
    state = _ensure_state(tool_context)
    # 1-6 단계: 최근 N개월 요약을 만든다.
    if top_k is None:
        top_k = state.get("stage_outputs", {}).get("1-5", {}).get("top_k", [])
    # 1-6 단계: sim_step 게이트를 확인한다.
    if not _gate_sim_step(state, 6):
        state["pending_action"] = {
            "action": "wait_step",
            "target_step": 6,
            "current_step": state.get("sim_step"),
        }
        return {
            "skipped": True,
            "reason": "step_gate",
            "expected_step": state.get("sim_step"),
        }
    recent_summary = {
        "months": months,
        "top_k": top_k or [],
        "defect_summary": [],
    }
    # 1-6 단계: 상태에 요약을 저장한다.
    state["stage_outputs"]["1-6"] = {"months": months}
    # 1-6 단계: 상태 완료 마크를 남긴다.
    state["stage_status"]["1-6"] = "done"
    state["sim_step"] = 6
    # 1-6 단계: 요약만 반환한다.
    return recent_summary


def create_root_agent(state=None):
    # 상태 요약을 만든다.
    state_summary = _build_state_summary(state)
    # 템플릿에 상태를 주입한다.
    instruction = SIMULATION_INSTRUCTION_TEMPLATE.replace("{{state_summary}}", state_summary)
    # 호환용 {state_summary}도 치환한다.
    instruction = instruction.replace("{state_summary}", state_summary)
    # 루트 에이전트를 생성한다.
    return LlmAgent(
        # 사용할 모델을 지정한다.
        model="gemini-2.5-flash",
        # 에이전트 이름을 지정한다.
        name="mlcc_simulation_agent",
        # 에이전트 역할을 설명한다.
        description="MLCC 시뮬레이션 단계 진행과 사용자 대화를 담당한다.",
        # 에이전트 규칙을 주입한다.
        instruction=instruction,
        # 단계별 툴을 등록한다.
        tools=[
            find_chip_prod_id,
            find_ref_lot_candidate,
            get_ref_lot_info,
            run_grid,
            fetch_recent_defect_summary,
        ],
    )


# 기본 루트 에이전트를 생성한다.
root_agents = create_root_agent()
