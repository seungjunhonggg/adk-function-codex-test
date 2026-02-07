from __future__ import annotations

from typing import Any

# 단계 번호와 단계 키를 매핑한다.
STEP_TO_STAGE = {
    2: "1-2",
    3: "1-3",
    4: "1-4",
    5: "1-5",
    6: "1-6",
}

# 단계 키와 단계 번호를 역매핑한다.
STAGE_TO_STEP = {value: key for key, value in STEP_TO_STAGE.items()}


def default_session_state() -> dict[str, Any]:
    # 세션 상태의 기본 구조를 만든다.
    return {
        "sim_step": 2,
        "input_params": {},
        "selections": {},
        "configs": {"top_k": 5, "months": 6},
        "stage_status": {stage: "idle" for stage in STEP_TO_STAGE.values()},
        "stage_outputs": {},
        "pending_action": None,
        "last_gap": None,
    }


def ensure_session_state(state: dict[str, Any] | None) -> dict[str, Any]:
    # 전달받은 상태를 기본 구조로 보정한다.
    base = default_session_state()
    if not isinstance(state, dict):
        return base
    merged = dict(base)
    merged.update(state)
    if not isinstance(merged.get("configs"), dict):
        merged["configs"] = base["configs"].copy()
    if not isinstance(merged.get("stage_status"), dict):
        merged["stage_status"] = base["stage_status"].copy()
    if not isinstance(merged.get("stage_outputs"), dict):
        merged["stage_outputs"] = {}
    if not isinstance(merged.get("input_params"), dict):
        merged["input_params"] = {}
    if not isinstance(merged.get("selections"), dict):
        merged["selections"] = {}
    for stage in STEP_TO_STAGE.values():
        merged["stage_status"].setdefault(stage, "idle")
    return merged


def invalidate_from_step(state: dict[str, Any], start_step: int) -> None:
    # 시작 단계 이후 결과를 모두 dirty 처리한다.
    stage_outputs = state.get("stage_outputs", {})
    stage_status = state.get("stage_status", {})
    for step in range(start_step, 7):
        stage_key = STEP_TO_STAGE.get(step)
        if not stage_key:
            continue
        stage_outputs.pop(stage_key, None)
        stage_status[stage_key] = "dirty"
    state["stage_outputs"] = stage_outputs
    state["stage_status"] = stage_status


def resolve_dirty_start(changed_paths: list[str]) -> int | None:
    # 변경 경로를 단계 시작점으로 매핑한다.
    dirty_step: int | None = None
    for path in changed_paths:
        step: int | None = None
        if path.startswith("input_params."):
            step = 2
        elif path == "selections.chip_prod_id_list":
            step = 3
        elif path == "selections.reference_lot_id":
            step = 4
        elif path == "configs.top_k":
            step = 5
        elif path == "configs.months":
            step = 6
        if step is None:
            continue
        if dirty_step is None or step < dirty_step:
            dirty_step = step
    return dirty_step


def apply_patch_to_state(state: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    # 패치 값을 상태에 반영하고 dirty 시작 단계를 계산한다.
    changed_paths: list[str] = []
    input_params = patch.get("input_params") or {}
    selections = patch.get("selections") or {}
    configs = patch.get("configs") or {}

    for key, value in input_params.items():
        if value is None:
            continue
        if state["input_params"].get(key) == value:
            continue
        state["input_params"][key] = value
        changed_paths.append(f"input_params.{key}")

    for key, value in selections.items():
        if value is None:
            continue
        if state["selections"].get(key) == value:
            continue
        state["selections"][key] = value
        changed_paths.append(f"selections.{key}")

    for key, value in configs.items():
        if value is None:
            continue
        if state["configs"].get(key) == value:
            continue
        state["configs"][key] = value
        changed_paths.append(f"configs.{key}")

    dirty_step = resolve_dirty_start(changed_paths)
    if dirty_step is not None:
        invalidate_from_step(state, dirty_step)
        state["sim_step"] = dirty_step
        state["pending_action"] = None
        state["last_gap"] = None

    return {
        "changed_paths": changed_paths,
        "dirty_step": dirty_step,
    }


def set_pending_confirmation(
    state: dict[str, Any],
    stage_key: str,
    summary: str,
) -> None:
    # 단계 완료 후 사용자 확인 대기 상태를 만든다.
    state["pending_action"] = {
        "type": "confirm_stage",
        "stage": stage_key,
        "summary": summary,
    }


def clear_pending_confirmation(state: dict[str, Any], stage_key: str | None = None) -> None:
    # 확인 대기 상태를 해제한다.
    pending = state.get("pending_action")
    if not isinstance(pending, dict):
        return
    if pending.get("type") != "confirm_stage":
        return
    if stage_key and pending.get("stage") != stage_key:
        return
    state["pending_action"] = None
