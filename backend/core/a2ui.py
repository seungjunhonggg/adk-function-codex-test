"""A2UI v0.10 프로토콜 메시지 빌더 및 SIM_INPUT 컴포넌트 정의."""

from __future__ import annotations

A2UI_VERSION = "v0.10"

# ---------------------------------------------------------------------------
# 메시지 빌더 함수
# ---------------------------------------------------------------------------


def create_surface(
    surface_id: str,
    catalog_id: str,
    send_data_model: bool = True,
) -> dict:
    return {
        "version": A2UI_VERSION,
        "createSurface": {
            "surfaceId": surface_id,
            "catalogId": catalog_id,
            "sendDataModel": send_data_model,
        },
    }


def update_components(surface_id: str, components: list[dict]) -> dict:
    return {
        "version": A2UI_VERSION,
        "updateComponents": {
            "surfaceId": surface_id,
            "components": components,
        },
    }


def update_data_model(
    surface_id: str,
    value: dict,
    path: str | None = None,
) -> dict:
    msg: dict = {
        "version": A2UI_VERSION,
        "updateDataModel": {
            "surfaceId": surface_id,
            "value": value,
        },
    }
    if path:
        msg["updateDataModel"]["path"] = path
    return msg


def delete_surface(surface_id: str) -> dict:
    return {
        "version": A2UI_VERSION,
        "deleteSurface": {"surfaceId": surface_id},
    }


# ---------------------------------------------------------------------------
# SIM_INPUT 폼 – A2UI 컴포넌트 트리
# ---------------------------------------------------------------------------

SIM_INPUT_SURFACE_ID = "sim_input_surface"
SIM_INPUT_CATALOG_ID = "sim_input_form"


def _build_sim_input_components() -> list[dict]:
    """SIM_INPUT 폼을 A2UI adjacency-list 컴포넌트 배열로 반환한다."""
    return [
        # ── root ──
        {
            "id": "root",
            "type": "Row",
            "childIds": ["left_col", "or_divider", "right_col"],
        },
        # ── 왼쪽: 조건 직접 입력 ──
        {
            "id": "left_col",
            "type": "Column",
            "childIds": [
                "left_header",
                "temp_picker",
                "size_picker",
                "cap_row",
                "volt_picker",
                "left_submit",
            ],
        },
        {
            "id": "left_header",
            "type": "Text",
            "content": "방법 1 — 조건 직접 입력",
            "variant": "h5",
        },
        {
            "id": "temp_picker",
            "type": "ChoicePicker",
            "label": "Temperature",
            "options": [
                {"label": v, "value": v}
                for v in ["X5R", "X7R", "X7S", "X6S", "X8R", "X8L"]
            ],
            "dataPath": "/temperature",
        },
        {
            "id": "size_picker",
            "type": "ChoicePicker",
            "label": "Size",
            "options": [
                {"label": v, "value": v}
                for v in [
                    "0402", "0603", "0805", "1005",
                    "1608", "2012", "3216", "3225",
                ]
            ],
            "dataPath": "/size",
        },
        {
            "id": "cap_row",
            "type": "Row",
            "childIds": ["cap_field", "cap_unit_picker"],
        },
        {
            "id": "cap_field",
            "type": "TextField",
            "label": "Capacity",
            "inputType": "number",
            "dataPath": "/capacity",
        },
        {
            "id": "cap_unit_picker",
            "type": "ChoicePicker",
            "label": "Unit",
            "options": [
                {"label": u, "value": u} for u in ["pF", "nF", "uF"]
            ],
            "dataPath": "/capacity_unit",
        },
        {
            "id": "volt_picker",
            "type": "ChoicePicker",
            "label": "Voltage",
            "options": [
                {"label": v, "value": v}
                for v in ["6.3V", "10V", "16V", "25V", "50V", "100V"]
            ],
            "dataPath": "/voltage",
        },
        {
            "id": "left_submit",
            "type": "Button",
            "label": "시뮬레이션 시작",
            "variant": "primary",
            "action": {"event": {"name": "submit_core"}},
        },
        # ── 구분선 ──
        {"id": "or_divider", "type": "Divider"},
        # ── 오른쪽: CHIP 기종으로 검색 ──
        {
            "id": "right_col",
            "type": "Column",
            "childIds": ["right_header", "chip_field", "right_submit"],
        },
        {
            "id": "right_header",
            "type": "Text",
            "content": "방법 2 — CHIP 기종으로 검색",
            "variant": "h5",
        },
        {
            "id": "chip_field",
            "type": "TextField",
            "label": "CHIP 기종 코드",
            "placeholder": "예: CL32Y106KBHNNNE",
            "dataPath": "/chip_prod_id",
        },
        {
            "id": "right_submit",
            "type": "Button",
            "label": "시뮬레이션 시작",
            "variant": "primary",
            "action": {"event": {"name": "submit_chip"}},
        },
    ]


_SIM_INPUT_INITIAL_DATA = {
    "temperature": "",
    "size": "",
    "capacity": "",
    "capacity_unit": "pF",
    "voltage": "",
    "chip_prod_id": "",
}


def build_sim_input_a2ui_messages() -> list[dict]:
    """SIM_INPUT 폼을 위한 A2UI 메시지 3개를 반환한다."""
    return [
        create_surface(SIM_INPUT_SURFACE_ID, SIM_INPUT_CATALOG_ID),
        update_components(SIM_INPUT_SURFACE_ID, _build_sim_input_components()),
        update_data_model(SIM_INPUT_SURFACE_ID, {**_SIM_INPUT_INITIAL_DATA}),
    ]
