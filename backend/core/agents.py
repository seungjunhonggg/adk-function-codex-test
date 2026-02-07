import json
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from . import db_production
from .db_production import InputParams

root_agent = LlmAgent(
        model="gemini-2.5-flash",
        name="mlcc_simulation_agent",
        description="MLCC 시뮬레이션 단계 진행과 사용자 확인을 처리한다.",
        instruction=instruction,
        tools=[
            find_chip_prod_id,
            find_ref_lot_candidate,
        ],
    )