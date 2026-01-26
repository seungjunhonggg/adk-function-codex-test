from typing import Any, Literal

from pydantic import BaseModel, Field


# 요청 스키마를 정의한다.
class ChatRequest(BaseModel):
    session_id: str
    message: str
    overrides: dict[str, Any] | None = None
    demo: bool = False


# 응답 스키마를 정의한다.
class ChatResponse(BaseModel):
    route: str
    blocks: list[dict[str, Any]]
    tables: dict[str, Any]
    charts: list[dict[str, Any]]


# 라우팅 결과 스키마를 정의한다.
class RouteDecision(BaseModel):
    route: Literal["casual", "simulation"]


# 커맨드 결과 스키마를 정의한다.
class CommandDecision(BaseModel):
    # 커맨드 액션을 정의한다.
    action: Literal["run", "update_input", "explain_stage", "reset"]
    target_stage: str | None = None


# 선택 결과 스키마를 정의한다.
class SelectionDecision(BaseModel):
    selected_ids: list[str] = []


# 입력 파라미터 스키마를 정의한다.
class InputParams(BaseModel):
    temperature: str | None = None
    voltage: str | None = None
    size: str | None = None
    capacity: str | None = None


# 변경 대상 스키마를 정의한다.
class UpdateSelections(BaseModel):
    chip_type_ids: list[str] | None = None
    reference_lot_id: str | None = None


class UpdateConfigs(BaseModel):
    top_k: int | None = None


class UpdateUserPrefs(BaseModel):
    chart_type: Literal["bar", "line", "scatter"] | None = None


class UpdateDecision(BaseModel):
    input_params: InputParams | None = None
    selections: UpdateSelections | None = None
    configs: UpdateConfigs | None = None
    user_prefs: UpdateUserPrefs | None = None
    missing_fields: list[str] = []


# 브리핑 블록 스키마를 정의한다.
class BriefingBlock(BaseModel):
    type: Literal["text", "table_ref", "chart_ref"]
    section: str | None = None
    # text 줄바꿈 규칙을 스키마 설명으로 전달한다.
    value: str | None = Field(
        default=None,
        description=(
            "text 블록 내용. 문장마다 **반드시** 줄바꿈(\\n)을 넣고 한 줄에 문장 1개만 작성."
            " 문장 구분은 마침표/물음표/느낌표 기준. 빈 줄 금지."
        ),
    )
    table_key: str | None = None
    chart_id: str | None = None


class BriefingOutput(BaseModel):
    blocks: list[BriefingBlock]


# 캐주얼 응답 구조를 정의한다.
class CasualOutput(BaseModel):
    answer: str


class ExplainOutput(BaseModel):
    answer: str


class GapQuestionOutput(BaseModel):
    question: str
