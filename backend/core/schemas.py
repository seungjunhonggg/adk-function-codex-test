from pydantic import BaseModel


class InputParams(BaseModel):
    # 1-1 단계: 사용자 입력값(4개 인자 또는 칩기종)을 저장한다.
    temperature: str | None = None
    voltage: str | None = None
    size: str | None = None
    capacity: str | None = None
    chip_prod_id: str | None = None
