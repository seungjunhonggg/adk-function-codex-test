from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.chat import router as chat_router

app = FastAPI()

# 채팅 라우터를 등록한다.
app.include_router(chat_router)

# 프론트엔드 경로를 계산한다.
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

# 정적 파일을 제공한다.
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    # 기본 화면을 반환한다.
    return FileResponse(frontend_dir / "index.html")
