from __future__ import annotations

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from . import db_production
from .agents import create_root_agent
from .state import ensure_session_state

# ADK 앱 이름을 고정한다.
APP_NAME = "mlcc_simulation"

# 데모/프로덕션 세션 서비스를 분리한다.
_DEMO_SESSION_SERVICE = InMemorySessionService()
_PROD_SESSION_SERVICE = InMemorySessionService()


class RunnerWrapper:
    def __init__(self, session_service: InMemorySessionService, use_db: bool):
        # 세션 서비스와 DB 사용 여부를 저장한다.
        self.session_service = session_service
        self.use_db = use_db

    async def _load_initial_state(self, session_id: str) -> dict:
        # 저장소에서 초기 상태를 읽고 기본 구조를 보정한다.
        if self.use_db:
            stored = db_production.fetch_session_state(session_id)
            return ensure_session_state(stored)
        return ensure_session_state({})

    async def _get_or_create_session(
        self,
        user_id: str,
        session_id: str,
        initial_state: dict,
    ):
        # 기존 세션을 조회한다.
        session = await self.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        # 세션이 없으면 기본 상태로 생성한다.
        if session is None:
            await self.session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state=initial_state,
            )
            session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
        return session

    async def run_async(self, user_id: str, session_id: str, new_message):
        # 초기 상태를 준비한다.
        initial_state = await self._load_initial_state(session_id)
        session = await self._get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
        )

        # 세션 상태를 기본 구조로 정규화한다.
        normalized = ensure_session_state(session.state if session else initial_state)
        if session is not None:
            session.state.update(normalized)

        # 상태를 반영한 루트 에이전트를 생성한다.
        runner = Runner(
            app_name=APP_NAME,
            agent=create_root_agent(normalized),
            session_service=self.session_service,
        )

        # 실행 이벤트를 스트리밍으로 전달한다.
        last_invocation_id = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            invocation_id = getattr(event, "invocation_id", None)
            if invocation_id:
                last_invocation_id = invocation_id
            yield event

        # 프로덕션 모드에서는 최신 상태를 DB에 저장한다.
        if self.use_db:
            latest_session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            if latest_session is not None:
                latest_state = ensure_session_state(latest_session.state or {})
                if last_invocation_id:
                    latest_state["last_invocation_id"] = last_invocation_id
                db_production.upsert_session_state(session_id, latest_state)

    async def get_state_async(self, user_id: str, session_id: str) -> dict:
        # 현재 세션 상태를 조회해서 반환한다.
        initial_state = await self._load_initial_state(session_id)
        session = await self._get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
        )
        normalized = ensure_session_state(session.state if session else initial_state)
        if session is not None:
            session.state.update(normalized)
        return normalized


# 데모/프로덕션 러너를 준비한다.
_DEMO_RUNNER = RunnerWrapper(_DEMO_SESSION_SERVICE, use_db=False)
_PROD_RUNNER = RunnerWrapper(_PROD_SESSION_SERVICE, use_db=True)


def get_runner(demo: bool = False) -> RunnerWrapper:
    # demo 플래그에 따라 적절한 러너를 반환한다.
    return _DEMO_RUNNER if demo else _PROD_RUNNER
