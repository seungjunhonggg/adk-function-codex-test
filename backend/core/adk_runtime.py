from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from . import db_production
from .agents import create_root_agent

# 앱 이름을 고정한다.
APP_NAME = "mlcc_simulation"

# 데모 세션 스토어를 준비한다.
_DEMO_SESSION_SERVICE = InMemorySessionService()
# 프로덕션 세션 스토어를 준비한다.
_PROD_SESSION_SERVICE = InMemorySessionService()


class RunnerWrapper:
    def __init__(self, session_service, use_db):
        # 세션 스토어를 저장한다.
        self.session_service = session_service
        # DB 사용 여부를 저장한다.
        self.use_db = use_db

    async def _get_or_create_session(self, user_id, session_id, initial_state):
        # 기존 세션을 조회한다.
        session = await self.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        # 세션이 없으면 새로 만든다.
        if session is None:
            await self.session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {},
            )
            # 생성 후 다시 조회한다.
            session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
        return session

    async def run_async(self, user_id, session_id, new_message):
        # 초기 상태를 준비한다.
        initial_state = {}
        # 프로덕션이면 저장된 상태를 불러온다.
        if self.use_db:
            stored_state = db_production.fetch_session_state(session_id)
            if isinstance(stored_state, dict):
                initial_state = stored_state
        # 세션을 준비한다.
        session = await self._get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
        )
        # 상태 요약을 넣은 에이전트를 만든다.
        agent = create_root_agent(session.state if session else initial_state)
        # 러너를 준비한다.
        runner = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=self.session_service,
        )
        # 러너 실행 결과를 전달한다.
        last_invocation_id = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            # 마지막 invocation_id를 저장한다.
            invocation_id = getattr(event, "invocation_id", None)
            if invocation_id:
                last_invocation_id = invocation_id
            yield event
        # 프로덕션이면 상태를 저장한다.
        if self.use_db:
            latest_session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            if latest_session is not None:
                latest_state = latest_session.state or {}
                if last_invocation_id:
                    latest_state["last_invocation_id"] = last_invocation_id
                db_production.upsert_session_state(session_id, latest_state)

    async def rewind_async(self, user_id, session_id, rewind_before_invocation_id):
        # 초기 상태를 준비한다.
        initial_state = {}
        # 프로덕션이면 저장된 상태를 불러온다.
        if self.use_db:
            stored_state = db_production.fetch_session_state(session_id)
            if isinstance(stored_state, dict):
                initial_state = stored_state
        # 세션을 준비한다.
        session = await self._get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
        )
        # 상태 요약을 넣은 에이전트를 만든다.
        agent = create_root_agent(session.state if session else initial_state)
        # 러너를 준비한다.
        runner = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=self.session_service,
        )
        # rewind를 수행한다.
        await runner.rewind_async(
            user_id=user_id,
            session_id=session_id,
            rewind_before_invocation_id=rewind_before_invocation_id,
        )
        # 프로덕션이면 상태를 저장한다.
        if self.use_db:
            latest_session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            if latest_session is not None:
                db_production.upsert_session_state(
                    session_id,
                    latest_session.state or {},
                )


# 데모 러너를 만든다.
_DEMO_RUNNER = RunnerWrapper(_DEMO_SESSION_SERVICE, use_db=False)
# 프로덕션 러너를 만든다.
_PROD_RUNNER = RunnerWrapper(_PROD_SESSION_SERVICE, use_db=True)


def get_runner(demo=False):
    # 데모 여부에 따라 러너를 반환한다.
    return _DEMO_RUNNER if demo else _PROD_RUNNER
