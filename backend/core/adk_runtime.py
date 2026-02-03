from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from .agents import RootAgent

# 앱 이름을 정의한다.
APP_NAME = "mlcc_adk"

# 세션 서비스를 준비한다.
_DEMO_SESSION_SERVICE = InMemorySessionService()
_PROD_SESSION_SERVICE = InMemorySessionService()

# 루트 에이전트를 준비한다.
_DEMO_AGENT = RootAgent(use_db=False)
_PROD_AGENT = RootAgent(use_db=True)

# 러너를 준비한다.
_DEMO_RUNNER = Runner(
    agent=_DEMO_AGENT,
    app_name=APP_NAME,
    session_service=_DEMO_SESSION_SERVICE,
)
_PROD_RUNNER = Runner(
    agent=_PROD_AGENT,
    app_name=APP_NAME,
    session_service=_PROD_SESSION_SERVICE,
)


def get_runner(is_demo: bool) -> Runner:
    # 데모 여부에 따라 러너를 반환한다.
    return _DEMO_RUNNER if is_demo else _PROD_RUNNER
