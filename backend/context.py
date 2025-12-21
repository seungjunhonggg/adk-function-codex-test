from contextvars import ContextVar

current_session_id = ContextVar("current_session_id", default="anonymous")
