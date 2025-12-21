import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SIM_API_URL = os.getenv("SIM_API_URL", "")
SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", "sessions.db")
DB_PATH = os.getenv("DB_PATH", "process_data.db")
