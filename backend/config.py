import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", )
if OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-nano")
TRACING_ENABLED = _env_bool("OPENAI_AGENTS_TRACING_ENABLED", True)
SIM_API_URL = os.getenv("SIM_API_URL", "")
SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", "sessions.db")
DB_PATH = os.getenv("DB_PATH", "process_data.db")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "workflow.json")
WORKFLOW_STORE_PATH = os.getenv("WORKFLOW_STORE_PATH", "workflows.json")
DB_CONNECTIONS_PATH = os.getenv("DB_CONNECTIONS_PATH", "db_connections.json")
LOT_DB_CONNECTION_ID = os.getenv("LOT_DB_CONNECTION_ID", "")
LOT_DB_SCHEMA = os.getenv("LOT_DB_SCHEMA", "public")
LOT_DB_TABLE = os.getenv("LOT_DB_TABLE", "")
LOT_DB_LOT_COLUMN = os.getenv("LOT_DB_LOT_COLUMN", "lot_id")
LOT_DB_COLUMNS = os.getenv("LOT_DB_COLUMNS", "")
LOT_DB_FILTER_OPERATOR = os.getenv("LOT_DB_FILTER_OPERATOR", "=")
LOT_PARAM_TEMPERATURE_COLUMN = os.getenv("LOT_PARAM_TEMPERATURE_COLUMN", "temperature")
LOT_PARAM_VOLTAGE_COLUMN = os.getenv("LOT_PARAM_VOLTAGE_COLUMN", "voltage")
LOT_PARAM_SIZE_COLUMN = os.getenv("LOT_PARAM_SIZE_COLUMN", "size")
LOT_PARAM_CAPACITY_COLUMN = os.getenv("LOT_PARAM_CAPACITY_COLUMN", "capacity")
LOT_PARAM_PRODUCTION_COLUMN = os.getenv("LOT_PARAM_PRODUCTION_COLUMN", "production_mode")
