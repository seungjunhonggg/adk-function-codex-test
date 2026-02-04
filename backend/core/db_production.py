from typing import Any, Callable
import asyncio
import json
import threading

from .schemas import InputParams
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from google.adk.tools import ToolContext
from pydantic import Field

# .env 파일 로드
load_dotenv()

class DatabaseHandler:
    def __init__(self):
        """환경 변수에서 DB 설정을 불러와 초기화합니다."""
        self.host = os.getenv("DB_HOST")
        self.dbname = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.port = os.getenv("DB_PORT")
        self.connection = None

    def connect(self):
        """DB 연결을 생성합니다."""
        try:
            if self.connection is None or self.connection.closed:
                self.connection = psycopg2.connect(
                    host=self.host,
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password,
                    port=self.port
                )
        except Exception as e:
            print(f"[DB Error] Connection failed: {e}")
            raise

    def execute_read(self, query, params=None):
        """
        SELECT 문과 같이 데이터를 조회할 때 사용합니다.
        결과를 딕셔너리 형태의 리스트로 반환합니다.
        """
        self.connect()
        try:
            # RealDictCursor를 사용하면 컬럼명:값 형태(Dict)로 결과를 받습니다.
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return result
        except Exception as e:
            print(f"[DB Error] Read query failed: {e}")
            return None
        # 연결은 유지하거나, 필요에 따라 finally에서 close() 할 수 있습니다.

    def execute_write(self, query, params=None):
        """
        INSERT, UPDATE, DELETE 문과 같이 데이터를 변경할 때 사용합니다.
        성공적으로 완료되면 commit, 실패하면 rollback 합니다.
        """
        self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                self.connection.commit()  # 변경사항 저장
                return True
        except Exception as e:
            self.connection.rollback()  # 에러 발생 시 되돌리기
            print(f"[DB Error] Write query failed: {e}")
            return False

    def close(self):
        """DB 연결을 종료합니다."""
        if self.connection:
            self.connection.close()

# 에이전트에서 바로 import해서 쓸 수 있도록 인스턴스 생성 (선택 사항)
db = DatabaseHandler()

# 에이전트 세션용 스키마를 정의한다.
AGENT_SCHEMA = "data_portal"
# 에이전트 세션 테이블명을 정의한다.
AGENT_SESSIONS_TABLE = f"{AGENT_SCHEMA}.agent_sessions"
# 에이전트 메시지 테이블명을 정의한다.
AGENT_MESSAGES_TABLE = f"{AGENT_SCHEMA}.agent_messages"
# 에이전트 상태 테이블명을 정의한다.
AGENT_STATE_TABLE = f"{AGENT_SCHEMA}.agent_session_state"
# 테이블 준비 여부를 캐시한다.
_AGENT_TABLES_READY = False
# 테이블 준비 락을 준비한다.
_AGENT_TABLES_LOCK = threading.Lock()
# 스레드별 DB 커넥션을 보관한다.
_AGENT_LOCAL = threading.local()


def _get_agent_db_config() -> dict[str, Any]:
    # 환경 변수에서 에이전트 DB 설정을 읽는다.
    return {
        "host": os.getenv("DB_HOST"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": os.getenv("DB_PORT"),
    }


def _get_agent_connection() -> psycopg2.extensions.connection:
    # 스레드 로컬 커넥션을 꺼낸다.
    connection = getattr(_AGENT_LOCAL, "connection", None)
    # 커넥션이 없거나 닫혔으면 새로 만든다.
    if connection is None or connection.closed:
        config = _get_agent_db_config()
        connection = psycopg2.connect(**config)
        _AGENT_LOCAL.connection = connection
    # 커넥션을 반환한다.
    return connection


def _ensure_agent_tables() -> None:
    # 이미 준비됐으면 종료한다.
    global _AGENT_TABLES_READY
    if _AGENT_TABLES_READY:
        return
    # 동시 생성 방지를 위해 락을 잡는다.
    with _AGENT_TABLES_LOCK:
        # 락 안에서 다시 확인한다.
        if _AGENT_TABLES_READY:
            return
        # 커넥션을 준비한다.
        connection = _get_agent_connection()
        # 스키마와 테이블을 만든다.
        with connection.cursor() as cursor:
            # 스키마를 만든다.
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {AGENT_SCHEMA}")
            # 세션 메타 테이블을 만든다.
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {AGENT_SESSIONS_TABLE} (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # 세션 IP 컬럼을 보장한다.
            cursor.execute(
                f"""
                ALTER TABLE {AGENT_SESSIONS_TABLE}
                ADD COLUMN IF NOT EXISTS client_ip TEXT
                """
            )
            # 메시지 테이블을 만든다.
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {AGENT_MESSAGES_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id)
                        REFERENCES {AGENT_SESSIONS_TABLE} (session_id)
                        ON DELETE CASCADE
                )
                """
            )
            # 메시지 인덱스를 만든다.
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_agent_messages_session_id
                ON {AGENT_MESSAGES_TABLE} (session_id, id)
                """
            )
            # 상태 테이블을 만든다.
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {AGENT_STATE_TABLE} (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        # 변경사항을 반영한다.
        connection.commit()
        # 준비 완료 플래그를 세운다.
        _AGENT_TABLES_READY = True


def fetch_session_state(session_id: str) -> dict[str, Any] | None:
    # 테이블을 준비한다.
    _ensure_agent_tables()
    # 커넥션을 준비한다.
    connection = _get_agent_connection()
    # 상태를 조회한다.
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT state FROM {AGENT_STATE_TABLE} WHERE session_id = %s",
            (session_id,),
        )
        row = cursor.fetchone()
    # 결과가 없으면 None을 반환한다.
    if not row:
        return None
    # 저장된 상태 문자열을 꺼낸다.
    raw_state = row[0]
    # 상태가 비어 있으면 None을 반환한다.
    if not raw_state:
        return None
    # JSON을 파싱해 반환한다.
    try:
        if isinstance(raw_state, str):
            return json.loads(raw_state)
        return raw_state
    except json.JSONDecodeError:
        return None


def upsert_session_state(session_id: str, state: dict[str, Any]) -> None:
    # 테이블을 준비한다.
    _ensure_agent_tables()
    # 커넥션을 준비한다.
    connection = _get_agent_connection()
    # 상태를 JSON으로 직렬화한다.
    payload = json.dumps(state, ensure_ascii=False)
    # upsert를 수행한다.
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {AGENT_STATE_TABLE} (session_id, state)
            VALUES (%s, %s)
            ON CONFLICT (session_id)
            DO UPDATE SET
                state = EXCLUDED.state,
                version = {AGENT_STATE_TABLE}.version + 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, payload),
        )
    # 변경사항을 반영한다.
    connection.commit()


def upsert_session_ip(session_id: str, client_ip: str | None) -> None:
    # IP가 없으면 처리하지 않는다.
    if not client_ip:
        return
    # 테이블을 준비한다.
    _ensure_agent_tables()
    # 커넥션을 준비한다.
    connection = _get_agent_connection()
    # 세션 IP를 저장한다.
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {AGENT_SESSIONS_TABLE} (session_id, client_ip)
            VALUES (%s, %s)
            ON CONFLICT (session_id)
            DO UPDATE SET
                client_ip = EXCLUDED.client_ip,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, client_ip),
        )
    # 변경사항을 반영한다.
    connection.commit()


class PostgresSession:
    def __init__(
        self,
        session_id: str,
        sessions_table: str = AGENT_SESSIONS_TABLE,
        messages_table: str = AGENT_MESSAGES_TABLE,
    ) -> None:
        # 세션 ID를 저장한다.
        self.session_id = session_id
        # 세션 테이블명을 저장한다.
        self.sessions_table = sessions_table
        # 메시지 테이블명을 저장한다.
        self.messages_table = messages_table

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        # 세션 메시지를 조회한다.
        def _get_items_sync() -> list[dict[str, Any]]:
            # 테이블을 준비한다.
            _ensure_agent_tables()
            # 커넥션을 준비한다.
            connection = _get_agent_connection()
            # 쿼리를 실행한다.
            with connection.cursor() as cursor:
                if limit is None:
                    cursor.execute(
                        f"""
                        SELECT message_data FROM {self.messages_table}
                        WHERE session_id = %s
                        ORDER BY id ASC
                        """,
                        (self.session_id,),
                    )
                    rows = cursor.fetchall()
                else:
                    cursor.execute(
                        f"""
                        SELECT message_data FROM {self.messages_table}
                        WHERE session_id = %s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (self.session_id, limit),
                    )
                    rows = cursor.fetchall()
                    rows = list(reversed(rows))
            # JSON으로 파싱한다.
            items: list[dict[str, Any]] = []
            for (message_data,) in rows:
                try:
                    items.append(json.loads(message_data))
                except json.JSONDecodeError:
                    continue
            # 결과를 반환한다.
            return items

        return await asyncio.to_thread(_get_items_sync)

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        # 추가할 아이템이 없으면 종료한다.
        if not items:
            return

        # 아이템을 저장한다.
        def _add_items_sync() -> None:
            # 테이블을 준비한다.
            _ensure_agent_tables()
            # 커넥션을 준비한다.
            connection = _get_agent_connection()
            # 트랜잭션을 연다.
            with connection.cursor() as cursor:
                # 세션 메타를 보장한다.
                cursor.execute(
                    f"""
                    INSERT INTO {self.sessions_table} (session_id)
                    VALUES (%s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (self.session_id,),
                )
                # 메시지를 준비한다.
                payload = [
                    (self.session_id, json.dumps(item, ensure_ascii=False))
                    for item in items
                ]
                # 메시지를 저장한다.
                cursor.executemany(
                    f"""
                    INSERT INTO {self.messages_table} (session_id, message_data)
                    VALUES (%s, %s)
                    """,
                    payload,
                )
                # 업데이트 시각을 갱신한다.
                cursor.execute(
                    f"""
                    UPDATE {self.sessions_table}
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                    """,
                    (self.session_id,),
                )
            # 변경사항을 반영한다.
            connection.commit()

        await asyncio.to_thread(_add_items_sync)

    async def pop_item(self) -> dict[str, Any] | None:
        # 마지막 메시지를 꺼낸다.
        def _pop_item_sync() -> dict[str, Any] | None:
            # 테이블을 준비한다.
            _ensure_agent_tables()
            # 커넥션을 준비한다.
            connection = _get_agent_connection()
            # 메시지를 삭제하고 반환한다.
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {self.messages_table}
                    WHERE id = (
                        SELECT id FROM {self.messages_table}
                        WHERE session_id = %s
                        ORDER BY id DESC
                        LIMIT 1
                    )
                    RETURNING message_data
                    """,
                    (self.session_id,),
                )
                row = cursor.fetchone()
            # 변경사항을 반영한다.
            connection.commit()
            # 결과가 없으면 None을 반환한다.
            if not row:
                return None
            # JSON을 파싱해 반환한다.
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None

        return await asyncio.to_thread(_pop_item_sync)

    async def clear_session(self) -> None:
        # 세션 메시지를 모두 삭제한다.
        def _clear_session_sync() -> None:
            # 테이블을 준비한다.
            _ensure_agent_tables()
            # 커넥션을 준비한다.
            connection = _get_agent_connection()
            # 삭제를 수행한다.
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.messages_table} WHERE session_id = %s",
                    (self.session_id,),
                )
                cursor.execute(
                    f"DELETE FROM {self.sessions_table} WHERE session_id = %s",
                    (self.session_id,),
                )
            # 변경사항을 반영한다.
            connection.commit()

        await asyncio.to_thread(_clear_session_sync)

# 라벨 매핑 테이블/컬럼명을 정의한다.
_COLUMN_LABEL_TABLE = "column_label_map"
_COLUMN_KEY_FIELD = "column_key"
_COLUMN_LABEL_FIELD = "korean_label"
_SIM_STEP_STAGE_MAP = {
    2: "1-2",
    3: "1-3",
    4: "1-4",
    5: "1-5",
    6: "1-6",
}


def _ensure_sim_step(state: dict, default_step: int) -> None:
    # sim_step 기본값을 보장한다.
    if "sim_step" not in state:
        state["sim_step"] = default_step


def _gate_sim_step(state: dict, required_step: int) -> bool:
    # sim_step 게이트를 확인한다.
    _ensure_sim_step(state, required_step)
    return state.get("sim_step") == required_step


def _invalidate_from_step(state: dict, start_step: int) -> None:
    # 시작 단계 이후 결과를 무효화한다.
    stage_outputs = state.get("stage_outputs", {})
    stage_status = state.get("stage_status", {})
    for step in range(start_step, 7):
        stage_key = _SIM_STEP_STAGE_MAP.get(step)
        if not stage_key:
            continue
        stage_outputs.pop(stage_key, None)
        stage_status[stage_key] = "dirty"
    state["stage_outputs"] = stage_outputs
    state["stage_status"] = stage_status


def fetch_column_label_map() -> dict[str, str]:
    # column_label_map 테이블에서 라벨 매핑을 조회한다.
    rows = _query_column_label_map()
    # 조회 결과를 매핑 dict로 변환한다.
    label_map: dict[str, str] = {}
    for row in rows:
        # 영문 컬럼명과 한글 라벨을 꺼낸다.
        column_key = row.get(_COLUMN_KEY_FIELD)
        column_label = row.get(_COLUMN_LABEL_FIELD)
        # 필수 값이 없으면 건너뛴다.
        if not column_key or not column_label:
            continue
        # 매핑 결과에 추가한다.
        label_map[column_key] = column_label
    return label_map

def find_chip_prod_id(tool_context: ToolContext | None, input_params=None, dirty=None):
    """
    1-2 단계 툴.
    사용 시점: sim_step=2일 때.
    입력: input_params(온도/전압/크기/용량 또는 chip_prod_id)
    출력: chip_prod_id_list 요약 + gap
    예시: {"chip_prod_id":"CL32Y106"} → {"chip_prod_id_list":[...], "gap":null}
    """
    # 1-2 단계: tool_context와 input_params를 정리한다.
    if isinstance(tool_context, InputParams) or isinstance(tool_context, dict):
        input_params = tool_context
        tool_context = None
    # 1-2 단계: 입력 파라미터를 InputParams로 통일한다.
    if isinstance(input_params, dict):
        input_params = InputParams(**input_params)
    if input_params is None:
        input_params = InputParams()

    # 1-2 단계: 파라미터를 추출한다.
    temperature = input_params.temperature
    voltage = input_params.voltage
    size = input_params.size
    capacity = input_params.capacity
    chip_prod_id = input_params.chip_prod_id
    # 1-2 단계: 쿼리 파라미터를 만든다.
    target_keys = ["temperature", "voltage", "size", "capacity"]
    params = input_params.model_dump(include=target_keys)

    # 1-2 단계: sim_step 게이트를 확인한다.
    if tool_context is not None:
        state = tool_context.state
        if "stage_outputs" not in state:
            state["stage_outputs"] = {}
        if "stage_status" not in state:
            state["stage_status"] = {}
        # 1-2 단계: 입력이 바뀌면 이후 결과를 무효화한다.
        prev_params = state.get("input_params")
        next_params = input_params.model_dump()
        if prev_params is None:
            state["sim_step"] = 2
        if prev_params and prev_params != next_params:
            _invalidate_from_step(state, 2)
            state["sim_step"] = 2
        # 1-2 단계: sim_step 게이트를 확인한다.
        if not _gate_sim_step(state, 2):
            state["pending_action"] = {
                "action": "wait_step",
                "target_step": 2,
                "current_step": state.get("sim_step"),
            }
            return {
                "skipped": True,
                "reason": "step_gate",
                "expected_step": state.get("sim_step"),
            }

    # 1. 메인 쿼리 로직
    if dirty is None:
        if chip_prod_id:
            params_chip = {'chip_prod_id': f'%{chip_prod_id}%'}
            print("기종으로 한다", chip_prod_id)
            
            sql = """
                SELECT DISTINCT ON (chip_prod_id) * FROM data_portal.mdh_contiguous_condition_view2 
                WHERE chip_prod_id LIKE %(chip_prod_id)s 
                ORDER BY chip_prod_id, design_input_date DESC;
            """
            results = db.execute_read(sql, params_chip)
        else:
            sql = """
                SELECT DISTINCT ON (chip_prod_id) * FROM data_portal.mdh_contiguous_condition_view2 
                WHERE temperature = %(temperature)s 
                  AND voltage = %(voltage)s 
                  AND size_detail = %(size)s::text 
                  AND base_volume = %(capacity)s 
                ORDER BY chip_prod_id, design_input_date DESC;
            """
            results = db.execute_read(sql, params)
            
    else:
        # dirty 값이 있는 경우 (재검색 로직)
        dirty_list = [dirty] if isinstance(dirty, str) else dirty
        params_chip = {'chip_prod_id': [f'%{item}%' for item in dirty_list]}
        
        print(f"칩기종 변환 요청해서 {dirty}로 칩기종 재검색 드루감")
        
        sql = """
            SELECT DISTINCT ON (chip_prod_id) * FROM data_portal.mdh_contiguous_condition_view2 
            WHERE chip_prod_id LIKE ANY (%(chip_prod_id)s) 
            ORDER BY chip_prod_id, design_input_date DESC;
        """
        results = db.execute_read(sql, params_chip)
        print(f"칩기종 변환 요청 결과: {results}")

    chip_prod_id_list = [row["chip_prod_id"] for row in results]

    # 1-2 단계: 상태를 준비한다.
    if tool_context is not None:
        state = tool_context.state
        # 1-2 단계: 입력 파라미터를 상태에 저장한다.
        state["input_params"] = input_params.model_dump()
        # 1-2 단계: 결과를 상태에 저장한다.
        state["stage_outputs"]["1-2"] = {
            "chip_prod_id_list": chip_prod_id_list,
            "candidate_count": len(chip_prod_id_list),
        }
        state["stage_status"]["1-2"] = "done"
        state["sim_step"] = 3

    # 1-2 단계: 결과가 있으면 요약을 반환한다.
    if results:
        if tool_context is None:
            return results, chip_prod_id_list, None
        return {
            "chip_prod_id_list": chip_prod_id_list[:20],
            "candidate_count": len(chip_prod_id_list),
            "truncated": len(chip_prod_id_list) > 20,
            "gap": None,
        }

    # 2. Fallback 로직 (결과가 없을 경우)
    if not chip_prod_id_list:
        if dirty is None:
            if chip_prod_id:
                # 기종명 기반 Fallback
                chip_prod_id_fullname = chip_prod_id if chip_prod_id.startswith('CL') else 'CL' + chip_prod_id
                voltage_code = chip_prod_id_fullname[9]
                
                reparams_voltage = searcher.get_neighbors(voltage_code)
                search_values = [item['code'] for item in reparams_voltage]
                
                new_chip_prod_id = [
                    chip_prod_id_fullname[:9] + char + chip_prod_id_fullname[10:] 
                    for char in search_values
                ]
                params_chip = {'chip_prod_id': [f'%{item}%' for item in new_chip_prod_id]}
                
                print("기종으로 위아래 조건 찾아서 재검색 한다", new_chip_prod_id)
                
                sql = """
                    SELECT DISTINCT ON (chip_prod_id) * FROM data_portal.mdh_contiguous_condition_view2 
                    WHERE chip_prod_id LIKE ANY (%(chip_prod_id)s) 
                    ORDER BY chip_prod_id, design_input_date DESC;
                """
                fallback_summary = f"해당 인자로 맞는 조건이 없어, 전압조건을 {search_values}으로 확대하여 재검색하였음."
                fallback_results = db.execute_read(sql, params_chip)
            
            else:
                # 파라미터 기반 Fallback
                reparams_voltage = searcher.get_neighbors(voltage)
                search_values = [item['val'] for item in reparams_voltage]
                params['voltage'] = search_values
                
                sql = """
                    SELECT DISTINCT ON (chip_prod_id) * FROM data_portal.mdh_contiguous_condition_view2 
                    WHERE temperature = %(temperature)s 
                      AND voltage = ANY(%(voltage)s) 
                      AND size_detail = %(size)s::text 
                      AND base_volume = %(capacity)s 
                    ORDER BY chip_prod_id, design_input_date DESC;
                """
                fallback_summary = f"해당 인자로 맞는 조건이 없어, 전압조건을 {search_values}으로 확대하여 재검색하였음."
                fallback_results = db.execute_read(sql, params)

            chip_prod_id_list = [row['chip_prod_id'] for row in fallback_results]
            print("chip_prod_id fallback 단계 ", search_values)
            
            if fallback_results:
                gap = {
                    "stage": "1-2",
                    "reason": "no_chip_type_match",
                    "fallback_summary": fallback_summary,
                    "candidate_count": len(fallback_results),
                    "table_key": "chip_type_candidates_table",
                    "id_field": "chip_prod_id",
                    "selection_field": "chip_prod_id",
                    "allow_multi": True,
                }
                if tool_context is not None:
                    state = tool_context.state
                    if "stage_outputs" not in state:
                        state["stage_outputs"] = {}
                    if "stage_status" not in state:
                        state["stage_status"] = {}
                    state["stage_outputs"]["1-2"] = {
                        "chip_prod_id_list": chip_prod_id_list,
                        "candidate_count": len(chip_prod_id_list),
                    }
                    state["stage_status"]["1-2"] = "done"
                    state["last_gap"] = gap
                if tool_context is None:
                    return fallback_results, chip_prod_id_list, gap
                return {
                    "chip_prod_id_list": chip_prod_id_list[:20],
                    "candidate_count": len(chip_prod_id_list),
                    "truncated": len(chip_prod_id_list) > 20,
                    "gap": gap,
                }

        # 최종 결과 없음
        fallback_summary = "해당 인자로 맞는 조건이 없어 전압조건을 확대하였으나 결과가 나오지 않았음."
        gap = {
            "stage": "1-2",
            "reason": "no_chip_type_match",
            "fallback_summary": fallback_summary,
            "candidate_count": 0,
            "table_key": "",
            "id_field": "",
            "selection_field": "",
            "allow_multi": True,
        }
        if tool_context is not None:
            state = tool_context.state
            if "stage_outputs" not in state:
                state["stage_outputs"] = {}
            if "stage_status" not in state:
                state["stage_status"] = {}
            state["stage_outputs"]["1-2"] = {
                "chip_prod_id_list": [],
                "candidate_count": 0,
            }
            state["stage_status"]["1-2"] = "done"
            state["last_gap"] = gap
        if tool_context is None:
            return [], [], gap
        return {
            "chip_prod_id_list": [],
            "candidate_count": 0,
            "truncated": False,
            "gap": gap,
        }
    
def _query_column_label_map() -> list[dict[str, Any]]:
    # 실제 DB 조회 로직을 구현한다.
    # 예시 SQL:
    # SELECT column_key, korean_label
    # FROM column_label_map
    # WHERE column_key IS NOT NULL;
    return []

from typing import Annotated, Optional

def find_ref_lot_candidate(
    tool_context: ToolContext | None,
    chip_prod_id_list: Annotated[Optional[list], Field(description="MLCC Chip production id list")] = None,
):
    """
    1-3 단계 툴.
    사용 시점: sim_step=3일 때.
    입력: chip_prod_id_list
    출력: ref_lot_id 요약 + gap
    예시: {"chip_prod_id_list":[...]} → {"ref_lot_id":"LOT-1", "gap":null}

    :param chip_prod_id_list: find_chip_prod_id를 사용해서 나온 MLCC 칩 기종 LIST.
    :return: 
    """
    # 1-3 단계: 상태를 준비한다.
    state = tool_context.state if tool_context is not None else None
    if state is not None and "stage_outputs" not in state:
        state["stage_outputs"] = {}
    if state is not None and "stage_status" not in state:
        state["stage_status"] = {}
    # 1-3 단계: 입력이 없으면 상태에서 꺼낸다.
    if not chip_prod_id_list and state is not None:
        chip_prod_id_list = state.get("stage_outputs", {}).get("1-2", {}).get("chip_prod_id_list", [])
    # 1-3 단계: 칩기종이 바뀌면 이후 결과를 무효화한다.
    if state is not None and chip_prod_id_list:
        prev_list = state.get("stage_outputs", {}).get("1-2", {}).get("chip_prod_id_list")
        if prev_list and prev_list != chip_prod_id_list:
            _invalidate_from_step(state, 3)
            state["sim_step"] = 3
    # 1-3 단계: sim_step 게이트를 확인한다.
    if state is not None and not _gate_sim_step(state, 3):
        state["pending_action"] = {
            "action": "wait_step",
            "target_step": 3,
            "current_step": state.get("sim_step"),
        }
        return {
            "skipped": True,
            "reason": "step_gate",
            "expected_step": state.get("sim_step"),
        }
    if not chip_prod_id_list:
        if state is not None:
            state["stage_outputs"]["1-3"] = {"ref_lot_id": None, "candidate_count": 0}
            state["stage_status"]["1-3"] = "done"
            state["last_gap"] = {
                "stage": "1-3",
                "reason": "no_chip_type_candidate",
                "fallback_summary": "칩기종 후보가 없어 레퍼런스 LOT를 찾지 못했음.",
                "candidate_count": 0,
            }
            return {"ref_lot_id": None, "candidate_count": 0, "gap": state["last_gap"]}
        return [], {}, None

    # 컬럼 정의
    lot_common_column = ["chip_prod_id", "lot_id", "cur_site_div"]
    lot_defect_column = [
        "design_input_date", "cutting_defect", "measure_defect", "bdv_avg", 
        "x_tr_short_defect_rate", "x_fr_ispass", "contact_defect", "pass_halt", 
        "pass_8585", "pass_burn_in", "x_df_ispass", "x_odb_pass_yn"
    ]
    
    target_columns = lot_common_column + lot_defect_column
    columns_clause = ", ".join(target_columns)

    # SQL 쿼리 구성
    sql = f"""
        SELECT {columns_clause}
        FROM data_portal.mdh_base_view_total_4
        WHERE 
            chip_prod_id = ANY (%s)
            AND SUBSTRING(screen_durable_spec_name, 6, 1) NOT IN ('F', 'L', 'G', 'K', 'E')
            AND SUBSTRING(screen_durable_spec_name, 11, 3) NOT IN ('3DJ', 'VLC', 'RHM', 'EXT', 'MPM', 'SHI')
            AND grinding_l_avg IS NOT NULL
            AND grinding_t_avg IS NOT NULL
            AND electrode_c_avg IS NOT NULL
            AND cast_dsgn_thk IS NOT NULL
            AND ldn_avr_value IS NOT NULL
            AND screen_chip_size_leng IS NOT NULL
            AND screen_mrgn_leng IS NOT NULL
            AND screen_chip_size_widh IS NOT NULL
            AND screen_mrgn_widh IS NOT NULL
            AND cover_sheet_thk IS NOT NULL
            AND top_cover_layer_num IS NOT NULL
            AND bot_cover_layer_num IS NOT NULL
            AND active_layer IS NOT NULL
            AND ni_paste_metal_xrf IS NOT NULL
            AND ni_paste_powder_xrf IS NOT NULL
            AND cutting_defect IN ('S 등급', 'A 등급', 'B 등급')
            AND x_fr_ispass IS DISTINCT FROM 'NG'
            AND (contact_defect = 0 OR contact_defect IS NULL)
            AND measure_defect IN ('S 등급', 'A 등급', 'B 등급')
            AND pass_halt IS DISTINCT FROM 'NG'
            AND pass_8585 IS DISTINCT FROM 'NG'
            AND pass_burn_in IS DISTINCT FROM 'NG'
            AND x_df_ispass IS DISTINCT FROM 'NG'
            AND x_odb_pass_yn IS DISTINCT FROM 'NG'
        ORDER BY 
            array_position(ARRAY['S 등급', 'A 등급', 'B 등급'], cutting_defect),
            array_position(ARRAY['S 등급', 'A 등급', 'B 등급'], measure_defect);
    """

    results = db.execute_read(sql, (chip_prod_id_list,))

    if not results:
        print("이전 쿼리 결과가 없어 상세 조회를 진행할 수 없습니다.")
        if state is not None:
            state["stage_outputs"]["1-3"] = {"ref_lot_id": None, "candidate_count": 0}
            state["stage_status"]["1-3"] = "done"
            state["last_gap"] = {
                "stage": "1-3",
                "reason": "no_ref_lot_candidate",
                "fallback_summary": "레퍼런스 LOT 후보가 없어 다음 단계를 진행할 수 없음.",
                "candidate_count": 0,
            }
            return {"ref_lot_id": None, "candidate_count": 0, "gap": state["last_gap"]}
        return [], {}, None

    ref_lot_candidates_results = [
        {key: row[key] for key in target_columns} for row in results
    ]

    ref_lot_info = ref_lot_candidates_results[0]
    ref_lot_id = ref_lot_info["lot_id"]

    # 1-3 단계: 상태에 요약을 저장한다.
    if state is not None:
        state["stage_outputs"]["1-3"] = {
            "ref_lot_id": ref_lot_id,
            "candidate_count": len(ref_lot_candidates_results),
            "ref_lot_info": ref_lot_info,
        }
        state["stage_status"]["1-3"] = "done"
        state["sim_step"] = 4
        # 1-3 단계: 요약만 반환한다.
        return {
            "ref_lot_id": ref_lot_id,
            "candidate_count": len(ref_lot_candidates_results),
            "gap": None,
        }
    return ref_lot_candidates_results, ref_lot_info, ref_lot_id


def get_first_lot_detail(results, dirty=None, table_name="mdh_base_view_total_4"):
    # 컬럼 정의
    lot_common_column = ["chip_prod_id", "lot_id", "cur_site_div"]
    lot_design_column = ["electrode_c_avg", "app_type", "active_powder_base", "ldn_cv_value", "cast_dsgn_thk"]
    
    target_columns = lot_common_column + lot_design_column
    
    # target_lot_id 결정
    if dirty is not None:
        target_lot_id = dirty
    else:
        try:
            target_lot_id = results[0]['lot_id']
        except (KeyError, IndexError):
            print("결과 데이터가 비어있거나 'lot_id' 컬럼이 없습니다.")
            return None

    print(f"첫 번째 LOT ID: {target_lot_id} 에 대한 전체 정보를 조회합니다.")

    # 상세 정보 조회
    sql_detail = f"SELECT * FROM data_portal.{table_name} WHERE lot_id = %s"
    detail_result = db.execute_read(sql_detail, (target_lot_id,))

    ref_lot_design_info = [
        {key: row[key] for key in target_columns} for row in detail_result
    ]

    return detail_result, ref_lot_design_info
