from typing import Any, Callable
import asyncio
import json
import threading

from .schemas import InputParams
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

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


def _query_column_label_map() -> list[dict[str, Any]]:
    # 실제 DB 조회 로직을 구현한다.
    # 예시 SQL:
    # SELECT column_key, korean_label
    # FROM column_label_map
    # WHERE column_key IS NOT NULL;
    return []


def build_simulation_from_db(
    input_params: InputParams,
    configs: dict[str, Any],
    selections: dict[str, Any],
    user_prefs: dict[str, Any],
    dirty_stages: list[str] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, Any] | None]:
    # 테이블 컨테이너를 준비한다.
    tables: dict[str, Any] = {}
    # 차트 컨테이너를 준비한다.
    charts: list[dict[str, Any]] = []
    # 단계 노트 컨테이너를 준비한다.
    stage_notes: dict[str, str] = {}
    # dirty 스테이지 집합을 준비한다.
    dirty_set = set(dirty_stages or [])
    # 데이터 공백 정보를 준비한다.
    gap: dict[str, Any] | None = None

    def _should_run(stage: str) -> bool:
        # dirty 정보가 없으면 전체 실행한다.
        if not dirty_set:
            return True
        # dirty에 포함된 단계만 실행한다.
        return stage in dirty_set

    def _emit_progress(stage: str) -> None:
        # 프론트 로깅 콜백이 있으면 단계 진행을 알린다.
        if not progress_cb:
            return
        # 콜백은 SSE에서 stage별 progress 이벤트를 보내도록 구현한다.
        progress_cb(stage)

    # 1-1: input_params_table
    if _should_run("1-1"):
        # 1-1 진행 로그를 보낸다.
        _emit_progress("1-1")
        # TODO: 입력값을 테이블로 정규화한다.
        # tables["input_params_table"] = [...]
        # stage_notes["1-1"] = "..."

    # 1-2: chip_type_candidates_table
    if _should_run("1-2"):
        # 1-2 진행 로그를 보낸다.
        _emit_progress("1-2")
        # 칩기종 후보 조회를 수행한다.
        chip_rows, chip_gap = find_chip_prod_id(input_params)
        # 조회 결과를 테이블에 넣는다.
        if chip_rows:
            tables["chip_type_candidates_table"] = chip_rows
        # gap이 있으면 여기서 멈춘다.
        if chip_gap:
            gap = chip_gap
            return tables, charts, stage_notes, gap

    # 1-3: reference_lot_candidates_table, reference_lot_table
    if _should_run("1-3"):
        # 1-3 진행 로그를 보낸다.
        _emit_progress("1-3")
        # TODO: 레퍼런스 LOT 후보와 선택 테이블을 만든다.
        # tables["reference_lot_candidates_table"] = [...]
        # tables["reference_lot_table"] = [...]
        # stage_notes["1-3"] = "..."

    # 1-4: payload 구성 근거(표는 reference_lot_table 사용)
    if _should_run("1-4"):
        # 1-4 진행 로그를 보낸다.
        _emit_progress("1-4")
        # TODO: stage_notes만 필요하면 여기서 작성한다.
        # stage_notes["1-4"] = "..."

    # 1-5: top_k_table
    if _should_run("1-5"):
        # 1-5 진행 로그를 보낸다.
        _emit_progress("1-5")
        # TODO: top-k 결과 테이블을 만든다.
        # tables["top_k_table"] = [...]
        # stage_notes["1-5"] = "..."

    # 1-6: recent_similar_table
    if _should_run("1-6"):
        # 1-6 진행 로그를 보낸다.
        _emit_progress("1-6")
        # TODO: 최근 유사 설계 테이블을 만든다.
        # tables["recent_similar_table"] = [...]
        # stage_notes["1-6"] = "..."

    # 1-7: defect_rate_table, defect_rate_summary
    if _should_run("1-7"):
        # 1-7 진행 로그를 보낸다.
        _emit_progress("1-7")
        # TODO: 불량률 테이블과 차트를 만든다.
        # tables["defect_rate_table"] = [...]
        # charts.append({...})
        # stage_notes["1-7"] = "..."

    # 1-8: 브리핑 근거 노트
    if _should_run("1-8"):
        # 1-8 진행 로그를 보낸다.
        _emit_progress("1-8")
        # TODO: 브리핑 근거 노트를 만든다.
        # stage_notes["1-8"] = "..."
    return tables, charts, stage_notes, gap


def find_chip_prod_id(params: InputParams) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    # 기본 조건 조회 쿼리를 준비한다.
    query = """
    SELECT ~~~
    FROM ~~~
    WHERE ~~~
    """
    # 기본 조건 조회를 실행한다.
    result = db.execute_read(query, params)
    # 기본 조건 결과가 있으면 바로 반환한다.
    if result:
        return result, None, "입력 조건에 맞는 칩기종 후보를 찾았습니다."
    # 대체 조건 쿼리를 준비한다.
    fallback_query = """
    SELECT ~~~
    FROM ~~~
    WHERE ~~~
    """
    # 대체 조건 요약을 준비한다.
    fallback_summary = "조건 일부를 완화"
    # 대체 조건 조회를 실행한다.
    fallback_rows = db.execute_read(fallback_query, params)
    # 대체 조건 결과가 있으면 gap을 만들어 반환한다.
    if fallback_rows:
        gap = {
            "stage": "1-2",
            "reason": "no_chip_type_match",
            "fallback_summary": fallback_summary,
            "candidate_count": len(fallback_rows),
            "table_key": "chip_type_candidates_table",
            "id_field": "chip_type_id",
            "selection_field": "chip_type_ids",
            "allow_multi": True,
        }
        return fallback_rows, gap
    # 대체 조건도 없으면 빈 결과로 반환한다.
    return [], None
