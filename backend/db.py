import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from .config import DB_PATH


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS process_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line TEXT NOT NULL,
    status TEXT NOT NULL,
    temperature REAL NOT NULL,
    voltage REAL NOT NULL,
    size REAL NOT NULL,
    capacity REAL NOT NULL,
    timestamp TEXT NOT NULL
)
"""


def init_db() -> None:
    path = Path(DB_PATH)
    conn = sqlite3.connect(path)
    try:
        conn.execute(SCHEMA_SQL)
        if _needs_seed(conn):
            _seed(conn)
            conn.commit()
    finally:
        conn.close()


def query_process_data(search: str = "", limit: int = 12) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if search:
            like = f"%{search}%"
            rows = conn.execute(
                """
                SELECT id, line, status, temperature, voltage, size, capacity, timestamp
                FROM process_records
                WHERE line LIKE ? OR status LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (like, like, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, line, status, temperature, voltage, size, capacity, timestamp
                FROM process_records
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _needs_seed(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) FROM process_records").fetchone()
    return row[0] == 0


def _seed(conn: sqlite3.Connection) -> None:
    now = datetime.utcnow()
    rows: List[Tuple[str, str, float, float, float, float, str]] = []
    for i in range(12):
        line = f"Line-{chr(65 + (i % 3))}"
        status = "OK" if i % 4 else "WARN"
        temperature = 110.0 + (i * 2.0)
        voltage = 3.5 + (i * 0.05)
        size = 10.0 + (i % 5) * 0.5
        capacity = 5.0 + (i % 4) * 0.25
        timestamp = (now - timedelta(minutes=15 * i)).isoformat() + "Z"
        rows.append((line, status, temperature, voltage, size, capacity, timestamp))

    conn.executemany(
        """
        INSERT INTO process_records
            (line, status, temperature, voltage, size, capacity, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
