import aiosqlite
from api.config import DB_PATH

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS swing_sessions (
    id           TEXT PRIMARY KEY,
    captured_at  TEXT NOT NULL,
    sample_count INTEGER,
    swing_count  INTEGER
)
"""

CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS swing_results (
    id                 TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    swing_index        INTEGER NOT NULL,
    captured_at        TEXT NOT NULL,
    verdict            TEXT NOT NULL,
    label              INTEGER NOT NULL,
    user_label         INTEGER,
    confidence         REAL,
    features_json      TEXT,
    importances_json   TEXT,
    phase_timing_json  TEXT,
    FOREIGN KEY (session_id) REFERENCES swing_sessions(id)
)
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_SESSIONS)
        await db.execute(CREATE_RESULTS)
        await db.commit()


async def get_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db
