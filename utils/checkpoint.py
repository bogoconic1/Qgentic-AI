import sqlite3
import json
import time
from pathlib import Path

_JSONB_FIELDS = [
    "version_scores",
    "successful_versions",
    "blacklisted_versions",
    "blacklisted_ideas",
    "successful_ideas",
    "global_suggestions",
    "input_list",
]

_COL_NAMES = [
    "id",
    "slug",
    "iteration",
    "model_name",
    "version",
    "best_score",
    "best_version",
    "best_code_file",
    "version_scores",
    "successful_versions",
    "blacklisted_versions",
    "blacklisted_ideas",
    "successful_ideas",
    "global_suggestions",
    "input_list",
    "last_input_tokens",
    "last_suggestion",
    "sota_suggestions_call_id",
    "created_at",
]

_SELECT_COLS = ", ".join(
    f"json({col})" if col in _JSONB_FIELDS else col for col in _COL_NAMES
)

# Resolve DB path relative to project root (Qgentic-AI/checkpoints.db)
_DB_PATH = Path(__file__).resolve().parent.parent / "checkpoints.db"


def create_db(db_path: str | Path = _DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            iteration TEXT NOT NULL,
            model_name TEXT NOT NULL,
            version INTEGER NOT NULL,
            best_score REAL,
            best_version INTEGER,
            best_code_file TEXT,
            version_scores BLOB,
            successful_versions BLOB,
            blacklisted_versions BLOB,
            blacklisted_ideas BLOB,
            successful_ideas BLOB,
            global_suggestions BLOB,
            input_list BLOB,
            last_input_tokens INTEGER,
            last_suggestion TEXT,
            sota_suggestions_call_id INTEGER,
            created_at REAL NOT NULL,
            UNIQUE(slug, iteration, version)
        )
    """)
    conn.commit()
    return conn


def save_checkpoint(
    conn: sqlite3.Connection,
    slug: str,
    iteration: str,
    model_name: str,
    version: int,
    state: dict,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO checkpoints (
            slug, iteration, model_name, version,
            best_score, best_version, best_code_file,
            version_scores, successful_versions, blacklisted_versions,
            blacklisted_ideas, successful_ideas, global_suggestions,
            input_list, last_input_tokens, last_suggestion,
            sota_suggestions_call_id,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, jsonb(?), jsonb(?), jsonb(?), jsonb(?), jsonb(?), jsonb(?), jsonb(?), ?, ?, ?, ?)
    """,
        (
            slug,
            iteration,
            model_name,
            version,
            state["best_score"],
            state["best_version"],
            state["best_code_file"],
            json.dumps(state["version_scores"]),
            json.dumps(state["successful_versions"]),
            json.dumps(state["blacklisted_versions"]),
            json.dumps(state["blacklisted_ideas"]),
            json.dumps(state["successful_ideas"]),
            json.dumps(state["global_suggestions"]),
            json.dumps(state["input_list"]),
            state["last_input_tokens"],
            state["last_suggestion"],
            state["sota_suggestions_call_id"],
            time.time(),
        ),
    )
    conn.commit()


def load_checkpoint(
    conn: sqlite3.Connection, slug: str, iteration: str, version: int
) -> dict | None:
    """Load a specific version's checkpoint (for rollback)."""
    row = conn.execute(
        f"""
        SELECT {_SELECT_COLS} FROM checkpoints
        WHERE slug = ? AND iteration = ? AND version = ?
    """,
        (slug, iteration, version),
    ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def load_latest_checkpoint(
    conn: sqlite3.Connection, slug: str, iteration: str
) -> dict | None:
    """Load the most recent checkpoint (for resume)."""
    row = conn.execute(
        f"""
        SELECT {_SELECT_COLS} FROM checkpoints
        WHERE slug = ? AND iteration = ?
        ORDER BY version DESC LIMIT 1
    """,
        (slug, iteration),
    ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def delete_checkpoints_after(
    conn: sqlite3.Connection, slug: str, iteration: str, version: int
) -> None:
    """Delete all checkpoints after a given version (for rollback)."""
    conn.execute(
        """
        DELETE FROM checkpoints
        WHERE slug = ? AND iteration = ? AND version > ?
    """,
        (slug, iteration, version),
    )
    conn.commit()


def _row_to_dict(row) -> dict:
    d = dict(zip(_COL_NAMES, row))
    for field in _JSONB_FIELDS:
        d[field] = json.loads(d[field])
    return d
