"""Persistent LangGraph checkpoint storage for multi-turn conversations."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

_PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = _PROJECT_DIR / ".checkpoints"
CHECKPOINT_DB_PATH = CHECKPOINT_DIR / "conversations.db"

_conn: sqlite3.Connection | None = None
_saver: SqliteSaver | None = None


def normalize_session_id(session: str) -> str:
    """Turn a user session name into a safe LangGraph thread_id."""
    safe = re.sub(r"[^\w\-]", "_", session.strip())
    return safe or "default"


def get_checkpointer() -> SqliteSaver:
    """SQLite checkpointer shared across app restarts (same path = same sessions)."""
    global _conn, _saver
    if _saver is None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
        _saver = SqliteSaver(_conn)
        _saver.setup()
    return _saver
