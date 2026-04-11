"""
core/database.py
────────────────
SQLite-backed chat history.

Each RAG gets its own chat.db file at store/{rag_id}/chat.db.
The schema is intentionally minimal — sessions and messages only.

Chat history is ephemeral by design: when the Docker container is
destroyed and re-launched, the store/ directory is reset and all
history is lost.  No Docker volume is mounted.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

_STORE_ROOT = os.environ.get("STORE_PATH", "./store")


def get_db_path(rag_id: str) -> str:
    """Return the absolute path of the SQLite file for this RAG."""
    return os.path.join(_STORE_ROOT, rag_id, "chat.db")


def _connect(rag_id: str) -> sqlite3.Connection:
    """Open (and return) a connection to this RAG's SQLite database."""
    path = get_db_path(rag_id)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(rag_id: str) -> None:
    """
    Create the sessions and messages tables if they do not yet exist.

    Safe to call multiple times (idempotent due to IF NOT EXISTS).

    Args:
        rag_id: The RAG whose database should be initialised.
    """
    conn = _connect(rag_id)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,   -- "user" or "assistant"
                content    TEXT    NOT NULL,
                sources    TEXT    DEFAULT '[]', -- JSON array of source dicts
                timestamp  TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_session(rag_id: str) -> str:
    """
    Create a new chat session and return its id.

    Args:
        rag_id: The RAG this session belongs to.

    Returns:
        A new UUID4 session id string.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(rag_id)
    try:
        conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def save_message(
    rag_id: str,
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[dict]] = None,
) -> None:
    """
    Persist one chat message.

    Args:
        rag_id:     The RAG this message belongs to.
        session_id: The session this message belongs to.
        role:       "user" or "assistant".
        content:    The text of the message.
        sources:    List of source dicts (for assistant messages).
    """
    import json

    now = datetime.now(timezone.utc).isoformat()
    sources_json = json.dumps(sources or [])
    conn = _connect(rag_id)
    try:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, sources, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, sources_json, now),
        )
        conn.commit()
    finally:
        conn.close()


def load_session_messages(rag_id: str, session_id: str) -> List[dict]:
    """
    Return all messages for a session, ordered by id (oldest first).

    Args:
        rag_id:     The RAG whose database to query.
        session_id: The session whose messages to return.

    Returns:
        List of dicts with keys: id, session_id, role, content, sources, timestamp.
        sources is already parsed from JSON to a Python list.
    """
    import json

    conn = _connect(rag_id)
    try:
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, sources, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    result = []
    for row in rows:
        msg = dict(row)
        msg["sources"] = json.loads(msg["sources"] or "[]")
        result.append(msg)
    return result


def list_sessions(rag_id: str) -> List[dict]:
    """
    Return all chat sessions for a RAG, newest first.

    Each entry includes: id, created_at, message_count.

    Args:
        rag_id: The RAG whose sessions to list.

    Returns:
        List of session summary dicts.
    """
    conn = _connect(rag_id)
    try:
        rows = conn.execute(
            """
            SELECT s.id, s.created_at,
                   COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
            """,
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]
