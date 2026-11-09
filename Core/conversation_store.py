"""Small, private SQLite store for recently completed conversation turns.

No book passages, provider keys, graph state, or raw prompts are persisted here.
The feature is optional; a configured path enables it. Access control belongs
at the API boundary, not in possession of a predictable session ID.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class ConversationStore:
    """SQLite-backed, bounded chat memory with inactivity expiry."""

    def __init__(self, path: str | Path, *, max_turns: int = 4,
                 ttl_seconds: int = 7 * 24 * 60 * 60):
        if max_turns < 1 or max_turns > 100:
            raise ValueError("max_turns must be between 1 and 100")
        if ttl_seconds < 60:
            raise ValueError("ttl_seconds must be at least 60")
        self.path = Path(path)
        if self.path.is_dir():
            raise ValueError("Conversation database path must be a file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    session_id TEXT PRIMARY KEY,
                    turns TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
            """)
            db.execute(
                "CREATE INDEX IF NOT EXISTS ix_conversation_expiry "
                "ON conversation_memory(updated_at)"
            )

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.execute("PRAGMA busy_timeout=10000")
        return db

    def _purge(self, db, now: int) -> None:
        db.execute(
            "DELETE FROM conversation_memory WHERE updated_at < ?",
            (now - self.ttl_seconds,),
        )

    @staticmethod
    def _valid_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not (8 <= len(session_id) <= 128):
            raise ValueError("Invalid session identifier")
        if not all(c.isalnum() or c in "-_" for c in session_id):
            raise ValueError("Invalid session identifier")
        if session_id.lower() == "default":
            raise ValueError("Shared default session is not permitted")
        return session_id

    def get(self, session_id: str) -> list[dict]:
        session_id = self._valid_id(session_id)
        now = int(time.time())
        with self._lock, self._connect() as db:
            self._purge(db, now)
            row = db.execute(
                "SELECT turns FROM conversation_memory WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if not row:
            return []
        turns = json.loads(row[0])
        return [dict(t) for t in turns] if isinstance(turns, list) else []

    def save(self, session_id: str, turns: list[dict]) -> None:
        session_id = self._valid_id(session_id)
        entries = [
            {"query": str(t.get("query", "")), "response": str(t.get("response", ""))}
            for t in turns[-self.max_turns:]
            if isinstance(t, dict)
        ]
        if not entries:
            self.clear(session_id)
            return
        payload = json.dumps(entries, ensure_ascii=False)
        now = int(time.time())
        with self._lock, self._connect() as db:
            self._purge(db, now)
            db.execute(
                "INSERT INTO conversation_memory(session_id, turns, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "turns=excluded.turns, updated_at=excluded.updated_at",
                (session_id, payload, now),
            )

    def clear(self, session_id: str) -> None:
        session_id = self._valid_id(session_id)
        with self._lock, self._connect() as db:
            db.execute(
                "DELETE FROM conversation_memory WHERE session_id=?",
                (session_id,),
            )
