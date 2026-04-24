"""
SessionStore — Session metadata persistence

Persists research session metadata (config, approval, conversation_history, status)
to a JSON file so it can be recovered after server restarts.

Structure:
  data/sessions.json  →  { session_id: { config, status, approval, conversation_history } }

Design principles:
  - asyncio.Lock protects against concurrent writes
  - in-memory dict used as the primary store (fast reads)
  - JSON file is flushed on every change
  - Automatically recovered from JSON on server startup
"""

from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path


class SessionStore:
    """
    Persistent session metadata store.
    Used like a dict, but automatically saves to file on changes.
    """

    def __init__(self, path: str = "./data/sessions.json"):
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._data: dict[str, dict] = self._load_sync()

    # ── Initialization ────────────────────────────────────────────────

    def _load_sync(self) -> dict[str, dict]:
        """Synchronous load on server startup (used in lifespan)."""
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    # ── Write ──────────────────────────────────────────────────────────

    async def save(self, session_id: str, session: dict) -> None:
        """Upsert session data to store + flush to file."""
        # conversation_history can be large, so save at most 20 turns
        save_data = {k: v for k, v in session.items() if k != "conversation_history"}
        history = session.get("conversation_history", [])
        save_data["conversation_history"] = history[-40:]   # up to 20 turns

        async with self._lock:
            self._data[session_id] = save_data
            await self._flush()

    async def _flush(self) -> None:
        """in-memory → JSON file (called inside Lock)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)   # atomic rename

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._data.pop(session_id, None)
            await self._flush()

    # ── Read ──────────────────────────────────────────────────────────

    def get(self, session_id: str) -> dict | None:
        return self._data.get(session_id)

    def all_sessions(self) -> dict[str, dict]:
        return dict(self._data)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._data

    def __len__(self) -> int:
        return len(self._data)
