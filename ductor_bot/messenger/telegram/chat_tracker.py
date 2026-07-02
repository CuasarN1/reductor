"""Lightweight chat activity tracker for /where visibility.

Tracks group joins/leaves (via ``my_chat_member`` events), rejected
group access attempts (via AuthMiddleware callback), and private chat
activity.  Persists to ``chat_activity.json`` in the ductor home.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ductor_bot.infra.json_store import atomic_json_save, load_json

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ChatRecord:
    """A single tracked chat/group."""

    chat_id: int
    chat_type: str = "private"  # "private" | "group" | "supergroup"
    title: str = ""
    first_seen: str = field(default_factory=_now_iso)
    last_seen: str = field(default_factory=_now_iso)
    status: str = "active"  # "active" | "left" | "kicked" | "auto_left"
    allowed: bool = True
    rejected_count: int = 0


@dataclass
class UserRecord:
    """A Telegram user the bot has seen in private/group updates."""

    user_id: int
    username: str = ""
    first_name: str = ""
    last_name: str = ""
    first_seen: str = field(default_factory=_now_iso)
    last_seen: str = field(default_factory=_now_iso)
    allowed: bool = False


class ChatTracker:
    """In-memory tracker backed by a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[int, ChatRecord] = {}
        self._users: dict[int, UserRecord] = {}
        self._load()

    # -- Public API -----------------------------------------------------------

    def record_join(
        self,
        chat_id: int,
        chat_type: str,
        title: str,
        *,
        allowed: bool,
    ) -> None:
        """Record a group join from ``my_chat_member``."""
        existing = self._records.get(chat_id)
        now = _now_iso()
        if existing:
            existing.chat_type = chat_type
            existing.title = title or existing.title
            existing.last_seen = now
            existing.status = "active"
            existing.allowed = allowed
        else:
            self._records[chat_id] = ChatRecord(
                chat_id=chat_id,
                chat_type=chat_type,
                title=title,
                first_seen=now,
                last_seen=now,
                status="active",
                allowed=allowed,
            )
        self._save()

    def record_leave(self, chat_id: int, status: str = "left") -> None:
        """Record a group leave/kick from ``my_chat_member`` or ``/leave``."""
        existing = self._records.get(chat_id)
        now = _now_iso()
        if existing:
            existing.status = status
            existing.last_seen = now
        else:
            self._records[chat_id] = ChatRecord(
                chat_id=chat_id,
                first_seen=now,
                last_seen=now,
                status=status,
            )
        self._save()

    def record_rejected(self, chat_id: int, chat_type: str, title: str) -> None:
        """Record a rejected group message from AuthMiddleware."""
        existing = self._records.get(chat_id)
        now = _now_iso()
        if existing:
            existing.rejected_count += 1
            existing.last_seen = now
            existing.title = title or existing.title
        else:
            self._records[chat_id] = ChatRecord(
                chat_id=chat_id,
                chat_type=chat_type,
                title=title,
                first_seen=now,
                last_seen=now,
                status="rejected",
                allowed=False,
                rejected_count=1,
            )
            self._save()
            return
        self._save()

    def get_all(self) -> list[ChatRecord]:
        """Return all records sorted by last_seen (newest first)."""
        return sorted(self._records.values(), key=lambda r: r.last_seen, reverse=True)

    def record_user(
        self,
        user_id: int,
        *,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
        allowed: bool = False,
    ) -> None:
        """Record a Telegram user so admins can later grant access by username."""
        existing = self._users.get(user_id)
        now = _now_iso()
        if existing:
            existing.username = username or existing.username
            existing.first_name = first_name or existing.first_name
            existing.last_name = last_name or existing.last_name
            existing.last_seen = now
            existing.allowed = allowed or existing.allowed
        else:
            self._users[user_id] = UserRecord(
                user_id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                first_seen=now,
                last_seen=now,
                allowed=allowed,
            )
        self._save()

    def resolve_username(self, username: str) -> int | None:
        """Return a user id for a previously seen Telegram username."""
        wanted = username.lstrip("@").lower()
        if not wanted:
            return None
        for rec in self._users.values():
            if rec.username.lower() == wanted:
                return rec.user_id
        return None

    def get_user(self, user_id: int) -> UserRecord | None:
        """Return a previously seen Telegram user record by id."""
        return self._users.get(user_id)

    # -- Persistence ----------------------------------------------------------

    def _load(self) -> None:
        raw = load_json(self._path)
        if not isinstance(raw, dict):
            return
        records: dict[str, Any] = raw.get("records", {})
        for key, val in records.items():
            if isinstance(val, dict) and "chat_id" in val:
                self._records[int(key)] = ChatRecord(**val)
        users: dict[str, Any] = raw.get("users", {})
        for key, val in users.items():
            if isinstance(val, dict) and "user_id" in val:
                self._users[int(key)] = UserRecord(**val)

    def _save(self) -> None:
        data = {
            "records": {str(k): asdict(v) for k, v in self._records.items()},
            "users": {str(k): asdict(v) for k, v in self._users.items()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_save(self._path, data)
