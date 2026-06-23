"""Durable Telegram delivery outbox.

Streaming delivery is best-effort.  The outbox stores authoritative final
messages on disk so a temporary proxy/Telegram outage does not lose the answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from aiogram.exceptions import TelegramAPIError, TelegramNetworkError

from ductor_bot.messenger.telegram.sender import SendRichOpts, send_rich

if TYPE_CHECKING:
    from aiogram import Bot

logger = logging.getLogger(__name__)

DEFAULT_RETRY_SECONDS = 30.0


def outbox_dir(ductor_home: Path) -> Path:
    return ductor_home / "state" / "telegram_outbox"


def _now() -> float:
    return time.time()


def _item_path(root: Path, delivery_id: str) -> Path:
    return root / f"{delivery_id}.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_item(path: Path) -> dict[str, Any] | None:
    try:
        return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Telegram outbox: failed to read %s", path, exc_info=True)
        return None


def enqueue_text(  # noqa: PLR0913
    root: Path,
    *,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    thread_id: int | None = None,
    allowed_roots: Sequence[Path] | None = None,
    delivery_id: str | None = None,
) -> str | None:
    clean = text.strip()
    if not clean:
        return None

    now = _now()
    did = delivery_id or f"{int(now)}-{uuid.uuid4().hex}"
    item = {
        "delivery_id": did,
        "chat_id": chat_id,
        "reply_to_message_id": reply_to_message_id,
        "thread_id": thread_id,
        "allowed_roots": [str(path) for path in allowed_roots] if allowed_roots else None,
        "text": clean,
        "status": "pending",
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
        "next_retry_at": now,
    }
    _atomic_write_json(_item_path(root, did), item)
    logger.info("Telegram outbox queued delivery_id=%s chat_id=%s", did, chat_id)
    return did


async def drain_once(
    bot: Bot,
    root: Path,
    *,
    limit: int = 20,
    now: float | None = None,
) -> int:
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    current = _now() if now is None else now
    sent = 0
    paths = await asyncio.to_thread(lambda: sorted(root.glob("*.json")))
    if paths:
        logger.info("Telegram outbox drain start pending=%d", len(paths))

    for path in paths[:limit]:
        item = _read_item(path)
        if not item:
            continue
        if float(item.get("next_retry_at") or 0) > current:
            continue

        delivery_id = str(item.get("delivery_id") or path.stem)
        attempts = int(item.get("attempts") or 0) + 1
        item["attempts"] = attempts
        item["updated_at"] = current

        allowed = item.get("allowed_roots")
        allowed_roots = [Path(value) for value in allowed] if isinstance(allowed, list) else None
        opts = SendRichOpts(
            reply_to_message_id=item.get("reply_to_message_id"),
            thread_id=item.get("thread_id"),
            allowed_roots=allowed_roots,
            raise_network_errors=True,
        )

        try:
            await send_rich(bot, int(item["chat_id"]), str(item["text"]), opts)
        except TelegramNetworkError:
            item["status"] = "pending"
            item["next_retry_at"] = current + min(DEFAULT_RETRY_SECONDS * attempts, 300.0)
            _atomic_write_json(path, item)
            logger.warning(
                "Telegram outbox send failed delivery_id=%s attempts=%d; retry_at=%.0f",
                delivery_id,
                attempts,
                item["next_retry_at"],
            )
            continue
        except TelegramAPIError:
            item["status"] = "pending"
            item["next_retry_at"] = current + min(DEFAULT_RETRY_SECONDS * attempts, 300.0)
            _atomic_write_json(path, item)
            logger.warning(
                "Telegram outbox API error delivery_id=%s attempts=%d; retry_at=%.0f",
                delivery_id,
                attempts,
                item["next_retry_at"],
                exc_info=True,
            )
            continue
        except Exception:
            item["status"] = "pending"
            item["next_retry_at"] = current + min(DEFAULT_RETRY_SECONDS * attempts, 300.0)
            _atomic_write_json(path, item)
            logger.warning(
                "Telegram outbox unexpected error delivery_id=%s",
                delivery_id,
                exc_info=True,
            )
            continue

        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        sent += 1
        logger.info("Telegram outbox sent delivery_id=%s attempts=%d", delivery_id, attempts)

    if paths:
        remaining = await asyncio.to_thread(lambda: len(list(root.glob("*.json"))))
        logger.info("Telegram outbox drain finish sent=%d remaining=%d", sent, remaining)
    return sent


async def drain_loop(bot: Bot, root: Path, *, interval_seconds: float = 30.0) -> None:
    logger.info("Telegram outbox drain loop started path=%s interval=%.1f", root, interval_seconds)
    try:
        while True:
            with contextlib.suppress(Exception):
                await drain_once(bot, root)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.debug("Telegram outbox drain loop cancelled")
        raise
