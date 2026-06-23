"""Tests for the durable Telegram outbox."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aiogram.exceptions import TelegramNetworkError


async def test_drain_once_sends_and_removes_item(tmp_path: Path) -> None:
    from ductor_bot.messenger.telegram.outbox import drain_once, enqueue_text

    bot = MagicMock()
    sent = MagicMock()
    sent.message_id = 42
    bot.send_message = AsyncMock(return_value=sent)

    delivery_id = enqueue_text(tmp_path, chat_id=1, text="hello")

    assert delivery_id is not None
    assert (tmp_path / f"{delivery_id}.json").exists()

    sent_count = await drain_once(bot, tmp_path)

    assert sent_count == 1
    assert not (tmp_path / f"{delivery_id}.json").exists()
    bot.send_message.assert_awaited_once()


async def test_drain_once_keeps_item_after_network_error(tmp_path: Path) -> None:
    from ductor_bot.messenger.telegram.outbox import drain_once, enqueue_text

    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=TelegramNetworkError(MagicMock(), "proxy unavailable"),
    )

    delivery_id = enqueue_text(tmp_path, chat_id=1, text="hello")

    assert delivery_id is not None
    path = tmp_path / f"{delivery_id}.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    sent_count = await drain_once(bot, tmp_path, now=before["next_retry_at"])

    assert sent_count == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["attempts"] == 1
    assert payload["next_retry_at"] == before["next_retry_at"] + 30.0
