"""Tests for Telegram upgrade/update notification UI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from ductor_bot.infra.version import VersionInfo
from ductor_bot.messenger.telegram.upgrade_handler import on_update_available


async def test_github_update_notification_links_release_without_pypi_upgrade() -> None:
    bot = MagicMock()
    bot.notify_upgrade = AsyncMock()
    info = VersionInfo(
        current="0.18.12",
        latest="0.19.0",
        update_available=True,
        summary="",
        source="github",
        release_url="https://github.com/CuasarN1/reductor/releases/tag/v0.19.0",
        source_repo="CuasarN1/reductor",
    )

    await on_update_available(bot, info)

    text, opts = bot.notify_upgrade.await_args.args
    assert "CuasarN1/reductor" in text
    keyboard = opts.reply_markup
    buttons = [button for row in keyboard.inline_keyboard for button in row]
    assert buttons[0].url == info.release_url
    assert all(button.callback_data != "upg:yes:0.19.0" for button in buttons)


async def test_pypi_update_notification_keeps_upgrade_callback() -> None:
    bot = MagicMock()
    bot.notify_upgrade = AsyncMock()
    info = VersionInfo(
        current="1.0.0",
        latest="2.0.0",
        update_available=True,
        summary="",
    )

    await on_update_available(bot, info)

    _text, opts = bot.notify_upgrade.await_args.args
    buttons = [button for row in opts.reply_markup.inline_keyboard for button in row]
    assert any(button.callback_data == "upg:yes:2.0.0" for button in buttons)
