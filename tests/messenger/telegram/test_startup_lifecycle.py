"""Tests for startup lifecycle detection and auto-recovery in _on_startup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from ductor_bot.infra.startup_state import StartupInfo, StartupKind
from ductor_bot.text.response_format import (
    recovery_notification_text,
    startup_notification_text,
)


class TestStartupNotification:
    def test_first_start_produces_message(self) -> None:
        text = startup_notification_text(StartupKind.FIRST_START.value)
        assert text
        assert "First start" in text

    def test_reboot_produces_message(self) -> None:
        text = startup_notification_text(StartupKind.SYSTEM_REBOOT.value)
        assert text
        assert "reboot" in text.lower()

    def test_restart_is_silent(self) -> None:
        text = startup_notification_text(StartupKind.SERVICE_RESTART.value)
        assert text == ""


async def test_broadcast_restart_sentinel_uses_startup_routing(tmp_path: Path) -> None:
    from ductor_bot.infra.restart import write_restart_sentinel
    from ductor_bot.messenger.telegram.startup import _handle_restart_sentinel

    sentinel = tmp_path / "restart-sentinel.json"
    write_restart_sentinel(
        chat_id=0,
        message="Bot is back.",
        sentinel_path=sentinel,
        broadcast=True,
    )
    bot = MagicMock()
    bot._orch.paths.ductor_home = tmp_path
    bot.notify_startup = AsyncMock()
    bot.notification_service.notify_all = AsyncMock()
    bot.notification_service.notify = AsyncMock()

    result = await _handle_restart_sentinel(bot)

    assert result is not None
    bot.notify_startup.assert_awaited_once_with("Bot is back.")
    bot.notification_service.notify_all.assert_not_called()
    bot.notification_service.notify.assert_not_called()


class TestRecoveryNotification:
    def test_foreground_recovery(self) -> None:
        text = recovery_notification_text("foreground", "fix the auth bug")
        assert "Interrupted" in text
        assert "fix the auth bug" in text

    def test_named_session_recovery(self) -> None:
        text = recovery_notification_text("named_session", "deploy", "redowl")
        assert "redowl" in text
        assert "deploy" in text


class TestStartupInfo:
    def test_round_trip(self) -> None:
        info = StartupInfo(
            kind=StartupKind.FIRST_START,
            boot_id="abc-123",
            started_at="2026-03-02T12:00:00+00:00",
        )
        assert info.kind == StartupKind.FIRST_START
        assert info.boot_id == "abc-123"
