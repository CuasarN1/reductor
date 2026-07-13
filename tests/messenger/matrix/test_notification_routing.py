"""Unit tests for Matrix startup/upgrade notification routing.

Parity coverage with ``tests/messenger/telegram/test_notification_routing.py``
so Matrix honours ``notifications.upgrade_targets`` separately from
``startup_targets`` and treats explicit-disabled target lists as silence
(no broadcast fallback).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ductor_bot.config import (
    AgentConfig,
    MatrixConfig,
    NotificationsConfig,
    NotificationTarget,
)


def _make_bot(
    *,
    startup_targets: list[NotificationTarget] | None = None,
    upgrade_targets: list[NotificationTarget] | None = None,
) -> tuple[Any, AsyncMock, AsyncMock, AsyncMock]:
    """Construct a MatrixBot-like object without heavy __init__.

    Returns ``(bot, notify_mock, notify_all_mock, broadcast_mock)``.
    """
    from ductor_bot.messenger.matrix import bot as bot_module

    cfg = AgentConfig(
        telegram_token="test-token",
        matrix=MatrixConfig(
            homeserver="https://example.invalid",
            user_id="@test:example.invalid",
            access_token="dummy",
        ),
        notifications=NotificationsConfig(
            startup_targets=startup_targets or [],
            upgrade_targets=upgrade_targets or [],
        ),
    )

    bot = bot_module.MatrixBot.__new__(bot_module.MatrixBot)
    bot._config = cfg

    notify_mock = AsyncMock()
    notify_all_mock = AsyncMock()
    broadcast_mock = AsyncMock()

    ns = MagicMock()
    ns.notify = notify_mock
    ns.notify_all = notify_all_mock
    bot._notification_service = ns
    # broadcast is a bound method on MatrixBot; replace for isolation.
    bot.broadcast = broadcast_mock  # type: ignore[method-assign]

    return bot, notify_mock, notify_all_mock, broadcast_mock


async def test_notify_startup_falls_back_when_no_targets() -> None:
    bot, notify, notify_all, _ = _make_bot()

    await bot.notify_startup("hello")

    notify_all.assert_awaited_once_with("hello")
    notify.assert_not_called()


async def test_notify_startup_uses_configured_targets() -> None:
    bot, notify, notify_all, _ = _make_bot(
        startup_targets=[NotificationTarget(enabled=True, chat_id=9001, topic_id=None)],
    )

    await bot.notify_startup("startup")

    notify_all.assert_not_called()
    notify.assert_awaited_once_with(9001, "startup")


async def test_notify_upgrade_uses_configured_targets_not_startup_targets() -> None:
    """Regression for v0.16.1 MED #2: upgrade events must read upgrade_targets."""
    bot, notify, _, broadcast = _make_bot(
        startup_targets=[NotificationTarget(enabled=True, chat_id=1111, topic_id=None)],
        upgrade_targets=[NotificationTarget(enabled=True, chat_id=2222, topic_id=None)],
    )

    await bot.notify_upgrade("update available")

    broadcast.assert_not_called()
    notify.assert_awaited_once_with(2222, "update available")


async def test_notify_upgrade_falls_back_to_broadcast_when_no_targets() -> None:
    bot, notify, _, broadcast = _make_bot()

    await bot.notify_upgrade("update available")

    broadcast.assert_awaited_once_with("update available")
    notify.assert_not_called()


async def test_notify_startup_silences_when_all_targets_disabled() -> None:
    """Regression for v0.16.1 MED #3.

    Users who list targets but disable them all intend silence — do not
    fall through to ``notify_all`` (that is the opposite of silence).
    """
    bot, notify, notify_all, _ = _make_bot(
        startup_targets=[NotificationTarget(enabled=False, chat_id=123, topic_id=None)],
    )

    await bot.notify_startup("startup note")

    notify_all.assert_not_called()
    notify.assert_not_called()


async def test_notify_upgrade_silences_when_all_targets_disabled() -> None:
    """Regression for v0.16.1 MED #3 — mirror of the startup case."""
    bot, notify, _, broadcast = _make_bot(
        upgrade_targets=[NotificationTarget(enabled=False, chat_id=456, topic_id=None)],
    )

    await bot.notify_upgrade("update available")

    broadcast.assert_not_called()
    notify.assert_not_called()


async def test_matrix_startup_wires_notify_upgrade_on_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The update observer callback must call notify_upgrade (not notify_startup)."""
    from ductor_bot.messenger.matrix import startup as startup_module

    bot, notify, notify_all, broadcast = _make_bot(
        upgrade_targets=[NotificationTarget(enabled=True, chat_id=5555, topic_id=None)],
    )
    bot._orchestrator = MagicMock()
    bot._agent_name = "main"
    bot._config.update_check = True
    bot._startup_hooks = []
    bot._update_observer = None

    # Capture the _on_update callback passed to UpdateObserver.
    captured: dict[str, Any] = {}

    class _FakeObserver:
        def __init__(self, *, notify: Any, check: Any = None) -> None:
            captured["notify"] = notify
            captured["check"] = check

        def start(self) -> None:
            captured["started"] = True

    # Stub the observer chain.
    def _is_upgradeable() -> bool:
        return True

    monkeypatch.setattr(startup_module, "consume_restart_marker", lambda **_kw: False)
    monkeypatch.setattr(startup_module, "consume_restart_sentinel", lambda **_kw: None)

    import ductor_bot.infra.install as install_mod
    import ductor_bot.infra.updater as updater_mod

    monkeypatch.setattr(install_mod, "is_upgradeable", _is_upgradeable)
    monkeypatch.setattr(updater_mod, "UpdateObserver", _FakeObserver)

    # Skip heavy orchestrator construction path: pre-set _orchestrator (non-primary).
    # run_matrix_startup skips the update wiring when primary=False, so we must
    # exercise the primary branch directly. Build a minimal fake orchestrator
    # and patch Orchestrator.create plus lifecycle helpers to no-ops.
    async def _noop(*_a: Any, **_kw: Any) -> None:
        return None

    async def _noop_sentinel(*_a: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr(startup_module, "_handle_restart_sentinel", _noop_sentinel)
    monkeypatch.setattr(startup_module, "_handle_startup_lifecycle", _noop)
    monkeypatch.setattr(startup_module, "_handle_recovery", _noop)
    monkeypatch.setattr(startup_module, "_consume_restart_marker", lambda _bot: "")

    fake_orch = MagicMock()
    fake_orch.wire_observers_to_bus = MagicMock()

    async def _create(*_a: Any, **_kw: Any) -> Any:
        return fake_orch

    import ductor_bot.orchestrator.core as core_mod

    monkeypatch.setattr(core_mod.Orchestrator, "create", staticmethod(_create))

    bot._orchestrator = None  # Force primary branch
    bot._bus = MagicMock()

    await startup_module.run_matrix_startup(bot)

    assert "notify" in captured, "UpdateObserver was not wired"
    assert "check" in captured, "UpdateObserver checker was not wired"

    # Fire the captured callback with a fake VersionInfo and assert routing.
    version_info = MagicMock()
    version_info.latest = "9.9.9"
    await captured["notify"](version_info)

    # notify_upgrade must be reached — notify_all (startup fallback) must not.
    notify_all.assert_not_called()
    # upgrade_targets has chat_id=5555 → notification_service.notify was called,
    # broadcast was NOT called (targets were configured).
    broadcast.assert_not_called()
    notify.assert_awaited_once()
    assert notify.await_args.args[0] == 5555


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
