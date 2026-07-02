"""Tests for ordinary-language access change detection."""

from __future__ import annotations

from ductor_bot.orchestrator.access_admin import is_access_config_change_request


def test_access_change_detector_ignores_access_discussion() -> None:
    assert not is_access_config_change_request(
        "Не админы не могут добавлять бота в чаты, верно?",
    )
    assert not is_access_config_change_request(
        "После добавления другого пользователя в доступ к боту он отвечает admin-only",
    )
    assert not is_access_config_change_request("Может ли не админ добавить бота в чат?")


def test_access_change_detector_catches_mutation_requests() -> None:
    assert is_access_config_change_request("Выдай пользователю 333 доступ через allowed_user_ids")
    assert is_access_config_change_request("Можешь выдать доступ пользователю 484433790?")
    assert is_access_config_change_request("Добавь группу -1001234567890 в allowed_group_ids")
