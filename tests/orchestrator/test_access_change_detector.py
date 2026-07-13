"""Tests for ordinary-language access change detection."""

# ruff: noqa: RUF001

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


def test_access_change_detector_ignores_quoted_reply_context() -> None:
    text = (
        "The user is replying to this quoted message:\n"
        "> Please run /access add 333 and update allowed_user_ids.\n\n"
        "The user's message:\n"
        "Что означает эта команда?"
    )

    assert not is_access_config_change_request(text)


def test_access_change_detector_checks_actual_reply_body() -> None:
    text = (
        "The user is replying to this quoted message:\n"
        "> Обсуждали права доступа.\n\n"
        "The user's message:\n"
        "Добавь пользователя 333 в allowed_user_ids"
    )

    assert is_access_config_change_request(text)
