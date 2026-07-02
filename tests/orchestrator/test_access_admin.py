"""Tests for /access admin commands."""

from __future__ import annotations

import json

from ductor_bot.cli.types import AgentResponse
from ductor_bot.config import ModelPolicyConfig
from ductor_bot.orchestrator.access_admin import cmd_access
from ductor_bot.orchestrator.core import Orchestrator
from ductor_bot.session.key import SessionKey


def _saved_config(orch: Orchestrator) -> dict[str, object]:
    return json.loads(orch.paths.config_path.read_text(encoding="utf-8"))


async def test_access_denies_non_admin(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]

    result = await cmd_access(orch, SessionKey(chat_id=2, user_id=2), "/access list")

    assert "admin-only" in result.text


async def test_owner_adds_user_and_policy(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]

    result = await cmd_access(
        orch,
        SessionKey(chat_id=1, user_id=1),
        "/access add 222 models=gpt-5.4-mini,gpt-5.4 efforts=low,medium switch=off",
    )

    saved = _saved_config(orch)
    policy = saved["model_policy"]
    assert "222" in result.text
    assert orch._config.allowed_user_ids == [1, 222]
    assert saved["allowed_user_ids"] == [1, 222]
    assert isinstance(policy, dict)
    assert policy["enabled"] is True
    assert policy["users"]["222"]["allowed_models"] == ["gpt-5.4-mini", "gpt-5.4"]
    assert policy["users"]["222"]["allowed_reasoning_efforts"] == ["low", "medium"]
    assert policy["users"]["222"]["allow_model_switch"] is False


async def test_access_add_invalid_policy_option_does_not_mutate(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]

    result = await cmd_access(
        orch,
        SessionKey(chat_id=1, user_id=1),
        "/access add 222 models=gpt-5.4-mini efforts=",
    )

    assert "empty" in result.text
    assert orch._config.allowed_user_ids == [1]
    assert "222" not in orch._config.model_policy.users


async def test_access_default_updates_default_policy(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]

    result = await cmd_access(
        orch,
        SessionKey(chat_id=1, user_id=1),
        "/access default models=gpt-5.4-mini efforts=low switch=off",
    )

    saved = _saved_config(orch)
    policy = saved["model_policy"]
    assert "Default policy updated" in result.text
    assert isinstance(policy, dict)
    assert policy["default"]["allowed_models"] == ["gpt-5.4-mini"]
    assert policy["default"]["allowed_reasoning_efforts"] == ["low"]
    assert policy["default"]["allow_model_switch"] is False


async def test_access_admin_role_can_manage_users(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1, 2]
    orch._config.model_policy = ModelPolicyConfig(admin_user_ids=[2])

    result = await cmd_access(orch, SessionKey(chat_id=2, user_id=2), "/access add 333")

    assert "333" in result.text
    assert orch._config.allowed_user_ids == [1, 2, 333]


async def test_owner_adds_group_access(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]
    orch._config.allowed_group_ids = []
    hot_updates: list[dict[str, object]] = []
    orch.set_config_hot_reload_handler(lambda _config, hot: hot_updates.append(hot))

    result = await cmd_access(
        orch,
        SessionKey(chat_id=1, user_id=1),
        "/access group add -1001234567890",
    )

    saved = _saved_config(orch)
    assert "added" in result.text
    assert orch._config.allowed_group_ids == [-1001234567890]
    assert saved["allowed_group_ids"] == [-1001234567890]
    assert hot_updates[-1]["allowed_group_ids"] == [-1001234567890]


async def test_owner_adds_group_access_idempotently(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]
    orch._config.allowed_group_ids = [-1001234567890]

    result = await cmd_access(
        orch,
        SessionKey(chat_id=1, user_id=1),
        "/access group add -1001234567890",
    )

    saved = _saved_config(orch)
    assert "already allowlisted" in result.text
    assert orch._config.allowed_group_ids == [-1001234567890]
    assert saved["allowed_group_ids"] == [-1001234567890]


async def test_access_group_rejects_positive_id_without_mutation(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]
    orch._config.allowed_group_ids = []

    result = await cmd_access(
        orch,
        SessionKey(chat_id=1, user_id=1),
        "/access group add 123",
    )

    assert "must be negative" in result.text
    assert orch._config.allowed_group_ids == []


async def test_access_admin_role_can_manage_groups(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1, 2]
    orch._config.model_policy = ModelPolicyConfig(admin_user_ids=[2])

    result = await cmd_access(
        orch,
        SessionKey(chat_id=2, user_id=2),
        "/access group approve -1002222222222",
    )

    assert "added" in result.text
    assert orch._config.allowed_group_ids == [-1002222222222]


async def test_owner_removes_group_access(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]
    orch._config.allowed_group_ids = [-1001234567890]

    result = await cmd_access(
        orch,
        SessionKey(chat_id=1, user_id=1),
        "/access group remove -1001234567890",
    )

    saved = _saved_config(orch)
    assert "removed" in result.text
    assert orch._config.allowed_group_ids == []
    assert saved["allowed_group_ids"] == []


async def test_non_admin_group_access_request_is_denied_before_cli(
    orch: Orchestrator,
) -> None:
    orch._config.allowed_user_ids = [1, 2]

    result = await orch.handle_message(
        SessionKey(chat_id=2, user_id=2),
        "Добавь группу -1001234567890 в allowed_group_ids",
    )

    assert "admin-only" in result.text
    orch._cli_service.execute.assert_not_awaited()


async def test_non_admin_user_access_request_is_denied_before_cli(
    orch: Orchestrator,
) -> None:
    orch._config.allowed_user_ids = [1, 2]

    result = await orch.handle_message(
        SessionKey(chat_id=2, user_id=2),
        "Выдай пользователю 333 доступ через allowed_user_ids",
    )

    assert "admin-only" in result.text
    orch._cli_service.execute.assert_not_awaited()


async def test_admin_access_request_gets_command_guidance_before_cli(
    orch: Orchestrator,
) -> None:
    orch._config.allowed_user_ids = [1]

    result = await orch.handle_message(
        SessionKey(chat_id=1, user_id=1),
        "Добавь группу -1001234567890 в allowed_group_ids",
    )

    assert "/access group add" in result.text
    orch._cli_service.execute.assert_not_awaited()


async def test_benign_access_config_question_reaches_cli(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1, 2]
    orch._cli_service.execute.return_value = AgentResponse(
        result="allowed_group_ids explained",
        session_id="sess-access-doc",
        is_error=False,
    )

    result = await orch.handle_message(
        SessionKey(chat_id=2, user_id=2),
        "Объясни, что значит allowed_group_ids",
    )

    assert result.text == "allowed_group_ids explained"
    orch._cli_service.execute.assert_awaited_once()


async def test_access_admin_command_adds_admin_and_full_policy(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]

    result = await cmd_access(orch, SessionKey(chat_id=1, user_id=1), "/access admin 222 on")

    saved = _saved_config(orch)
    policy = saved["model_policy"]
    assert "admin=on" in result.text
    assert isinstance(policy, dict)
    assert saved["allowed_user_ids"] == [1, 222]
    assert policy["admin_user_ids"] == [222]
    assert policy["users"]["222"]["allowed_models"] == ["*"]
    assert policy["users"]["222"]["allowed_reasoning_efforts"] == ["*"]
    assert policy["users"]["222"]["allow_model_switch"] is True


async def test_access_remove_user_removes_policy_and_admin_role(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1, 222]
    orch._config.model_policy = ModelPolicyConfig(admin_user_ids=[222])
    orch._config.model_policy.users["222"] = ModelPolicyConfig().default

    result = await cmd_access(orch, SessionKey(chat_id=1, user_id=1), "/access remove 222")

    saved = _saved_config(orch)
    policy = saved["model_policy"]
    assert "removed" in result.text
    assert saved["allowed_user_ids"] == [1]
    assert isinstance(policy, dict)
    assert policy["admin_user_ids"] == []
    assert "222" not in policy["users"]


async def test_access_refuses_to_remove_owner(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1, 222]

    result = await cmd_access(orch, SessionKey(chat_id=1, user_id=1), "/access remove 1")

    assert "Refusing to remove the owner" in result.text
    assert orch._config.allowed_user_ids == [1, 222]


async def test_access_is_registered_in_orchestrator(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [1]

    result = await orch.handle_message(SessionKey(chat_id=1, user_id=1), "/access list")

    assert "Access" in result.text
