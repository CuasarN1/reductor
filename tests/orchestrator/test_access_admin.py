"""Tests for /access admin commands."""

from __future__ import annotations

import json

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
