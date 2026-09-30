"""Focused tests for the two-stage execution-model router."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ductor_bot.cli.codex_cache import CodexModelCache
from ductor_bot.cli.codex_discovery import CodexModelInfo
from ductor_bot.cli.param_resolver import TaskOverrides
from ductor_bot.cli.types import AgentResponse
from ductor_bot.config import ModelPolicyConfig, ModelPolicyRule, ModelRouterConfig
from ductor_bot.model_policy import SelectedModelTarget
from ductor_bot.orchestrator.core import NamedSessionRequest, Orchestrator
from ductor_bot.session import SessionKey
from ductor_bot.tasks.models import TaskSubmit


def _codex_model(model_id: str) -> CodexModelInfo:
    return CodexModelInfo(
        id=model_id,
        display_name=model_id,
        description="",
        supported_efforts=("low", "medium", "high"),
        default_effort="medium",
        is_default=False,
    )


def _enable_codex_inventory(orch: Orchestrator) -> None:
    cache = CodexModelCache(
        "now",
        [
            _codex_model("gpt-6-luna"),
            _codex_model("gpt-6-sol"),
            _codex_model("gpt-6-astra"),
        ],
    )
    orch._providers._available_providers = frozenset({"codex"})
    orch._providers._codex_cache_fn = lambda: cache


async def test_router_only_receives_non_admin_policy_candidates(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        router=ModelRouterConfig(enabled=True),
        default=ModelPolicyRule(
            allowed_models=["gpt-6-sol"],
            allowed_reasoning_efforts=["low", "medium"],
            allow_model_switch=True,
        ),
    )
    orch._cli_service.execute_router = AsyncMock(
        return_value=AgentResponse(result='{"candidate_id":"c0","reasoning_effort":"medium"}')
    )

    selected = await orch.select_execution_target(
        SessionKey(chat_id=-100, user_id=22),
        "Implement a small change",
    )

    assert selected is not None
    assert selected.model == "gpt-6-sol"
    assert selected.reasoning_effort == "medium"
    request = orch._cli_service.execute_router.call_args.args[0]
    payload = json.loads(request.prompt)
    assert payload["candidates"] == [
        {
            "candidate_id": "c0",
            "provider": "codex",
            "model": "gpt-6-sol",
            "description": "",
            "reasoning_efforts": ["low", "medium"],
        }
    ]


async def test_policy_admin_receives_unrestricted_available_pool(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.allowed_user_ids = [7]
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        router=ModelRouterConfig(enabled=True),
        default=ModelPolicyRule(
            allowed_models=["gpt-6-luna"],
            allowed_reasoning_efforts=["low"],
            allow_model_switch=False,
        ),
    )
    orch._cli_service.execute_router = AsyncMock(
        return_value=AgentResponse(result='{"candidate_id":"c2","reasoning_effort":"high"}')
    )

    selected = await orch.select_execution_target(SessionKey(chat_id=7, user_id=7), "Hard task")

    assert selected is not None
    assert selected.model == "gpt-6-astra"
    assert selected.reasoning_effort == "high"
    request = orch._cli_service.execute_router.call_args.args[0]
    payload = json.loads(request.prompt)
    assert [item["model"] for item in payload["candidates"]] == [
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6-astra",
    ]


async def test_invalid_router_output_uses_legacy_heuristic(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        router=ModelRouterConfig(enabled=True),
        default=ModelPolicyRule(
            allowed_models=["gpt-6-luna", "gpt-6-astra"],
            allowed_reasoning_efforts=["low", "high"],
            allow_model_switch=False,
        ),
    )
    orch._cli_service.execute_router = AsyncMock(
        return_value=AgentResponse(result="```json\n{}\n```")
    )

    selected = await orch.select_execution_target(
        SessionKey(chat_id=-100, user_id=22),
        "Deep, thorough full implementation and architecture review " + "x" * 1300,
    )

    assert selected is not None
    assert selected.model == "gpt-6-astra"
    assert selected.reasoning_effort == "high"


async def test_router_rejects_effort_not_offered_to_classifier(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        router=ModelRouterConfig(enabled=True),
        default=ModelPolicyRule(
            allowed_models=["gpt-6-sol"],
            allowed_reasoning_efforts=["low"],
            allow_model_switch=False,
        ),
    )
    orch._cli_service.execute_router = AsyncMock(
        return_value=AgentResponse(result='{"candidate_id":"c0","reasoning_effort":"high"}')
    )

    selected = await orch.select_execution_target(
        SessionKey(chat_id=-100, user_id=22),
        "hello",
    )

    assert selected is not None
    assert selected.model == "gpt-6-sol"
    assert selected.reasoning_effort == "low"


async def test_explicit_model_override_skips_router(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        router=ModelRouterConfig(enabled=True),
    )
    router_execute = AsyncMock()
    orch._cli_service.execute_router = router_execute
    orch._cli_service.execute = AsyncMock(return_value=AgentResponse(result="ok"))

    result = await orch.handle_message(SessionKey(chat_id=1), "@sonnet answer this")

    assert result.text == "ok"
    router_execute.assert_not_awaited()
    execution_request = orch._cli_service.execute.call_args.args[0]
    assert execution_request.model_override == "sonnet"


async def test_router_target_is_sticky_for_followup_turns(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        router=ModelRouterConfig(enabled=True),
    )
    orch._cli_service.execute_router = AsyncMock(
        side_effect=[
            AgentResponse(result='{"candidate_id":"c0","reasoning_effort":"low"}'),
            AgentResponse(result='{"candidate_id":"c0","reasoning_effort":"high"}'),
        ]
    )
    orch._cli_service.execute = AsyncMock(
        side_effect=[
            AgentResponse(result="first", session_id="session-luna"),
            AgentResponse(result="second", session_id="session-luna"),
        ]
    )
    key = SessionKey(chat_id=1)

    first = await orch.handle_message(key, "Implement the first step")
    second = await orch.handle_message(key, "Now adjust that implementation")

    assert first.text == "first"
    assert second.text == "second"
    assert orch._cli_service.execute_router.await_count == 2
    followup_router_request = orch._cli_service.execute_router.await_args_list[1].args[0]
    followup_payload = json.loads(followup_router_request.prompt)
    assert [item["model"] for item in followup_payload["candidates"]] == ["gpt-6-luna"]
    second_request = orch._cli_service.execute.await_args_list[1].args[0]
    assert second_request.model_override == "gpt-6-luna"
    assert second_request.provider_override == "codex"
    assert second_request.reasoning_effort_override == "high"
    assert second_request.resume_session == "session-luna"


def test_router_and_admin_pool_hot_reload_into_cli_service(orch: Orchestrator) -> None:
    orch._config.allowed_user_ids = [77]
    orch._config.model_policy = ModelPolicyConfig(
        admin_user_ids=[88],
        router=ModelRouterConfig(enabled=True, model="gpt-6-sol"),
    )

    orch._on_config_hot_reload(
        orch._config,
        {
            "allowed_user_ids": [77],
            "model_policy": orch._config.model_policy.model_dump(mode="json"),
        },
    )

    service_config = orch._cli_service.update_config.call_args.args[0]
    assert service_config.model_policy.router.enabled is True
    assert service_config.model_policy.router.model == "gpt-6-sol"
    assert service_config.model_policy_admin_user_ids == (77, 88)


async def test_unqualified_named_session_uses_router(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        router=ModelRouterConfig(enabled=True),
    )
    orch._cli_service.execute_router = AsyncMock(
        return_value=AgentResponse(result='{"candidate_id":"c2","reasoning_effort":"high"}')
    )
    background = MagicMock()
    background.submit.return_value = "task-1"
    orch._observers.background = background

    task_id, name = await orch.submit_named_session(
        10,
        "Do a difficult review",
        NamedSessionRequest(message_id=1, thread_id=None, user_id=10),
    )

    assert task_id == "task-1"
    submitted = background.submit.call_args.args[0]
    assert submitted.model_override == "gpt-6-astra"
    assert submitted.provider_override == "codex"
    assert submitted.reasoning_effort_override == "high"
    assert submitted.model_selection_origin == "policy"
    named = orch.get_named_session(10, name)
    assert named is not None
    assert named.reasoning_effort == "high"


async def test_task_infrastructure_override_obeys_allowlist_not_manual_switch(
    orch: Orchestrator,
) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        router=ModelRouterConfig(enabled=True),
        default=ModelPolicyRule(
            allowed_models=["gpt-6-sol"],
            allowed_reasoning_efforts=["medium"],
            allow_model_switch=False,
        ),
    )
    orch._cli_service.execute_router = AsyncMock()
    submit = TaskSubmit(
        chat_id=-100,
        prompt="review this",
        message_id=0,
        thread_id=7,
        parent_agent="main",
        model_override="gpt-6-sol",
        thinking_override="medium",
        user_id=42,
        model_selection_origin="infrastructure",
    )

    selected = await orch.route_task_submit(submit, None)

    assert selected.model == "gpt-6-sol"
    assert selected.reasoning_effort == "medium"
    orch._cli_service.execute_router.assert_not_awaited()


async def test_task_override_uses_originating_user_policy(orch: Orchestrator) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        router=ModelRouterConfig(enabled=True),
        default=ModelPolicyRule(allowed_models=["gpt-6-luna"], allow_model_switch=False),
        users={
            "42": ModelPolicyRule(
                allowed_models=["gpt-6-sol"],
                allowed_reasoning_efforts=["medium"],
                allow_model_switch=False,
            )
        },
    )
    submit = TaskSubmit(
        chat_id=-100,
        prompt="review this",
        message_id=0,
        thread_id=None,
        parent_agent="main",
        model_override="gpt-6-astra",
        thinking_override="medium",
        user_id=42,
        model_selection_origin="infrastructure",
    )

    with pytest.raises(ValueError, match="not allowed"):
        await orch.route_task_submit(submit, None)


async def test_user_origin_override_still_respects_manual_switch_policy(
    orch: Orchestrator,
) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        default=ModelPolicyRule(
            allowed_models=["gpt-6-sol"],
            allowed_reasoning_efforts=["medium"],
            allow_model_switch=False,
        ),
    )
    submit = TaskSubmit(
        chat_id=-100,
        prompt="review this",
        message_id=0,
        thread_id=None,
        parent_agent="main",
        model_override="gpt-6-sol",
        thinking_override="medium",
        user_id=42,
        model_selection_origin="user",
    )

    with pytest.raises(ValueError, match="Manual model selection"):
        await orch.route_task_submit(submit, None)


async def test_unpinned_cron_routes_but_pinned_cron_skips_classifier(
    orch: Orchestrator,
) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        router=ModelRouterConfig(enabled=True),
    )
    orch._cli_service.execute_router = AsyncMock(
        return_value=AgentResponse(result='{"candidate_id":"c0","reasoning_effort":"low"}')
    )
    key = SessionKey(chat_id=42, user_id=42)

    routed = await orch.route_task_overrides(key, "small recurring check", TaskOverrides())
    pinned = await orch.route_task_overrides(
        key,
        "deep review",
        TaskOverrides(
            provider="codex",
            model="gpt-6-sol",
            reasoning_effort="high",
            cli_parameters=["--flag"],
        ),
    )
    pinned_with_routed_effort = await orch.route_task_overrides(
        key,
        "deep review",
        TaskOverrides(provider="codex", model="gpt-6-sol"),
    )

    assert routed.model == "gpt-6-luna"
    assert routed.reasoning_effort == "low"
    assert pinned == TaskOverrides(
        provider="codex",
        model="gpt-6-sol",
        reasoning_effort="high",
        cli_parameters=["--flag"],
    )
    assert pinned_with_routed_effort.model == "gpt-6-sol"
    assert pinned_with_routed_effort.reasoning_effort == "low"
    assert orch._cli_service.execute_router.await_count == 2
    pinned_payload = json.loads(orch._cli_service.execute_router.await_args_list[1].args[0].prompt)
    assert [item["model"] for item in pinned_payload["candidates"]] == ["gpt-6-sol"]


async def test_resume_without_router_keeps_model_and_selects_allowed_effort(
    orch: Orchestrator,
) -> None:
    _enable_codex_inventory(orch)
    orch._config.model_policy = ModelPolicyConfig(
        enabled=True,
        router=ModelRouterConfig(enabled=False),
        default=ModelPolicyRule(
            allowed_models=["gpt-6-sol"],
            allowed_reasoning_efforts=["low"],
            allow_model_switch=False,
        ),
    )

    selected = await orch.resolve_policy_execution_target(
        SessionKey(chat_id=-100, user_id=42),
        "continue",
        session_target=SelectedModelTarget("gpt-6-sol", "codex", "high"),
    )

    assert selected == SelectedModelTarget("gpt-6-sol", "codex", "low")
