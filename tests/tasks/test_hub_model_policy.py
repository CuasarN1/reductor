"""Focused model-policy coverage for delegated task submit/resume."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ductor_bot.cli.process_registry import ProcessRegistry
from ductor_bot.cli.service import CLIService, CLIServiceConfig
from ductor_bot.cli.types import CLIResponse
from ductor_bot.config import ModelPolicyConfig, ModelPolicyRule, ModelRegistry
from ductor_bot.model_policy import SelectedModelTarget
from ductor_bot.tasks.hub import TaskHub
from ductor_bot.tasks.models import TaskSubmit
from ductor_bot.tasks.registry import TaskRegistry


def _service(policy: ModelPolicyConfig) -> CLIService:
    return CLIService(
        config=CLIServiceConfig(
            working_dir="/tmp",
            default_model="sonnet",
            provider="claude",
            max_turns=None,
            max_budget_usd=None,
            permission_mode="bypassPermissions",
            model_policy=policy,
        ),
        models=ModelRegistry(),
        available_providers=frozenset({"claude"}),
        process_registry=ProcessRegistry(),
    )


def _hub(tmp_path: Path, service: CLIService) -> tuple[TaskHub, TaskRegistry]:
    registry = TaskRegistry(tmp_path / "tasks.json", tmp_path / "tasks")
    config = MagicMock(enabled=True, max_parallel=5, timeout_seconds=60.0)
    hub = TaskHub(
        registry,
        MagicMock(workspace=tmp_path),
        cli_service=service,
        config=config,
    )
    hub.set_result_handler("main", AsyncMock())
    return hub, registry


def _submit(*, model: str = "") -> TaskSubmit:
    return TaskSubmit(
        chat_id=42,
        prompt="do the work",
        message_id=1,
        thread_id=None,
        parent_agent="main",
        model_override=model,
        provider_override="claude" if model else "",
    )


async def test_submit_cannot_smuggle_disallowed_model(tmp_path: Path) -> None:
    service = _service(
        ModelPolicyConfig(
            enabled=True,
            default=ModelPolicyRule(
                allowed_models=["sonnet"],
                allow_model_switch=True,
            ),
        )
    )
    hub, registry = _hub(tmp_path, service)

    with patch("ductor_bot.cli.service.create_cli") as create_cli:
        task_id = hub.submit(_submit(model="opus"))
        await asyncio.sleep(0.1)

    entry = registry.get(task_id)
    assert entry is not None
    assert entry.status == "failed"
    assert "Model `opus` is not allowed" in entry.error
    create_cli.assert_not_called()
    await hub.shutdown()


async def test_submit_then_resume_reuses_allowed_infrastructure_model(tmp_path: Path) -> None:
    service = _service(
        ModelPolicyConfig(
            enabled=True,
            default=ModelPolicyRule(
                allowed_models=["sonnet"],
                allow_model_switch=False,
            ),
        )
    )
    hub, registry = _hub(tmp_path, service)
    cli_response = CLIResponse(result="done", session_id="sess-1")

    with patch("ductor_bot.cli.service.create_cli") as create_cli:
        cli = AsyncMock()
        cli.send.return_value = cli_response
        create_cli.return_value = cli

        task_id = hub.submit(_submit())
        await asyncio.sleep(0.1)
        entry = registry.get(task_id)
        assert entry is not None
        assert entry.status == "done"
        assert entry.provider == "claude"
        assert entry.model == "sonnet"

        hub.resume(task_id, "continue")
        await asyncio.sleep(0.1)

    entry = registry.get(task_id)
    assert entry is not None
    assert entry.status == "done"
    assert cli.send.await_count == 2
    assert create_cli.call_count == 2
    await hub.shutdown()


async def test_routed_submit_persists_target_reasoning_and_identity(tmp_path: Path) -> None:
    service = _service(ModelPolicyConfig())
    hub, registry = _hub(tmp_path, service)
    route = AsyncMock(
        return_value=SelectedModelTarget(
            model="gpt-6-sol",
            provider="codex",
            reasoning_effort="high",
        )
    )
    hub.set_route_handler("main", route)
    submit = _submit()
    submit.user_id = 77
    submit.transport = "tg"
    submit.model_selection_origin = "infrastructure"

    with patch("ductor_bot.cli.service.create_cli") as create_cli:
        cli = AsyncMock()
        cli.send.return_value = CLIResponse(result="done", session_id="sess-1")
        create_cli.return_value = cli
        task_id = await hub.submit_routed(submit)
        await asyncio.sleep(0.1)

    entry = registry.get(task_id)
    assert entry is not None
    assert entry.provider == "codex"
    assert entry.model == "gpt-6-sol"
    assert entry.thinking == "high"
    assert entry.user_id == 77
    assert entry.model_selection_origin == "policy"
    request = create_cli.call_args.args[0]
    assert request.reasoning_effort == "high"
    route.assert_awaited_once()
    await hub.shutdown()


async def test_routing_denial_happens_before_registry_creation(tmp_path: Path) -> None:
    service = _service(ModelPolicyConfig())
    hub, registry = _hub(tmp_path, service)
    hub.set_route_handler(
        "main",
        AsyncMock(side_effect=ValueError("Model `opus` is not allowed")),
    )

    with pytest.raises(ValueError, match="not allowed"):
        await hub.submit_routed(_submit(model="opus"))

    assert registry.list_all() == []
    await hub.shutdown()


async def test_capacity_rejection_skips_router_and_artifact_creation(tmp_path: Path) -> None:
    service = _service(ModelPolicyConfig())
    hub, registry = _hub(tmp_path, service)
    hub._config.max_parallel = 1
    route = AsyncMock(return_value=SelectedModelTarget("sonnet", "claude"))
    hub.set_route_handler("main", route)
    release = asyncio.Event()

    async def _block_execution(*_args: object, **_kwargs: object) -> CLIResponse:
        await release.wait()
        return CLIResponse(result="done", session_id="sess-1")

    with patch("ductor_bot.cli.service.create_cli") as create_cli:
        cli = AsyncMock()
        cli.send.side_effect = _block_execution
        create_cli.return_value = cli
        hub.submit(_submit())
        await asyncio.sleep(0)

        with pytest.raises(ValueError, match="Too many background tasks"):
            await hub.submit_routed(_submit())

        assert len(registry.list_all()) == 1
        route.assert_not_awaited()
        release.set()
        await asyncio.sleep(0.1)

    await hub.shutdown()


async def test_routed_submit_fails_closed_without_agent_router(tmp_path: Path) -> None:
    service = _service(ModelPolicyConfig())
    hub, registry = _hub(tmp_path, service)

    with pytest.raises(ValueError, match="Task routing unavailable"):
        await hub.submit_routed(_submit())

    assert registry.list_all() == []
    await hub.shutdown()


async def test_routed_resume_pins_model_and_can_lower_reasoning(tmp_path: Path) -> None:
    service = _service(ModelPolicyConfig())
    hub, registry = _hub(tmp_path, service)
    route = AsyncMock(
        side_effect=[
            SelectedModelTarget("gpt-6-sol", "codex", "high"),
            SelectedModelTarget("gpt-6-sol", "codex", "low"),
        ]
    )
    hub.set_route_handler("main", route)

    with patch("ductor_bot.cli.service.create_cli") as create_cli:
        cli = AsyncMock()
        cli.send.return_value = CLIResponse(result="done", session_id="sess-1")
        create_cli.return_value = cli
        task_id = await hub.submit_routed(_submit())
        await asyncio.sleep(0.1)
        await hub.resume_routed(task_id, "continue", parent_agent="main")
        await asyncio.sleep(0.1)

    entry = registry.get(task_id)
    assert entry is not None
    assert entry.provider == "codex"
    assert entry.model == "gpt-6-sol"
    assert entry.thinking == "low"
    resume_target = route.await_args_list[1].args[1]
    assert resume_target == SelectedModelTarget("gpt-6-sol", "codex", "high")
    resume_config = create_cli.call_args.args[0]
    assert resume_config.model == "gpt-6-sol"
    assert resume_config.reasoning_effort == "low"
    await hub.shutdown()


async def test_concurrent_routed_resume_classifies_only_once(tmp_path: Path) -> None:
    service = _service(ModelPolicyConfig())
    hub, registry = _hub(tmp_path, service)
    resume_route_entered = asyncio.Event()
    release_resume_route = asyncio.Event()
    resumed_execution_started = asyncio.Event()
    release_resumed_execution = asyncio.Event()
    route_calls = 0

    async def _route(
        _submit: TaskSubmit,
        session_target: SelectedModelTarget | None,
    ) -> SelectedModelTarget:
        nonlocal route_calls
        route_calls += 1
        if session_target is not None:
            resume_route_entered.set()
            await release_resume_route.wait()
        return SelectedModelTarget("gpt-6-sol", "codex", "low")

    hub.set_route_handler("main", _route)
    execution_calls = 0

    async def _execute(*_args: object, **_kwargs: object) -> CLIResponse:
        nonlocal execution_calls
        execution_calls += 1
        if execution_calls == 2:
            resumed_execution_started.set()
            await release_resumed_execution.wait()
        return CLIResponse(result="done", session_id="sess-1")

    with patch("ductor_bot.cli.service.create_cli") as create_cli:
        cli = AsyncMock()
        cli.send.side_effect = _execute
        create_cli.return_value = cli
        task_id = await hub.submit_routed(_submit())
        await asyncio.sleep(0.1)

        first = asyncio.create_task(hub.resume_routed(task_id, "first", parent_agent="main"))
        await resume_route_entered.wait()
        second = asyncio.create_task(hub.resume_routed(task_id, "second", parent_agent="main"))
        release_resume_route.set()
        await first
        await resumed_execution_started.wait()

        with pytest.raises(ValueError, match=r"still running|already running"):
            await second

        assert route_calls == 2  # Initial submit plus exactly one resume classification.
        release_resumed_execution.set()
        await asyncio.sleep(0.1)

    entry = registry.get(task_id)
    assert entry is not None
    assert entry.thinking == "low"
    await hub.shutdown()
