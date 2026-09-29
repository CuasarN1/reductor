"""Focused model-policy coverage for delegated task submit/resume."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ductor_bot.cli.process_registry import ProcessRegistry
from ductor_bot.cli.service import CLIService, CLIServiceConfig
from ductor_bot.cli.types import CLIResponse
from ductor_bot.config import ModelPolicyConfig, ModelPolicyRule, ModelRegistry
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
