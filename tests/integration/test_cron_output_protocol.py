"""Exact cron output protocols preserve ACK silence and real alert delivery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ductor_bot.bus.adapters import from_cron_result
from ductor_bot.cli.codex_cache import CodexModelCache
from ductor_bot.config import AgentConfig
from ductor_bot.cron.manager import CronJob, CronManager
from ductor_bot.cron.observer import CronObserver
from ductor_bot.messenger.telegram.transport import TelegramTransport
from ductor_bot.workspace.paths import DuctorPaths

ACK = "Message sent successfully. delivered to Telegram"
ALERT = "ALERT: usage limit reached"


@pytest.mark.parametrize("chat_id", [0, 42], ids=["broadcast", "unicast"])
@pytest.mark.parametrize("output", [ACK, ALERT, f"{ALERT}\n{ACK}"])
async def test_task_protocol_telegram_delivery(tmp_path: Path, chat_id: int, output: str) -> None:
    paths = DuctorPaths(
        ductor_home=tmp_path / "home",
        home_defaults=tmp_path / "fw" / "workspace",
        framework_root=tmp_path / "fw",
    )
    folder = paths.cron_tasks_dir / "monitor"
    folder.mkdir(parents=True)
    (folder / "TASK_DESCRIPTION.md").write_text(
        f"Return the alert verbatim when present. If there is no alert, return exactly:\n{ACK}",
        encoding="utf-8",
    )
    job = CronJob(
        id="monitor",
        title="Monitor",
        description="Check for alerts",
        schedule="* * * * *",
        task_folder="monitor",
        agent_instruction="Read and follow TASK_DESCRIPTION.md",
        chat_id=chat_id,
    )
    assert job.silent_on_success is False
    manager = CronManager(jobs_path=paths.cron_jobs_path)
    manager.add_job(job)
    observer = CronObserver(
        paths,
        manager,
        config=AgentConfig(provider="claude", model="sonnet"),
        codex_cache=CodexModelCache(last_updated="2026-10-11T00:00:00Z", models=[]),
    )
    bot = MagicMock()
    bot._config.allowed_user_ids = [101]
    bot.file_roots.return_value = [tmp_path]
    transport = TelegramTransport(bot)

    async def deliver(title, result, status, target, topic, transport_name):
        env = from_cron_result(
            title, result, status, chat_id=target, topic_id=topic, transport=transport_name
        )
        if target:
            await transport.deliver(env)
        else:
            await transport.deliver_broadcast(env)

    handler = AsyncMock(side_effect=deliver)
    observer.set_result_handler(handler)
    # Model output is stubbed; exercise the real parser, observer, adapter and transport.
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (json.dumps({"result": output}).encode(), b"")
    with (
        patch("ductor_bot.cron.execution.which", return_value="/usr/bin/claude"),
        patch("asyncio.create_subprocess_exec", return_value=proc) as spawn,
        patch("ductor_bot.messenger.telegram.transport.send_rich", new_callable=AsyncMock) as send,
        patch.object(transport, "_send_queued", new_callable=AsyncMock) as queued,
    ):
        await observer._execute_job(job.id, job.agent_instruction, job.task_folder)

    prompt = spawn.call_args.args[-1]
    assert "TASK_DESCRIPTION.md defines an exact output protocol, follow it verbatim" in prompt
    assert "including any required service ACK" in prompt
    handler.assert_awaited_once_with("Monitor", output, "success", chat_id, None, "tg")
    if output == ACK:
        send.assert_not_awaited()
        queued.assert_not_awaited()
    else:
        delivery = send if chat_id else queued
        delivery.assert_awaited_once()
        text = delivery.await_args.args[2 if chat_id else 1]
        assert text == f"**TASK: Monitor**\n\n{ALERT}"
    assert manager.get_job(job.id).last_run_status == "success"
