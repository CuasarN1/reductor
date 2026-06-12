"""Tests for Antigravity CLI provider integration."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ductor_bot.cli.antigravity_events import (
    parse_antigravity_json,
    parse_antigravity_stream_line,
)
from ductor_bot.cli.antigravity_provider import AntigravityCLI, _finish_stream_process
from ductor_bot.cli.base import CLIConfig
from ductor_bot.cli.stream_events import (
    AssistantTextDelta,
    ResultEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from ductor_bot.config import ModelRegistry


def test_antigravity_plain_text_line_becomes_text_delta() -> None:
    events = parse_antigravity_stream_line("hello")

    assert len(events) == 1
    assert isinstance(events[0], AssistantTextDelta)
    assert events[0].text == "hello"


def test_antigravity_thought_marker_is_split_from_text() -> None:
    events = parse_antigravity_stream_line("[Thought: plan]\nfinal answer")

    assert [type(event) for event in events] == [ThinkingEvent, AssistantTextDelta]
    assert events[0].text == "[Thought: plan]"
    assert events[1].text == "final answer"


def test_antigravity_structured_tool_events() -> None:
    tool_use = json.dumps({"type": "tool_use", "name": "Read", "id": "tool-1"})
    tool_result = json.dumps({"type": "tool_result", "tool_id": "tool-1", "output": "ok"})

    use_events = parse_antigravity_stream_line(tool_use)
    result_events = parse_antigravity_stream_line(tool_result)

    assert isinstance(use_events[0], ToolUseEvent)
    assert use_events[0].tool_name == "Read"
    assert isinstance(result_events[0], ToolResultEvent)
    assert result_events[0].output == "ok"


def test_antigravity_result_event() -> None:
    events = parse_antigravity_stream_line(
        json.dumps({"type": "result", "content": "done", "session_id": "conv-1"})
    )

    assert len(events) == 1
    assert isinstance(events[0], ResultEvent)
    assert events[0].result == "done"
    assert events[0].session_id == "conv-1"


def test_antigravity_batch_json_extracts_common_content_keys() -> None:
    assert parse_antigravity_json('{"result":"ok"}') == "ok"
    assert parse_antigravity_json("plain") == "plain"


def test_antigravity_command_uses_print_and_conversation() -> None:
    cli = AntigravityCLI(CLIConfig(provider="antigravity", model="antigravity-default"))

    cmd = cli._build_command(resume_session="conv-1")

    assert cmd[:2] == ["agy", "--print"]
    assert "--model" not in cmd
    assert "--conversation" in cmd
    assert "conv-1" in cmd


def test_antigravity_command_includes_selected_model() -> None:
    cli = AntigravityCLI(CLIConfig(provider="antigravity", model="claude-sonnet-4-5"))

    cmd = cli._build_command()

    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-5"


def test_antigravity_streaming_command_uses_prompt_interactive() -> None:
    cli = AntigravityCLI(CLIConfig(provider="antigravity", permission_mode="bypassPermissions"))

    cmd = cli._build_command(streaming=True, continue_session=True)

    assert "--prompt-interactive" in cmd
    assert "--continue" in cmd
    assert "--dangerously-skip-permissions" in cmd


def test_antigravity_command_includes_cli_parameters() -> None:
    cli = AntigravityCLI(
        CLIConfig(
            provider="antigravity",
            cli_parameters=["--log-file", "agy.log"],
        )
    )

    cmd = cli._build_command()

    assert cmd[-2:] == ["--log-file", "agy.log"]


def test_antigravity_ignores_docker_container() -> None:
    cli = AntigravityCLI(
        CLIConfig(
            provider="antigravity",
            model="antigravity-default",
            docker_container="ductor-sandbox",
            working_dir=".",
        )
    )

    cmd, cwd = cli._host_command(["agy", "--print", "hello"])

    assert cmd[:2] == ["agy", "--print"]
    assert "docker" not in cmd
    assert cwd


async def test_finish_stream_process_waits_before_killing() -> None:
    proc = AsyncMock(spec=asyncio.subprocess.Process)
    proc.returncode = None
    proc.pid = 123
    proc.wait = AsyncMock(side_effect=lambda: setattr(proc, "returncode", 0))
    stderr_task = asyncio.create_task(_bytes_result(b""))

    with patch("ductor_bot.cli.antigravity_provider.force_kill_process_tree") as kill:
        await _finish_stream_process(proc, stderr_task)

    kill.assert_not_called()


async def _bytes_result(value: bytes) -> bytes:
    return value


def test_antigravity_model_prefix_routes_to_provider() -> None:
    assert ModelRegistry().provider_for("antigravity-default") == "antigravity"


def _make_oneshot_process(stdout: bytes = b'{"response": "ok"}') -> AsyncMock:
    proc = AsyncMock(spec=asyncio.subprocess.Process)
    proc.returncode = 0
    proc.pid = 12345
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    return proc


def _make_streaming_process(returncode: int = 0) -> AsyncMock:
    proc = AsyncMock(spec=asyncio.subprocess.Process)
    proc.returncode = returncode
    proc.pid = 12345
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    stdout_mock = AsyncMock()
    stdout_mock.readline = AsyncMock(return_value=b"")
    proc.stdout = stdout_mock

    stderr_mock = AsyncMock()
    stderr_mock.read = AsyncMock(return_value=b"")
    proc.stderr = stderr_mock

    stdin_mock = MagicMock()
    stdin_mock.write = MagicMock()
    stdin_mock.drain = AsyncMock()
    stdin_mock.close = MagicMock()
    proc.stdin = stdin_mock

    return proc


def _make_cli(**overrides: Any) -> AntigravityCLI:
    return AntigravityCLI(
        CLIConfig(
            provider="antigravity",
            model="antigravity-default",
            working_dir=".",
            **overrides,
        )
    )


class TestAgentEnvInjection:
    """agy subprocesses must receive the DUCTOR_* agent identification env."""

    async def test_send_injects_agent_env(self) -> None:
        cli = _make_cli(chat_id=77, transport="tg")
        proc = _make_oneshot_process()

        with patch(
            "ductor_bot.cli.antigravity_provider.asyncio.create_subprocess_exec",
            return_value=proc,
        ) as spawn:
            await cli.send("hello")

        env = spawn.call_args.kwargs["env"]
        assert env["DUCTOR_AGENT_NAME"] == "main"
        assert env["DUCTOR_CHAT_ID"] == "77"
        assert env["DUCTOR_TRANSPORT"] == "tg"
        assert "DUCTOR_HOME" in env
        assert "DUCTOR_SHARED_MEMORY_PATH" in env

    async def test_send_streaming_injects_agent_env(self) -> None:
        cli = _make_cli(chat_id=77)
        proc = _make_streaming_process()

        with patch(
            "ductor_bot.cli.antigravity_provider.asyncio.create_subprocess_exec",
            return_value=proc,
        ) as spawn:
            async for _event in cli.send_streaming("hello"):
                pass

        env = spawn.call_args.kwargs["env"]
        assert env["DUCTOR_AGENT_NAME"] == "main"
        assert env["DUCTOR_CHAT_ID"] == "77"
        assert "DUCTOR_HOME" in env
