"""Tests for Antigravity CLI provider integration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ductor_bot.cli.antigravity_events import parse_antigravity_json
from ductor_bot.cli.antigravity_provider import (
    AntigravityCLI,
    _agy_state_root,
    _read_transcript_answer,
)
from ductor_bot.cli.base import CLIConfig
from ductor_bot.cli.stream_events import AssistantTextDelta, ResultEvent
from ductor_bot.config import ModelRegistry


def test_antigravity_batch_json_extracts_common_content_keys() -> None:
    assert parse_antigravity_json('{"result":"ok"}') == "ok"
    assert parse_antigravity_json("plain") == "plain"


def test_antigravity_command_uses_print_and_conversation() -> None:
    cli = AntigravityCLI(CLIConfig(provider="antigravity", model="antigravity-default"))

    cmd = cli._build_command("hi there", resume_session="conv-1")

    assert cmd[0] == "agy"
    # --print and its prompt value must be last and adjacent (--print swallows
    # the next token as the prompt).
    assert cmd[-2:] == ["--print", "hi there"]
    assert "--model" not in cmd
    assert "--conversation" in cmd
    assert "conv-1" in cmd


def test_antigravity_command_grounds_in_workspace() -> None:
    cli = AntigravityCLI(CLIConfig(provider="antigravity", working_dir="."))

    cmd = cli._build_command("hi")

    assert "--add-dir" in cmd
    assert cmd[cmd.index("--add-dir") + 1] == str(cli._agy_workspace)


def test_antigravity_dotted_workspace_mapped_to_symlink(tmp_path: Path) -> None:
    # agy rejects dot-prefixed ancestors (e.g. ~/.ductor/workspace), so ductor
    # exposes the workspace via a non-dotted symlink that still resolves back.
    dotted = tmp_path / ".ductor" / "workspace"
    dotted.mkdir(parents=True)

    cli = AntigravityCLI(CLIConfig(provider="antigravity", working_dir=str(dotted)))

    assert "/." not in cli._agy_workspace.as_posix()
    assert cli._agy_workspace.resolve() == dotted.resolve()
    cmd = cli._build_command("hi")
    assert cmd[cmd.index("--add-dir") + 1] == str(cli._agy_workspace)


def test_antigravity_plain_workspace_unchanged(tmp_path: Path) -> None:
    plain = tmp_path / "proj"
    plain.mkdir()

    cli = AntigravityCLI(CLIConfig(provider="antigravity", working_dir=str(plain)))

    assert cli._agy_workspace == plain.resolve()


def test_antigravity_command_includes_selected_model() -> None:
    cli = AntigravityCLI(CLIConfig(provider="antigravity", model="claude-sonnet-4-5"))

    cmd = cli._build_command("hi")

    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-5"
    # --model must come before --print, else --print eats it as the prompt.
    assert cmd.index("--model") < cmd.index("--print")


def test_antigravity_command_continue_and_bypass() -> None:
    cli = AntigravityCLI(CLIConfig(provider="antigravity", permission_mode="bypassPermissions"))

    cmd = cli._build_command("hi", continue_session=True)

    assert "--continue" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--prompt-interactive" not in cmd
    assert cmd[-2:] == ["--print", "hi"]


def test_antigravity_command_includes_cli_parameters() -> None:
    cli = AntigravityCLI(
        CLIConfig(
            provider="antigravity",
            cli_parameters=["--log-file", "agy.log"],
        )
    )

    cmd = cli._build_command("hi")

    # cli_parameters come before the trailing --print/prompt pair.
    idx = cmd.index("--log-file")
    assert cmd[idx + 1] == "agy.log"
    assert idx < cmd.index("--print")
    assert cmd[-2:] == ["--print", "hi"]


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


def test_antigravity_model_prefix_routes_to_provider() -> None:
    assert ModelRegistry().provider_for("antigravity-default") == "antigravity"


def _make_oneshot_process(stdout: bytes = b"hello world") -> AsyncMock:
    proc = AsyncMock(spec=asyncio.subprocess.Process)
    proc.returncode = 0
    proc.pid = 12345
    proc.communicate = AsyncMock(return_value=(stdout, b""))
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


class TestStreaming:
    """Streaming delegates to the one-shot --print path (agy has no stream)."""

    async def test_send_streaming_emits_text_then_result(self) -> None:
        cli = _make_cli()
        proc = _make_oneshot_process(b"the answer")

        with patch(
            "ductor_bot.cli.antigravity_provider.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            events = [event async for event in cli.send_streaming("hi")]

        assert [type(event) for event in events] == [AssistantTextDelta, ResultEvent]
        assert events[0].text == "the answer"
        assert events[1].result == "the answer"
        assert events[1].is_error is False

    async def test_send_streaming_skips_empty_text_delta(self) -> None:
        cli = _make_cli()
        proc = _make_oneshot_process(b"")

        with patch(
            "ductor_bot.cli.antigravity_provider.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            events = [event async for event in cli.send_streaming("hi")]

        assert [type(event) for event in events] == [ResultEvent]
        assert events[0].result == ""


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
        proc = _make_oneshot_process()

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


# -- Transcript reading (workaround for antigravity-cli#76) --------------------


_PLANNER: dict[str, str] = {"source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE"}


@pytest.fixture(autouse=True)
def isolated_agy_state(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point agy's state root at an empty temp dir for every test.

    Without this, ``send()`` would read the real ``~/.gemini`` transcripts on
    the dev machine; the transcript tests populate this root explicitly.
    """
    root = tmp_path_factory.mktemp("agy-state")
    with patch("ductor_bot.cli.antigravity_provider._agy_state_root", return_value=root):
        yield root


def _write_transcript(root: Path, conv_id: str, entries: list[dict[str, str]]) -> None:
    logs = root / "brain" / conv_id / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "transcript.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries), encoding="utf-8"
    )


def _map_cwd(root: Path, cwd: Path, conv_id: str) -> None:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "last_conversations.json").write_text(
        json.dumps({str(cwd): conv_id}), encoding="utf-8"
    )


class TestTranscriptReading:
    """The answer is read from agy's transcript, not its (dropped) stdout."""

    def test_returns_last_planner_response(self, isolated_agy_state: Path, tmp_path: Path) -> None:
        cwd = (tmp_path / "ws").resolve()
        _map_cwd(isolated_agy_state, cwd, "conv-1")
        _write_transcript(
            isolated_agy_state,
            "conv-1",
            [
                {
                    "source": "USER_EXPLICIT",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "content": "hi",
                },
                {**_PLANNER, "content": "intermediate plan"},
                {"source": "MODEL", "type": "LIST_DIRECTORY", "status": "DONE", "content": "ls"},
                {**_PLANNER, "content": "final answer"},
            ],
        )

        assert _read_transcript_answer(cwd) == "final answer"

    def test_ignores_tool_steps_only(self, isolated_agy_state: Path, tmp_path: Path) -> None:
        cwd = (tmp_path / "ws").resolve()
        _map_cwd(isolated_agy_state, cwd, "conv-1")
        _write_transcript(
            isolated_agy_state,
            "conv-1",
            [{"source": "MODEL", "type": "LIST_DIRECTORY", "status": "DONE", "content": "ls"}],
        )

        assert _read_transcript_answer(cwd) is None

    def test_falls_back_to_newest_brain_when_cwd_unmapped(
        self, isolated_agy_state: Path, tmp_path: Path
    ) -> None:
        _write_transcript(isolated_agy_state, "conv-x", [{**_PLANNER, "content": "from newest"}])

        assert _read_transcript_answer((tmp_path / "unmapped").resolve()) == "from newest"

    def test_returns_none_without_state(self, isolated_agy_state: Path, tmp_path: Path) -> None:
        assert _read_transcript_answer((tmp_path / "ws").resolve()) is None


class TestStateRootCrossPlatform:
    """agy's state dir resolves from the subprocess env on every platform."""

    def test_prefers_userprofile_then_home(self) -> None:
        windows = _agy_state_root({"USERPROFILE": "/win/home", "HOME": "/unix/home"})
        assert windows == Path("/win/home") / ".gemini" / "antigravity-cli"

        unix = _agy_state_root({"HOME": "/unix/home"})
        assert unix == Path("/unix/home") / ".gemini" / "antigravity-cli"

    def test_falls_back_to_user_home(self) -> None:
        assert _agy_state_root({}) == Path.home() / ".gemini" / "antigravity-cli"


class TestSendUsesTranscript:
    """send() prefers the transcript answer over (possibly empty) stdout."""

    async def test_prefers_transcript_over_stdout(self, isolated_agy_state: Path) -> None:
        cli = _make_cli()
        _map_cwd(isolated_agy_state, cli._agy_workspace, "conv-1")
        _write_transcript(isolated_agy_state, "conv-1", [{**_PLANNER, "content": "clean answer"}])
        proc = _make_oneshot_process(b"verbose narration on stdout")

        with patch(
            "ductor_bot.cli.antigravity_provider.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            resp = await cli.send("hi")

        assert resp.result == "clean answer"

    async def test_falls_back_to_stdout_without_transcript(self, isolated_agy_state: Path) -> None:
        cli = _make_cli()
        proc = _make_oneshot_process(b"plain stdout answer")

        with patch(
            "ductor_bot.cli.antigravity_provider.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            resp = await cli.send("hi")

        assert resp.result == "plain stdout answer"
