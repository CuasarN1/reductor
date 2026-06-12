"""Async wrapper around the Antigravity CLI (agy)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ductor_bot.cli.antigravity_events import (
    parse_antigravity_json,
    parse_antigravity_stream_line,
)
from ductor_bot.cli.base import (
    BaseCLI,
    CLIConfig,
    _feed_stdin_and_close,
)
from ductor_bot.cli.executor import build_subprocess_env
from ductor_bot.cli.process_registry import ProcessRegistry, TrackedProcess
from ductor_bot.cli.stream_events import ResultEvent, StreamEvent, SystemInitEvent
from ductor_bot.cli.types import CLIResponse
from ductor_bot.config import ANTIGRAVITY_MODELS
from ductor_bot.infra.platform import CREATION_FLAGS as _CREATION_FLAGS
from ductor_bot.infra.process_tree import force_kill_process_tree

if TYPE_CHECKING:
    from ductor_bot.cli.timeout_controller import TimeoutController

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300.0  # 5 minutes, matches agy --print-timeout default


@dataclass(slots=True)
class _AntigravityStreamState:
    """Mutable stream-state for Antigravity event processing."""

    last_session_id: str | None = None
    saw_result: bool = False

    def track(self, event: StreamEvent) -> None:
        """Track session + final-result information from one stream event."""
        if isinstance(event, (SystemInitEvent, ResultEvent)) and event.session_id:
            self.last_session_id = event.session_id
        if isinstance(event, ResultEvent):
            self.saw_result = True
            if not event.session_id:
                event.session_id = self.last_session_id


class AntigravityCLI(BaseCLI):
    """Async wrapper around the Antigravity CLI (agy).

    agy flags reference:
      --print / -p <prompt>   Non-interactive single-shot
      --prompt-interactive    Interactive mode with initial prompt via stdin
      --continue / -c         Continue most recent conversation
      --conversation <id>     Resume a specific conversation
      --dangerously-skip-permissions  Auto-approve all tools
      --print-timeout <dur>   Timeout for --print mode (default 5m)
      --sandbox               Sandbox mode
      --add-dir <dir>         Add workspace directory
      --log-file <path>       Override log file
    """

    def __init__(self, config: CLIConfig) -> None:
        self._config = config
        self._working_dir = Path(config.working_dir).resolve()
        self._cli = "agy"
        logger.info("AntigravityCLI: cwd=%s model=%s", self._working_dir, config.model)

    def _build_command(
        self,
        *,
        streaming: bool = False,
        resume_session: str | None = None,
        continue_session: bool = False,
    ) -> list[str]:
        """Build the agy command list.

        Non-streaming uses ``--print`` (prompt passed as argument).
        Streaming uses ``--prompt-interactive`` (prompt piped via stdin).
        """
        cmd = [self._cli]

        if not streaming:
            cmd += ["--print"]
        else:
            cmd += ["--prompt-interactive"]

        if self._config.model and self._config.model not in ANTIGRAVITY_MODELS:
            cmd += ["--model", self._config.model]

        # Session resume / continue
        if resume_session:
            cmd += ["--conversation", resume_session]
        elif continue_session:
            cmd += ["--continue"]

        # Auto-approve when bypass mode is set
        if self._config.permission_mode == "bypassPermissions":
            cmd += ["--dangerously-skip-permissions"]

        cmd.extend(self._config.cli_parameters)
        return cmd

    def _host_command(self, cmd: list[str]) -> tuple[list[str], str]:
        """Return a host-execution command.

        Antigravity is a host CLI. The standard Docker sandbox image does not
        include ``agy`` or its user auth state, so running it through
        ``docker exec`` produces an OCI "agy not found" error.
        """
        if self._config.docker_container:
            logger.info("Antigravity runs on host; ignoring Docker container for agy")
        return cmd, str(self._working_dir)

    # -- Process tracking -----------------------------------------------------

    def _track_process(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[ProcessRegistry | None, TrackedProcess | None]:
        """Register a subprocess in ProcessRegistry if tracking is enabled."""
        reg = self._config.process_registry
        tracked = (
            reg.register(
                self._config.chat_id,
                process,
                self._config.process_label,
                topic_id=self._config.topic_id,
            )
            if reg
            else None
        )
        return reg, tracked

    @staticmethod
    def _untrack_process(reg: ProcessRegistry | None, tracked: TrackedProcess | None) -> None:
        """Unregister a previously tracked subprocess."""
        if tracked is not None and reg is not None:
            reg.unregister(tracked)

    # -- Non-streaming --------------------------------------------------------

    async def send(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
        timeout_controller: TimeoutController | None = None,
    ) -> CLIResponse:
        """Send a non-streaming prompt via ``agy --print``."""
        effective_timeout = timeout_seconds or _DEFAULT_TIMEOUT
        cmd = self._build_command(
            resume_session=resume_session,
            continue_session=continue_session,
        )
        # --print takes prompt as a positional argument
        cmd.append(prompt)

        cmd, cwd = self._host_command(cmd)
        safe_cmd = _safe_command_for_logging(cmd)
        logger.debug("Antigravity send (non-streaming): %s", safe_cmd)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=build_subprocess_env(self._config),
            creationflags=_CREATION_FLAGS,
        )

        reg, tracked = self._track_process(proc)

        try:
            timed_out = False
            try:
                communicate_coro = proc.communicate()
                if timeout_controller is not None:
                    stdout_bytes, stderr_bytes = await timeout_controller.run_with_timeout(
                        communicate_coro,
                    )
                else:
                    async with asyncio.timeout(effective_timeout):
                        stdout_bytes, stderr_bytes = await communicate_coro
            except TimeoutError:
                timed_out = True
                logger.warning("Antigravity send timed out")
                force_kill_process_tree(proc.pid)
                stdout_bytes, stderr_bytes = await proc.communicate()
                return CLIResponse(
                    result="Timeout",
                    is_error=True,
                    timed_out=True,
                    returncode=proc.returncode,
                    stderr=stderr_bytes.decode(errors="replace")[:2000] if stderr_bytes else "",
                )
        finally:
            self._untrack_process(reg, tracked)
            if not timed_out and proc.returncode is None:
                force_kill_process_tree(proc.pid)

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

        result_text = parse_antigravity_json(stdout)
        is_error = proc.returncode not in (None, 0)

        return CLIResponse(
            result=result_text,
            is_error=is_error,
            returncode=proc.returncode,
            stderr=stderr,
        )

    # -- Streaming ------------------------------------------------------------

    async def send_streaming(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
        timeout_controller: TimeoutController | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream events from ``agy --prompt-interactive``."""
        effective_timeout = timeout_seconds or _DEFAULT_TIMEOUT
        cmd = self._build_command(
            streaming=True,
            resume_session=resume_session,
            continue_session=continue_session,
        )
        cmd, cwd = self._host_command(cmd)

        safe_cmd = _safe_command_for_logging(cmd)
        logger.debug("Antigravity send_streaming: %s", safe_cmd)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=build_subprocess_env(self._config),
            limit=4 * 1024 * 1024,
            creationflags=_CREATION_FLAGS,
        )

        if proc.stderr is None:
            msg = "Antigravity subprocess created without stderr pipe"
            raise RuntimeError(msg)

        stderr_task = asyncio.create_task(proc.stderr.read())
        reg, tracked = self._track_process(proc)

        state = _AntigravityStreamState(last_session_id=resume_session)
        timed_out = False

        try:
            await _feed_stdin_and_close(proc, prompt)
            try:
                async for event in self._stream_events(
                    proc,
                    state,
                    effective_timeout,
                    timeout_controller=timeout_controller,
                ):
                    yield event
            except TimeoutError:
                timed_out = True
                yield ResultEvent(
                    type="result",
                    result="Timeout",
                    is_error=True,
                    session_id=state.last_session_id,
                )
        finally:
            stderr_bytes = await _finish_stream_process(proc, stderr_task)
            self._untrack_process(reg, tracked)

        # Emit synthetic result if stream ended without one
        final_event = _build_stream_exit_event(
            returncode=proc.returncode,
            stderr_bytes=stderr_bytes,
            state=state,
        )
        if final_event is not None and not timed_out:
            yield final_event

    async def _stream_events(
        self,
        proc: asyncio.subprocess.Process,
        state: _AntigravityStreamState,
        timeout_seconds: float,
        *,
        timeout_controller: TimeoutController | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Read lines and yield normalized stream events."""
        if proc.stdout is None:
            msg = "Antigravity subprocess created without stdout pipe"
            raise RuntimeError(msg)

        if timeout_controller is not None:
            async for event in _stream_events_with_controller(
                proc,
                state,
                timeout_controller=timeout_controller,
            ):
                yield event
        else:
            async for event in _stream_events_plain(
                proc,
                state,
                timeout_seconds=timeout_seconds,
            ):
                yield event


# -- Module-level helpers -----------------------------------------------------


async def _stream_events_plain(
    proc: asyncio.subprocess.Process,
    state: _AntigravityStreamState,
    *,
    timeout_seconds: float,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream events with a fixed timeout."""
    assert proc.stdout is not None
    async with asyncio.timeout(timeout_seconds):
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode(errors="replace")
            for event in parse_antigravity_stream_line(line):
                state.track(event)
                yield event


async def _stream_events_with_controller(
    proc: asyncio.subprocess.Process,
    state: _AntigravityStreamState,
    *,
    timeout_controller: TimeoutController,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream events with a TimeoutController that supports extension."""
    assert proc.stdout is not None
    while True:
        try:
            async with asyncio.timeout(timeout_controller.activity_extension_seconds):
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace")
                for event in parse_antigravity_stream_line(line):
                    state.track(event)
                    yield event
        except TimeoutError:
            if timeout_controller.try_extend():
                continue
            raise


async def _finish_stream_process(
    proc: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[bytes],
) -> bytes:
    """Ensure process shutdown and return collected stderr."""
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except TimeoutError:
            force_kill_process_tree(proc.pid)
            await proc.wait()
    else:
        await proc.wait()
    return await stderr_task


def _build_stream_exit_event(
    *,
    returncode: int | None,
    stderr_bytes: bytes,
    state: _AntigravityStreamState,
) -> ResultEvent | None:
    """Build a synthetic final ResultEvent when the stream lacked one."""
    if state.saw_result:
        return None

    if returncode == 0:
        return ResultEvent(
            type="result",
            result="",
            is_error=False,
            returncode=returncode,
            session_id=state.last_session_id,
        )

    detail = stderr_bytes.decode(errors="replace").strip()
    if not detail:
        detail = f"Antigravity exited with code {returncode}"
    return ResultEvent(
        type="result",
        result=detail[:500],
        is_error=True,
        returncode=returncode,
        session_id=state.last_session_id,
    )


def _safe_command_for_logging(cmd: list[str]) -> list[str]:
    """Return a command safe for debug logs."""
    safe = [part if len(part) <= 80 else part[:80] + "..." for part in cmd]
    if "--print" in cmd and safe:
        safe[-1] = "<prompt>"
    return safe
