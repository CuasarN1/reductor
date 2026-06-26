"""Async wrapper around the OpenAI Codex CLI."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from shutil import copy2, which
from typing import TYPE_CHECKING

from ductor_bot.cli.base import (
    _IS_WINDOWS,
    BaseCLI,
    CLIConfig,
    docker_wrap,
)
from ductor_bot.cli.codex_events import (
    CodexThinkingFilter,
    parse_codex_jsonl,
    parse_codex_stream_event,
)
from ductor_bot.cli.executor import (
    SubprocessResult,
    SubprocessSpec,
    run_oneshot_subprocess,
    run_streaming_subprocess,
)
from ductor_bot.cli.stream_events import (
    AssistantTextDelta,
    ResultEvent,
    StreamEvent,
    SystemInitEvent,
)
from ductor_bot.cli.types import CLIResponse

if TYPE_CHECKING:
    from ductor_bot.cli.timeout_controller import TimeoutController

logger = logging.getLogger(__name__)

_CODEX_STDIN_NOTICE_PREFIXES = (
    "Reading prompt from stdin",
    "Reading additional input from stdin",
)
_CODEX_NO_FINAL_RESPONSE = "Codex failed before producing a final response."


class _StreamState:
    """Mutable accumulator for streaming session data."""

    __slots__ = ("accumulated_text", "generated_files", "last_error_message", "thread_id")

    def __init__(self) -> None:
        self.accumulated_text: list[str] = []
        self.generated_files: list[str] = []
        self.thread_id: str | None = None
        # Captures the message from any ResultEvent(is_error=True) seen in
        # the stream (e.g. Codex `turn.failed`), so _codex_final_result can
        # surface the real cause instead of downstream stderr artefacts.
        self.last_error_message: str | None = None

    def track(self, event: StreamEvent) -> None:
        """Update state from a single stream event."""
        if isinstance(event, SystemInitEvent) and event.session_id:
            self.thread_id = event.session_id
        elif isinstance(event, AssistantTextDelta) and event.text:
            self.accumulated_text.append(event.text)
        elif isinstance(event, ResultEvent) and event.is_error and event.result:
            self.last_error_message = event.result

    def add_generated_file(self, path: Path) -> None:
        """Track a materialized generated file once."""
        path_str = str(path)
        if path_str not in self.generated_files:
            self.generated_files.append(path_str)


class CodexCLI(BaseCLI):
    """Async wrapper around the OpenAI Codex CLI."""

    def __init__(self, config: CLIConfig) -> None:
        self._config = config
        self._working_dir = Path(config.working_dir).resolve()
        self._cli = "codex" if config.docker_container else self._find_cli()
        logger.info("Codex CLI wrapper: cwd=%s, model=%s", self._working_dir, config.model)

    @staticmethod
    def _find_cli() -> str:
        path = which("codex")
        if not path:
            msg = "codex CLI not found on PATH. Install via: npm install -g @openai/codex"
            raise FileNotFoundError(msg)
        return path

    def _compose_prompt(self, prompt: str) -> str:
        """Inject system context into user prompt (Codex has no --system-prompt)."""
        cfg = self._config
        parts: list[str] = []
        if cfg.system_prompt:
            parts.append(cfg.system_prompt)
        parts.append(prompt)
        if cfg.append_system_prompt:
            parts.append(cfg.append_system_prompt)
        return "\n\n".join(parts)

    def _sandbox_flags(self) -> list[str]:
        """Return sandbox/approval flags based on permission_mode."""
        cfg = self._config
        if cfg.permission_mode == "bypassPermissions":
            return ["--dangerously-bypass-approvals-and-sandbox"]
        if cfg.sandbox_mode == "full-access":
            return ["--sandbox", "danger-full-access"]
        if cfg.sandbox_mode == "workspace-write":
            return ["--full-auto"]
        return ["--sandbox", cfg.sandbox_mode]

    def _build_resume_command(
        self, final_prompt: str, session_id: str, *, json_output: bool
    ) -> list[str]:
        """Build command to resume an existing Codex session."""
        cmd = [self._cli, "exec", "resume"]
        if json_output:
            cmd.append("--json")
        cmd += self._sandbox_flags()
        cmd += ["--", session_id]
        cmd.append("-" if _IS_WINDOWS else final_prompt)
        return cmd

    def _build_command(
        self,
        prompt: str,
        resume_session: str | None = None,
        *,
        json_output: bool = True,
    ) -> list[str]:
        cfg = self._config
        final_prompt = self._compose_prompt(prompt)

        if resume_session:
            return self._build_resume_command(final_prompt, resume_session, json_output=json_output)

        cmd = [self._cli, "exec"]
        if json_output:
            cmd.append("--json")
        cmd += ["--color", "never"]
        cmd += self._sandbox_flags()
        cmd.append("--skip-git-repo-check")

        if cfg.model:
            cmd += ["--model", cfg.model]
        if cfg.reasoning_effort and cfg.reasoning_effort != "default":
            cmd += ["-c", f"model_reasoning_effort={cfg.reasoning_effort}"]
        if cfg.instructions:
            cmd += ["--instructions", cfg.instructions]
        for img in cfg.images:
            cmd += ["--image", img]

        # Add extra CLI parameters before the separator
        if cfg.cli_parameters:
            cmd.extend(cfg.cli_parameters)

        cmd.append("--")
        # On Windows, .CMD wrappers mangle arguments with special characters.
        # Use "-" so Codex reads stdin without emitting the "Reading prompt..."
        # notice as a stdout prelude before JSONL.
        cmd.append("-" if _IS_WINDOWS else final_prompt)
        return cmd

    def _docker_wrap(self, cmd: list[str]) -> tuple[list[str], str | None]:
        """Keep stdin open for Dockerized Codex on Windows so prompts reach the CLI."""
        return docker_wrap(
            cmd,
            self._config,
            interactive=_IS_WINDOWS,
        )

    async def send(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
        timeout_controller: TimeoutController | None = None,
    ) -> CLIResponse:
        """Send a prompt and return the final result."""
        if continue_session:
            logger.debug("continue_session is not supported by Codex CLI, ignoring")
        cmd = self._build_command(prompt, resume_session, json_output=True)
        exec_cmd, use_cwd = self._docker_wrap(cmd)
        _log_cmd(exec_cmd)
        return await run_oneshot_subprocess(
            config=self._config,
            spec=SubprocessSpec(exec_cmd, use_cwd, prompt, timeout_seconds, timeout_controller),
            parse_output=lambda stdout, stderr, returncode: self._parse_output(
                stdout,
                stderr,
                returncode,
                generated_output_dir=self._generated_output_dir(),
            ),
            provider_label="Codex",
        )

    async def send_streaming(
        self,
        prompt: str,
        resume_session: str | None = None,
        continue_session: bool = False,
        timeout_seconds: float | None = None,
        timeout_controller: TimeoutController | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send a prompt and yield stream events as they arrive."""
        cmd = self._build_command(prompt, resume_session, json_output=True)
        exec_cmd, use_cwd = self._docker_wrap(cmd)
        _log_cmd(exec_cmd, streaming=True)

        state = _StreamState()
        thinking_filter = CodexThinkingFilter()

        async def line_handler(line: str) -> AsyncGenerator[StreamEvent, None]:
            if not line:
                return
            for raw_event in parse_codex_stream_event(line):
                for event in thinking_filter.process(raw_event):
                    state.track(event)
                    yield event
            for generated_file in _materialize_codex_generated_images(
                line,
                self._generated_output_dir(),
                state.thread_id,
            ):
                state.add_generated_file(generated_file)
            for event in thinking_filter.flush():
                state.track(event)
                yield event

        async def post_handler(result: SubprocessResult) -> AsyncGenerator[StreamEvent, None]:
            yield _codex_final_result(
                result,
                state.accumulated_text,
                state.thread_id,
                state.last_error_message,
                generated_files=state.generated_files,
            )

        async for event in run_streaming_subprocess(
            config=self._config,
            spec=SubprocessSpec(exec_cmd, use_cwd, prompt, timeout_seconds, timeout_controller),
            line_handler=line_handler,
            provider_label="Codex",
            post_handler=post_handler,
        ):
            yield event

    @staticmethod
    def _parse_output(
        stdout: bytes,
        stderr: bytes,
        returncode: int | None,
        *,
        generated_output_dir: Path | None = None,
    ) -> CLIResponse:
        """Parse Codex subprocess output into a CLIResponse."""
        stderr_text = stderr.decode(errors="replace")[:2000] if stderr else ""
        if stderr_text:
            logger.warning("Codex stderr (exit=%s): %s", returncode, stderr_text[:500])

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            logger.error("Codex returned empty output (exit=%s)", returncode)
            return CLIResponse(
                result=_strip_codex_stdin_notices(stderr_text),
                is_error=True,
                returncode=returncode,
                stderr=stderr_text,
            )

        is_error = returncode != 0
        result_text, thread_id, usage = parse_codex_jsonl(raw)
        generated_files = _materialize_codex_generated_images(
            raw,
            generated_output_dir,
            thread_id,
        )
        cleaned_stdout = _strip_codex_stdin_notices(raw)
        cleaned_stderr = _strip_codex_stdin_notices(stderr_text)
        parsed_error = _extract_codex_error_detail(raw)
        if result_text:
            result = _append_file_tags(result_text, generated_files)
        elif generated_files and not is_error:
            result = _append_file_tags("", generated_files)
        elif is_error:
            stdout_fallback = "" if _is_codex_protocol_only(raw) else cleaned_stdout
            result = parsed_error or cleaned_stderr or stdout_fallback or _CODEX_NO_FINAL_RESPONSE
        elif _is_codex_protocol_only(raw):
            result = _CODEX_NO_FINAL_RESPONSE
        else:
            result = raw
        response = CLIResponse(
            session_id=thread_id,
            result=result,
            is_error=is_error or not (result_text or generated_files),
            returncode=returncode,
            stderr=stderr_text,
            usage=usage or {},
        )

        if response.is_error:
            logger.error("Codex error exit=%s: %s", returncode, response.result[:300])
        else:
            logger.info(
                "Codex done session=%s tokens=%d",
                (response.session_id or "?")[:8],
                response.total_tokens,
            )

        return response

    def _generated_output_dir(self) -> Path:
        """Directory where generated media can be exposed via <file:...> tags."""
        return self._working_dir / "output_to_user"


def _codex_final_result(
    result: SubprocessResult,
    accumulated_text: list[str],
    thread_id: str | None,
    last_error_message: str | None = None,
    *,
    generated_files: list[str] | None = None,
) -> ResultEvent:
    """Build the final ResultEvent after the stream loop completes.

    On non-zero exit the user-facing detail is chosen in this order:

    1. ``last_error_message`` — captured from in-stream ResultEvent(is_error=True),
       e.g. Codex ``turn.failed`` (the real cause).
    2. Joined ``accumulated_text`` — partial assistant output that arrived
       before the failure.
    3. ``stderr_text`` — raw stderr (often a downstream artefact such as
       ``thread … not found`` after a failed Codex turn).
    4. ``"(no output)"`` — fallback.
    """
    stderr_text = result.stderr_bytes.decode(errors="replace")[:2000] if result.stderr_bytes else ""
    stderr_detail = _strip_codex_stdin_notices(stderr_text)

    if result.process.returncode != 0:
        error_detail = (
            last_error_message or "\n".join(accumulated_text) or stderr_detail or "(no output)"
        )
        logger.error(
            "Codex stream exited with code %d: %s",
            result.process.returncode,
            error_detail[:300],
        )
        return ResultEvent(
            type="result",
            result=error_detail[:500],
            is_error=True,
            returncode=result.process.returncode,
        )

    return ResultEvent(
        type="result",
        session_id=thread_id,
        result=_append_file_tags("\n".join(accumulated_text), generated_files or []),
        is_error=False,
        returncode=result.process.returncode,
    )


def _append_file_tags(text: str, file_paths: list[str]) -> str:
    """Append file tags to the final text so messenger transports send them."""
    unique_paths = list(dict.fromkeys(path for path in file_paths if path))
    if not unique_paths:
        return text
    tags = "\n".join(f"<file:{path}>" for path in unique_paths)
    stripped = text.rstrip()
    if not stripped:
        return tags
    return f"{stripped}\n\n{tags}"


def _materialize_codex_generated_images(
    raw: str,
    output_dir: Path | None,
    session_id: str | None,
) -> list[Path]:
    """Persist Codex image_generation_end payloads under output_to_user."""
    if output_dir is None:
        return []

    paths: list[Path] = []
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        path = _materialize_codex_generated_image(data, output_dir, session_id)
        if path is not None and path not in paths:
            paths.append(path)
    return paths


def _materialize_codex_generated_image(
    data: dict[str, object],
    output_dir: Path,
    session_id: str | None,
) -> Path | None:
    """Persist one Codex image_generation_end event and return the public path."""
    if data.get("type") != "image_generation_end":
        return None

    safe_call_id = _codex_generated_image_safe_call_id(data)
    if safe_call_id is None:
        return None

    target = output_dir / f"codex_{safe_call_id}.png"
    if not _ensure_codex_generated_output_dir(output_dir):
        return None

    source = _codex_generated_image_source(session_id, safe_call_id)
    if _path_has_content(target) or _materialize_codex_generated_image_target(
        data,
        target,
        source,
    ):
        return target

    logger.warning("Codex generated image was not materialized for call_id=%s", safe_call_id)
    return None


def _codex_generated_image_safe_call_id(data: dict[str, object]) -> str | None:
    """Extract a path-safe Codex image call ID."""
    call_id = data.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        logger.warning("Codex image_generation_end without call_id")
        return None
    safe_call_id = _safe_codex_file_stem(call_id)
    if not safe_call_id:
        logger.warning("Codex image_generation_end has unsafe call_id: %r", call_id)
        return None
    return safe_call_id


def _ensure_codex_generated_output_dir(output_dir: Path) -> bool:
    """Create the generated-image output directory if needed."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Failed to create Codex generated image directory: %s", output_dir)
        return False
    return True


def _materialize_codex_generated_image_target(
    data: dict[str, object],
    target: Path,
    source: Path | None,
) -> bool:
    """Write or copy a Codex generated image into *target*."""
    payload = data.get("result")
    if (
        isinstance(payload, str)
        and payload.strip()
        and _write_codex_generated_image_payload(
            payload,
            target,
        )
    ):
        return True

    if source is None or not source.exists():
        return False
    try:
        copy2(source, target)
    except OSError:
        logger.warning("Failed to copy Codex generated image: %s", source)
        return False
    return True


def _path_has_content(path: Path) -> bool:
    """Return True when *path* exists and has non-zero size."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _write_codex_generated_image_payload(payload: str, target: Path) -> bool:
    """Decode a Codex base64 image payload into *target*."""
    encoded = payload.strip()
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        target.write_bytes(image_bytes)
    except (binascii.Error, OSError, ValueError):
        logger.warning("Failed to decode Codex generated image payload")
        return False
    return True


def _codex_generated_image_source(session_id: str | None, safe_call_id: str) -> Path | None:
    """Return Codex CLI's own generated image path, when it can be inferred."""
    if not session_id:
        return None
    return Path.home() / ".codex" / "generated_images" / session_id / f"{safe_call_id}.png"


def _safe_codex_file_stem(value: str) -> str:
    """Keep Codex-generated filenames path-safe without changing common call IDs."""
    return "".join(ch for ch in value.strip() if ch.isalnum() or ch in {"-", "_", "."}).strip(".-_")


def _is_codex_stdin_notice(line: str) -> bool:
    """Return True for Codex's informational stdin prelude lines."""
    stripped = line.strip()
    return any(stripped.startswith(prefix) for prefix in _CODEX_STDIN_NOTICE_PREFIXES)


def _strip_codex_stdin_notices(text: str) -> str:
    """Remove stdin notice lines from Codex stdout/stderr text."""
    return "\n".join(line for line in text.splitlines() if not _is_codex_stdin_notice(line)).strip()


def _is_codex_protocol_only(raw: str) -> bool:
    """Return True when stdout contains only Codex JSONL protocol events."""
    saw_protocol_event = False
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped or _is_codex_stdin_notice(stripped):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        if not isinstance(data, dict):
            return False
        if (
            isinstance(data.get("type"), str)
            or isinstance(data.get("item"), dict)
            or isinstance(data.get("thread_id"), str)
            or isinstance(data.get("usage"), dict)
        ):
            saw_protocol_event = True
            continue
        return False
    return saw_protocol_event


def _extract_codex_error_detail(raw: str) -> str:
    """Extract a structured Codex error message from JSONL stdout."""
    detail = ""
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped or _is_codex_stdin_notice(stripped):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("type") == "turn.failed":
            error = data.get("error", {})
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    detail = message.strip()
        item = data.get("item")
        if isinstance(item, dict) and item.get("type") == "error":
            message = item.get("message")
            if isinstance(message, str) and message.strip():
                detail = message.strip()
    return detail


def _log_cmd(cmd: list[str], *, streaming: bool = False) -> None:
    """Log the CLI command with truncated long values."""
    safe_cmd = [(c[:80] + "...") if len(c) > 80 else c for c in cmd]
    prefix = "Codex stream cmd" if streaming else "Codex cmd"
    logger.info("%s: %s", prefix, " ".join(safe_cmd))
