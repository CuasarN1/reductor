"""NDJSON parser for the Antigravity CLI.

Translates Antigravity-specific events into normalized StreamEvents.
Handles both structured NDJSON and plain-text fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ductor_bot.cli.stream_events import (
    AssistantTextDelta,
    ResultEvent,
    StreamEvent,
    SystemInitEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
)

logger = logging.getLogger(__name__)

# Antigravity may emit thinking markers similar to Gemini.
_THOUGHT_MARKER_RE = re.compile(r"^\[Thought:\s*[^\]]+\]")


def parse_antigravity_stream_line(line: str) -> list[StreamEvent]:
    """Parse a single line from Antigravity CLI into normalized stream events.

    Tries JSON first; falls back to plain-text ``AssistantTextDelta``.
    """
    stripped = line.strip()
    if not stripped:
        return []

    # Try NDJSON parse
    try:
        data: dict[str, Any] = json.loads(stripped)
    except json.JSONDecodeError:
        # Plain text line — emit as assistant text delta
        return _split_thought_and_text(stripped)

    return _parse_event(data)


def _parse_event(data: dict[str, Any]) -> list[StreamEvent]:
    """Route a parsed JSON object to the appropriate handler."""
    event_type = data.get("type", "")
    role = data.get("role", "")

    # -- Structured event types -----------------------------------------------
    event_handlers: dict[str, Any] = {
        "init": lambda d: [SystemInitEvent(type="init", session_id=d.get("session_id", ""))],
        "result": lambda d: [_parse_result(d)],
        "error": lambda d: [_parse_error(d)],
        "tool_use": _parse_tool_use,
        "assistant_tool_use": _parse_tool_use,
        "tool_result": lambda d: [_parse_tool_result(d)],
        "tool_result_message": lambda d: [_parse_tool_result(d)],
    }

    handler = event_handlers.get(event_type)
    if handler is not None:
        result: list[StreamEvent] = handler(data)
        return result

    # -- Role-based fallback ---------------------------------------------------
    if role in ("assistant", "model"):
        return _parse_assistant_message(data)

    return []


def _parse_result(data: dict[str, Any]) -> ResultEvent:
    """Parse a result event with usage stats and error detection."""
    content = data.get("content", "")
    result_text = content if isinstance(content, str) else str(content)

    is_error = bool(data.get("is_error", False))
    error_msg = data.get("error", "")
    if error_msg and not is_error:
        is_error = True
        result_text = result_text or str(error_msg)

    return ResultEvent(
        type="result",
        result=result_text,
        session_id=data.get("session_id", ""),
        is_error=is_error,
        duration_ms=data.get("duration_ms"),
        usage=data.get("usage", {}),
    )


def _parse_error(data: dict[str, Any]) -> ResultEvent:
    """Parse a dedicated error event."""
    msg = (
        data.get("message", "")
        or data.get("error", "")
        or data.get("detail", "")
        or "Unknown Antigravity error"
    )
    return ResultEvent(
        type="result",
        result=str(msg),
        is_error=True,
    )


def _parse_tool_use(data: dict[str, Any]) -> list[StreamEvent]:
    """Parse a tool_use event."""
    name = data.get("name", "") or data.get("tool_name", "")
    if not name:
        return []
    return [
        ToolUseEvent(
            type="assistant",
            tool_name=name,
            tool_id=_as_optional_str(data.get("id")),
            parameters=_as_dict(data.get("input") or data.get("parameters")),
        ),
    ]


def _parse_tool_result(data: dict[str, Any]) -> ToolResultEvent:
    """Parse a tool_result event."""
    return ToolResultEvent(
        type="tool_result",
        tool_id=str(data.get("tool_id", "") or data.get("id", "")),
        status=str(data.get("status", "")),
        output=str(data.get("output", "") or data.get("content", "")),
    )


def _parse_assistant_message(data: dict[str, Any]) -> list[StreamEvent]:
    """Parse an assistant/model message — handles structured content blocks."""
    content = data.get("content", "")

    # String content — simple text
    if isinstance(content, str):
        if not content:
            return []
        return _split_thought_and_text(content)

    # Structured content blocks (list of {type, text, ...})
    if isinstance(content, list):
        return _parse_content_blocks(content)

    return []


def _parse_content_blocks(blocks: list[Any]) -> list[StreamEvent]:
    """Parse a list of content blocks into stream events."""
    events: list[StreamEvent] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                events.extend(_split_thought_and_text(text))

        elif block_type == "tool_use":
            name = block.get("name", "")
            if name:
                events.append(
                    ToolUseEvent(
                        type="assistant",
                        tool_name=name,
                        tool_id=_as_optional_str(block.get("id")),
                        parameters=_as_dict(block.get("input")),
                    ),
                )

        elif block_type == "thinking":
            text = block.get("text", "")
            if text:
                events.append(ThinkingEvent(type="assistant", text=text))

    return events


def _split_thought_and_text(text: str) -> list[StreamEvent]:
    r"""Split text into ThinkingEvent + AssistantTextDelta if a thought marker.

    Mirrors the Gemini pattern: ``[Thought: <value>]`` is routed to a
    ThinkingEvent so transports that skip thinking don't leak it.
    """
    match = _THOUGHT_MARKER_RE.match(text)
    if match is None:
        return [AssistantTextDelta(type="assistant", text=text)]

    marker = match.group(0)
    remainder = text[match.end() :].lstrip("\n")
    events: list[StreamEvent] = [ThinkingEvent(type="assistant", text=marker)]
    if remainder:
        events.append(AssistantTextDelta(type="assistant", text=remainder))
    return events


# -- Non-streaming (batch) JSON parsing ----------------------------------------


def parse_antigravity_json(raw: str) -> str:
    """Extract result text from Antigravity CLI JSON batch output.

    Tries to parse as JSON; falls back to raw text truncated to 2000 chars.
    """
    if not raw:
        return ""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Try common content keys
            for key in ("content", "result", "text", "message"):
                val = parsed.get(key)
                if isinstance(val, str) and val:
                    return val
            return str(parsed)
        return str(parsed)
    except json.JSONDecodeError:
        return raw[:2000]


# -- Helpers ------------------------------------------------------------------


def _as_optional_str(val: Any) -> str | None:
    if val is None:
        return None
    return str(val)


def _as_dict(val: Any) -> dict[str, Any] | None:
    if isinstance(val, dict):
        return val
    return None
