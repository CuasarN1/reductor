"""Privacy gates for shared operator memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ductor_bot.infra.atomic_io import atomic_text_save
from ductor_bot.model_policy import is_model_policy_admin, subject_id_for_key

if TYPE_CHECKING:
    from ductor_bot.config import AgentConfig
    from ductor_bot.session.key import SessionKey
    from ductor_bot.workspace.paths import DuctorPaths


GLOBAL_MEMORY_RELATIVE_PATH = "memory_system/MAINMEMORY.md"


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """Resolved long-term memory file for one request scope."""

    path: Path
    relative_path: str
    is_global: bool


_MEMORY_TARGET_TERMS = (
    "mainmemory",
    "main memory",
    "memory_system/mainmemory.md",
    "memory_system\\mainmemory.md",
    "memory_system/mainmemory",
    "memory_system\\mainmemory",
)

_OTHER_USER_HISTORY_TERMS = (
    "what did the owner ask",
    "what did admin ask",
    "owner asked",
    "admin asked",
    "owner conversation",
    "admin conversation",
    "другого пользователя",
    "история владельца",
    "история админа",
    "переписка владельца",
    "переписка админа",
    "что владелец спрашивал",
    "что админ спрашивал",
    "что другой пользователь спрашивал",
)


def can_access_global_memory(config: AgentConfig, key: SessionKey) -> bool:
    """Return whether this request may read or inject global MAINMEMORY.md."""
    return is_model_policy_admin(config, subject_id_for_key(key))


def memory_scope_for_key(paths: DuctorPaths, config: AgentConfig, key: SessionKey) -> MemoryScope:
    """Resolve the memory file that should be visible to this request."""
    if can_access_global_memory(config, key) and key.transport == "tg" and key.chat_id > 0:
        return MemoryScope(
            path=paths.mainmemory_path,
            relative_path=GLOBAL_MEMORY_RELATIVE_PATH,
            is_global=True,
        )

    subject_id = subject_id_for_key(key)
    if key.transport == "tg" and key.chat_id > 0 and subject_id is not None:
        relative_path = f"memory_system/users/{subject_id}/MAINMEMORY.md"
    else:
        chat_part = f"{key.transport}_{key.chat_id}"
        if key.topic_id is None:
            relative_path = f"memory_system/chats/{chat_part}/MAINMEMORY.md"
        else:
            relative_path = f"memory_system/chats/{chat_part}/topics/{key.topic_id}/MAINMEMORY.md"

    return MemoryScope(
        path=paths.workspace / relative_path,
        relative_path=relative_path,
        is_global=False,
    )


def ensure_memory_scope(scope: MemoryScope) -> None:
    """Create an empty scoped memory file so agents can update it safely."""
    if scope.path.exists():
        return
    atomic_text_save(scope.path, "# Main Memory\n")


def memory_scope_instruction(scope: MemoryScope) -> str:
    """System prompt fragment that steers agents to the resolved memory file."""
    if scope.is_global:
        return ""
    return (
        "## MEMORY SCOPE\n"
        f"For this chat, use `{scope.relative_path}` as the long-term memory file.\n"
        f"Do not read or write `{GLOBAL_MEMORY_RELATIVE_PATH}` for this chat unless an "
        "admin explicitly asks for global memory."
    )


def apply_memory_scope_to_prompt(prompt: str, scope: MemoryScope | str) -> str:
    """Rewrite hard-coded global memory prompts to the resolved scoped path."""
    relative_path = scope.relative_path if isinstance(scope, MemoryScope) else scope
    if relative_path == GLOBAL_MEMORY_RELATIVE_PATH:
        return prompt

    scoped = prompt
    if GLOBAL_MEMORY_RELATIVE_PATH in scoped:
        scoped = scoped.replace(GLOBAL_MEMORY_RELATIVE_PATH, relative_path)
    elif "MAINMEMORY.md" in scoped:
        scoped = scoped.replace("MAINMEMORY.md", relative_path)

    return (
        f"{scoped}\n\nMemory scope: use `{relative_path}` for this session. "
        f"Do not read or write `{GLOBAL_MEMORY_RELATIVE_PATH}`."
    )


def global_memory_denied_text() -> str:
    """User-facing denial for private shared memory."""
    return "Global memory is admin-only."


def is_global_memory_disclosure_request(text: str) -> bool:
    """Best-effort detector for prompts asking to disclose shared memory/history."""
    normalized = text.casefold()
    if any(term in normalized for term in _MEMORY_TARGET_TERMS):
        return True
    return any(term in normalized for term in _OTHER_USER_HISTORY_TERMS)
