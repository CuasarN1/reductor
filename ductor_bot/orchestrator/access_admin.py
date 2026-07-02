"""Admin-only access and per-user model-policy commands."""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from re import fullmatch
from typing import TYPE_CHECKING, cast

from ductor_bot.config import ModelPolicyRule, update_config_file_async
from ductor_bot.model_policy import is_model_policy_admin, subject_id_for_key
from ductor_bot.orchestrator.registry import OrchestratorResult

if TYPE_CHECKING:
    from ductor_bot.orchestrator.core import Orchestrator
    from ductor_bot.session.key import SessionKey

_AccessHandler = Callable[["Orchestrator", list[str]], Awaitable[OrchestratorResult]]
_PolicyPatch = dict[str, object]

_INHERIT = {"clear", "default", "inherit", "unset"}
_TRUE = {"1", "allow", "allowed", "enable", "enabled", "on", "true", "yes"}
_FALSE = {"0", "deny", "denied", "disable", "disabled", "false", "no", "off"}
_OPTION_ALIASES = {
    "admin": "admin",
    "effort": "efforts",
    "efforts": "efforts",
    "model": "models",
    "models": "models",
    "model_switch": "switch",
    "reasoning": "efforts",
    "switch": "switch",
}
_POLICY_OPTION_KEYS = frozenset({"efforts", "models", "switch"})
_USERNAME_PATTERN = r"@?[A-Za-z][A-Za-z0-9_]{4,31}"


async def cmd_access(orch: Orchestrator, key: SessionKey, text: str) -> OrchestratorResult:
    """Handle /access admin commands."""
    caller_id = subject_id_for_key(key)
    if not is_model_policy_admin(orch._config, caller_id):
        return OrchestratorResult(text="Access management is admin-only.")

    tokens, error = _split_command(text)
    if error is not None:
        return OrchestratorResult(text=error)
    if not tokens or tokens[0].lower() in {"help", "-h", "--help"}:
        return OrchestratorResult(text=_help_text())

    action = tokens[0].lower()
    args = tokens[1:]
    handler = _ACTION_HANDLERS.get(action)
    if handler is not None:
        return await handler(orch, args)
    return OrchestratorResult(text=f"Unknown /access action `{action}`.\n\n{_help_text()}")


def _split_command(text: str) -> tuple[list[str], str | None]:
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return [], f"Could not parse /access command: {exc}"
    return parts[1:], None


def _help_text() -> str:
    return (
        "Access management\n"
        "\n"
        "Commands:\n"
        "- `/access list`\n"
        "- `/access add <user_id|@username> [models=...] [efforts=...] [switch=on|off] "
        "[admin=on|off]`\n"
        "- `/access policy <user_id|@username> [models=...] [efforts=...] [switch=on|off] "
        "[admin=on|off]`\n"
        "- `/access default [models=...] [efforts=...] [switch=on|off]`\n"
        "- `/access group list`\n"
        "- `/access group add <group_id>`\n"
        "- `/access group remove <group_id>`\n"
        "- `/access admin <user_id|@username> on|off`\n"
        "- `/access remove <user_id|@username>`\n"
        "\n"
        "Examples:\n"
        "- `/access add 123456789 models=gpt-5.4-mini,gpt-5.4 efforts=low,medium "
        "switch=off`\n"
        "- `/access add @somebody`\n"
        "- `/access policy 123456789 models=* efforts=* switch=on`\n"
        "- `/access group add -1001234567890`"
    )


def _parse_user_id(raw: str) -> tuple[int | None, str | None]:
    try:
        user_id = int(raw)
    except ValueError:
        return None, f"Invalid Telegram user ID `{raw}`."
    if user_id <= 0:
        return None, f"Telegram user ID must be positive: `{raw}`."
    return user_id, None


def _normalize_username(raw: str) -> str | None:
    value = raw.strip()
    if not fullmatch(_USERNAME_PATTERN, value):
        return None
    return value.lstrip("@")


async def _resolve_user_ref(orch: Orchestrator, raw: str) -> tuple[int | None, str | None]:
    """Resolve a user target accepted by /access: numeric id or @username."""
    user_id, id_error = _parse_user_id(raw)
    if user_id is not None:
        return user_id, None
    if raw.strip().lstrip("-").isdigit():
        return None, id_error

    username = _normalize_username(raw)
    if username is None:
        return None, f"Invalid Telegram user target `{raw}`; use a numeric ID or @username."

    resolved = await orch.resolve_access_username(username)
    if resolved is None:
        return (
            None,
            f"Could not resolve Telegram username `@{username}` to a private user. "
            "Ask them to message the bot once, then retry, or use their numeric Telegram ID.",
        )
    return resolved, None


def _parse_group_id(raw: str) -> tuple[int | None, str | None]:
    try:
        group_id = int(raw)
    except ValueError:
        return None, f"Invalid Telegram group ID `{raw}`."
    if group_id >= 0:
        return None, f"Telegram group ID must be negative: `{raw}`."
    return group_id, None


def _parse_bool(raw: str, *, allow_inherit: bool = False) -> tuple[bool | None, str | None]:
    value = raw.strip().lower()
    if allow_inherit and value in _INHERIT:
        return None, None
    if value in _TRUE:
        return True, None
    if value in _FALSE:
        return False, None
    allowed = "on/off"
    if allow_inherit:
        allowed += "/inherit"
    return None, f"Invalid boolean value `{raw}`; use {allowed}."


def _parse_list(raw: str) -> tuple[list[str] | None, str | None]:
    value = raw.strip()
    if value.lower() in _INHERIT:
        return None, None
    if value.lower() in {"all", "any"}:
        return ["*"], None

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None, f"List value `{raw}` is empty."
    return items, None


def _parse_options(
    args: list[str],
    *,
    allowed: frozenset[str],
) -> tuple[list[str], dict[str, str], str | None]:
    positionals: list[str] = []
    options: dict[str, str] = {}
    for arg in args:
        if "=" not in arg:
            positionals.append(arg)
            continue
        raw_key, value = arg.split("=", 1)
        key = _OPTION_ALIASES.get(raw_key.strip().lower())
        if key is None or key not in allowed:
            return [], {}, f"Unknown option `{raw_key}`."
        options[key] = value
    return positionals, options, None


async def _parse_policy_target(
    orch: Orchestrator,
    args: list[str],
    *,
    usage: str,
    allowed: frozenset[str],
) -> tuple[int | None, dict[str, str], OrchestratorResult | None]:
    positionals, options, error = _parse_options(args, allowed=allowed)
    if error is not None:
        return None, {}, OrchestratorResult(text=error)
    if len(positionals) != 1:
        return None, {}, OrchestratorResult(text=usage)

    user_id, error = await _resolve_user_ref(orch, positionals[0])
    if error is not None or user_id is None:
        return None, {}, OrchestratorResult(text=error or "Invalid Telegram user target.")
    return user_id, options, None


def _parse_policy_options(options: dict[str, str]) -> tuple[_PolicyPatch, str | None]:
    patch: _PolicyPatch = {}
    if "models" in options:
        models, error = _parse_list(options["models"])
        if error is not None:
            return {}, error
        patch["allowed_models"] = models

    if "efforts" in options:
        efforts, error = _parse_list(options["efforts"])
        if error is not None:
            return {}, error
        patch["allowed_reasoning_efforts"] = efforts

    if "switch" in options:
        switch, error = _parse_bool(options["switch"], allow_inherit=True)
        if error is not None:
            return {}, error
        patch["allow_model_switch"] = switch

    return patch, None


def _apply_policy_patch(rule: ModelPolicyRule, patch: _PolicyPatch) -> None:
    if "allowed_models" in patch:
        rule.allowed_models = cast("list[str] | None", patch["allowed_models"])
    if "allowed_reasoning_efforts" in patch:
        rule.allowed_reasoning_efforts = cast(
            "list[str] | None",
            patch["allowed_reasoning_efforts"],
        )
    if "allow_model_switch" in patch:
        rule.allow_model_switch = cast("bool | None", patch["allow_model_switch"])


def _owner_id(orch: Orchestrator) -> int | None:
    users = orch._config.allowed_user_ids
    return users[0] if users else None


def _add_allowed_user(orch: Orchestrator, user_id: int) -> bool:
    if user_id in orch._config.allowed_user_ids:
        return False
    orch._config.allowed_user_ids.append(user_id)
    return True


def _add_allowed_group(orch: Orchestrator, group_id: int) -> bool:
    if group_id in orch._config.allowed_group_ids:
        return False
    orch._config.allowed_group_ids.append(group_id)
    return True


def _remove_allowed_group(orch: Orchestrator, group_id: int) -> bool:
    before = list(orch._config.allowed_group_ids)
    orch._config.allowed_group_ids = [gid for gid in before if gid != group_id]
    return group_id in before


def _set_policy_admin(orch: Orchestrator, user_id: int, *, enabled: bool) -> bool:
    admins = list(dict.fromkeys(orch._config.model_policy.admin_user_ids))
    changed = False
    if enabled and user_id not in admins:
        admins.append(user_id)
        changed = True
    if not enabled and user_id in admins:
        admins.remove(user_id)
        changed = True
    orch._config.model_policy.admin_user_ids = admins
    return changed


def _full_access_rule() -> ModelPolicyRule:
    return ModelPolicyRule(
        allowed_models=["*"],
        allowed_reasoning_efforts=["*"],
        allow_model_switch=True,
    )


def _user_rule(orch: Orchestrator, user_id: int, *, admin: bool = False) -> ModelPolicyRule:
    users = orch._config.model_policy.users
    key = str(user_id)
    if key not in users:
        users[key] = _full_access_rule() if admin else ModelPolicyRule(allow_model_switch=False)
    return users[key]


async def _persist(orch: Orchestrator, *, include_groups: bool = False) -> None:
    updates: dict[str, object] = {
        "allowed_user_ids": list(orch._config.allowed_user_ids),
        "model_policy": orch._config.model_policy.model_dump(mode="json"),
    }
    hot: dict[str, object] = {
        "allowed_user_ids": orch._config.allowed_user_ids,
        "model_policy": orch._config.model_policy,
    }
    if include_groups:
        updates["allowed_group_ids"] = list(orch._config.allowed_group_ids)
        hot["allowed_group_ids"] = orch._config.allowed_group_ids

    await update_config_file_async(orch.paths.config_path, **updates)
    orch._cli_service.update_model_policy(orch._config.model_policy)
    handler = getattr(orch, "_config_hot_reload_handler", None)
    if handler is not None:
        handler(orch._config, hot)


def _format_list(values: list[str] | None) -> str:
    if values is None:
        return "inherit"
    if values == ["*"]:
        return "*"
    return ",".join(values)


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "inherit"
    return "on" if value else "off"


def _format_rule(rule: ModelPolicyRule | None) -> str:
    if rule is None:
        return "policy=default"
    return (
        f"models={_format_list(rule.allowed_models)} "
        f"efforts={_format_list(rule.allowed_reasoning_efforts)} "
        f"switch={_format_bool(rule.allow_model_switch)}"
    )


def _format_effective_user(orch: Orchestrator, user_id: int) -> str:
    tags: list[str] = []
    if user_id == _owner_id(orch):
        tags.append("owner")
    if user_id in orch._config.model_policy.admin_user_ids:
        tags.append("admin")
    tag_text = f" ({', '.join(tags)})" if tags else ""
    rule = orch._config.model_policy.users.get(str(user_id))
    return f"- `{user_id}`{tag_text}: {_format_rule(rule)}"


def _list_access(orch: Orchestrator) -> str:
    policy = orch._config.model_policy
    lines = [
        "Access",
        f"- policy: {'on' if policy.enabled else 'off'}",
        f"- owner: `{_owner_id(orch)}`" if _owner_id(orch) is not None else "- owner: none",
        f"- admins: {_format_admins(policy.admin_user_ids)}",
        f"- default: {_format_rule(policy.default)}",
        "",
        "Users:",
    ]
    if not orch._config.allowed_user_ids:
        lines.append("- none")
    else:
        lines.extend(_format_effective_user(orch, user_id) for user_id in orch._config.allowed_user_ids)
    lines.extend(["", "Groups:"])
    if not orch._config.allowed_group_ids:
        lines.append("- none")
    else:
        lines.extend(f"- `{group_id}`" for group_id in orch._config.allowed_group_ids)
    return "\n".join(lines)


def _format_admins(admins: list[int]) -> str:
    if not admins:
        return "none"
    return ", ".join(f"`{user_id}`" for user_id in admins)


async def _list_access_result(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    if args:
        return OrchestratorResult(text="Usage: `/access list`")
    return OrchestratorResult(text=_list_access(orch))


async def _add_user(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    positionals, options, error = _parse_options(
        args,
        allowed=frozenset({"admin", "efforts", "models", "switch"}),
    )
    if error is not None:
        return OrchestratorResult(text=error)
    if len(positionals) != 1:
        return OrchestratorResult(text="Usage: `/access add <user_id|@username> [models=...] [efforts=...] [switch=on|off] [admin=on|off]`")

    user_id, error = await _resolve_user_ref(orch, positionals[0])
    if error is not None or user_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram user target.")

    admin_state: bool | None = None
    if "admin" in options:
        admin_state, error = _parse_bool(options["admin"])
        if error is not None:
            return OrchestratorResult(text=error)
    policy_options = {key: value for key, value in options.items() if key in _POLICY_OPTION_KEYS}
    policy_patch, error = _parse_policy_options(policy_options)
    if error is not None:
        return OrchestratorResult(text=error)

    added = _add_allowed_user(orch, user_id)
    policy = orch._config.model_policy
    policy.enabled = True

    if admin_state is not None:
        _set_policy_admin(orch, user_id, enabled=admin_state)

    rule = _user_rule(orch, user_id, admin=admin_state is True)
    _apply_policy_patch(rule, policy_patch)

    await _persist(orch)
    status = "added" if added else "already allowlisted"
    admin_text = " admin=on" if admin_state is True else " admin=off" if admin_state is False else ""
    return OrchestratorResult(
        text=f"Access updated: `{user_id}` {status}{admin_text}.\n{_format_effective_user(orch, user_id)}"
    )


async def _set_user_policy(  # noqa: PLR0911
    orch: Orchestrator, args: list[str]
) -> OrchestratorResult:
    user_id, options, parse_error = await _parse_policy_target(
        orch,
        args,
        usage=(
            "Usage: `/access policy <user_id|@username> [models=...] [efforts=...] "
            "[switch=on|off] [admin=on|off]`"
        ),
        allowed=frozenset({"admin", "efforts", "models", "switch"}),
    )
    if parse_error is not None:
        return parse_error
    assert user_id is not None

    if user_id not in orch._config.allowed_user_ids:
        return OrchestratorResult(text=f"`{user_id}` is not allowlisted. Use `/access add {user_id}` first.")
    if not options:
        rule = orch._config.model_policy.users.get(str(user_id))
        return OrchestratorResult(text=f"`{user_id}`: {_format_rule(rule)}")

    policy_options = {key: value for key, value in options.items() if key in _POLICY_OPTION_KEYS}
    policy_patch, error = _parse_policy_options(policy_options)
    if error is not None:
        return OrchestratorResult(text=error)

    policy = orch._config.model_policy
    policy.enabled = True

    admin_state: bool | None = None
    if "admin" in options:
        admin_state, error = _parse_bool(options["admin"])
        if error is not None:
            return OrchestratorResult(text=error)
        if admin_state is None:
            return OrchestratorResult(text="Invalid admin state.")
        _set_policy_admin(orch, user_id, enabled=admin_state)

    rule = _user_rule(orch, user_id, admin=admin_state is True and not policy_options)
    _apply_policy_patch(rule, policy_patch)

    await _persist(orch)
    return OrchestratorResult(text=f"Policy updated.\n{_format_effective_user(orch, user_id)}")


async def _set_default_policy(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    positionals, options, error = _parse_options(
        args,
        allowed=frozenset({"efforts", "models", "switch"}),
    )
    if error is not None:
        return OrchestratorResult(text=error)
    if positionals:
        return OrchestratorResult(text="Usage: `/access default [models=...] [efforts=...] [switch=on|off]`")
    if not options:
        return OrchestratorResult(text=f"default: {_format_rule(orch._config.model_policy.default)}")

    policy_patch, error = _parse_policy_options(options)
    if error is not None:
        return OrchestratorResult(text=error)

    policy = orch._config.model_policy
    policy.enabled = True
    _apply_policy_patch(policy.default, policy_patch)

    await _persist(orch)
    return OrchestratorResult(text=f"Default policy updated: {_format_rule(policy.default)}")


def _list_groups(orch: Orchestrator, positionals: list[str]) -> OrchestratorResult:
    if len(positionals) != 1:
        return OrchestratorResult(text="Usage: `/access group list`")
    if not orch._config.allowed_group_ids:
        return OrchestratorResult(text="Allowed groups:\n- none")
    lines = ["Allowed groups:"]
    lines.extend(f"- `{group_id}`" for group_id in orch._config.allowed_group_ids)
    return OrchestratorResult(text="\n".join(lines))


async def _add_group(orch: Orchestrator, positionals: list[str]) -> OrchestratorResult:
    if len(positionals) != 2:
        return OrchestratorResult(text="Usage: `/access group add <group_id>`")
    group_id, error = _parse_group_id(positionals[1])
    if error is not None or group_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram group ID.")
    added = _add_allowed_group(orch, group_id)
    await _persist(orch, include_groups=True)
    status = "added" if added else "already allowlisted"
    return OrchestratorResult(text=f"Group access updated: `{group_id}` {status}.")


async def _remove_group(orch: Orchestrator, positionals: list[str]) -> OrchestratorResult:
    if len(positionals) != 2:
        return OrchestratorResult(text="Usage: `/access group remove <group_id>`")
    group_id, error = _parse_group_id(positionals[1])
    if error is not None or group_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram group ID.")
    removed = _remove_allowed_group(orch, group_id)
    await _persist(orch, include_groups=True)
    status = "removed" if removed else "was not allowlisted"
    return OrchestratorResult(text=f"Group access updated: `{group_id}` {status}.")


async def _manage_group(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    positionals, options, error = _parse_options(args, allowed=frozenset())
    if error is not None:
        return OrchestratorResult(text=error)
    if options or not positionals:
        return OrchestratorResult(text="Usage: `/access group list|add|remove <group_id>`")

    action = positionals[0].lower()
    if action in {"list", "ls"}:
        return _list_groups(orch, positionals)

    if action in {"add", "approve"}:
        return await _add_group(orch, positionals)

    if action in {"remove", "rm", "revoke"}:
        return await _remove_group(orch, positionals)

    return OrchestratorResult(text=f"Unknown /access group action `{action}`.")


async def _set_admin(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    positionals, options, error = _parse_options(args, allowed=frozenset())
    if error is not None:
        return OrchestratorResult(text=error)
    if options or len(positionals) != 2:
        return OrchestratorResult(text="Usage: `/access admin <user_id|@username> on|off`")

    user_id, error = await _resolve_user_ref(orch, positionals[0])
    if error is not None or user_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram user target.")
    enabled, error = _parse_bool(positionals[1])
    if error is not None or enabled is None:
        return OrchestratorResult(text=error or "Invalid admin state.")

    if enabled:
        _add_allowed_user(orch, user_id)
        _set_policy_admin(orch, user_id, enabled=True)
        orch._config.model_policy.enabled = True
        _user_rule(orch, user_id, admin=True)
        result = "admin=on"
    else:
        _set_policy_admin(orch, user_id, enabled=False)
        result = "admin=off"
        if user_id == _owner_id(orch):
            result += " (owner remains admin via first allowed_user_ids)"

    await _persist(orch)
    return OrchestratorResult(text=f"Access updated: `{user_id}` {result}.")


async def _remove_user(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    positionals, options, error = _parse_options(args, allowed=frozenset())
    if error is not None:
        return OrchestratorResult(text=error)
    if options or len(positionals) != 1:
        return OrchestratorResult(text="Usage: `/access remove <user_id|@username>`")

    user_id, error = await _resolve_user_ref(orch, positionals[0])
    if error is not None or user_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram user target.")
    if user_id == _owner_id(orch):
        return OrchestratorResult(text="Refusing to remove the owner user. Put another owner first in allowed_user_ids manually if you need to rotate ownership.")

    before = list(orch._config.allowed_user_ids)
    orch._config.allowed_user_ids = [uid for uid in before if uid != user_id]
    orch._config.model_policy.users.pop(str(user_id), None)
    _set_policy_admin(orch, user_id, enabled=False)

    await _persist(orch)
    status = "removed" if user_id in before else "was not allowlisted"
    return OrchestratorResult(text=f"Access updated: `{user_id}` {status}.")


_ACTION_HANDLERS: dict[str, _AccessHandler] = {
    "add": _add_user,
    "admin": _set_admin,
    "default": _set_default_policy,
    "group": _manage_group,
    "list": _list_access_result,
    "policy": _set_user_policy,
    "remove": _remove_user,
}


def access_change_denied_text() -> str:
    """User-facing denial for access allowlist changes from non-admin users."""
    return (
        "Access allowlist changes are admin-only. Ask an owner/admin to use "
        "`/access add <user_id>` or `/access group add <group_id>`."
    )


def access_change_command_guidance_text() -> str:
    """User-facing guidance for admins who ask for access edits in prose."""
    return (
        "Use the admin-only access commands for allowlist changes: "
        "`/access add <user_id>` or `/access group add <group_id>`."
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _contains_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def is_access_config_change_request(text: str) -> bool:
    """Return True when text appears to request access allowlist changes."""
    normalized = text.casefold()
    if _contains_any(normalized, ("allowed_user_ids", "allowed_group_ids")) and _contains_any(
        normalized,
        _CONFIG_KEY_CHANGE_TERMS,
    ):
        return True
    return _is_user_access_change_request(normalized) or _is_group_access_change_request(normalized)


def _is_user_access_change_request(normalized: str) -> bool:
    has_user = _contains_any(
        normalized,
        (
            "user",
            "users",
            "telegram id",
            "telegram user",
            "пользовател",
            "юзер",
            "человек",
        ),
    )
    if not has_user:
        return False

    access_terms = (
        "allowlist",
        "whitelist",
        "access",
        "authorize",
        "authorise",
        "grant",
        "admin",
        "доступ",
        "разреш",
        "авториз",
        "админ",
    )
    if not (
        _contains_any(normalized, access_terms)
        and _contains_any(normalized, _ACCESS_CHANGE_TERMS)
    ):
        return False

    # Avoid treating explanatory/past-tense questions like
    # "после добавления другого пользователя..." as access changes. Without a
    # concrete id or a direct imperative, the CLI can safely answer the question.
    return _contains_digit(normalized) or _contains_any(normalized, _DIRECT_USER_ACCESS_TERMS)


def is_group_access_change_request(text: str) -> bool:
    """Best-effort detector for ordinary-language group allowlist change requests.

    Slash commands are handled by ``cmd_access``. This guard catches non-command
    requests before they reach a CLI agent with filesystem access.
    """
    normalized = text.casefold()
    return _is_group_access_change_request(normalized)


def _is_group_access_change_request(normalized: str) -> bool:
    if "allowed_group_ids" in normalized:
        return _contains_any(normalized, _CONFIG_KEY_CHANGE_TERMS)

    has_group = _contains_any(normalized, ("group", "chat", "групп", "чат"))
    if not has_group:
        return False

    if "-100" in normalized:
        return True

    access_terms = (
        "allowlist",
        "whitelist",
        "access",
        "authorize",
        "authorise",
        "approve",
        "grant",
        "доступ",
        "разреш",
        "одобр",
        "авториз",
    )
    if _contains_any(normalized, access_terms) and _contains_any(
        normalized,
        _ACCESS_CHANGE_TERMS,
    ):
        return _contains_any(normalized, _DIRECT_GROUP_ACCESS_TERMS)

    return False


_ACCESS_CHANGE_TERMS = (
    "add",
    "allow",
    "approve",
    "authorize",
    "authorise",
    "change",
    "edit",
    "grant",
    "modify",
    "remove",
    "revoke",
    "set",
    "update",
    "добав",
    "выдай",
    "выдат",
    "измен",
    "одобр",
    "помен",
    "разреш",
    "авториз",
    "удал",
    "отзов",
)

_CONFIG_KEY_CHANGE_TERMS = tuple(term for term in _ACCESS_CHANGE_TERMS if term != "allow")

_DIRECT_USER_ACCESS_TERMS = (
    "add user",
    "allow user",
    "approve user",
    "authorize user",
    "authorise user",
    "grant access",
    "remove user",
    "revoke access",
    "добавь пользовател",
    "выдай доступ",
    "выдать доступ",
    "дай доступ",
    "дать доступ",
    "разреши пользовател",
    "одобри пользовател",
    "авторизуй пользовател",
    "удали пользовател",
    "отзови доступ",
)

_DIRECT_GROUP_ACCESS_TERMS = (
    "add group",
    "add chat",
    "allow group",
    "allow chat",
    "approve group",
    "approve chat",
    "authorize group",
    "authorize chat",
    "authorise group",
    "authorise chat",
    "grant access",
    "remove group",
    "remove chat",
    "revoke access",
    "добавь групп",
    "добавь чат",
    "выдай доступ",
    "выдать доступ",
    "дай доступ",
    "дать доступ",
    "разреши групп",
    "разреши чат",
    "разреши этому чату",
    "одобри групп",
    "одобри чат",
    "авторизуй групп",
    "авторизуй чат",
    "удали групп",
    "удали чат",
    "отзови доступ",
)
