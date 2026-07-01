"""Admin-only access and per-user model-policy commands."""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
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
        "- `/access add <user_id> [models=...] [efforts=...] [switch=on|off] "
        "[admin=on|off]`\n"
        "- `/access policy <user_id> [models=...] [efforts=...] [switch=on|off] "
        "[admin=on|off]`\n"
        "- `/access default [models=...] [efforts=...] [switch=on|off]`\n"
        "- `/access admin <user_id> on|off`\n"
        "- `/access remove <user_id>`\n"
        "\n"
        "Examples:\n"
        "- `/access add 123456789 models=gpt-5.4-mini,gpt-5.4 efforts=low,medium "
        "switch=off`\n"
        "- `/access policy 123456789 models=* efforts=* switch=on`"
    )


def _parse_user_id(raw: str) -> tuple[int | None, str | None]:
    try:
        user_id = int(raw)
    except ValueError:
        return None, f"Invalid Telegram user ID `{raw}`."
    if user_id <= 0:
        return None, f"Telegram user ID must be positive: `{raw}`."
    return user_id, None


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


def _parse_policy_target(
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

    user_id, error = _parse_user_id(positionals[0])
    if error is not None or user_id is None:
        return None, {}, OrchestratorResult(text=error or "Invalid Telegram user ID.")
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


async def _persist(orch: Orchestrator) -> None:
    await update_config_file_async(
        orch.paths.config_path,
        allowed_user_ids=list(orch._config.allowed_user_ids),
        model_policy=orch._config.model_policy.model_dump(mode="json"),
    )
    orch._cli_service.update_model_policy(orch._config.model_policy)
    handler = getattr(orch, "_config_hot_reload_handler", None)
    if handler is not None:
        handler(
            orch._config,
            {
                "allowed_user_ids": orch._config.allowed_user_ids,
                "model_policy": orch._config.model_policy,
            },
        )


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
        return OrchestratorResult(text="Usage: `/access add <user_id> [models=...] [efforts=...] [switch=on|off] [admin=on|off]`")

    user_id, error = _parse_user_id(positionals[0])
    if error is not None or user_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram user ID.")

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


async def _set_user_policy(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    user_id, options, parse_error = _parse_policy_target(
        args,
        usage=(
            "Usage: `/access policy <user_id> [models=...] [efforts=...] "
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


async def _set_admin(orch: Orchestrator, args: list[str]) -> OrchestratorResult:
    positionals, options, error = _parse_options(args, allowed=frozenset())
    if error is not None:
        return OrchestratorResult(text=error)
    if options or len(positionals) != 2:
        return OrchestratorResult(text="Usage: `/access admin <user_id> on|off`")

    user_id, error = _parse_user_id(positionals[0])
    if error is not None or user_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram user ID.")
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
        return OrchestratorResult(text="Usage: `/access remove <user_id>`")

    user_id, error = _parse_user_id(positionals[0])
    if error is not None or user_id is None:
        return OrchestratorResult(text=error or "Invalid Telegram user ID.")
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
    "list": _list_access_result,
    "policy": _set_user_policy,
    "remove": _remove_user,
}
