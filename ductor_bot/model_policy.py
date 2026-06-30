"""Per-user model and reasoning policy helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ductor_bot.config import ModelPolicyConfig, ModelPolicyRule

if TYPE_CHECKING:
    from ductor_bot.config import AgentConfig
    from ductor_bot.session.key import SessionKey

_WILDCARD = "*"


@dataclass(frozen=True, slots=True)
class ResolvedModelPolicy:
    """Effective model policy for one user."""

    enabled: bool
    allowed_models: tuple[str, ...] | None
    allowed_reasoning_efforts: tuple[str, ...] | None
    allow_model_switch: bool


@dataclass(frozen=True, slots=True)
class SelectedModelTarget:
    """Model/reasoning target selected by policy for one request."""

    model: str
    provider: str
    reasoning_effort: str | None = None


def subject_id_for_key(key: SessionKey) -> int | None:
    """Return the Telegram user id to apply policy to, with private-chat fallback."""
    if key.user_id is not None:
        return key.user_id
    if key.transport == "tg" and key.chat_id > 0:
        return key.chat_id
    return None


def subject_id_for_request(user_id: int | None, chat_id: int, transport: str) -> int | None:
    """Return the policy subject for a CLI request."""
    if user_id is not None:
        return user_id
    if transport == "tg" and chat_id > 0:
        return chat_id
    return None


def resolve_model_policy(
    policy: ModelPolicyConfig,
    user_id: int | None,
) -> ResolvedModelPolicy:
    """Resolve inherited default/user model policy values."""
    if not policy.enabled:
        return ResolvedModelPolicy(
            enabled=False,
            allowed_models=None,
            allowed_reasoning_efforts=None,
            allow_model_switch=True,
        )

    default = policy.default
    user_rule = policy.users.get(str(user_id)) if user_id is not None else None
    allowed_models = _resolve_list(user_rule, default, "allowed_models")
    allowed_efforts = _resolve_list(user_rule, default, "allowed_reasoning_efforts")
    allow_switch = _resolve_bool(user_rule, default, "allow_model_switch", fallback=False)

    return ResolvedModelPolicy(
        enabled=True,
        allowed_models=allowed_models,
        allowed_reasoning_efforts=allowed_efforts,
        allow_model_switch=allow_switch,
    )


def can_switch_models(config: AgentConfig, user_id: int | None) -> bool:
    """Return whether this user may persistently switch models via /model."""
    return resolve_model_policy(config.model_policy, user_id).allow_model_switch


def is_model_allowed(
    config: AgentConfig,
    user_id: int | None,
    model_id: str,
    *,
    provider: str = "",
) -> bool:
    """Return whether a model is allowed for this user."""
    resolved = resolve_model_policy(config.model_policy, user_id)
    allowed = resolved.allowed_models
    if not resolved.enabled or allowed is None or _WILDCARD in allowed:
        return True
    return any(_model_pattern_matches(pattern, model_id, provider) for pattern in allowed)


def filter_allowed_models(
    config: AgentConfig,
    user_id: int | None,
    model_ids: list[str],
    *,
    provider: str = "",
) -> list[str]:
    """Return only models allowed for this user."""
    return [
        model_id
        for model_id in model_ids
        if is_model_allowed(config, user_id, model_id, provider=provider)
    ]


def is_reasoning_effort_allowed(
    config: AgentConfig,
    user_id: int | None,
    effort: str,
) -> bool:
    """Return whether a Codex reasoning effort is allowed for this user."""
    resolved = resolve_model_policy(config.model_policy, user_id)
    allowed = resolved.allowed_reasoning_efforts
    if not resolved.enabled or allowed is None or _WILDCARD in allowed:
        return True
    return effort in allowed


def filter_allowed_reasoning_efforts(
    config: AgentConfig,
    user_id: int | None,
    efforts: tuple[str, ...],
) -> tuple[str, ...]:
    """Return only reasoning efforts allowed for this user."""
    return tuple(
        effort for effort in efforts if is_reasoning_effort_allowed(config, user_id, effort)
    )


def model_switch_denied_text() -> str:
    """User-facing denial for persistent model switching."""
    return "Manual model selection is disabled for this user."


def model_denied_text(model_id: str) -> str:
    """User-facing model denial."""
    return f"Model `{model_id}` is not allowed for this user."


def reasoning_denied_text(effort: str) -> str:
    """User-facing reasoning-effort denial."""
    return f"Reasoning effort `{effort}` is not allowed for this user."


def no_models_available_text() -> str:
    """User-facing text when policy hides every model in a selector step."""
    return "No models are available for this user."


def no_reasoning_efforts_available_text() -> str:
    """User-facing text when policy hides every reasoning effort."""
    return "No reasoning efforts are available for this user."


def select_model_target_for_prompt(  # noqa: PLR0913
    config: AgentConfig,
    user_id: int | None,
    prompt: str,
    *,
    default_model: str,
    provider_for: Callable[[str], str],
    supported_efforts: tuple[str, ...] | None = None,
) -> SelectedModelTarget | None:
    """Choose a policy-approved model/reasoning target for an ordinary user.

    The admin-controlled ``allowed_models`` order is intentional: cheaper or
    preferred models should be listed first, stronger fallbacks later.
    """
    resolved = resolve_model_policy(config.model_policy, user_id)
    if not resolved.enabled or resolved.allow_model_switch:
        return None

    candidates = _explicit_allowed_model_candidates(resolved)
    if not candidates and is_model_allowed(
        config, user_id, default_model, provider=provider_for(default_model)
    ):
        candidates = (default_model,)
    if not candidates:
        return None

    complexity = _prompt_complexity(prompt)
    model = _select_model_from_candidates(candidates, complexity)
    provider = provider_for(model)
    effort = _select_reasoning_effort(config, user_id, complexity, supported_efforts)
    return SelectedModelTarget(model=model, provider=provider, reasoning_effort=effort)


def request_policy_denial(
    config: AgentConfig,
    user_id: int | None,
    model_id: str,
    *,
    provider: str,
    reasoning_effort: str,
) -> str | None:
    """Return denial text for an effective CLI target, or None when allowed."""
    return request_policy_denial_for_policy(
        config.model_policy,
        user_id,
        model_id,
        provider=provider,
        reasoning_effort=reasoning_effort,
    )


def request_policy_denial_for_policy(
    policy: ModelPolicyConfig,
    user_id: int | None,
    model_id: str,
    *,
    provider: str,
    reasoning_effort: str,
) -> str | None:
    """Return denial text for an effective CLI target using a policy object."""
    if not _is_model_allowed_for_policy(policy, user_id, model_id, provider=provider):
        return model_denied_text(model_id)
    if provider == "codex" and not _is_effort_allowed_for_policy(policy, user_id, reasoning_effort):
        return reasoning_denied_text(reasoning_effort)
    return None


def _resolve_list(
    user_rule: ModelPolicyRule | None,
    default: ModelPolicyRule,
    attr: str,
) -> tuple[str, ...] | None:
    user_value = getattr(user_rule, attr) if user_rule is not None else None
    value = user_value if user_value is not None else getattr(default, attr)
    return tuple(value) if value is not None else None


def _resolve_bool(
    user_rule: ModelPolicyRule | None,
    default: ModelPolicyRule,
    attr: str,
    *,
    fallback: bool,
) -> bool:
    user_value = getattr(user_rule, attr) if user_rule is not None else None
    value = user_value if user_value is not None else getattr(default, attr)
    return fallback if value is None else bool(value)


def _model_pattern_matches(pattern: str, model_id: str, provider: str) -> bool:
    if pattern == model_id:
        return True
    if provider and pattern in {f"{provider}:*", f"provider:{provider}"}:
        return True
    if pattern.endswith(_WILDCARD):
        return model_id.startswith(pattern[:-1])
    return False


def _explicit_allowed_model_candidates(policy: ResolvedModelPolicy) -> tuple[str, ...]:
    allowed = policy.allowed_models
    if not allowed or _WILDCARD in allowed:
        return ()
    return tuple(
        item
        for item in allowed
        if _WILDCARD not in item and not item.startswith("provider:") and not item.endswith(":*")
    )


def _prompt_complexity(prompt: str) -> int:
    """Return 0=simple, 1=normal, 2=complex, 3=very complex."""
    lower = prompt.casefold()
    length = len(prompt)
    complex_markers = (
        "debug",
        "fix",
        "bug",
        "implement",
        "refactor",
        "architecture",
        "design",
        "test",
        "review",
        "analyze",
        "compare",
        "reason",
        "ошибка",
        "исправ",
        "рефактор",
        "архитектур",
        "проанализ",
        "сравни",
        "тест",
        "код",
    )
    very_complex_markers = (
        "deep",
        "thorough",
        "full implementation",
        "end to end",
        "с нуля",  # noqa: RUF001
        "подробно",
        "глубоко",
        "полная реализация",
    )
    score = 0
    if length > 300:
        score += 1
    if length > 1200:
        score += 1
    if any(marker in lower for marker in complex_markers):
        score += 1
    if any(marker in lower for marker in very_complex_markers):
        score += 1
    return min(score, 3)


def _select_model_from_candidates(candidates: tuple[str, ...], complexity: int) -> str:
    if len(candidates) == 1 or complexity <= 0:
        return candidates[0]
    if complexity >= 3:
        return candidates[-1]
    return candidates[min(len(candidates) - 1, len(candidates) // 2)]


def _select_reasoning_effort(
    config: AgentConfig,
    user_id: int | None,
    complexity: int,
    supported_efforts: tuple[str, ...] | None,
) -> str | None:
    desired_by_complexity = ("low", "medium", "high", "xhigh")
    desired = desired_by_complexity[complexity]
    allowed = filter_allowed_reasoning_efforts(
        config,
        user_id,
        supported_efforts or ("low", "medium", "high", "xhigh"),
    )
    if not allowed:
        return None
    order = {effort: index for index, effort in enumerate(desired_by_complexity)}
    desired_index = order[desired]
    ranked = sorted(allowed, key=lambda effort: order.get(effort, desired_index))
    not_above_desired = [
        effort for effort in ranked if order.get(effort, desired_index) <= desired_index
    ]
    if not_above_desired:
        return not_above_desired[-1]
    return ranked[0]


def _is_model_allowed_for_policy(
    policy: ModelPolicyConfig,
    user_id: int | None,
    model_id: str,
    *,
    provider: str = "",
) -> bool:
    resolved = resolve_model_policy(policy, user_id)
    allowed = resolved.allowed_models
    if not resolved.enabled or allowed is None or _WILDCARD in allowed:
        return True
    return any(_model_pattern_matches(pattern, model_id, provider) for pattern in allowed)


def _is_effort_allowed_for_policy(
    policy: ModelPolicyConfig,
    user_id: int | None,
    effort: str,
) -> bool:
    resolved = resolve_model_policy(policy, user_id)
    allowed = resolved.allowed_reasoning_efforts
    if not resolved.enabled or allowed is None or _WILDCARD in allowed:
        return True
    return effort in allowed
