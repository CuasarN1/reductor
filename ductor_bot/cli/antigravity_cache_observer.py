"""Background observer for periodic Antigravity model cache refresh."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ductor_bot.cli.antigravity_cache import AntigravityModelCache
from ductor_bot.cli.model_cache import BaseModelCacheObserver


class AntigravityCacheObserver(BaseModelCacheObserver):
    """Refreshes the Antigravity model cache periodically.

    Loads initial cache at startup and refreshes every 60 minutes.
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        on_refresh: Callable[[tuple[str, ...]], None] | None = None,
    ) -> None:
        """Initialize observer with cache file path.

        Args:
            cache_path: Path to JSON cache file.
            on_refresh: Optional callback invoked with the model list after
                        each successful cache load/refresh.
        """
        super().__init__(cache_path)
        self._on_refresh = on_refresh
        self._cache: AntigravityModelCache | None = None

    def _provider_name(self) -> str:
        return "Antigravity"

    async def _load_cache(self, *, initial: bool) -> AntigravityModelCache:
        return await AntigravityModelCache.load_or_refresh(self._cache_path, force_refresh=initial)

    def _model_count(self) -> int:
        return len(self._cache.models) if self._cache else 0

    def _last_updated(self) -> str:
        return self._cache.last_updated if self._cache else ""

    def _on_cache_loaded(self) -> None:
        """Invoke on_refresh callback if set."""
        if self._on_refresh and self._cache and self._cache.models:
            self._on_refresh(self._cache.models)

    def get_cache(self) -> AntigravityModelCache | None:
        """Return current cache (may be None if never loaded)."""
        return self._cache
