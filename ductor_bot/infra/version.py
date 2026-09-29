"""Package version checking against PyPI and GitHub Releases."""

from __future__ import annotations

import importlib.metadata
import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

_PYPI_URL = "https://pypi.org/pypi/ductor/json"
_DEFAULT_REDUCTOR_GITHUB_REPO = "CuasarN1/reductor"
_PACKAGE_NAME = "ductor"
_TIMEOUT = aiohttp.ClientTimeout(total=10)


def get_current_version() -> str:
    """Return the installed version of ductor."""
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse dotted version string into a comparable tuple."""
    v = _normalize_version(v)
    parts: list[int] = []
    for segment in v.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            break
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """Result of a PyPI version check."""

    current: str
    latest: str
    update_available: bool
    summary: str
    source: str = "pypi"
    release_url: str = ""
    source_repo: str = ""


def _normalize_version(value: str) -> str:
    """Normalize common release tag spellings to a dotted version string."""
    return value.strip().removeprefix("v").removeprefix("V")


def _is_prerelease_version(value: str) -> bool:
    normalized = _normalize_version(value)
    return "-" in normalized or any(ch.isalpha() for ch in normalized)


def _github_releases_url(repo: str, releases_url: str = "") -> str:
    if releases_url.strip():
        return releases_url.strip().rstrip("/")
    return f"https://api.github.com/repos/{repo.strip()}/releases"


def _github_tags_url(repo: str, releases_url: str = "") -> str:
    releases = releases_url.strip().rstrip("/")
    if releases.endswith("/releases"):
        return f"{releases[: -len('/releases')]}/tags"
    return f"https://api.github.com/repos/{repo.strip()}/tags"


async def check_pypi(*, fresh: bool = False) -> VersionInfo | None:
    """Check PyPI for the latest version. Returns None on failure.

    When ``fresh=True``, request with no-cache headers and a cache-busting
    query parameter to reduce stale CDN/cache responses.
    """
    current = get_current_version()
    headers = None
    params = None
    if fresh:
        headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
        params = {"_": str(time.time_ns())}

    try:
        async with (
            aiohttp.ClientSession(timeout=_TIMEOUT) as session,
            session.get(_PYPI_URL, headers=headers, params=params) as resp,
        ):
            if resp.status != 200:
                return None
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError, ValueError):
        logger.debug("PyPI version check failed", exc_info=True)
        return None

    info = data.get("info", {})
    latest = info.get("version", "")
    if not latest:
        return None

    summary = info.get("summary", "")
    update_available = _parse_version(latest) > _parse_version(current)
    return VersionInfo(
        current=current,
        latest=latest,
        update_available=update_available,
        summary=summary,
    )


async def check_github_release(
    *,
    repo: str = _DEFAULT_REDUCTOR_GITHUB_REPO,
    releases_url: str = "",
    include_prereleases: bool = False,
    fresh: bool = False,
) -> VersionInfo | None:
    """Check a GitHub release/tag stream for the latest ReDuctor version.

    Releases are preferred because they have notes and stable URLs. If a fork
    only pushes tags, the checker falls back to the tags API.
    """
    current = get_current_version()
    headers = {"Accept": "application/vnd.github+json"}
    params: dict[str, object] = {"per_page": 20}
    if fresh:
        headers.update({"Cache-Control": "no-cache", "Pragma": "no-cache"})
        params["_"] = str(time.time_ns())

    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=headers) as session:
            release = await _fetch_latest_github_release(
                session,
                _github_releases_url(repo, releases_url),
                params,
                include_prereleases=include_prereleases,
            )
            if release is None:
                release = await _fetch_latest_github_tag(
                    session,
                    _github_tags_url(repo, releases_url),
                    params,
                    include_prereleases=include_prereleases,
                )
    except (aiohttp.ClientError, TimeoutError, ValueError):
        logger.debug("GitHub version check failed for %s", repo, exc_info=True)
        return None

    if release is None:
        return None

    latest = release["version"]
    update_available = _parse_version(latest) > _parse_version(current)
    return VersionInfo(
        current=current,
        latest=latest,
        update_available=update_available,
        summary=release.get("summary", ""),
        source="github",
        release_url=release.get("url", ""),
        source_repo=repo,
    )


async def _fetch_latest_github_release(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, object],
    *,
    include_prereleases: bool,
) -> dict[str, str] | None:
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("draft"):
            continue
        if item.get("prerelease") and not include_prereleases:
            continue
        tag = str(item.get("tag_name") or item.get("name") or "").strip()
        version = _normalize_version(tag)
        if _is_prerelease_version(version) and not include_prereleases:
            continue
        if not _parse_version(version):
            continue
        body = str(item.get("body") or "").strip()
        title = str(item.get("name") or tag).strip()
        summary = title
        if not summary and body:
            summary = body.splitlines()[0]
        return {
            "version": version,
            "summary": summary,
            "url": str(item.get("html_url") or ""),
        }
    return None


async def _fetch_latest_github_tag(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, object],
    *,
    include_prereleases: bool,
) -> dict[str, str] | None:
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("name") or "").strip()
        version = _normalize_version(tag)
        if _is_prerelease_version(version) and not include_prereleases:
            continue
        if not _parse_version(version):
            continue
        return {
            "version": version,
            "summary": tag,
            "url": str(item.get("html_url") or ""),
        }
    return None


async def fetch_changelog(
    version: str,
    *,
    repo: str = _DEFAULT_REDUCTOR_GITHUB_REPO,
    releases_url: str = "",
) -> str | None:
    """Fetch release notes for *version* from GitHub Releases.

    Tries ``v{version}`` tag first, then ``{version}`` without prefix.
    Returns the release body (Markdown) or ``None`` on failure.
    """
    headers = {"Accept": "application/vnd.github+json"}
    base_url = _github_releases_url(repo, releases_url)
    for tag in (f"v{version}", version):
        url = f"{base_url}/tags/{tag}"
        try:
            async with (
                aiohttp.ClientSession(timeout=_TIMEOUT, headers=headers) as session,
                session.get(url) as resp,
            ):
                if resp.status != 200:
                    continue
                data = await resp.json()
                body: str = data.get("body", "")
                if body:
                    return body.strip()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            logger.debug("GitHub release fetch failed for tag %s", tag, exc_info=True)
    return None
