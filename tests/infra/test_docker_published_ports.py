"""Tests for Docker user-configured published ports (``-p`` flags)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ductor_bot.config import DockerConfig
from ductor_bot.infra.docker import _build_published_port_flags
from ductor_bot.workspace.paths import DuctorPaths

# ---------------------------------------------------------------------------
# _build_published_port_flags
# ---------------------------------------------------------------------------


class TestBuildPublishedPortFlags:
    """Unit tests for the _build_published_port_flags helper."""

    def test_empty_list_returns_no_flags(self) -> None:
        assert _build_published_port_flags([]) == []

    def test_single_entry_produces_one_flag_pair(self) -> None:
        flags = _build_published_port_flags(["127.0.0.1:8080:80"])
        assert flags == ["-p", "127.0.0.1:8080:80"]

    def test_multiple_entries_produce_multiple_flag_pairs(self) -> None:
        flags = _build_published_port_flags(["8080:80", "192.168.1.1:5050:5051"])
        assert flags == ["-p", "8080:80", "-p", "192.168.1.1:5050:5051"]

    def test_blank_entries_are_skipped(self) -> None:
        flags = _build_published_port_flags(["", "   ", "8080:80"])
        assert flags == ["-p", "8080:80"]

    def test_non_string_entries_are_skipped(self) -> None:
        # Defensive: malformed config (e.g. JSON ints) should not blow up.
        flags = _build_published_port_flags(["8080:80", 1234, None])  # type: ignore[list-item]
        assert flags == ["-p", "8080:80"]

    def test_entries_are_stripped(self) -> None:
        flags = _build_published_port_flags(["  8080:80  "])
        assert flags == ["-p", "8080:80"]


# ---------------------------------------------------------------------------
# DockerManager._start_container with published_ports
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_paths(tmp_path: Path) -> DuctorPaths:
    home = tmp_path / ".ductor"
    home.mkdir()
    ws = home / "workspace"
    ws.mkdir()
    (ws / "tools").mkdir()
    fw = tmp_path / "framework"
    fw.mkdir()
    return DuctorPaths(ductor_home=home, home_defaults=fw / "workspace", framework_root=fw)


class TestDockerManagerPublishedPorts:
    """Integration tests: verify -p flags appear in the docker run command."""

    async def test_empty_published_ports_produces_no_p_flag(
        self, docker_paths: DuctorPaths
    ) -> None:
        config = DockerConfig(enabled=True, published_ports=[])
        from ductor_bot.infra.docker import DockerManager

        mgr = DockerManager(config, docker_paths)
        run_args: list[str] = []

        async def mock_exec(*args: str, **_kwargs: object) -> tuple[int, str]:
            cmd = " ".join(args)
            if "docker info" in cmd:
                return 0, "ok"
            if "image inspect" in cmd:
                return 0, "ok"
            if "container inspect" in cmd:
                return 1, ""
            if "rm -f" in cmd:
                return 0, ""
            if "docker run" in cmd:
                run_args.extend(args)
                return 0, "cid"
            return 0, ""

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(mgr, "_exec", side_effect=mock_exec),
        ):
            result = await mgr.setup()

        assert result is not None  # Container starts.
        assert "-p" not in run_args

    async def test_single_published_port_in_run_cmd(self, docker_paths: DuctorPaths) -> None:
        config = DockerConfig(enabled=True, published_ports=["127.0.0.1:8080:80"])
        from ductor_bot.infra.docker import DockerManager

        mgr = DockerManager(config, docker_paths)
        run_args: list[str] = []

        async def mock_exec(*args: str, **_kwargs: object) -> tuple[int, str]:
            cmd = " ".join(args)
            if "docker info" in cmd:
                return 0, "ok"
            if "image inspect" in cmd:
                return 0, "ok"
            if "container inspect" in cmd:
                return 1, ""
            if "rm -f" in cmd:
                return 0, ""
            if "docker run" in cmd:
                run_args.extend(args)
                return 0, "cid"
            return 0, ""

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(mgr, "_exec", side_effect=mock_exec),
        ):
            await mgr.setup()

        # The pair must appear adjacent so docker parses them correctly.
        assert "-p" in run_args
        idx = run_args.index("-p")
        assert run_args[idx + 1] == "127.0.0.1:8080:80"

    async def test_multiple_published_ports_in_run_cmd(self, docker_paths: DuctorPaths) -> None:
        config = DockerConfig(
            enabled=True,
            published_ports=["192.168.8.24:5050:5051", "8080:80/udp"],
        )
        from ductor_bot.infra.docker import DockerManager

        mgr = DockerManager(config, docker_paths)
        run_args: list[str] = []

        async def mock_exec(*args: str, **_kwargs: object) -> tuple[int, str]:
            cmd = " ".join(args)
            if "docker info" in cmd:
                return 0, "ok"
            if "image inspect" in cmd:
                return 0, "ok"
            if "container inspect" in cmd:
                return 1, ""
            if "rm -f" in cmd:
                return 0, ""
            if "docker run" in cmd:
                run_args.extend(args)
                return 0, "cid"
            return 0, ""

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(mgr, "_exec", side_effect=mock_exec),
        ):
            await mgr.setup()

        # Collect every value that follows a -p flag.
        published = [run_args[i + 1] for i, arg in enumerate(run_args) if arg == "-p"]
        assert published == ["192.168.8.24:5050:5051", "8080:80/udp"]


# ---------------------------------------------------------------------------
# DockerConfig Pydantic model
# ---------------------------------------------------------------------------


class TestDockerConfigPublishedPorts:
    """Verify the Pydantic model handles the published_ports field."""

    def test_default_published_ports_empty(self) -> None:
        config = DockerConfig()
        assert config.published_ports == []

    def test_published_ports_from_dict(self) -> None:
        config = DockerConfig(published_ports=["127.0.0.1:8080:80", "5050:5051"])
        assert config.published_ports == ["127.0.0.1:8080:80", "5050:5051"]

    def test_published_ports_serialization(self) -> None:
        config = DockerConfig(published_ports=["8080:80"])
        data = config.model_dump()
        assert data["published_ports"] == ["8080:80"]
