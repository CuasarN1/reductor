"""Tests for /upgrade Telegram command."""

from __future__ import annotations

from unittest.mock import patch

from ductor_bot.infra.version import VersionInfo
from ductor_bot.orchestrator.commands import cmd_upgrade
from ductor_bot.orchestrator.core import Orchestrator


def _checker(info: VersionInfo | None):
    async def check() -> VersionInfo | None:
        return info

    return check


class TestCmdUpgrade:
    """Test /upgrade command handler."""

    async def test_shows_update_with_buttons(self, orch: Orchestrator) -> None:
        info = VersionInfo(
            current="1.0.0", latest="2.0.0", update_available=True, summary="Big update"
        )
        with (
            patch("ductor_bot.infra.install.detect_install_mode", return_value="pipx"),
            patch(
                "ductor_bot.orchestrator.commands.make_update_checker", return_value=_checker(info)
            ),
        ):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        assert "Update Available" in result.text
        assert "1.0.0" in result.text
        assert "2.0.0" in result.text
        assert result.buttons is not None

        all_buttons = [b for row in result.buttons.rows for b in row]
        callback_values = [b.callback_data for b in all_buttons]
        assert "upg:yes:2.0.0" in callback_values
        assert "upg:no" in callback_values
        assert "upg:cl:2.0.0" in callback_values

    async def test_shows_up_to_date_with_changelog_button(self, orch: Orchestrator) -> None:
        info = VersionInfo(current="2.0.0", latest="2.0.0", update_available=False, summary="")
        with (
            patch("ductor_bot.infra.install.detect_install_mode", return_value="pip"),
            patch(
                "ductor_bot.orchestrator.commands.make_update_checker", return_value=_checker(info)
            ),
        ):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        assert "up to date" in result.text.lower()
        assert "2.0.0" in result.text
        assert result.buttons is not None
        buttons = result.buttons.rows[0]
        assert any(b.callback_data == "upg:cl:2.0.0" for b in buttons)

    async def test_handles_pypi_failure(self, orch: Orchestrator) -> None:
        with (
            patch("ductor_bot.infra.install.detect_install_mode", return_value="pipx"),
            patch(
                "ductor_bot.orchestrator.commands.make_update_checker", return_value=_checker(None)
            ),
        ):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        assert "could not reach" in result.text.lower() or "source" in result.text.lower()
        assert result.buttons is None

    async def test_button_text_is_user_friendly(self, orch: Orchestrator) -> None:
        info = VersionInfo(current="1.0.0", latest="1.1.0", update_available=True, summary="Patch")
        with (
            patch("ductor_bot.infra.install.detect_install_mode", return_value="pipx"),
            patch(
                "ductor_bot.orchestrator.commands.make_update_checker", return_value=_checker(info)
            ),
        ):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        all_buttons = [b for row in result.buttons.rows for b in row]
        labels = [b.text for b in all_buttons]
        assert any("upgrade" in label.lower() or "yes" in label.lower() for label in labels)
        assert any("not" in label.lower() or "no" in label.lower() for label in labels)
        assert any("changelog" in label.lower() for label in labels)

    async def test_version_info_in_up_to_date(self, orch: Orchestrator) -> None:
        info = VersionInfo(current="3.5.1", latest="3.5.1", update_available=False, summary="")
        with (
            patch("ductor_bot.infra.install.detect_install_mode", return_value="pip"),
            patch(
                "ductor_bot.orchestrator.commands.make_update_checker", return_value=_checker(info)
            ),
        ):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        assert "3.5.1" in result.text
        assert "latest" in result.text.lower()

    async def test_update_available_has_changelog_button(self, orch: Orchestrator) -> None:
        info = VersionInfo(current="1.0.0", latest="2.0.0", update_available=True, summary="Update")
        with (
            patch("ductor_bot.infra.install.detect_install_mode", return_value="pipx"),
            patch(
                "ductor_bot.orchestrator.commands.make_update_checker", return_value=_checker(info)
            ),
        ):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        all_buttons = [b for row in result.buttons.rows for b in row]
        assert any(b.callback_data == "upg:cl:2.0.0" for b in all_buttons)
        assert any(b.callback_data == "upg:yes:2.0.0" for b in all_buttons)

    async def test_github_source_checks_fork_without_upgrade_button(
        self,
        orch: Orchestrator,
    ) -> None:
        info = VersionInfo(
            current="0.19.1",
            latest="0.19.2",
            update_available=True,
            summary="Fork release",
            source="github",
            release_url="https://github.com/CuasarN1/reductor/releases/tag/v0.19.2",
            source_repo="CuasarN1/reductor",
        )
        with (
            patch("ductor_bot.infra.install.detect_install_mode", return_value="pipx"),
            patch(
                "ductor_bot.orchestrator.commands.make_update_checker", return_value=_checker(info)
            ),
        ):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        assert "CuasarN1/reductor" in result.text
        assert result.buttons is not None
        all_buttons = [b for row in result.buttons.rows for b in row]
        assert any(b.callback_data == "upg:cl:0.19.2" for b in all_buttons)
        assert all(not b.callback_data.startswith("upg:yes:") for b in all_buttons)

    async def test_dev_mode_rejects_upgrade(self, orch: Orchestrator) -> None:
        with patch("ductor_bot.infra.install.detect_install_mode", return_value="dev"):
            result = await cmd_upgrade(orch, 1, "/upgrade")

        assert "source" in result.text.lower() or "git pull" in result.text.lower()
        assert result.buttons is None
