# Changelog

## v0.19.2 - 2026-07-16

### Fixed

- `/upgrade` checks the configured ReDuctor GitHub release source instead of
  PyPI by default.
- Reverted the unsafe Codex `gpt-5.6` overlay: ReDuctor now trusts local Codex
  discovery and does not show GPT-5.6 unless the Codex account actually exposes
  it.

## v0.19.1 - 2026-07-16

### Added

- Added GPT-5.6 as the newest/recommended Codex fallback model for new or
  discovery-fallback installs.
- Added `/diagnose` visibility for the installed Codex CLI version.
- Added `/diagnose` warning when local Codex model discovery does not expose
  GPT-5.6, with guidance to update Codex CLI and restart ReDuctor or wait for
  account rollout.

### Changed

- Direct `/model <codex-id>` now rejects unknown Codex model IDs when the
  local Codex model cache is loaded, instead of silently switching to a model
  the local CLI may not be able to run.
- Cron/webhook model validation now explains that an unknown Codex model was
  not returned by local Codex CLI discovery and lists available models.
- Refreshed bundled config, cron, webhook, and agent tool rules to list
  GPT-5.6 before GPT-5.5.

### Upgrade Notes

GPT-5.6 availability depends on the user's installed Codex CLI and OpenAI
account rollout. After upgrading ReDuctor, users should update Codex CLI and
restart the bot so `codex app-server` model discovery is refreshed.

## v0.19.0 - 2026-07-13

ReDuctor 0.19.0 backports the most relevant Ductor 0.19 runtime fixes while
preserving this fork's Telegram outbox, access management, model policy, proxy,
and restart behavior.

### Highlights

- Backported safer background task handling:
  - task subprocesses now receive `DUCTOR_TASK_ID`
  - task resume is rejected while the same task is still running
  - completed task results are preserved when cancel races completion
- Scoped process interruption to the active Telegram topic/session so `/new`,
  `/reset`, model switches, and recovery paths do not kill unrelated tasks or
  named sessions.
- Switched Codex prompts to stdin for one-shot and streaming execution, reducing
  long-prompt argv-limit failures.
- Re-asserted the configured Codex model on `codex exec resume`.
- Made Telegram startup notification fan-out resilient to unreachable recipients.
- Refreshed bundled Codex/Gemini fallback model lists and generated RULES files.
- Fixed `/access` false positives when replying to quoted Telegram messages:
  access-change detection now analyzes only the user's actual reply body.

### Fork Update Notifications

- Added GitHub-based update notifications for ReDuctor deployments.
- Default update source is the fork repository: `CuasarN1/reductor`.
- GitHub Releases are checked first; tags are used as a fallback.
- Prerelease-looking tags such as `v0.20.0-rc1` are ignored unless prereleases
  are explicitly enabled.
- Existing `notifications.upgrade_targets` routing is reused for update notices.
- GitHub update notifications link to the fork release notes and do not show the
  PyPI "Upgrade now" action, so users do not accidentally install upstream
  `ductor` over the fork.
- `/upgrade` still uses the existing PyPI package flow.

Relevant config keys:

```json
{
  "update_check": true,
  "notifications": {
    "update_source": "github",
    "update_github_repo": "CuasarN1/reductor",
    "update_github_releases_url": "",
    "update_include_prereleases": false,
    "upgrade_targets": []
  }
}
```

### Upgrade Notes

Install from the fork release/tag, not from PyPI:

```bash
uv tool install --force git+https://github.com/CuasarN1/reductor.git@v0.19.0
```

Existing deployments older than 0.19.0 do not yet have fork update
notifications, so they need one manual update to receive future release
notices.

### Verification

- Focused pytest suite: 649 passed.
- Follow-up version/update/access notification subset: 56 passed.
- Ruff on changed Python files: passed.
- `py_compile` on changed Python files: passed.
- `git diff --check`: passed.
