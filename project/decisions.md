---
title: Decisions
updated: 2026-10-13
---

# Run command

- **Generic `run` instead of `make`** — the old `make` command (removed in `4eb8302`, reason not recorded) is replaced by `run`, which runs any command. Same processing-job path: `SageMakerClient.run()` → `_run_processing(["run", *command])`.
- **`exec "$@"` in `executor.sh`, not `sh -c "$*"`** — arguments reach the tool unchanged. Checked: with `sh -c "$*"`, `python -c 'print("a b")'` is a syntax error and `echo 'x; echo INJECTED'` runs a second command. `exec` also makes the tool PID 1, so its exit code fails the SageMaker job. Shell features need an explicit `-- sh -c "..."`.
- **Trailing arguments after `--`, not a `--command` flag** — a string flag would need re-splitting (`shlex`) and nested quoting, and goes against `docker run` / `uv run` convention. A command already held in one string can use `-- sh -c "$CMD"`.
- **Option parsing stops at the command** (`allow_interspersed_args=False`) — without it, `easy_sm run -e ml.m5.large -n job dbt run -e prod` silently submitted instance type `prod`. easy_sm options must come before the command.
- **Guard against outdated projects** (`require_run_support`) — an image built from an old `executor.sh` falls through to the model server on `run` and runs until the 24 h processing timeout. The guard checks the project's local `executor.sh` only; an image that wasn't rebuilt and pushed is not detected.
- **`local run` returns the command's exit code** — so it works in CI. Other local commands still exit 0 on failure; left unchanged to avoid breaking existing users.

# Security

- **Env vars passed as separate arguments** — `process_local.sh` used to join `--env` values with commas and expand them unquoted; checked that `MSG=hi --privileged -v /:/host` became real docker flags. Now each `KEY=VALUE` is one argument rebuilt via `"$@"`. `local process --env` refuses to run with the old script (it silently dropped all but the first variable).
- **`~/.aws` stays writable in local containers** — `:ro` would crash SSO users: botocore's `SSOTokenProvider` writes the refreshed token to `~/.aws/sso/cache` without catching errors (checked in botocore `tokens.py`). Proper fix is passing temporary credentials; open.
- **`local deploy` binds to `127.0.0.1`** — previously published on all interfaces. `localhost` URLs unchanged; `stop_local.sh` still matches the container.
- **PyPI Trusted Publishing, `pypi` environment limited to `v*` tags** (PR #10) — replaces the long-lived `PYPI_TOKEN`. No required reviewer and no tag ruleset: solo repo, chosen 2026-10-13.
- **Actions pinned to commit SHAs, Dependabot keeps them current** — pins are the versions the floating tags resolved to on 2026-10-13, so no behaviour change.

# Code and release

- **No `cloud_app`** — it was never registered; cloud commands are plain functions registered in `__main__.py`, where names and `context_settings` belong. `--help` output for all 14 commands identical before and after removal.
- **Image tag from config** — `docker_tag` lives in `{app_name}.json` (`2f97df7`); there is no global `--docker-tag` option.
- **SageMaker SDK pinned to `==2.257.3`** (`ce837b7`) — exact pin; reason not recorded beyond "stability". An earlier move to SDK v3 (`a19683f`) was superseded by this pin.
- **Minor version for features** — new commands bump minor (env vars → 1.1.0, `run` → 1.3.0); fixes bump patch.
- **Test against coloured output** — Rich splits flags with ANSI codes when colour is on (GitHub Actions), so tests strip ANSI before matching CLI errors. Reproduce with `FORCE_COLOR=1`.
