---
title: Decisions
updated: 2026-10-13
---

# Principles

- **Don't break existing projects** (2026-10-13) — template scripts are copied into user projects by `init` and `executor.sh` is baked into the image at build, so a fix only reaches users after `easy_sm update-scripts` (+ rebuild/push). New behaviour therefore checks for outdated copies and stops with instructions (`process_local.sh` `--env`, `require_run_support`) instead of changing behaviour silently. Same reason `~/.aws` stayed writable and existing local commands keep exit code 0.
- **Unix-style CLI** (`b66fef7`) — cloud commands at the top level (no `cloud` prefix), app name auto-detected from the single `*.json`, IAM role from `SAGEMAKER_ROLE`, and output is only the data (S3 path, job or endpoint name) so commands pipe into each other.

# CLI design

- **Typer, not Click** (`e5ade8c`) — options declared with `Annotated` type hints; Click groups became Typer sub-apps.
- **No `cloud_app`** — it was never registered; cloud commands are plain functions registered in `__main__.py`, where names and `context_settings` belong. `--help` for all 14 commands identical before and after removal.
- **Image tag from config, not a CLI flag** (`2f97df7`) — `docker_tag` lives in `{app_name}.json` so each app/model version can use its own tag; the global `--docker-tag` option was removed (user docs still mention it: open).
- **Wait by default, `--no-wait` to opt out** (`0eba322`) — `train` and `process` (and `run`) wait for the job; reverses the fire-and-forget behaviour from the v3 simplification (`81e2c05`), which returned an S3 path before the job had finished.
- **Short flags don't collide within a command** (`e25c95d`) — `-c` is always instance count; serverless max concurrency became `-mc`.
- **`requirements_dir` kept** — renaming it to `requirements_file_name` was tried and reverted the same day (`9d7365e` → `472c0a1`) because requirements can live in different folders for different scenarios (e.g. dev vs prod), so the config points at a location, not a fixed file name.

# Run command

- **Generic `run` instead of `make`** — the old `make` command (removed in `4eb8302` to simplify the project and focus on Python functionality) is replaced by `run`, which runs any command. Same processing-job path: `SageMakerClient.run()` → `_run_processing(["run", *command])`.
- **Runs from `easy_sm_base/processing/`** — same directory `process` uses (`aa3da62`: processing jobs run inside it so local runs mirror SageMaker), so a `Makefile` or `dbt_project.yml` goes there.
- **`exec "$@"` in `executor.sh`, not `sh -c "$*"`** — arguments reach the tool unchanged. Checked: with `sh -c "$*"`, `python -c 'print("a b")'` is a syntax error and `echo 'x; echo INJECTED'` runs a second command. `exec` also makes the tool PID 1, so its exit code fails the SageMaker job. Shell features need an explicit `-- sh -c "..."`.
- **Trailing arguments after `--`, not a `--command` flag** — a string flag would need re-splitting (`shlex`) and nested quoting, and goes against `docker run` / `uv run` convention. A command already held in one string can use `-- sh -c "$CMD"`.
- **Option parsing stops at the command** (`allow_interspersed_args=False`, `0a4251a`) — without it, `easy_sm run -e ml.m5.large -n job dbt run -e prod` silently submitted instance type `prod`. easy_sm options must come before the command; a command starting with `-` needs `--`.
- **Guard against outdated projects** (`require_run_support`) — an image built from an old `executor.sh` falls through to the model server on `run` and runs until the 24 h processing timeout. The guard checks the project's local `executor.sh` only; an image that wasn't rebuilt and pushed is not detected. Risk accepted for 1.3.0; `--max-runtime` is on the todolist.
- **`executor.sh` still serves on unknown arguments** — making the fallback strict was considered and dropped: it can't help old images (the real risk) and SageMaker hosting relies on the `serve` / no-argument path.
- **`local run` returns the command's exit code** — so it works in CI. Other local commands still exit 0 on failure; left unchanged to avoid breaking existing users.

# Security

- **Env vars passed as separate arguments** (`ce2c366`) — `process_local.sh` used to join `--env` values with commas and expand them unquoted; checked that `MSG=hi --privileged -v /:/host` became real docker flags. Now each `KEY=VALUE` is one argument rebuilt via `"$@"`. `local process --env` refuses to run with the old script (it silently dropped all but the first variable).
- **`~/.aws` stays writable in local containers** — `:ro` would crash SSO users: botocore's `SSOTokenProvider` writes the refreshed token to `~/.aws/sso/cache` without catching errors (checked in botocore `tokens.py`). Proper fix is passing temporary credentials; open.
- **`local deploy` binds to `127.0.0.1`** (`ce2c366`) — previously published on all interfaces. `localhost` URLs unchanged; `stop_local.sh` still matches the container.
- **No secrets in config or job definitions** — credentials were removed from `{app}.json` in favour of boto3's credential chain (`d64173c`); docs warn that `--env` values and `run` arguments are stored in plain text in the SageMaker job.
- **App name validated** (`4eb8302`) — alphanumeric, `-` and `_` only, to block path traversal via `{app_name}.json`; scripts installed `0o755`, not `0o777`.
- **PyPI Trusted Publishing, `pypi` environment limited to `v*` tags** (PR #10) — replaces the long-lived `PYPI_TOKEN`. No required reviewer and no tag ruleset: solo repo, chosen 2026-10-13.
- **Least-privilege publish jobs** (PR #10) — `build` (read) / `publish` (`id-token: write` only) / `release` (`contents: write`), default `permissions: {}`, so build steps that install packages never get a publishing credential. Archived `actions/create-release` replaced by `gh release create`.
- **Releases only from tags** (PR #10) — `workflow_dispatch` (added in `75db68b` "for testing") removed from publish: it let anyone with write access publish from any branch.
- **Actions pinned to commit SHAs, Dependabot keeps them current** (PR #10) — pins are the versions the floating tags resolved to on 2026-10-13, so no behaviour change; without Dependabot, pins would silently miss security fixes.

# SageMaker SDK

- **SDK pinned to `==2.257.3`** (`ce837b7`) — history: pinned `<3.0.0` "for compatibility" (`9e38f4a`), moved to v3 (`a19683f`), relaxed back to `>=2.243.0` the next day (`ddb0a3b`), then pinned exactly with imports switched from `sagemaker.core.*` to `sagemaker.*`. Reason for the exact pin not recorded beyond "stability".
- **Unique endpoint config per deploy** (`62c900e`, `01dd9fa`) — a fixed `{endpoint}-config` name made redeploys fail with "already existing endpoint configuration"; old configs are cleaned up best-effort, and `delete-endpoint --delete-config` looks up the current config name first.
- **Silence SDK logging before import** (`0a820ed`) — the SDK logs at import time and only honours a log level set beforehand; otherwise the line pollutes pipe-friendly output.

# CI and release

- **Tests on every push, publish only on `v*` tags** (`d73519c`, `4af1623`) — keeps the badge honest; workflow renamed to `publish.yml` to drop the old failing run history. PyPI trusted publisher is bound to that file name.
- **Tag only after CI passes on `main`** (2026-10-13) — a PyPI version can't be re-uploaded. Paid off on 1.3.0: the first `main` run failed (ANSI-sensitive test) and was fixed (`d663439`) before tagging.
- **Minor version for features, patch for fixes** — env vars → 1.1.0, `docker_tag` → 1.2.0, `run` → 1.3.0. Version lives in `setup.py`, `easy_sm/__init__.py`, `mkdocs.yml`.
- **Python `>=3.13`** (`4614cc0`) — 3.14 was tried and dropped because of pydantic problems on the 3.14 release candidate.
- **Pinned `ruff==0.16.2`** (`f7865b9`) — an unpinned ruff in CI picked up new default rules and broke the build without any code change.
- **mypy `warn_return_any = False`** (`49592cf`) — boto3 / SageMaker SDK calls return `Any` (no stubs).

# Testing and docs

- **Mock AWS and Docker in Python tests; run shell scripts for real** — Python tests mock subprocess / boto3 / SDK; `test_local_scripts.py` runs the template scripts with a fake `docker` that records its arguments. Each new script test was checked to fail against the old script.
- **Test against coloured output** — Rich splits flags with ANSI codes when colour is on (GitHub Actions), so tests strip ANSI before matching CLI errors. Reproduce with `FORCE_COLOR=1`.
- **Docs describe only what the code does** (`cfc06a4`, `e25c95d`, `c26de9c`) — unsupported features (hyperparameters, `--tags`) were removed, and `validate_docs.py` + the pre-commit hook guard against drift. Known gap: it compares no flags, because no command page has a `Flag` table header.
- **Three project docs at the repo root, no README** (2026-10-13) — `summary.md`, `decisions.md`, `todolist.txt`. A root `README.md` repo map duplicated CLAUDE.md's structure map and would be mistaken for the user guide (`.github/README.md`), so it was dropped. `summary.md` keeps only state no tool shows, pointing to todolist sections, because version / PRs / CI are one command away and duplicated todos drift.
