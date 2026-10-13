---
title: Summary
updated: 2026-10-13
---

- **Released:** 1.3.0 on PyPI and GitHub (2026-10-13), adding `run` / `local run` and the local-script security fixes. 160 tests pass in CI; mypy, ruff and doc validation are clean.
- **`run` command:** runs any CLI tool (e.g. `make`, `dbt`) as a SageMaker processing job or locally, from `easy_sm_base/processing/` ([decisions.md](./decisions.md#run-command)). Checked for real in local Docker; **never run on SageMaker yet**.
- **Security fixes in 1.3.0:** `local process --env` can no longer inject docker flags; `local deploy` binds to `127.0.0.1`. Existing projects get them through `easy_sm update-scripts`.
- **Release hardening in review:** [PR #10](https://github.com/prteek/easy_sm/pull/10) moves publishing to PyPI Trusted Publishing and pins actions to SHAs. Tests pass; blocked on adding the trusted publisher on pypi.org, which is waiting on a PyPI login problem (2026-10-13). Manual steps are in [todolist.txt](./todolist.txt) under Release. The `pypi` GitHub environment (`v*` tags only) already exists.
- **Open security items:** writable `~/.aws` mount in local containers; an old image + `run` starts the model server and bills until the 24 h timeout; `push.sh` edits `~/.aws/config` permanently. See [todolist.txt](./todolist.txt).
- **Known doc gaps:** user docs describe a global `--docker-tag` option that doesn't exist, and `validate_docs.py` compares no flags (no command page has a `Flag` table header).
