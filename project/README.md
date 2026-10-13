# easy_sm project docs

easy_sm is a Python CLI (published on PyPI as `easy-sm`) for prototyping SageMaker training, processing and deployment locally in Docker, then running the same image on SageMaker. These four files track the state of the work so it can be picked up cold. The user guide is [`.github/README.md`](../.github/README.md); code conventions and commands are in [`CLAUDE.md`](../CLAUDE.md).

- `project/README.md`, `summary.md`, `decisions.md`, `todolist.txt` — project docs (this folder; maintained with the `project-docs` skill).
- `easy_sm/__main__.py` — CLI entry point; registers every top-level command and the `local` sub-app.
- `easy_sm/commands/` — command implementations: `cloud.py` (SageMaker commands incl. `process`, `run`), `local.py` (`local train/deploy/process/run/stop`), `build.py`, `push.py`, `initialize.py`, `update.py` (`update-scripts`), `helpers.py` (subprocess, config loading, env parsing, outdated-script guards).
- `easy_sm/sagemaker/sagemaker.py` — `SageMakerClient`, the wrapper around the SageMaker SDK and boto3.
- `easy_sm/config/config.py` — `Config` / `ConfigManager` for `{app_name}.json` (incl. `docker_tag`).
- `easy_sm/template/easy_sm_base/` — files copied into user projects by `init`: Dockerfile, `executor.sh` (container entrypoint), `build.sh`, `push.sh`, `local_test/*.sh`.
- `tests/` — pytest suite (160 tests); `test_local_scripts.py` runs the real shell scripts against a fake `docker`.
- `docs/`, `mkdocs.yml`, `requirements-docs.txt` — MkDocs site, deployed to https://prteek.github.io/easy_sm/.
- `scripts/validate_docs.py` — checks docs against command code; run by `.githooks/pre-commit` and `/validate-docs`.
- `.github/workflows/` — `test.yml` (pytest, mypy, ruff), `publish.yml` (PyPI + GitHub release on `v*` tags), `docs.yml` (Pages).
- `.claude/skills/` — `refresh`, `validate-docs`, `project-docs`.
- `setup.py`, `easy_sm/__init__.py`, `mkdocs.yml` — the three places the version number lives.
- `DOCS_WORKFLOW.md` — documentation validation workflow.
