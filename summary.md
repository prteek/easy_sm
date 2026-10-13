---
title: Summary
updated: 2026-10-13
---

Only what tools can't tell you. For version, open PRs and CI, run `git tag`, `gh pr list` and `gh run list`.

- **Unreviewed:** Dependabot PR #11 bumps all 8 actions by a major version (*Release*).
- **Untested:** `easy_sm run` has only been run in local Docker, never on SageMaker (*Run command*).
- **Risks:** an image built before 1.3.0 starts the model server on `run` and bills until the 24 h timeout; local containers get a writable `~/.aws`; `push.sh` edits `~/.aws/config` permanently (*Run command*, *Security*).
- **Misleading docs:** user docs describe a non-existent `--docker-tag` option, and `validate_docs.py` compares no flags (*Docs*).
- **Local only, by choice:** branch `docs/project-docs` (these docs and the `project-docs` skill) is not pushed (*Housekeeping*).
