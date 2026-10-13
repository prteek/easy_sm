---
name: project-docs
description: Set up or update the project docs at the repo root (README.md, summary.md, decisions.md, todolist.txt) that track the state of work on easy_sm. Use when asked to "set up/update project docs", to work through items in todolist.txt, or after finishing a piece of work (a feature, fix, release or PR) to record status and decisions.
---

# Project docs

Four short files at the repo root, so work on easy_sm can be picked up cold. They track *the work*; the user guide is `.github/README.md` (GitHub shows it in place of the root `README.md`) and code conventions are in `CLAUDE.md`. Link to those rather than repeat them.

| File | Answers |
|---|---|
| `README.md` | What is this repo and where is everything? |
| `summary.md` | Where are we now (released version, open PRs, open risks)? |
| `decisions.md` | What did we choose, and why? |
| `todolist.txt` | What's left to do? |

## Rules

- Dates are absolute (`YYYY-MM-DD`). `summary.md` and `decisions.md` start with front matter `title:` and `updated:`; bump `updated` on every edit.
- **README.md** — one-paragraph purpose, then one line per file or folder saying what it is. Tracked files only (check `.gitignore`).
- **summary.md** — current state as short bullets: latest released version, what's merged but unreleased, open PRs and what blocks them, CI status, open risks. Rewrite stale bullets; don't append history.
- **decisions.md** — sections by area (e.g. `# Run command`, `# Security`, `# Code and release`). Bullets: **short name** — the rule, why, and the evidence (a test result, a reproduced failure, a commit hash). Record checked facts, not intentions. If the reason isn't recorded anywhere, say "reason not recorded" rather than guess. Rewrite a decision in place when it changes.
- **todolist.txt** — Todo+ format: sections end in `:`, items `☐` pending, `✔` done, `✘` cancelled. Finished items move to `Archive:` (last section, newest first) as `✔ <task> (<result>) @done(YY-MM-DD HH:MM)`. Never invent a time: take it from the commit (`git log --date=format:"%y-%m-%d %H:%M"`), else use `@done(YY-MM-DD)`.

## Sources of truth

Check facts before writing them; don't copy numbers from memory or from other docs.

- Version: `setup.py`, `easy_sm/__init__.py`, `mkdocs.yml` (all three must agree); latest tag: `git tag --sort=-v:refname | head -1`; on PyPI: `https://pypi.org/pypi/easy-sm/json`.
- Test count: `pytest -q --collect-only | tail -1`.
- PRs and CI: `gh pr list`, `gh run list --branch main --limit 3`.
- History and reasons: `git log --format="%h %ad %s" --date=short`, `git show -s <hash>`.

## Working

- **Starting:** create the four files with what is known; leave `TBD` where unknown.
- **After each piece of work** (commit, PR, release): move the todo to `Archive:` with its result, update the matching `summary.md` bullet, add any choice or caveat to `decisions.md`, and add new files to `README.md`.
- **Todo list as work queue:** "do item N" means the Nth pending `☐` in the named section. Don't reorder or reword the user's items. Prerequisites go above the item they block; out-of-scope findings become new items rather than silent fixes.
- **Code changes still follow CLAUDE.md:** explain the plan and wait for OK before writing code, run tests/mypy/ruff, and run `/validate-docs` after doc edits.
- **Keep it short:** one line per item, one or two sentences per decision or bullet, the result plus the one number that proves it. Link rather than repeat.
- **Human edits win:** re-read a file before editing and build on the user's version.
