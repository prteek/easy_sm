---
name: easy_sm
description: Train, run and deploy this project on AWS SageMaker with the easy_sm CLI. Use when asked to train a model, run a script or command as a SageMaker job, deploy or call an endpoint, check jobs or logs, or try any of these locally in Docker.
---

# easy_sm

easy_sm runs this project's code (`src/`) in one Docker image, locally (`--local`) or on SageMaker. Settings live in `easy_sm.toml`; check what will be used with `easy_sm env`.

## Rules

- **Run `--dry-run` first** for `train`, `run`, `push` and `endpoints deploy`. It prints the exact SageMaker request (or docker command) as JSON and changes nothing. Show it to the user before anything that costs money.
- Ask the user before `endpoints delete` and `jobs cancel`: they act immediately, with no confirmation prompt.
- Never put AWS credentials in `easy_sm.toml`. They come from the environment, like the AWS CLI. `easy_sm env` shows the account and credential source in use; check it before cloud work.
- stdout carries only data (a tag, a path, a job or endpoint name, a response, JSON); messages, progress and logs go to stderr. Capture results with `$(...)`.
- Use `--format json` on `env`, `jobs ls`, `jobs inspect`, `endpoints ls` and `endpoints describe` to read results.
- Cloud jobs can take minutes to hours. For long jobs use `--no-wait`, then check with `easy_sm jobs inspect <job> --format json` or `easy_sm jobs logs <job>`.

## Workflow

```bash
easy_sm env                                   # settings, image tag, AWS account
easy_sm train --local                         # train in Docker on data/; prints the model folder
easy_sm run --local clean.py --date 2026-11-01
easy_sm endpoints serve --local --port 8765   # foreground; Ctrl-C stops it
easy_sm endpoints invoke --local --port 8765 -d '{"inputs": [[1.0]]}'

easy_sm data upload                           # data/ → S3; prints the S3 folder
easy_sm train --dry-run                       # check the request first
easy_sm train                                 # prints the job name, streams logs, exit 1 on failure
easy_sm run --instance ml.m5.xlarge clean.py
easy_sm endpoints deploy --latest --dry-run
easy_sm endpoints deploy --latest             # newest completed job's model AND the image it trained with
easy_sm endpoints invoke -d @sample.json
easy_sm jobs ls --format json
easy_sm endpoints delete <name>               # ask the user first
```

## Commands

| Command | Does | Prints (stdout) |
|---|---|---|
| `init [PATH] [--name]` | Create a project; never overwrites | created files |
| `env [--format json]` | Settings, sources, image tag, AWS account | table or JSON |
| `build` | Build the image if the code changed | image tag |
| `push [--dry-run]` | Build and push to ECR if missing | ECR image URI |
| `train [--local] [--input] [--output] [--instance] [--count] [--timeout] [--no-wait] [-q] [--dry-run]` | Calls `train(input_dir, model_dir)` in `src/train.py` | job name (cloud) or model folder (local) |
| `run [options] COMMAND...` | Any command from `src/`; `x.py` runs with python. Options go before the command | job name (cloud) |
| `endpoints deploy [NAME] --latest \| --job JOB \| --model S3 [--dry-run]` | Create or update an endpoint; fails if it rolls back | endpoint name |
| `endpoints invoke [NAME] [--local] -d DATA\|@file` | One prediction request | response body |
| `endpoints serve --local [--port]` | Local endpoint for the model from `train --local` | nothing |
| `endpoints ls`, `endpoints describe [NAME]` | Endpoint status, size, model, image | table or JSON |
| `endpoints delete NAME [--delete-config]` | Delete at once, no prompt | what was deleted |
| `jobs ls [-n N] [-q]` | This app's jobs, newest first | table, names (`-q`) or JSON |
| `jobs inspect JOB`, `jobs logs JOB [-f]` | Details and failure reason; log lines | details, log lines |
| `jobs cancel JOB` | Stop a running job | job name |
| `jobs artifacts JOB` | `model.tar.gz` of a completed training job | S3 path |
| `data upload [DIR] [S3]` | Upload a folder (hidden files skipped) | S3 folder |
| `skill [PATH]` | Write this guide to `.claude/skills/easy_sm/SKILL.md`; never overwrites | path written |

## Exit codes

0 done · 1 a check or the job failed (reason on stderr) · 2 the command was called wrongly · 130 Ctrl-C while following a cloud job (the job keeps running) · `run --local` passes through the command's own exit code.

## Your code

- `src/train.py`: `train(input_dir: str, model_dir: str) -> None`. Read data from `input_dir`, save everything needed for prediction into `model_dir`. An exception fails the job and becomes its failure reason.
- `src/predict.py`: `class Handler`, created once as `Handler(model_dir)`, called as `handler(data)` per request. JSON requests arrive parsed; CSV arrives as a list of rows of strings. Return something JSON-serialisable (e.g. `.tolist()`, not a numpy array).
- Dependencies: `uv add <package>`; the image installs them from `uv.lock`.
