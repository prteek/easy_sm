# easy_sm

Train, run and deploy on AWS SageMaker. Every command works locally in Docker first (`--local`), then the same command runs in the cloud.

> **Version 2 is on the `v2` branch and not released yet.** Version 1.4.0 is on PyPI and the `main` branch.

You need Docker, [uv](https://docs.astral.sh/uv/), AWS credentials (as for the AWS CLI) and a SageMaker execution role:

```bash
uv tool install easy_sm
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/my-sagemaker-role
```

## Example 1: train and deploy a model

```bash
easy_sm init churn && cd churn       # creates src/train.py, src/predict.py, data/train.csv
easy_sm train --local                # train in Docker on data/
easy_sm data upload                  # copy data/ to S3
easy_sm train                        # train on SageMaker; prints the job name
easy_sm endpoints deploy --latest    # serve that job's model; prints the endpoint name
easy_sm endpoints invoke -d '{"inputs": [[1.0], [2.0]]}'
easy_sm endpoints delete churn       # stop paying for the endpoint
```

Your code is two functions:

```python
# src/train.py
def train(input_dir: str, model_dir: str) -> None:
    ...  # read data from input_dir, save the model into model_dir

# src/predict.py
class Handler:
    def __init__(self, model_dir: str): ...   # load the model once
    def __call__(self, data): ...             # one request in, JSON-able result out
```

## Example 2: run a script

```bash
# put your script in src/process.py
easy_sm run --local process.py --date 2026-11-01   # in Docker
easy_sm run process.py --date 2026-11-01           # as a SageMaker processing job
```

Arguments after the script name go to the script. Any command works, e.g. `easy_sm run make features`.

## Commands

| Command | What it does |
|---|---|
| `easy_sm init NAME` | Create a project |
| `easy_sm env` | Show settings, the image tag and the AWS account in use |
| `easy_sm train [--local]` | Train with `src/train.py` |
| `easy_sm run [--local] COMMAND` | Run a script or command from `src/` |
| `easy_sm data upload` | Upload `data/` to S3 |
| `easy_sm endpoints deploy --latest` | Deploy the newest trained model |
| `easy_sm endpoints invoke -d DATA` | Send one prediction request |
| `easy_sm endpoints serve --local` | Serve the local model on 127.0.0.1 |
| `easy_sm endpoints ls` / `describe` / `delete NAME` | List, show, delete endpoints |
| `easy_sm jobs ls` / `inspect` / `logs` / `cancel` / `artifacts` | List and manage jobs |
| `easy_sm build` / `push` | Build the image / push it to ECR (other commands do this when needed) |
| `easy_sm skill` | Write a guide for coding agents to `.claude/skills/easy_sm/SKILL.md` |

Add `--dry-run` to `train`, `run`, `push` or `endpoints deploy` to see what would be sent without changing anything, and `--format json` to `env`, `jobs` and `endpoints` listings for scripts. Settings such as instance types live in `easy_sm.toml`; a flag overrides them for one run. AWS credentials never go in that file.

## Credits

The design of easy_sm 2 borrows heavily from the [Hugging Face Hub CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli): commands grouped by resource then action, the main object as a positional argument, `run` taking the command directly like `hf jobs run`, user inference code as a small handler class like Inference Endpoints' `handler.py`, deploying from a training job the way `hf` deploys from a repo, and a skill file for coding agents like `hf skills add`. Thanks to the Hugging Face team for a CLI worth learning from.
