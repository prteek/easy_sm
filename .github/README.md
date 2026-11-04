# easy_sm

Prototype SageMaker training, jobs and endpoints locally with Docker, then run the same commands in the cloud.

> **Version 2 is being rewritten.** Only `easy_sm init` and `easy_sm env` exist so far. Version 1.4.0 is on PyPI and on the `main` branch.

## Start a project

```bash
easy_sm init churn
cd churn
easy_sm env        # every setting, its value, and where the value came from
```

`init` creates:

| File | What it is |
|---|---|
| `easy_sm.toml` | Project settings: AWS region and role, instance types for training, jobs and endpoints |
| `pyproject.toml` | Your dependencies (`uv add pandas`) |
| `src/train.py` | `train(input_dir, model_dir)`: read data, save a model |
| `src/predict.py` | `Handler(model_dir)`, called with each prediction request |
| `data/train.csv` | Sample data |

## Settings

Each setting comes from the first of: a command-line flag, an environment variable (`AWS_REGION`, `SAGEMAKER_ROLE`), `easy_sm.toml`, a built-in default. Use another file with `easy_sm --config prod.toml ...`.

AWS credentials are never in `easy_sm.toml`: they come from your environment exactly as for the AWS CLI (`AWS_PROFILE`, access keys, OIDC in CI, an instance role). `easy_sm env` shows which account and credential source are in use.

## Credits

The design of easy_sm 2 borrows heavily from the [Hugging Face Hub CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli): commands grouped by resource then action, the main object as a positional argument, `run` taking the command directly like `hf jobs run`, user inference code as a small handler class like Inference Endpoints' `handler.py`, deploying from a training job the way `hf` deploys from a repo, and (planned) a skill file for coding agents like `hf skills add`. Thanks to the Hugging Face team for a CLI worth learning from.
