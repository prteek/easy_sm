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
| `easy_sm.toml` | Project settings: AWS profile/region/role, instance types for training, jobs and endpoints |
| `pyproject.toml` | Your dependencies (`uv add pandas`) |
| `src/train.py` | `train(input_dir, model_dir)`: read data, save a model |
| `src/predict.py` | `Handler(model_dir)`, called with each prediction request |
| `data/train.csv` | Sample data |

## Settings

Each setting comes from the first of: a command-line flag, an environment variable (`AWS_PROFILE`, `AWS_REGION`, `SAGEMAKER_ROLE`), `easy_sm.toml`, a built-in default. Use another file with `easy_sm --config prod.toml ...`.
