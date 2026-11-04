# easy_sm

Train, run and deploy on AWS SageMaker. Every command works locally in Docker first (`--local`), then the same command runs in the cloud.

You need Docker, [uv](https://docs.astral.sh/uv/), AWS credentials (as for the AWS CLI) and a SageMaker execution role:

```bash
uv tool install easy_sm
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/my-sagemaker-role
```

Version 2 is a rewrite and does not read 1.x projects. Inside a project, `uv run easy_sm ...` uses the version locked in its `uv.lock`; upgrade it with `uv lock --upgrade-package easy_sm`.

## AWS setup

**The SageMaker execution role** (`SAGEMAKER_ROLE`, or `[aws] role` in `easy_sm.toml`) is the identity your jobs and endpoints run as. It needs:

- a trust policy that lets `sagemaker.amazonaws.com` assume it (easy_sm never assumes it itself, so it doesn't need to trust your user)
- ECR: pull the project image (`ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`)
- S3: read training and `run` inputs, write models and `run` outputs, read `model.tar.gz` for endpoints (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`)
- CloudWatch Logs: write job and endpoint logs (`logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`)
- anything your own code calls (other buckets, secrets, databases)

A minimal role for easy_sm's defaults: replace `REGION`, `ACCOUNT` and `APP` (your `app_name`).

Trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Principal": { "Service": "sagemaker.amazonaws.com" }, "Action": "sts:AssumeRole" }
  ]
}
```

Permissions policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
    {
      "Sid": "PullProjectImage", "Effect": "Allow",
      "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
      "Resource": "arn:aws:ecr:REGION:ACCOUNT:repository/APP"
    },
    { "Sid": "ListBucket", "Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::sagemaker-REGION-ACCOUNT" },
    {
      "Sid": "ReadWriteProjectData", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::sagemaker-REGION-ACCOUNT/APP/*"
    },
    {
      "Sid": "Logs", "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
      "Resource": "arn:aws:logs:REGION:ACCOUNT:log-group:/aws/sagemaker/*"
    },
    { "Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*" }
  ]
}
```

If `train.input`, `train.output`, `run.input` or `run.output` point at other buckets, add them to the two S3 statements; KMS-encrypted buckets also need `kms:Decrypt` and `kms:GenerateDataKey` on their key. The managed policy `AmazonSageMakerFullAccess` is a broader alternative that also covers buckets with `sagemaker` in their name.

**You (or CI) running easy_sm** need:

- `sts:GetCallerIdentity`
- `iam:PassRole` on the execution role, to hand it to SageMaker
- ECR: `ecr:GetAuthorizationToken`, `DescribeRepositories`, `CreateRepository`, `DescribeImages` and push (`BatchCheckLayerAvailability`, `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, `PutImage`)
- S3: `s3:ListBucket`, `s3:PutObject`, and `s3:CreateBucket` if the default bucket doesn't exist yet
- SageMaker: create, describe, list and stop training and processing jobs; create and describe models and endpoint configs; create, update, describe, list, invoke and delete endpoints; delete endpoint configs (only with `--delete-config`)
- CloudWatch Logs: `logs:DescribeLogStreams`, `logs:GetLogEvents` (to show job logs)

**Good to know:**

- Credentials come from your environment exactly as for the AWS CLI; they never go in `easy_sm.toml`. Run `easy_sm env` to see which account and credential source are in use before cloud work.
- The role, the ECR repository, the S3 data and the jobs must be in the same region (`AWS_REGION`, `[aws] region`, or your AWS CLI default).
- easy_sm creates two things on first use: an ECR repository named after the app (tags are immutable), and the bucket `sagemaker-<region>-<account>` when no S3 paths are configured.
- New accounts often have a quota of 0 for some instance types; a job then fails with `ResourceLimitExceeded` until a quota increase is approved.
- Endpoints cost money for every hour they are `InService`, used or not; serverless endpoints only when called. easy_sm never deletes models, endpoint configs, images or S3 data; clean up with `endpoints delete` and the AWS console.
- Images are built for `linux/amd64`, the machines SageMaker runs. On an Apple Silicon Mac that works through emulation, so the first build is slower.

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
