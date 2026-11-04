# easy_sm

Train and deploy machine learning models on AWS SageMaker, right from your laptop. Try everything locally first, then run the very same command in the cloud.

```bash
uv tool install easy_sm
```

All you need is [Docker](https://www.docker.com/products/docker-desktop/) and [uv](https://docs.astral.sh/uv/). AWS comes later.

## Try it in two minutes

No AWS account needed. Everything runs in Docker on your machine.

```bash
easy_sm init churn && cd churn     # a ready-to-run sample project
easy_sm train --local              # trains on data/train.csv, saves the model
easy_sm endpoints serve --local    # serves it at 127.0.0.1:8080 (Ctrl-C to stop)
```

In a second terminal, ask it for a prediction:

```bash
easy_sm endpoints invoke --local -d '{"inputs": [[1.0], [2.0]]}'
# [5.05, 5.05]
```

That's a trained model answering requests. Now make it yours.

## Write your own model

A project is two small Python files in `src/`.

```python
# src/train.py: read data, save a model
def train(input_dir: str, model_dir: str) -> None:
    ...

# src/predict.py: load the model once, answer each request
class Handler:
    def __init__(self, model_dir: str): ...
    def __call__(self, data): ...    # return anything JSON can hold
```

Put shared code, like feature steps, in its own file such as `src/features.py`, and import it from both. Keep any class you save inside a model there too, not in `train.py`, so the endpoint can load it.

Need a library? `uv add scikit-learn`. easy_sm rebuilds the image only when your code or dependencies change.

## Take it to the cloud

Once [AWS is set up](#set-up-aws-once), drop `--local`. Same commands, now on SageMaker.

```bash
easy_sm data upload                  # copy data/ to S3
easy_sm train                        # train on SageMaker, with logs as it runs
easy_sm endpoints deploy --latest    # serve the model you just trained
easy_sm endpoints invoke -d '{"inputs": [[1.0], [2.0]]}'
easy_sm endpoints delete churn       # endpoints bill by the hour, so delete when done
```

## Run any script

Data prep, a report, a dbt build: anything that should run near your data.

```bash
easy_sm run --local process.py --date 2026-11-01   # on your machine
easy_sm run process.py --date 2026-11-01           # as a SageMaker job
```

Your script lives in `src/`, and everything after its name is passed straight to it. Python tools you `uv add` are ready to use. For system tools like `git` or `make`, see [Customise the image](#customise-the-image).

## Everything easy_sm can do

**Get started**

| | |
|---|---|
| `easy_sm init NAME` | Create a project |
| `easy_sm env` | See your settings, the AWS account in use and the image tag |

**Train and run**

| | |
|---|---|
| `easy_sm train` | Train with `src/train.py` |
| `easy_sm run COMMAND` | Run a script or command from `src/` |
| `easy_sm data upload` | Copy `data/` to S3 |

**Deploy**

| | |
|---|---|
| `easy_sm endpoints deploy --latest` | Serve your newest trained model |
| `easy_sm endpoints serve --local` | Serve a locally trained model |
| `easy_sm endpoints invoke -d DATA` | Ask for one prediction |
| `easy_sm endpoints ls` · `describe` · `delete NAME` | See and remove endpoints |

**Keep an eye on things**

| | |
|---|---|
| `easy_sm jobs ls` | Your recent jobs |
| `easy_sm jobs inspect` · `logs` · `cancel` · `artifacts` | Details, logs, stop a job, find its model |

**Customise**

| | |
|---|---|
| `easy_sm build` · `push` | Build the image, or push it to AWS. Other commands do this for you |
| `easy_sm dockerfile` | Show the Dockerfile easy_sm uses |
| `easy_sm skill` | Teach your coding agent to use easy_sm |

Add `--local` to `train`, `run`, `endpoints serve` and `endpoints invoke` to use Docker instead of AWS. Add `--dry-run` to `train`, `run`, `push` or `endpoints deploy` to see exactly what would happen without doing it. Add `--format json` to `env` and the `jobs` and `endpoints` listings for scripts. Every command explains itself with `--help`.

## Settings

Instance types, S3 paths and other defaults live in `easy_sm.toml` in your project. A flag overrides them for a single run:

```bash
easy_sm train --instance ml.m5.2xlarge
```

## Customise the image

easy_sm builds the image for you. To add system tools, save its Dockerfile into your project and edit it:

```bash
easy_sm dockerfile > Dockerfile    # then add e.g. RUN apt-get install -y git after FROM
```

easy_sm uses your Dockerfile from then on. Delete it to go back. After upgrading easy_sm, `easy_sm dockerfile | diff - Dockerfile` shows what changed.

## Set up AWS once

1. **Credentials.** easy_sm uses the same credentials as the AWS CLI. If `aws sts get-caller-identity` works, you're set.
2. **A SageMaker role.** Your jobs and endpoints run as this role. Tell easy_sm which one:

   ```bash
   export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/my-sagemaker-role
   ```

   or set `role` under `[aws]` in `easy_sm.toml`.
3. **Check.** `easy_sm env` shows the account, region and role easy_sm will use.

On first use, easy_sm creates an ECR repository named after your project and, if you haven't set S3 paths, the bucket `sagemaker-<region>-<account>`.

<details markdown="1">
<summary><strong>Permissions the SageMaker role needs</strong></summary>

The role needs a trust policy that lets SageMaker use it:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Principal": { "Service": "sagemaker.amazonaws.com" }, "Action": "sts:AssumeRole" }
  ]
}
```

And this permissions policy, with `REGION`, `ACCOUNT` and `APP` (your `app_name`) filled in:

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

- Using other buckets for `train.input`, `train.output`, `run.input` or `run.output`? Add them to the two S3 statements. KMS-encrypted buckets also need `kms:Decrypt` and `kms:GenerateDataKey` on their key.
- Add anything your own code calls: other buckets, secrets, databases.
- Prefer something broader? The managed policy `AmazonSageMakerFullAccess` works too.

</details>

<details markdown="1">
<summary><strong>Permissions you (or CI) need</strong></summary>

- `sts:GetCallerIdentity`
- `iam:PassRole` on the SageMaker role
- ECR: `ecr:GetAuthorizationToken`, `DescribeRepositories`, `CreateRepository`, `DescribeImages`, and push (`BatchCheckLayerAvailability`, `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, `PutImage`)
- S3: `s3:ListBucket`, `s3:PutObject`, and `s3:CreateBucket` if the default bucket doesn't exist yet
- SageMaker: create, describe, list and stop training and processing jobs; create and describe models and endpoint configs; create, update, describe, list, invoke and delete endpoints; delete endpoint configs (only with `--delete-config`)
- CloudWatch Logs: `logs:DescribeLogStreams`, `logs:GetLogEvents`

</details>

## Good to know

- **Cost.** Endpoints bill for every hour they're running, used or not. Serverless endpoints bill only when called. easy_sm never deletes anything on its own, so clean up with `easy_sm endpoints delete` and the AWS console.
- **Quotas.** New AWS accounts often have a limit of 0 for some instance types. A job then fails with `ResourceLimitExceeded` until you request an increase.
- **One region.** Your role, images, data and jobs all need to be in the same region (`AWS_REGION`, `region` under `[aws]`, or your AWS CLI default).
- **Apple Silicon.** SageMaker runs Intel machines, so images are built for them. On an M-series Mac the first build takes a little longer.
- **Credentials stay out of your project.** They never go in `easy_sm.toml`.
- **Same version for the whole team.** Inside a project, `uv run easy_sm ...` uses the version pinned in `uv.lock`. Upgrade with `uv lock --upgrade-package easy_sm`.
- **Coming from 1.x?** Version 2 is a fresh start and doesn't read 1.x projects.

## Credits

The design of easy_sm 2 borrows heavily from the [Hugging Face Hub CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli): commands grouped by resource then action, the main object as a positional argument, `run` taking the command directly like `hf jobs run`, user inference code as a small handler class like Inference Endpoints' `handler.py`, deploying from a training job the way `hf` deploys from a repo, and a skill file for coding agents like `hf skills add`. Thanks to the Hugging Face team for a CLI worth learning from.
