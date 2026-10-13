# run

Run any CLI command (`make`, `dbt`, a Python module, a shell script...) as a processing job on AWS SageMaker.

## Synopsis

```bash
easy_sm run --base-job-name NAME --ec2-type TYPE [OPTIONS] -- COMMAND [ARGS...]
```

## Description

The `run` command submits a SageMaker processing job that runs your command inside your project's Docker image. It works like [`process`](process.md), but instead of running a single Python file it runs whatever command you give it.

- Everything after `--` is the command. It reaches the container exactly as you typed it, with no shell re-parsing, so spaces, quotes and `$` characters are kept.
- easy_sm's own options must come **before** the command. Once the command starts, every remaining argument belongs to it.
- The command runs from the `easy_sm_base/processing/` directory, so put files like your `Makefile` or `dbt_project.yml` there.
- The tool must be installed in the image. `make` is included in the default Python base image; anything else (e.g. `dbt-core`) must be in your requirements file.
- If the command exits with a non-zero code, the SageMaker job is marked as failed.

By default, the command waits for the job to finish before returning. Pass `--no-wait` to submit the job and return immediately.

The Docker image tag comes from `docker_tag` in your `{app_name}.json` (default `latest`).

!!! tip "Put `--` before the command"
    easy_sm stops reading its own options at the first word of your command, so flags like `dbt run -e prod` always go to your tool, even though `-e` is also an easy_sm option. `--` is still recommended: it makes the boundary obvious to readers, and it's required if your command itself starts with `-`.

    ```bash
    # Correct: easy_sm options first, then the command
    easy_sm run -e ml.m5.large -n job -- dbt run -e prod

    # Wrong: -e ml.m5.large is passed to dbt, and easy_sm reports that --ec2-type is missing
    easy_sm run -n job dbt run -e ml.m5.large
    ```

## Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--base-job-name` | `-n` | string | **Yes** | - | Prefix for the SageMaker processing job |
| `--ec2-type` | `-e` | string | **Yes** | - | EC2 instance type (e.g., `ml.m5.large`) |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | AWS IAM role ARN for SageMaker |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--instance-count` | `-c` | integer | No | `1` | Number of EC2 instances |
| `--s3-input-location` | `-i` | string | No | None | S3 location for input data |
| `--s3-output-location` | `-o` | string | No | None | S3 location to save output |
| `--input-sharded` | `-is` | boolean | No | `false` | Shard input data across instances |
| `--env` | | string | No | None | Environment variables in `KEY=VALUE` format (repeatable) |
| `--no-wait` | | boolean | No | `false` | Don't wait for the processing job to finish |

## Examples

### Run a make target

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

easy_sm run -n build-features -e ml.m5.large -- make features
```

With a Makefile in `easy_sm_base/processing/`:

```makefile
features:
	python build_features.py --input /opt/ml/processing/input --output /opt/ml/processing/output
```

### Pass flags to your tool

```bash
easy_sm run -n parallel-build -e ml.m5.xlarge -- make -j4 all
```

### Run dbt

```bash
easy_sm run -n dbt-nightly -e ml.m5.large \
  --env DBT_TARGET=prod \
  -- dbt build --select tag:nightly --target prod
```

### Run a Python module

```bash
easy_sm run -n backfill -e ml.m5.large -- python -m mypackage.backfill --since 2024-01-01
```

### With input and output data

```bash
easy_sm run -n clean-data -e ml.m5.large \
  -i s3://my-bucket/raw \
  -o s3://my-bucket/clean \
  -- make clean
```

Input data is available at `/opt/ml/processing/input/` and anything written to `/opt/ml/processing/output/` is uploaded to the output location.

### Pipes and `&&`

The command is not run through a shell, so `|`, `&&` and `>` are passed to your tool as plain arguments. If you need shell features, ask for a shell explicitly:

```bash
easy_sm run -n pipeline -e ml.m5.large -- sh -c "make extract && make load"
```

### Submit without waiting

```bash
easy_sm run -n nightly -e ml.m5.large --no-wait -- make nightly
```

## Output Format

Outputs the base job name, so it can be piped into other commands:

```
build-features
```

## Testing locally first

Run the same command in Docker on your machine with [`local run`](local.md#run):

```bash
easy_sm local run -a my-app -- make features
```

## Upgrading an existing project

Projects created before `run` existed need the new container entrypoint. If yours is outdated, easy_sm stops with an error explaining what to do:

```bash
easy_sm update-scripts -a my-app
easy_sm build -a my-app
easy_sm push -a my-app
```

The image must be rebuilt and pushed because `executor.sh` is copied into the image at build time.

## Security notes

- The command runs with the permissions of the job's IAM role. Give that role access only to what the job needs.
- Command arguments and `--env` values are stored in plain text in the SageMaker job definition and are visible in the console and through `DescribeProcessingJob`. Don't put secrets in them; fetch secrets from AWS Secrets Manager inside the job instead.

## Prerequisites

- Docker image built with `easy_sm build` and pushed with `easy_sm push`
- IAM role (via `-r` or `SAGEMAKER_ROLE`)
- The tool you're running installed in the image

## Related Commands

- [`process`](process.md) - Run a single Python file as a processing job
- [`local run`](local.md#run) - Run a command locally in Docker
- [`update-scripts`](update-scripts.md) - Update project scripts to support `run`
