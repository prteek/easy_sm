# Command Reference

Complete reference for all easy_sm CLI commands.

## Command Structure

Cloud commands are at the top level for simplicity. Local operations are under the `local` sub-command.

```text
easy_sm [--docker-tag TAG] COMMAND [OPTIONS]
```

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--docker-tag`, `-t` | Docker image tag | `latest` |
| `--help` | Show help message | - |
| `--version` | Show version | - |

## Commands by Category

### Initialization

| Command | Description |
|---------|-------------|
| [`init`](init.md) | Initialize new easy_sm project with template files |

### Build & Push

| Command | Description |
|---------|-------------|
| [`build`](build.md) | Build Docker image locally |
| [`push`](push.md) | Push Docker image to AWS ECR |
| [`update-scripts`](update-scripts.md) | Update shell scripts with latest secure versions |

### Local Operations

| Command | Description |
|---------|-------------|
| [`local train`](local.md#train) | Train model locally in Docker |
| [`local deploy`](local.md#deploy) | Deploy model locally on port 8080 |
| [`local process`](local.md#process) | Run processing job locally |
| [`local stop`](local.md#stop) | Stop local deployment |

### Cloud Training

| Command | Description |
|---------|-------------|
| [`train`](train.md) | Train model on AWS SageMaker |

### Cloud Deployment

| Command | Description |
|---------|-------------|
| [`deploy`](deploy.md#provisioned) | Deploy to provisioned SageMaker endpoint |
| [`deploy-serverless`](deploy.md#serverless) | Deploy to serverless SageMaker endpoint |

### Cloud Processing

| Command | Description |
|---------|-------------|
| [`process`](process.md) | Run processing job on SageMaker |
| [`batch-transform`](batch-transform.md) | Run batch predictions |

### Management

| Command | Description |
|---------|-------------|
| [`list-endpoints`](endpoints.md#list) | List all SageMaker endpoints |
| [`list-training-jobs`](train.md#list) | List recent training jobs |
| [`get-model-artifacts`](train.md#get-artifacts) | Get S3 model path from training job |
| [`delete-endpoint`](endpoints.md#delete) | Delete a SageMaker endpoint |
| `upload-data` | Upload data to S3 |

## Common Options

Many commands share common options:

### App Name

```bash
--app-name, -a TEXT     App name (auto-detected if not specified)
```

Auto-detected from `*.json` config file in current directory. Override when multiple configs exist.

### IAM Role

```bash
--iam-role-arn, -r TEXT    AWS IAM role ARN (or set SAGEMAKER_ROLE env var)
```

Required for cloud operations. Set via environment variable or flag:

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
```

### Instance Configuration

```bash
--instance-type, -e TEXT         EC2 instance type
--num-instances INTEGER          Number of instances
```

Common instance types:

- **Training**: `ml.m5.large`, `ml.m5.xlarge`, `ml.p3.2xlarge` (GPU)
- **Inference**: `ml.t2.medium`, `ml.m5.large`, `ml.c5.xlarge`

See [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/) for full list.

### S3 Paths

```bash
--input-data, -i TEXT        S3 input path (s3://bucket/path)
--output-path, -o TEXT       S3 output path (s3://bucket/path)
```

S3 paths must use the `s3://` protocol.

## Usage Patterns

### Auto-Detection

Most commands work without specifying app name or IAM role:

```bash
# App name auto-detected from *.json file
# IAM role from SAGEMAKER_ROLE env var
easy_sm build
easy_sm train -n job-name -e ml.m5.large -i s3://... -o s3://...
```

### Explicit Configuration

Override auto-detection when needed:

```bash
easy_sm build -a my-app
easy_sm train -a my-app -r arn:aws:iam::123456789012:role/OtherRole ...
```

### Docker Tags

Use versioned Docker tags:

```bash
easy_sm --docker-tag v1.0 build
easy_sm --docker-tag v1.0 push
easy_sm --docker-tag v1.0 train ...
```

### Piped Workflows

Combine commands with Unix pipes:

```bash
# Get latest training job model and deploy
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

See [Piped Workflows](../user-guide/piped-workflows.md) for more examples.

## Command Details

Click on any command for detailed documentation:

- [init](init.md) - Initialize projects
- [build](build.md) - Build Docker images
- [push](push.md) - Push to ECR
- [local](local.md) - Local operations
- [train](train.md) - Train on SageMaker
- [deploy](deploy.md) - Deploy endpoints
- [process](process.md) - Processing jobs
- [batch-transform](batch-transform.md) - Batch predictions
- [endpoints](endpoints.md) - Manage endpoints
- [update-scripts](update-scripts.md) - Update shell scripts
