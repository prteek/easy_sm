# train

Train ML models on AWS SageMaker.

## Synopsis

```bash
easy_sm [--docker-tag TAG] train --base-job-name NAME --ec2-type TYPE \
  --input-s3-dir S3_PATH --output-s3-dir S3_PATH [OPTIONS]
```

## Description

The `train` command submits a training job to AWS SageMaker using your Docker image. It creates a training job that:

1. Pulls your Docker image from ECR
2. Downloads training data from S3
3. Runs your training code
4. Uploads the trained model to S3

The command outputs the S3 location of the trained model, making it easy to pipe into deployment commands.

## Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--base-job-name` | `-n` | string | **Yes** | - | Prefix for the SageMaker training job |
| `--ec2-type` | `-e` | string | **Yes** | - | EC2 instance type (e.g., `ml.m5.large`) |
| `--input-s3-dir` | `-i` | string | **Yes** | - | S3 location for input training data |
| `--output-s3-dir` | `-o` | string | **Yes** | - | S3 location to save trained model |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` env var | AWS IAM role ARN for SageMaker |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--instance-count` | `-c` | integer | No | `1` | Number of EC2 instances for training |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

## Examples

### Basic training job

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

easy_sm train \
  -n my-training-job \
  -e ml.m5.large \
  -i s3://my-bucket/training-data \
  -o s3://my-bucket/models
```

Output:
```
s3://my-bucket/models/my-training-job/output/model.tar.gz
```

### Training with specific IAM role

```bash
easy_sm train \
  -n my-training-job \
  -e ml.m5.large \
  -i s3://my-bucket/data \
  -o s3://my-bucket/output \
  -r arn:aws:iam::123456789012:role/CustomRole
```

### Training with multiple instances

For distributed training:

```bash
easy_sm train \
  -n distributed-job \
  -e ml.m5.xlarge \
  -c 4 \
  -i s3://my-bucket/data \
  -o s3://my-bucket/output
```

### Training with specific Docker tag

```bash
easy_sm -t v1.2.0 train \
  -n production-training \
  -e ml.m5.large \
  -i s3://my-bucket/data \
  -o s3://my-bucket/output
```

### Complete workflow: build, push, train

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# Build and push
easy_sm -t v1.0.0 build
easy_sm -t v1.0.0 push

# Train
easy_sm -t v1.0.0 train \
  -n my-job-v1 \
  -e ml.m5.large \
  -i s3://my-bucket/data \
  -o s3://my-bucket/models
```

## Output Format

The command outputs the S3 path to the trained model:

```
s3://bucket/output-path/job-name/output/model.tar.gz
```

This output is designed for piping into other commands:

```bash
# Save model path
MODEL=$(easy_sm train -n my-job -e ml.m5.large \
  -i s3://bucket/data -o s3://bucket/output)

# Deploy the model
easy_sm deploy -n my-endpoint -e ml.m5.large -m $MODEL
```

## Prerequisites

- Docker image pushed to ECR with [`easy_sm push`](push.md)
- Training data uploaded to S3 (use [`upload-data`](#upload-data) if needed)
- IAM role with SageMaker permissions (either in `SAGEMAKER_ROLE` env var or via `-r` flag)
- IAM role must have:
  - Trust relationship with `sagemaker.amazonaws.com`
  - Permissions: `s3:GetObject`, `s3:PutObject`, `ecr:GetDownloadUrlForLayer`, etc.

## Training Data Structure

Your training data in S3 should be organized as:

```
s3://my-bucket/training-data/
├── train.csv
├── validation.csv
└── metadata.json
```

In the training container, this data is available at:

```
/opt/ml/input/data/training/
```

## Training Code Requirements

Your training code at `training/training.py` should implement:

```python
import os
import joblib

def train(input_data_path, model_save_path):
    """
    Train model on SageMaker.

    Args:
        input_data_path: /opt/ml/input/data/training
        model_save_path: /opt/ml/model
    """
    # Load training data
    train_data = load_data(input_data_path)

    # Train model
    model = train_model(train_data)

    # Save model
    joblib.dump(model, os.path.join(model_save_path, 'model.mdl'))
    print("Training completed")
```

## Container Paths

SageMaker uses these standard paths:

| Path | Purpose |
|------|---------|
| `/opt/ml/input/data/training/` | Input training data |
| `/opt/ml/model/` | Save trained model here |
| `/opt/ml/output/` | Training metrics and logs |

## Model Output

After training, SageMaker automatically:

1. Creates a tarball of `/opt/ml/model/`
2. Uploads it to S3 as `model.tar.gz`
3. The path is: `{output_s3_dir}/{job_name}/output/model.tar.gz`

This model can be used directly with [`deploy`](deploy.md).

## Instance Types

Common instance types for training:

| Instance Type | vCPUs | Memory | Use Case |
|--------------|-------|---------|----------|
| `ml.m5.large` | 2 | 8 GB | Small datasets, testing |
| `ml.m5.xlarge` | 4 | 16 GB | Medium datasets |
| `ml.m5.2xlarge` | 8 | 32 GB | Large datasets |
| `ml.c5.xlarge` | 4 | 8 GB | Compute-intensive |
| `ml.c5.2xlarge` | 8 | 16 GB | Heavy compute |
| `ml.p3.2xlarge` | 8 | 61 GB + GPU | Deep learning |
| `ml.p3.8xlarge` | 32 | 244 GB + 4 GPUs | Large deep learning |

See [AWS documentation](https://aws.amazon.com/sagemaker/pricing/) for full list and pricing.

## Distributed Training

For distributed training across multiple instances:

```bash
easy_sm train \
  -n distributed-job \
  -e ml.m5.xlarge \
  -c 4 \
  -i s3://bucket/data \
  -o s3://bucket/output
```

Your training code needs to handle distribution. SageMaker provides:

- `SM_HOSTS`: List of all hosts
- `SM_CURRENT_HOST`: Current host name
- `SM_NUM_GPUS`: Number of GPUs available

Example:

```python
import os
import json

def train(input_data_path, model_save_path):
    # Get distributed training info
    hosts = json.loads(os.environ.get('SM_HOSTS', '[]'))
    current_host = os.environ.get('SM_CURRENT_HOST', '')
    num_gpus = int(os.environ.get('SM_NUM_GPUS', 0))

    if len(hosts) > 1:
        # Distributed training logic
        train_distributed(hosts, current_host)
    else:
        # Single-instance training
        train_single()
```

## Monitoring Training

After submitting a training job, monitor it in:

1. **AWS Console**: SageMaker → Training → Training jobs
2. **CloudWatch Logs**: `/aws/sagemaker/TrainingJobs`
3. **CLI**:
   ```bash
   aws sagemaker describe-training-job --training-job-name my-job
   ```

## Troubleshooting

### "SAGEMAKER_ROLE environment variable not set"

**Problem**: IAM role not provided.

**Solution**: Set the environment variable:
```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
```

Or pass it explicitly:
```bash
easy_sm train -r arn:aws:iam::123456789012:role/SageMakerRole ...
```

### Training job fails immediately

**Problem**: Docker image not found in ECR.

**Solution**: Push the image first:
```bash
easy_sm push
```

### "Access Denied" errors

**Problem**: IAM role lacks S3 or ECR permissions.

**Solution**: Add required permissions to the SageMaker execution role:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket/*",
                "arn:aws:s3:::my-bucket"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:BatchCheckLayerAvailability"
            ],
            "Resource": "*"
        }
    ]
}
```

### Out of memory errors

**Problem**: Model too large for instance memory.

**Solution**: Use a larger instance type:
```bash
# Upgrade from ml.m5.large to ml.m5.xlarge
easy_sm train -e ml.m5.xlarge ...
```

### Training takes too long

**Problem**: Slow training on small instances.

**Solution**:
1. Use compute-optimized or GPU instances
2. Use distributed training with `-c` flag
3. Optimize your training code

---

## upload-data

Upload local data to S3 for training.

### Synopsis

```bash
easy_sm upload-data --input-dir PATH --target-dir S3_PATH [OPTIONS]
```

### Description

Uploads a local directory to S3 for use as training input data.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--input-dir` | `-i` | path | **Yes** | - | Local directory containing data |
| `--target-dir` | `-t` | string | **Yes** | - | S3 location (e.g., `s3://bucket/data`) |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | IAM role ARN |
| `--app-name` | `-a` | string | No | Auto-detected | App name |

### Examples

```bash
# Upload training data
easy_sm upload-data \
  -i ./local-data \
  -t s3://my-bucket/training-data

# Output: s3://my-bucket/training-data
```

### Output

The command outputs the S3 path where data was uploaded:

```
s3://my-bucket/training-data
```

---

## list-training-jobs

List recent SageMaker training jobs.

### Synopsis

```bash
easy_sm list-training-jobs [OPTIONS]
```

### Description

Lists recent training jobs with their status and creation time. Supports pipe-friendly output for automation.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--max-results` | `-m` | integer | No | `5` | Maximum number of jobs to return |
| `--names-only` | `-n` | boolean | No | `false` | Output only job names (one per line) |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | IAM role ARN |
| `--app-name` | `-a` | string | No | Auto-detected | App name |

### Examples

#### List recent training jobs

```bash
easy_sm list-training-jobs
```

Output:
```
my-training-job-1  Completed  2024-01-15 10:23:45+00:00
my-training-job-2  InProgress  2024-01-15 11:30:12+00:00
my-training-job-3  Failed  2024-01-14 09:15:33+00:00
```

#### List more jobs

```bash
easy_sm list-training-jobs -m 20
```

#### Names only (pipe-friendly)

```bash
easy_sm list-training-jobs -n
```

Output:
```
my-training-job-1
my-training-job-2
my-training-job-3
```

#### Get latest job name

```bash
easy_sm list-training-jobs -n -m 1
```

Output:
```
my-training-job-1
```

### Pipe-Friendly Usage

```bash
# Get latest completed job
easy_sm list-training-jobs -m 10 | grep Completed | head -1

# Get all production jobs
easy_sm list-training-jobs -n -m 20 | grep "prod-"

# Get latest model and deploy
MODEL=$(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
easy_sm deploy -n my-endpoint -e ml.m5.large -m $MODEL
```

---

## get-model-artifacts

Get S3 model path from a training job.

### Synopsis

```bash
easy_sm get-model-artifacts --training-job-name JOB_NAME [OPTIONS]
```

### Description

Retrieves the S3 location of the model artifacts (`model.tar.gz`) for a completed training job. Essential for deployment workflows.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--training-job-name` | `-j` | string | **Yes** | - | Training job name |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | IAM role ARN |
| `--app-name` | `-a` | string | No | Auto-detected | App name |

### Examples

#### Get model path

```bash
easy_sm get-model-artifacts -j my-training-job-123
```

Output:
```
s3://my-bucket/models/my-training-job-123/output/model.tar.gz
```

#### Use in deployment pipeline

```bash
# Get latest job
JOB=$(easy_sm list-training-jobs -n -m 1)

# Get its model
MODEL=$(easy_sm get-model-artifacts -j $JOB)

# Deploy
easy_sm deploy -n my-endpoint -e ml.m5.large -m $MODEL
```

#### One-liner deployment

```bash
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

### Troubleshooting

**Problem**: "Training job not found"

**Solution**: Verify the job name:
```bash
easy_sm list-training-jobs -n
```

**Problem**: Job not completed yet

**Solution**: Wait for training to complete:
```bash
aws sagemaker describe-training-job --training-job-name my-job \
  --query 'TrainingJobStatus'
```

---

## Complete Training Workflow

### End-to-End Example

```bash
# 1. Set up environment
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# 2. Build and push Docker image
easy_sm -t v1.0.0 build
easy_sm -t v1.0.0 push

# 3. Upload training data
easy_sm upload-data \
  -i ./training-data \
  -t s3://my-bucket/data

# 4. Train model
easy_sm -t v1.0.0 train \
  -n my-training-job \
  -e ml.m5.large \
  -i s3://my-bucket/data \
  -o s3://my-bucket/models

# 5. List training jobs
easy_sm list-training-jobs

# 6. Get model artifacts
MODEL=$(easy_sm get-model-artifacts -j my-training-job)

# 7. Deploy
easy_sm -t v1.0.0 deploy \
  -n my-endpoint \
  -e ml.m5.large \
  -m $MODEL
```

### Automated Pipeline

```bash
#!/bin/bash
set -e

export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
VERSION="v$(date +%Y%m%d-%H%M%S)"

# Build and push
easy_sm -t $VERSION build
easy_sm -t $VERSION push

# Train
easy_sm -t $VERSION train \
  -n training-$VERSION \
  -e ml.m5.xlarge \
  -i s3://my-bucket/data \
  -o s3://my-bucket/models

# Get model and deploy
MODEL=$(easy_sm get-model-artifacts -j training-$VERSION)
easy_sm -t $VERSION deploy \
  -n production-endpoint \
  -e ml.m5.large \
  -m $MODEL

echo "Deployed $VERSION to production-endpoint"
```

## Related Commands

- [`build`](build.md) - Build Docker image
- [`push`](push.md) - Push image to ECR
- [`upload-data`](#upload-data) - Upload training data to S3
- [`deploy`](deploy.md) - Deploy trained models
- [`local train`](local.md#train) - Test training locally first

## See Also

- [Training Guide](../user-guide/cloud-deployment.md)
- [SageMaker Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)
- [Instance Types and Pricing](https://aws.amazon.com/sagemaker/pricing/)
