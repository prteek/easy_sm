# Cloud Deployment Guide

This guide covers deploying machine learning models to AWS SageMaker using easy_sm, including training jobs, model deployment, batch predictions, and processing jobs.

## Overview

Easy_sm provides commands for deploying to AWS SageMaker with minimal configuration. All commands auto-detect your app configuration and IAM role from environment variables.

## Prerequisites

- AWS CLI configured with credentials
- SageMaker execution role (see [AWS Setup Guide](aws-setup.md))
- Docker image pushed to ECR (`easy_sm push`)
- `SAGEMAKER_ROLE` environment variable set

## Environment Setup

Before using cloud commands, set your SageMaker IAM role:

```bash
# Set once per session
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# Or add to ~/.bashrc for persistence
echo 'export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole' >> ~/.bashrc
```

**Note**: You can override auto-detection with `-r/--iam-role-arn` flag on any command.

## Cloud Training

### Overview

The `train` command launches SageMaker training jobs using your Docker image from ECR.

### Basic Usage

```bash
# Train on SageMaker
easy_sm train -n job-name -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output
```

### Parameters

| Parameter | Flag | Required | Description |
|-----------|------|----------|-------------|
| Job Name | `-n, --base-job-name` | Yes | Prefix for training job name |
| Instance Type | `-e, --ec2-type` | Yes | EC2 instance type (e.g., ml.m5.large) |
| Input Path | `-i, --input-s3-dir` | Yes | S3 path to training data |
| Output Path | `-o, --output-s3-dir` | Yes | S3 path for model artifacts |
| Instance Count | `-c, --instance-count` | No | Number of instances (default: 1) |
| App Name | `-a, --app-name` | No | Override auto-detected app name |
| IAM Role | `-r, --iam-role-arn` | No | Override SAGEMAKER_ROLE env var |

### Multi-Instance Training

```bash
# Train on multiple instances for distributed training
easy_sm train -n my-training-job -e ml.m5.xlarge \
  -i s3://my-bucket/training-data \
  -o s3://my-bucket/model-output \
  -c 2
```

### Instance Type Selection

**CPU Instances** (general purpose):
- `ml.m5.large` - 2 vCPU, 8 GB RAM - Good for small models
- `ml.m5.xlarge` - 4 vCPU, 16 GB RAM - Medium workloads
- `ml.m5.2xlarge` - 8 vCPU, 32 GB RAM - Larger datasets
- `ml.m5.4xlarge` - 16 vCPU, 64 GB RAM - Heavy compute

**GPU Instances** (deep learning):
- `ml.p3.2xlarge` - 1x V100 GPU, 8 vCPU, 61 GB RAM
- `ml.p3.8xlarge` - 4x V100 GPU, 32 vCPU, 244 GB RAM
- `ml.g4dn.xlarge` - 1x T4 GPU, 4 vCPU, 16 GB RAM - Cost-effective

**Selection Guide**:
- Small tabular models → `ml.m5.large`
- Medium ML models → `ml.m5.xlarge`
- Large models/datasets → `ml.m5.2xlarge` or higher
- Deep learning (CNNs, NLP) → `ml.p3.2xlarge` or `ml.g4dn.xlarge`

### Training Output

The command outputs the S3 path to the trained model:

```bash
easy_sm train -n my-job -e ml.m5.large -i s3://bucket/in -o s3://bucket/out
# Output: s3://bucket/out/my-job/output/model.tar.gz
```

This output is designed to be piped to deployment commands (see [Piped Workflows](piped-workflows.md)).

### Training Job Management

```bash
# List recent training jobs
easy_sm list-training-jobs -m 10

# List job names only (pipe-friendly)
easy_sm list-training-jobs -n -m 5

# Get model artifacts from completed job
easy_sm get-model-artifacts -j my-training-job
```

## Model Deployment

Easy_sm supports two deployment types:
1. **Provisioned Endpoints** - Dedicated instances, predictable performance
2. **Serverless Endpoints** - Auto-scaling, pay-per-invocation

### Provisioned Endpoints

#### Overview

Provisioned endpoints run on dedicated EC2 instances with predictable performance and pricing.

#### Basic Usage

```bash
easy_sm deploy -n endpoint-name -e ml.m5.large \
  -m s3://bucket/model.tar.gz
```

#### Parameters

| Parameter | Flag | Required | Description |
|-----------|------|----------|-------------|
| Endpoint Name | `-n, --endpoint-name` | Yes | Unique name for endpoint |
| Instance Type | `-e, --instance-type` | Yes | EC2 instance type |
| Model Path | `-m, --s3-model-location` | Yes | S3 path to model.tar.gz |
| Instance Count | `-c, --instance-count` | No | Number of instances (default: 1) |
| App Name | `-a, --app-name` | No | Override auto-detected app name |
| IAM Role | `-r, --iam-role-arn` | No | Override SAGEMAKER_ROLE env var |

#### Example with Multiple Instances

```bash
# Deploy with 3 instances for high availability
easy_sm deploy -n prod-endpoint -e ml.m5.xlarge \
  -m s3://bucket/models/model.tar.gz \
  -c 3
```

#### Instance Type Selection

**CPU Instances** (cost-effective for most models):
- `ml.t2.medium` - 2 vCPU, 4 GB RAM - Minimal/testing
- `ml.m5.large` - 2 vCPU, 8 GB RAM - Small models
- `ml.m5.xlarge` - 4 vCPU, 16 GB RAM - Medium models
- `ml.c5.xlarge` - 4 vCPU, 8 GB RAM - Compute-optimized

**GPU Instances** (deep learning inference):
- `ml.g4dn.xlarge` - 1x T4 GPU, 4 vCPU, 16 GB RAM
- `ml.p3.2xlarge` - 1x V100 GPU, 8 vCPU, 61 GB RAM

**Selection Guide**:
- Development/testing → `ml.t2.medium`
- Production (small models) → `ml.m5.large`
- Production (medium models) → `ml.m5.xlarge`
- High throughput → `ml.c5.xlarge` or multiple instances
- Deep learning inference → `ml.g4dn.xlarge`

#### When to Use Provisioned Endpoints

- **Predictable traffic patterns** - Steady request rate
- **Low latency requirements** - <100ms response time
- **High throughput** - Thousands of requests per minute
- **Long-running inference** - Complex models with longer execution time

### Serverless Endpoints

#### Overview

Serverless endpoints auto-scale based on traffic and charge per invocation. Ideal for variable workloads.

#### Basic Usage

```bash
easy_sm deploy-serverless -n endpoint-name -s 2048 \
  -m s3://bucket/model.tar.gz
```

#### Parameters

| Parameter | Flag | Required | Description |
|-----------|------|----------|-------------|
| Endpoint Name | `-n, --endpoint-name` | Yes | Unique name for endpoint |
| Memory Size | `-s, --memory-size-in-mb` | Yes | Memory in MB (1024, 2048, 3072, 4096, 5120, 6144) |
| Model Path | `-m, --s3-model-location` | Yes | S3 path to model.tar.gz |
| Max Concurrency | `-mc, --max-concurrency` | No | Max concurrent invocations (default: 5) |
| App Name | `-a, --app-name` | No | Override auto-detected app name |
| IAM Role | `-r, --iam-role-arn` | No | Override SAGEMAKER_ROLE env var |

#### Memory Size Selection

Available memory sizes: 1024, 2048, 3072, 4096, 5120, 6144 MB

**Selection Guide**:
- Small models (<100MB) → 2048 MB
- Medium models (100-500MB) → 4096 MB
- Large models (>500MB) → 6144 MB

#### Example with Concurrency Control

```bash
# Deploy with higher concurrency for variable traffic
easy_sm deploy-serverless -n api-endpoint -s 4096 \
  -m s3://bucket/models/model.tar.gz \
  -c 100
```

#### When to Use Serverless Endpoints

- **Variable traffic** - Sporadic or unpredictable requests
- **Low request volume** - <1000 requests per day
- **Cost optimization** - Pay only for invocations
- **Development/testing** - No idle costs
- **Cold start tolerance** - Can accept 1-3 second startup delay

#### Serverless Limitations

- **Cold start latency** - First request after idle period takes 1-3 seconds
- **Max invocation time** - 60 seconds per request
- **Memory limits** - Max 6144 MB
- **No GPU support** - CPU inference only

### Provisioned vs Serverless Comparison

| Factor | Provisioned | Serverless |
|--------|-------------|------------|
| **Cost** | Fixed hourly rate | Pay per invocation |
| **Latency** | Consistent (<100ms) | Variable (cold starts) |
| **Scaling** | Manual (set instances) | Automatic |
| **Idle Cost** | Continuous | None |
| **Best For** | Steady traffic | Variable traffic |
| **GPU Support** | Yes | No |
| **Max Memory** | Instance-dependent | 6144 MB |

### Deployment Output

Both commands output the endpoint name:

```bash
easy_sm deploy -n my-endpoint -e ml.m5.large -m s3://bucket/model.tar.gz
# Output: my-endpoint
```

## Endpoint Management

### List Endpoints

```bash
# List all endpoints
easy_sm list-endpoints

# Output format:
# EndpointName: my-endpoint
# Status: InService
# CreationTime: 2025-01-15 10:30:00
# ───────────────────────────────────
```

### Delete Endpoint

```bash
# Delete endpoint (keeps endpoint configuration)
easy_sm delete-endpoint -n endpoint-name

# Delete endpoint and configuration
easy_sm delete-endpoint -n endpoint-name --delete-config
```

### Making Predictions

Once deployed, invoke endpoints using AWS SDK or CLI:

**Python (boto3)**:
```python
import boto3

client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='text/csv',
    Body='1.0,2.0,3.0,4.0,5.0'
)

prediction = response['Body'].read()
print(prediction)
```

**AWS CLI**:
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name my-endpoint \
  --content-type text/csv \
  --body '1.0,2.0,3.0,4.0,5.0' \
  output.txt
```

## Batch Transform

### Overview

Batch transform runs batch predictions on large datasets without deploying an endpoint.

### Basic Usage

```bash
easy_sm batch-transform -e ml.m5.large --num-instances 1 \
  -m s3://bucket/model.tar.gz \
  -i s3://bucket/input-data \
  -o s3://bucket/predictions
```

### Parameters

| Parameter | Flag | Required | Description |
|-----------|------|----------|-------------|
| Instance Type | `-e, --ec2-type` | Yes | EC2 instance type |
| Instance Count | `--num-instances` | Yes | Number of instances |
| Model Path | `-m, --s3-model-location` | Yes | S3 path to model.tar.gz |
| Input Path | `-i, --s3-input-location` | Yes | S3 path to input data |
| Output Path | `-o, --s3-output-location` | Yes | S3 path for predictions |
| Wait | `-w, --wait` | No | Wait until job completes (default: false) |
| Job Name | `-n, --job-name` | No | Custom job name (auto-generated if not provided) |
| App Name | `-a, --app-name` | No | Override auto-detected app name |
| IAM Role | `-r, --iam-role-arn` | No | Override SAGEMAKER_ROLE env var |

### Example with Multiple Instances

```bash
# Process large dataset with parallel instances
easy_sm batch-transform -e ml.m5.xlarge --num-instances 5 \
  -m s3://bucket/models/model.tar.gz \
  -i s3://bucket/batch-data \
  -o s3://bucket/batch-predictions
```

### When to Use Batch Transform

- **Large datasets** - Millions of predictions
- **Periodic predictions** - Daily/weekly batch jobs
- **No real-time requirements** - Can wait minutes/hours
- **Cost optimization** - No idle endpoint costs
- **High throughput** - Parallel processing across instances

## Processing Jobs

### Overview

Processing jobs run data transformation and feature engineering on SageMaker.

### Basic Usage

```bash
easy_sm process -f script.py -e ml.m5.large -n job-name
```

### Parameters

| Parameter | Flag | Required | Description |
|-----------|------|----------|-------------|
| Script File | `-f, --processing-file` | Yes | Python script in processing/ directory |
| Instance Type | `-e, --ec2-type` | Yes | EC2 instance type |
| Job Name | `-n, --job-name` | Yes | Unique name for processing job |
| Input Path | `-i, --s3-input-location` | No | S3 path to input data |
| Output Path | `-o, --s3-output-location` | No | S3 path for output |
| App Name | `-a, --app-name` | No | Override auto-detected app name |
| IAM Role | `-r, --iam-role-arn` | No | Override SAGEMAKER_ROLE env var |

### Processing Script Structure

Your processing script should be in `app-name/easy_sm_base/processing/`:

```python
import pandas as pd
import os

def process():
    """
    Processing function for data transformation.
    """
    # Input/output paths in SageMaker container
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # Load data
    df = pd.read_csv(os.path.join(input_path, 'data.csv'))

    # Transform data
    df_processed = transform_features(df)

    # Save output
    df_processed.to_csv(
        os.path.join(output_path, 'processed.csv'),
        index=False
    )

if __name__ == '__main__':
    process()
```

### Example with Input/Output

```bash
# Run processing with S3 input/output
easy_sm process -f preprocess.py -e ml.m5.large -n preprocessing-job \
  -i s3://bucket/raw-data \
  -o s3://bucket/processed-data
```

### When to Use Processing Jobs

- **Data preprocessing** - Feature engineering before training
- **ETL workflows** - Extract, transform, load operations
- **Data validation** - Quality checks and profiling
- **Post-processing** - Transform model outputs

## Data Upload

### Upload Data to S3

```bash
# Upload local directory to S3
easy_sm upload-data -i ./local-data -t s3://bucket/training-data
```

This uploads files to S3 for use in training or processing jobs.

## Complete Cloud Workflow

### End-to-End Example

```bash
# 1. Set environment
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# 2. Build and push Docker image
easy_sm build
easy_sm push

# 3. Upload training data
easy_sm upload-data -i ./data -t s3://my-bucket/training-data

# 4. Train model
easy_sm train -n training-job-001 -e ml.m5.xlarge \
  -i s3://my-bucket/training-data \
  -o s3://my-bucket/models

# 5. Get model path
MODEL_PATH=$(easy_sm get-model-artifacts -j training-job-001)

# 6. Deploy to endpoint
easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL_PATH

# 7. Test endpoint
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name prod-endpoint \
  --content-type text/csv \
  --body '1.0,2.0,3.0' \
  output.txt
```

## Monitoring and Troubleshooting

### CloudWatch Logs

Training and endpoint logs are in CloudWatch:

**Training Jobs:**
```
/aws/sagemaker/TrainingJobs/<job-name>
```

**Endpoints:**
```
/aws/sagemaker/Endpoints/<endpoint-name>
```

View logs via AWS Console or CLI:

```bash
# Get training job logs
aws logs tail /aws/sagemaker/TrainingJobs/my-training-job --follow

# Get endpoint logs
aws logs tail /aws/sagemaker/Endpoints/my-endpoint --follow
```

### Common Issues

**Issue: Training job fails immediately**

Check CloudWatch logs for errors. Common causes:
- Missing dependencies in requirements.txt
- Incorrect S3 paths
- IAM role lacks S3 permissions
- Code errors in training script

```bash
# View training job logs
aws logs tail /aws/sagemaker/TrainingJobs/my-job --follow
```

**Issue: Endpoint creation fails**

- Model artifacts missing or corrupted
- Serving code has errors
- Insufficient IAM permissions
- Instance type not available in region

```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name my-endpoint

# View endpoint logs
aws logs tail /aws/sagemaker/Endpoints/my-endpoint --follow
```

**Issue: Endpoint returns errors**

- Input format doesn't match expected format
- Model file not found in container
- Serving code exception

```bash
# Test endpoint with verbose output
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name my-endpoint \
  --content-type text/csv \
  --body '1.0,2.0,3.0' \
  --debug \
  output.txt
```

**Issue: High latency on serverless endpoint**

- Cold start delay (normal for first request after idle)
- Increase memory size for faster initialization
- Consider provisioned endpoint for consistent latency

**Issue: "AlgorithmError" during training**

- Check training code for exceptions
- Verify data format matches expected input
- Check CloudWatch logs for stack traces

### Cost Monitoring

**View SageMaker costs:**
- AWS Console → Cost Explorer → Filter by Service → SageMaker
- Set up cost alerts in CloudWatch

**Cost optimization tips:**
1. **Serverless for low traffic** - No idle costs
2. **Right-size instances** - Don't over-provision
3. **Delete unused endpoints** - Provisioned endpoints charge 24/7
4. **Use batch transform** - More cost-effective than endpoints for batch workloads
5. **Multi-instance training** - Faster training reduces billable time

### Performance Optimization

**Training:**
- Use larger instances for faster training
- Use distributed training (`-c` flag) for large datasets
- Use GPU instances (ml.p3.x) for deep learning

**Endpoints:**
- Add more instances for higher throughput
- Use auto-scaling for variable traffic
- Choose CPU vs GPU based on model type

**Batch Transform:**
- Increase instance count for parallel processing
- Use larger instances for faster per-record inference

## Next Steps

- **Piped Workflows**: See [Piped Workflows Guide](piped-workflows.md) for command composition
- **AWS Setup**: See [AWS Setup Guide](aws-setup.md) for IAM and S3 configuration
- **Local Testing**: See [Local Development Guide](local-development.md) for testing before cloud deployment
