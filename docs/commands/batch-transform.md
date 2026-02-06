# batch-transform

Run batch predictions on AWS SageMaker.

## Synopsis

```bash
easy_sm [--docker-tag TAG] batch-transform --s3-model-location S3_PATH \
  --s3-input-location S3_PATH --s3-output-location S3_PATH \
  --num-instances COUNT --ec2-type TYPE [OPTIONS]
```

## Description

The `batch-transform` command runs batch inference on SageMaker, processing large datasets without deploying a persistent endpoint. It's cost-effective for:

- Periodic batch predictions
- Large-scale inference jobs
- One-time predictions on datasets
- Offline scoring

Unlike endpoints, batch transform jobs:
- Process data in S3 and write results back to S3
- Run once and terminate (no ongoing costs)
- Handle large files automatically
- Don't require endpoint management

## Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--s3-model-location` | `-m` | string | **Yes** | - | S3 location to model tar.gz |
| `--s3-input-location` | `-i` | string | **Yes** | - | S3 location of input data files |
| `--s3-output-location` | `-o` | string | **Yes** | - | S3 location to save predictions |
| `--num-instances` | - | integer | **Yes** | - | Number of EC2 instances |
| `--ec2-type` | `-e` | string | **Yes** | - | EC2 instance type (e.g., `ml.m5.large`) |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | AWS IAM role ARN |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--wait` | `-w` | boolean | No | `false` | Wait until job completes |
| `--job-name` | `-n` | string | No | Auto-generated | Custom job name |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

## Examples

### Basic batch transform

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

easy_sm batch-transform \
  -e ml.m5.large \
  --num-instances 1 \
  -m s3://my-bucket/models/model.tar.gz \
  -i s3://my-bucket/input-data \
  -o s3://my-bucket/predictions
```

### Large-scale batch job

Process large datasets with multiple instances:

```bash
easy_sm batch-transform \
  -e ml.m5.xlarge \
  --num-instances 5 \
  -m s3://bucket/model.tar.gz \
  -i s3://bucket/large-dataset \
  -o s3://bucket/predictions
```

### Wait for completion

Block until the job finishes:

```bash
easy_sm batch-transform \
  -e ml.m5.large \
  --num-instances 1 \
  -m s3://bucket/model.tar.gz \
  -i s3://bucket/data \
  -o s3://bucket/output \
  --wait
```

Output:
```
Completed
```

Or if failed:
```
Failed
```
(Exit code 1)

### Custom job name

```bash
easy_sm batch-transform \
  -n my-batch-job-2024-01 \
  -e ml.m5.large \
  --num-instances 2 \
  -m s3://bucket/model.tar.gz \
  -i s3://bucket/data \
  -o s3://bucket/output
```

### With specific Docker tag

```bash
easy_sm -t v1.0.0 batch-transform \
  -e ml.m5.large \
  --num-instances 1 \
  -m s3://bucket/model.tar.gz \
  -i s3://bucket/data \
  -o s3://bucket/output
```

## Output Format

Without `--wait`:
- No output (job submitted asynchronously)

With `--wait`:
- Outputs final job status: `Completed`, `Failed`, or `Stopped`
- Exit code 0 for success, 1 for failure

## Prerequisites

- Trained model in S3 (from [`train`](train.md) command)
- Docker image pushed to ECR
- Input data files in S3
- IAM role with SageMaker and S3 permissions
- Inference code in `prediction/serve`

## Input Data Format

### File Structure

Place input files in S3:

```
s3://my-bucket/input-data/
├── batch1.csv
├── batch2.csv
└── batch3.csv
```

SageMaker processes each file and creates corresponding output files:

```
s3://my-bucket/predictions/
├── batch1.csv.out
├── batch2.csv.out
└── batch3.csv.out
```

### Input File Formats

Batch transform supports various formats:

**CSV:**
```csv
1.0,2.0,3.0,4.0
5.0,6.0,7.0,8.0
9.0,10.0,11.0,12.0
```

**JSON Lines (JSONL):**
```json
{"features": [1.0, 2.0, 3.0, 4.0]}
{"features": [5.0, 6.0, 7.0, 8.0]}
{"features": [9.0, 10.0, 11.0, 12.0]}
```

**Binary formats** (if your serving code supports it)

## Serving Code Requirements

Your `prediction/serve` code must implement the inference functions:

```python
import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    """Load model."""
    return joblib.load(os.path.join(model_dir, 'model.mdl'))

def input_fn(request_body, content_type):
    """
    Parse input for batch transform.

    Args:
        request_body: Raw input line/record
        content_type: Input format (e.g., 'text/csv')

    Returns:
        Parsed input ready for prediction
    """
    if content_type == 'text/csv':
        # Parse CSV line
        values = [float(x) for x in request_body.strip().split(',')]
        return np.array(values).reshape(1, -1)
    elif content_type == 'application/json':
        # Parse JSON
        data = json.loads(request_body)
        return np.array(data['features']).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make prediction."""
    return model.predict(input_data)

def output_fn(prediction, accept):
    """
    Format output for batch transform.

    Args:
        prediction: Model prediction
        accept: Output format

    Returns:
        Formatted prediction string
    """
    if accept == 'application/json':
        return json.dumps({"prediction": prediction.tolist()})
    else:
        return str(prediction[0])
```

## How Batch Transform Works

1. **Split input**: SageMaker splits input data across instances
2. **Process in parallel**: Each instance processes its chunk
3. **Generate predictions**: Calls your inference code for each record
4. **Write output**: Saves predictions to S3 output location
5. **Terminate**: Instances shut down when complete

```
S3 Input Files
    ↓
Split across instances
    ↓
Instance 1: batch1.csv → predictions
Instance 2: batch2.csv → predictions
Instance 3: batch3.csv → predictions
    ↓
S3 Output Files
```

## Output Files

For each input file, SageMaker creates an output file:

| Input | Output |
|-------|--------|
| `data.csv` | `data.csv.out` |
| `input.json` | `input.json.out` |
| `batch_001.txt` | `batch_001.txt.out` |

Output format matches your `output_fn` implementation.

## Performance Optimization

### Multiple Instances

For large datasets, use multiple instances:

```bash
# Single instance: processes files sequentially
easy_sm batch-transform --num-instances 1 ...

# Multiple instances: parallel processing
easy_sm batch-transform --num-instances 10 ...
```

### Instance Types

Choose based on workload:

| Instance Type | vCPUs | Memory | Best For |
|--------------|-------|---------|----------|
| `ml.m5.large` | 2 | 8 GB | Small jobs |
| `ml.m5.xlarge` | 4 | 16 GB | Medium jobs |
| `ml.c5.2xlarge` | 8 | 16 GB | CPU-intensive |
| `ml.p3.2xlarge` | 8 | 61 GB + GPU | Deep learning |

### File Size Considerations

- **Small files** (<10 MB): Use fewer instances, increase parallelism per instance
- **Large files** (>100 MB): Use more instances for parallel processing
- **Many small files**: SageMaker distributes across instances automatically

## Monitoring

### AWS Console

Monitor in SageMaker Console:
- **SageMaker → Inference → Batch transform jobs**
- View progress, logs, and metrics

### AWS CLI

```bash
# Describe job
aws sagemaker describe-transform-job \
  --transform-job-name my-batch-job

# List jobs
aws sagemaker list-transform-jobs \
  --sort-by CreationTime \
  --sort-order Descending

# View CloudWatch logs
aws logs tail /aws/sagemaker/TransformJobs \
  --follow
```

### Check Output

After completion, verify output in S3:

```bash
aws s3 ls s3://my-bucket/predictions/

# Download predictions
aws s3 cp s3://my-bucket/predictions/ ./predictions/ --recursive
```

## Cost Comparison

### Batch Transform vs Endpoint

**Scenario**: 10,000 predictions, once per day

**Batch Transform** (ml.m5.large, 10 minutes):
- Cost: 10 min × $0.119/hour / 60 = $0.0198 per day
- Monthly: ~$0.60

**Provisioned Endpoint** (ml.m5.large, 24/7):
- Cost: 24 hours × $0.119/hour = $2.86 per day
- Monthly: ~$85

**Batch transform is 140x cheaper for periodic jobs!**

### When to Use Batch Transform

✅ **Use batch transform for:**
- Periodic predictions (daily, weekly, monthly)
- Large datasets processed offline
- One-time scoring jobs
- Cost-sensitive workloads
- Non-latency-critical applications

❌ **Use endpoints for:**
- Real-time predictions
- Interactive applications
- Low-latency requirements
- Continuous traffic

## Use Cases

### 1. Daily Customer Scoring

```bash
#!/bin/bash
# daily_scoring.sh

export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# Export customer data daily
# (assume data exported to S3)

# Run batch predictions
easy_sm batch-transform \
  -e ml.m5.large \
  --num-instances 3 \
  -m s3://models/churn-model/model.tar.gz \
  -i s3://data/customers/$(date +%Y-%m-%d) \
  -o s3://predictions/churn/$(date +%Y-%m-%d) \
  --wait

if [ $? -eq 0 ]; then
  echo "Predictions complete, loading to database..."
  # Load predictions to database
else
  echo "Prediction job failed!"
  exit 1
fi
```

### 2. Large-Scale Offline Scoring

```bash
# Score 1 million records
easy_sm batch-transform \
  -e ml.c5.2xlarge \
  --num-instances 20 \
  -m s3://bucket/model.tar.gz \
  -i s3://bucket/million-records \
  -o s3://bucket/scores \
  --wait
```

### 3. Monthly Financial Forecasts

```bash
# Run monthly
easy_sm batch-transform \
  -n forecast-$(date +%Y-%m) \
  -e ml.m5.xlarge \
  --num-instances 5 \
  -m s3://models/forecast/model.tar.gz \
  -i s3://data/historical/$(date +%Y-%m) \
  -o s3://forecasts/$(date +%Y-%m) \
  --wait
```

## Troubleshooting

### Job fails immediately

**Problem**: Job goes directly to "Failed" status.

**Solution**: Check CloudWatch logs:

```bash
aws logs tail /aws/sagemaker/TransformJobs --follow
```

Common issues:
- Model file not found in S3
- Docker image missing in ECR
- Serving code errors

### Input/output mismatch

**Problem**: Output format doesn't match expected.

**Solution**: Check your `output_fn` implementation:

```python
def output_fn(prediction, accept):
    # Return consistent format
    return json.dumps({"prediction": prediction.tolist()})
```

### Out of memory errors

**Problem**: Instance runs out of memory during inference.

**Solution**: Use larger instance type:

```bash
# Upgrade from ml.m5.large to ml.m5.xlarge
easy_sm batch-transform -e ml.m5.xlarge ...
```

### Job takes too long

**Problem**: Slow processing.

**Solution**: Increase parallelism:

```bash
# Add more instances
easy_sm batch-transform --num-instances 10 ...

# Or use faster instances
easy_sm batch-transform -e ml.c5.2xlarge ...
```

### Missing output files

**Problem**: Some input files don't have corresponding output files.

**Solution**: Check logs for errors on specific files. Ensure all input files are valid.

### "ModelError" in logs

**Problem**: Serving code throws exceptions.

**Solution**: Test locally first:

```bash
easy_sm local deploy
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  --data-binary @test_input.csv
```

## Complete Batch Transform Workflow

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# 1. Train model
MODEL=$(easy_sm train \
  -n training-job \
  -e ml.m5.large \
  -i s3://bucket/training-data \
  -o s3://bucket/models)

# 2. Prepare batch input data
easy_sm upload-data \
  -i ./batch-input \
  -t s3://bucket/batch-input

# 3. Run batch transform
easy_sm batch-transform \
  -e ml.m5.large \
  --num-instances 3 \
  -m $MODEL \
  -i s3://bucket/batch-input \
  -o s3://bucket/predictions \
  --wait

# 4. Download predictions
aws s3 cp s3://bucket/predictions/ ./predictions/ --recursive

# 5. Process predictions
python process_predictions.py ./predictions/
```

## Automated Batch Scoring Pipeline

```bash
#!/bin/bash
# batch_pipeline.sh
set -e

export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
DATE=$(date +%Y-%m-%d)

echo "Starting batch scoring pipeline for $DATE"

# 1. Export data (your data export logic)
echo "Exporting data..."
python export_data.py --date $DATE --output s3://bucket/daily-data/$DATE/

# 2. Run batch transform
echo "Running batch predictions..."
easy_sm batch-transform \
  -n batch-job-$DATE \
  -e ml.m5.xlarge \
  --num-instances 5 \
  -m s3://bucket/models/latest/model.tar.gz \
  -i s3://bucket/daily-data/$DATE \
  -o s3://bucket/predictions/$DATE \
  --wait

# 3. Validate predictions
echo "Validating predictions..."
python validate_predictions.py \
  --input s3://bucket/predictions/$DATE \
  --output s3://bucket/validated/$DATE

# 4. Load to database
echo "Loading predictions to database..."
python load_to_db.py s3://bucket/validated/$DATE

echo "Pipeline completed successfully!"
```

## Related Commands

- [`train`](train.md) - Train models for batch inference
- [`deploy`](deploy.md) - Alternative: deploy endpoint for real-time inference
- [`upload-data`](train.md#upload-data) - Upload input data to S3

## See Also

- [Batch Inference Guide](../user-guide/cloud-deployment.md)
- [SageMaker Batch Transform Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- [Cost Optimization Guide](../user-guide/cloud-deployment.md)
