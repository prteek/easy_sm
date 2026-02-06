# process

Run processing jobs on AWS SageMaker.

## Synopsis

```bash
easy_sm [--docker-tag TAG] process --file FILE --base-job-name NAME --ec2-type TYPE [OPTIONS]
```

## Description

The `process` command submits a processing job to AWS SageMaker that executes a Python script for data preprocessing, feature engineering, model evaluation, or other data processing tasks.

Processing jobs run in the same Docker container as training but execute custom Python scripts instead of the training entry point.

## Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--file` | `-f` | string | **Yes** | - | Python file name to run (relative to `processing/` directory) |
| `--base-job-name` | `-n` | string | **Yes** | - | Prefix for the SageMaker processing job |
| `--ec2-type` | `-e` | string | **Yes** | - | EC2 instance type (e.g., `ml.m5.large`) |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | AWS IAM role ARN for SageMaker |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--instance-count` | `-c` | integer | No | `1` | Number of EC2 instances |
| `--s3-input-location` | `-i` | string | No | None | S3 location for input data |
| `--s3-output-location` | `-o` | string | No | None | S3 location to save output |
| `--input-sharded` | `-is` | boolean | No | `false` | Shard input data across instances |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

## Examples

### Basic processing job

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

easy_sm process \
  -f preprocess.py \
  -n data-preprocessing \
  -e ml.m5.large
```

Output:
```
data-preprocessing
```

### Processing with input and output

```bash
easy_sm process \
  -f feature_engineering.py \
  -n feature-job \
  -e ml.m5.large \
  -i s3://my-bucket/raw-data \
  -o s3://my-bucket/processed-data
```

### Distributed processing

Process data across multiple instances:

```bash
easy_sm process \
  -f process_large_dataset.py \
  -n distributed-processing \
  -e ml.m5.xlarge \
  -c 4 \
  -i s3://bucket/large-dataset \
  -o s3://bucket/output \
  --input-sharded
```

With `--input-sharded`, each instance receives a portion of the input data.

### Processing with specific Docker tag

```bash
easy_sm -t v1.0.0 process \
  -f validate_model.py \
  -n model-validation \
  -e ml.m5.large \
  -i s3://bucket/models \
  -o s3://bucket/validation-results
```

### Model evaluation job

```bash
easy_sm process \
  -f evaluate.py \
  -n model-evaluation \
  -e ml.m5.large \
  -i s3://bucket/test-data \
  -o s3://bucket/metrics
```

## Output Format

The command outputs the job name prefix:

```
data-preprocessing
```

This can be used in scripts:

```bash
JOB_NAME=$(easy_sm process -f preprocess.py -n preprocessing -e ml.m5.large)
echo "Started processing job: $JOB_NAME"
```

## Prerequisites

- Docker image pushed to ECR with [`easy_sm push`](push.md)
- Processing script in `{app_name}/easy_sm_base/processing/` directory
- IAM role with SageMaker permissions
- Input data in S3 (if using `-i` flag)

## Processing Script Requirements

Place your processing script in the `processing/` directory:

```
{app_name}/easy_sm_base/processing/
├── preprocess.py
├── feature_engineering.py
└── evaluate.py
```

### Script Template

```python
# processing/preprocess.py
import os
import pandas as pd
import json

def process():
    """
    Main processing function.
    """
    # Standard SageMaker paths
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    print(f"Reading data from {input_path}")

    # Read input data
    df = pd.read_csv(os.path.join(input_path, 'raw_data.csv'))

    print(f"Processing {len(df)} rows")

    # Process data
    df_processed = preprocess_data(df)

    # Save processed data
    output_file = os.path.join(output_path, 'processed_data.csv')
    df_processed.to_csv(output_file, index=False)

    print(f"Saved processed data to {output_file}")

    # Save metadata
    metadata = {
        'input_rows': len(df),
        'output_rows': len(df_processed),
        'features': list(df_processed.columns)
    }
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    print("Processing completed")

def preprocess_data(df):
    """Your preprocessing logic here."""
    # Clean data
    df = df.dropna()

    # Feature engineering
    df['new_feature'] = df['feature1'] * df['feature2']

    # Normalize
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df

if __name__ == '__main__':
    process()
```

## Container Paths

Processing jobs use standard SageMaker paths:

| Path | Purpose |
|------|---------|
| `/opt/ml/processing/input/` | Input data from S3 (`--s3-input-location`) |
| `/opt/ml/processing/output/` | Output data uploaded to S3 (`--s3-output-location`) |
| `/opt/ml/processing/code/` | Your processing script |

## Data Flow

```
S3 Input (--s3-input-location)
    ↓
/opt/ml/processing/input/
    ↓
Your processing script (-f file.py)
    ↓
/opt/ml/processing/output/
    ↓
S3 Output (--s3-output-location)
```

## Use Cases

### 1. Data Preprocessing

Clean and transform raw data before training:

```python
# processing/preprocess.py
import pandas as pd
import os

def process():
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # Read raw data
    df = pd.read_csv(os.path.join(input_path, 'raw_data.csv'))

    # Clean
    df = df.dropna()
    df = df[df['value'] > 0]

    # Transform
    df['log_value'] = np.log(df['value'])

    # Save
    df.to_csv(os.path.join(output_path, 'clean_data.csv'), index=False)
```

Run:
```bash
easy_sm process -f preprocess.py -n preprocess -e ml.m5.large \
  -i s3://bucket/raw -o s3://bucket/clean
```

### 2. Feature Engineering

Generate features from processed data:

```python
# processing/feature_engineering.py
import pandas as pd
import os

def process():
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    df = pd.read_csv(os.path.join(input_path, 'clean_data.csv'))

    # Create features
    df['feature_interaction'] = df['feature1'] * df['feature2']
    df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1)

    # Time-based features
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Save
    df.to_csv(os.path.join(output_path, 'features.csv'), index=False)
```

### 3. Model Evaluation

Evaluate trained models on test data:

```python
# processing/evaluate.py
import joblib
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def process():
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # Load model
    model = joblib.load(os.path.join(input_path, 'model/model.mdl'))

    # Load test data
    test_df = pd.read_csv(os.path.join(input_path, 'test/test.csv'))
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='weighted'
    )

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model Evaluation Results:")
    print(json.dumps(metrics, indent=2))
```

Run:
```bash
easy_sm process -f evaluate.py -n evaluate -e ml.m5.large \
  -i s3://bucket/test-data -o s3://bucket/metrics
```

### 4. Data Validation

Validate data quality before training:

```python
# processing/validate.py
import pandas as pd
import json
import os

def process():
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    df = pd.read_csv(os.path.join(input_path, 'data.csv'))

    # Validation checks
    validation_report = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_ranges': {}
    }

    # Check numeric ranges
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        validation_report['numeric_ranges'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean())
        }

    # Save report
    with open(os.path.join(output_path, 'validation_report.json'), 'w') as f:
        json.dump(validation_report, f, indent=2)

    # Fail if data quality issues
    if validation_report['duplicates'] > 100:
        raise ValueError(f"Too many duplicates: {validation_report['duplicates']}")
```

## Distributed Processing

For large datasets, use multiple instances with sharding:

```bash
easy_sm process \
  -f process_large_data.py \
  -n distributed-job \
  -e ml.m5.xlarge \
  -c 4 \
  --input-sharded \
  -i s3://bucket/large-dataset \
  -o s3://bucket/output
```

With `--input-sharded`:
- Input data is automatically split across instances
- Each instance processes its shard independently
- Outputs are combined in S3

### Distributed Processing Script

```python
# processing/process_large_data.py
import os
import json

def process():
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # Get current instance info
    resource_config = json.loads(
        os.environ.get('SM_RESOURCE_CONFIG', '{}')
    )
    current_host = resource_config.get('current_host', 'unknown')
    hosts = resource_config.get('hosts', [])

    print(f"Processing on {current_host} (total hosts: {len(hosts)})")

    # Process local shard
    files = os.listdir(input_path)
    for file in files:
        process_file(os.path.join(input_path, file))

    # Save output
    output_file = os.path.join(output_path, f'output_{current_host}.csv')
    save_results(output_file)
```

## Instance Types

Common instance types for processing:

| Instance Type | vCPUs | Memory | Use Case |
|--------------|-------|---------|----------|
| `ml.m5.large` | 2 | 8 GB | Small datasets |
| `ml.m5.xlarge` | 4 | 16 GB | Medium datasets |
| `ml.m5.4xlarge` | 16 | 64 GB | Large datasets |
| `ml.c5.2xlarge` | 8 | 16 GB | CPU-intensive processing |
| `ml.r5.2xlarge` | 8 | 64 GB | Memory-intensive processing |

## Monitoring

Monitor processing jobs in AWS Console:
- **SageMaker → Processing → Processing jobs**
- **CloudWatch Logs**: `/aws/sagemaker/ProcessingJobs`

Or via CLI:

```bash
# Describe job
aws sagemaker describe-processing-job \
  --processing-job-name my-processing-job

# View logs
aws logs tail /aws/sagemaker/ProcessingJobs \
  --log-stream-name my-processing-job/... \
  --follow
```

## Troubleshooting

### "Processing file not found"

**Problem**: Script doesn't exist in `processing/` directory.

**Solution**: Ensure the file exists:

```bash
ls {app_name}/easy_sm_base/processing/
# Should show: preprocess.py
```

### Script execution errors

**Problem**: Script fails during processing.

**Solution**: Test locally first:

```bash
# Test locally
easy_sm local process -f preprocess.py

# Check logs
docker logs $(docker ps -q --filter name=my-ml-app)
```

### "No space left on device"

**Problem**: Instance runs out of disk space.

**Solution**:
1. Use larger instance type with more storage
2. Process data in chunks
3. Clean up intermediate files in script

### Input/output path errors

**Problem**: Can't find input or output paths.

**Solution**: Use standard paths:

```python
# Always use these paths
input_path = '/opt/ml/processing/input'
output_path = '/opt/ml/processing/output'

# Never use
# input_path = './input'  # Wrong!
```

### Module import errors

**Problem**: Can't import required packages.

**Solution**: Add packages to `requirements.txt` and rebuild:

```bash
# Add to requirements.txt
echo "scikit-learn==1.3.0" >> requirements.txt

# Rebuild and push
easy_sm build
easy_sm push
```

## Complete Processing Workflow

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# 1. Upload raw data
easy_sm upload-data \
  -i ./raw-data \
  -t s3://my-bucket/raw

# 2. Preprocess data
easy_sm process \
  -f preprocess.py \
  -n preprocess \
  -e ml.m5.large \
  -i s3://my-bucket/raw \
  -o s3://my-bucket/clean

# 3. Feature engineering
easy_sm process \
  -f feature_engineering.py \
  -n features \
  -e ml.m5.large \
  -i s3://my-bucket/clean \
  -o s3://my-bucket/features

# 4. Train model
easy_sm train \
  -n training \
  -e ml.m5.large \
  -i s3://my-bucket/features \
  -o s3://my-bucket/models

# 5. Evaluate model
MODEL=$(easy_sm get-model-artifacts -j training)
easy_sm process \
  -f evaluate.py \
  -n evaluate \
  -e ml.m5.large \
  -i s3://my-bucket/test-data \
  -o s3://my-bucket/metrics
```

## Related Commands

- [`train`](train.md) - Train models after processing data
- [`local process`](local.md#process) - Test processing scripts locally
- [`upload-data`](train.md#upload-data) - Upload data to S3

## See Also

- [Processing Guide](../user-guide/cloud-deployment.md)
- [Data Preparation](../user-guide/cloud-deployment.md)
- [SageMaker Processing Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
