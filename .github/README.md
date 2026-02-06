# easy_sm

A Python CLI tool that simplifies AWS SageMaker workflows by enabling rapid local prototyping with Docker before deploying to the cloud.

**Credits**: This project borrows heavily from [Sagify](https://github.com/Kenza-AI/sagify). Check it out especially if you want to work with LLMs on SageMaker.

> **Note**: This is an experimental package. APIs may evolve between releases.

## Features

- **Local Development**: Train, process, and deploy models locally in Docker containers that mimic SageMaker
- **Cloud Deployment**: Deploy trained models to AWS SageMaker with minimal configuration changes
- **Docker Integration**: Automatically build and manage Docker images
- **Endpoint Management**: Deploy and manage SageMaker endpoints (provisioned and serverless)
- **Job Monitoring**: List and filter training jobs

## Requirements

- Python >=3.13
- Docker (for local development)
- AWS CLI configured with credentials

## Installation

```bash
pip install easy-sm
```

**From source:**
```bash
git clone <repository-url>
cd easy_sm
pip install -e .
```

## Quick Start

### 1. Initialize Project

```bash
easy_sm init
```

Follow the prompts to configure:
- App name
- AWS profile and region
- Python version
- Requirements file location

### 2. Add Your Code

Copy your code to the appropriate folder:
- **Training**: `app-name/easy_sm_base/training/training.py`
- **Processing**: `app-name/easy_sm_base/processing/`
- **Serving**: `app-name/easy_sm_base/prediction/serve`

### 3. Build and Test Locally

```bash
# App name auto-detected from *.json file
easy_sm build
easy_sm local train
```

### 4. Deploy to SageMaker

```bash
# Set role once (or add to ~/.bashrc)
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# Push and train
easy_sm push
easy_sm train -n job-name -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output
```

## Commands

Cloud commands are at the top level for simplicity. Local operations are under `local` sub-command.

```text
Commands:
  init                  Initialize SageMaker template
  build                 Build Docker image
  push                  Push Docker image to AWS ECR
  update-scripts        Update shell scripts with latest secure versions

  # Cloud Operations (default)
  upload-data           Upload data to S3
  train                 Train models on SageMaker
  deploy                Deploy to provisioned endpoint
  deploy-serverless     Deploy to serverless endpoint
  batch-transform       Run batch predictions
  process               Run processing jobs
  list-endpoints        List all endpoints
  list-training-jobs    List recent training jobs
  get-model-artifacts   Get S3 path from training job
  delete-endpoint       Delete an endpoint

  # Local Operations
  local                 Local operations (train, deploy, process, stop)
```

### Context Auto-Detection

The CLI auto-detects context to minimize repetitive flags:

- **App name**: Auto-detected from `*.json` config file in current directory
- **IAM role**: Read from `SAGEMAKER_ROLE` environment variable
- **AWS profile/region**: From config file (uses boto3 defaults)

You can always override with `-a/--app-name` and `-r/--iam-role-arn` flags.

### Setup Environment

```bash
# Set IAM role once (add to ~/.bashrc for persistence)
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# Navigate to project directory (config auto-detected)
cd my-app/
```

### Local Commands

```bash
# Train locally
easy_sm local train

# Run processing job
easy_sm local process -f script.py

# Deploy locally (starts server on port 8080)
easy_sm local deploy

# Stop local deployment
easy_sm local stop
```

### Cloud Commands

```bash
# Upload data to S3
easy_sm upload-data -i ./data -t s3://bucket/data

# Train on SageMaker
easy_sm train -n job-name -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output

# Deploy to provisioned endpoint
easy_sm deploy -n endpoint-name -e ml.m5.large \
  -m s3://bucket/model.tar.gz

# Deploy to serverless endpoint
easy_sm deploy-serverless -n endpoint-name -s 2048 \
  -m s3://bucket/model.tar.gz

# Run batch transform
easy_sm batch-transform -e ml.m5.large --num-instances 1 \
  -m s3://bucket/model.tar.gz -i s3://bucket/input -o s3://bucket/output

# Run processing job
easy_sm process -f script.py -e ml.m5.large -n job-name

# List endpoints
easy_sm list-endpoints

# List training jobs (pipe-friendly with -n flag)
easy_sm list-training-jobs -m 10
easy_sm list-training-jobs -n -m 1  # Just names, one per line

# Get model artifacts from training job
easy_sm get-model-artifacts -j training-job-name

# Delete endpoint
easy_sm delete-endpoint -n endpoint-name [--delete-config]
```

### Update Scripts

If you have an existing project, update shell scripts with security fixes:

```bash
# Auto-detects app from *.json file
easy_sm update-scripts

# Or specify explicitly
easy_sm update-scripts -a app-name
```

## AWS Setup

### 1. AWS Profile

Configure in `~/.aws/config`:
```ini
[profile dev]
region = eu-west-1
output = json
```

With credentials in `~/.aws/credentials`:
```ini
[dev]
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
```

### 2. SageMaker Execution Role

Create an IAM role (e.g., `arn:aws:iam::123456789012:role/SageMakerExecutionRole`) with:
- SageMaker permissions
- S3 access for your buckets
- ECR access

Add trust relationship for your user:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::123456789012:user/your-user",
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

## Training Example

### 1. Training Code

Create `app-name/easy_sm_base/training/training.py`:

```python
import pandas as pd
import joblib
import os

def train(input_data_path, model_save_path, hyperparams_path=None):
    # Load data
    data = pd.read_csv(os.path.join(input_data_path, 'data.csv'))

    # Train model
    model = train_your_model(data)

    # Save model
    joblib.dump(model, os.path.join(model_save_path, 'model.mdl'))
```

### 2. Test Data

Place sample data at:
`app-name/easy_sm_base/local_test/test_dir/input/data/training/`

### 3. Train

```bash
# Build container
easy_sm build

# Test locally
easy_sm local train

# Push and train on SageMaker
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
easy_sm push
easy_sm train -n my-training-job -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output
```

## Deployment Example

### 1. Serving Code

Create `app-name/easy_sm_base/prediction/serve` with:

```python
import joblib
import os

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, 'model.mdl'))

def predict_fn(input_data, model):
    return model.predict(input_data)
```

### 2. Deploy

```bash
# Local testing
easy_sm build
easy_sm local train
easy_sm local deploy

# Test endpoint
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0,3.0'

# Cloud deployment
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
easy_sm push
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m s3://bucket/model.tar.gz

# Or serverless
easy_sm deploy-serverless -n my-endpoint -s 2048 \
  -m s3://bucket/model.tar.gz
```

## Configuration

Projects use a JSON config file (`app-name.json`):

```json
{
    "image_name": "my-app",
    "aws_profile": "dev",
    "aws_region": "eu-west-1",
    "python_version": "3.13",
    "easy_sm_module_dir": "my-app",
    "requirements_dir": "requirements.txt"
}
```

## Project Structure

```
my-project/
├── app-name.json                    # Configuration
├── requirements.txt                 # Dependencies
└── app-name/
    └── easy_sm_base/
        ├── Dockerfile               # Customize if needed
        ├── training/
        │   ├── train                # Entry point
        │   └── training.py          # Your training code
        ├── prediction/
        │   └── serve                # Your serving code
        ├── processing/              # Processing scripts
        └── local_test/
            └── test_dir/            # Local test data
```

## Tips

### Piped Workflows

The CLI is designed for Unix-style composition:

```bash
# Get latest training job, extract model, and deploy
JOB=$(easy_sm list-training-jobs -n -m 1)
MODEL=$(easy_sm get-model-artifacts -j $JOB)
easy_sm deploy -n my-endpoint -e ml.m5.large -m $MODEL
```

**One-liner deployment**:
```bash
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

**Filter and process training jobs**:
```bash
# Get all training jobs and filter completed ones
easy_sm list-training-jobs -m 20 | grep Completed

# Get model from specific job
easy_sm list-training-jobs -n -m 10 | grep "prod-" | head -1 | xargs -I {} easy_sm get-model-artifacts -j {}
```

### Other Tips

- **Save training output**: `easy_sm train ... | tee train_output.txt`
- **Custom Docker**: Modify `app-name/easy_sm_base/Dockerfile`
- **Docker tags**: Use `-t` flag: `easy_sm build -t v1.0`
- **Override auto-detection**: Use `-a app-name` or `-r $ROLE` flags when needed

## License

MIT License

## Author

Created by Prateek (prteek@icloud.com)
