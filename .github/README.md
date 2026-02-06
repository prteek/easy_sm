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
easy_sm build -a app-name
easy_sm local train -a app-name
```

### 4. Deploy to SageMaker

```bash
easy_sm push -a app-name
easy_sm cloud train -n job-name -r $SAGEMAKER_EXECUTION_ROLE -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output -a app-name
```

## Commands

```text
Commands:
  init            Initialize SageMaker template
  build           Build Docker image
  push            Push Docker image to AWS ECR
  update-scripts  Update shell scripts with latest secure versions
  local           Local operations (train, deploy, process, stop)
  cloud           Cloud operations (train, deploy, process, etc.)
```

### Local Commands

```bash
# Train locally
easy_sm local train -a app-name

# Run processing job
easy_sm local process -a app-name -f script.py

# Deploy locally (starts server on port 8080)
easy_sm local deploy -a app-name [-p 8080]

# Stop local deployment
easy_sm local stop -a app-name [-p 8080]
```

### Cloud Commands

```bash
# Upload data to S3
easy_sm cloud upload-data -a app-name -i ./data -t s3://bucket/data -r $ROLE

# Train on SageMaker
easy_sm cloud train -a app-name -n job-name -r $ROLE -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output

# Deploy to provisioned endpoint
easy_sm cloud deploy -a app-name -n endpoint-name -r $ROLE \
  -m s3://bucket/model.tar.gz -e ml.m5.large

# Deploy to serverless endpoint
easy_sm cloud deploy-serverless -a app-name -n endpoint-name -r $ROLE \
  -m s3://bucket/model.tar.gz -s 2048

# Run batch transform
easy_sm cloud batch-transform -a app-name -r $ROLE -e ml.m5.large \
  -m s3://bucket/model.tar.gz -i s3://bucket/input -o s3://bucket/output \
  --num-instances 1

# Run processing job
easy_sm cloud process -a app-name -f script.py -r $ROLE -e ml.m5.large \
  -n job-name

# List endpoints
easy_sm cloud list-endpoints -a app-name -r $ROLE

# List training jobs
easy_sm cloud list-training-jobs -a app-name -r $ROLE [-m 10]

# Delete endpoint
easy_sm cloud delete-endpoint -a app-name -n endpoint-name -r $ROLE [--delete-config]
```

### Update Scripts

If you have an existing project, update shell scripts with security fixes:

```bash
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
easy_sm build -a app-name

# Test locally
easy_sm local train -a app-name

# Push and train on SageMaker
easy_sm push -a app-name
easy_sm cloud train -a app-name -n my-training-job -r $ROLE \
  -e ml.m5.large -i s3://bucket/input -o s3://bucket/output
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
easy_sm build -a app-name
easy_sm local train -a app-name
easy_sm local deploy -a app-name

# Test endpoint
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0,3.0'

# Cloud deployment
easy_sm push -a app-name
easy_sm cloud deploy -a app-name -n my-endpoint -r $ROLE \
  -m s3://bucket/model.tar.gz -e ml.m5.large

# Or serverless
easy_sm cloud deploy-serverless -a app-name -n my-endpoint -r $ROLE \
  -m s3://bucket/model.tar.gz -s 2048
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

- **Save training output**: `easy_sm cloud train ... | tee train_output.txt`
- **Extract model location**: `grep -o -E "s3://[^ ]+" train_output.txt`
- **Custom Docker**: Modify `app-name/easy_sm_base/Dockerfile`
- **Docker tags**: Use `-t` flag: `easy_sm build -a app-name -t v1.0`

## License

MIT License

## Author

Created by Prateek (prteek@icloud.com)
