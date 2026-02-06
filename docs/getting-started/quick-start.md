# Quick Start

Get up and running with easy_sm in 5 minutes. This guide walks through initializing a project, adding code, testing locally, and deploying to SageMaker.

## Prerequisites

Before starting, ensure you have:

- [x] Installed easy_sm (`pip install easy-sm`)
- [x] Docker running locally
- [x] AWS CLI configured with credentials
- [x] SageMaker IAM role ARN

!!! tip "First Time Setup"
    If you haven't configured AWS or created a SageMaker role, see the [AWS Setup](../user-guide/aws-setup.md) guide first.

## Step 1: Initialize Project

Create a new easy_sm project:

```bash
easy_sm init
```

Follow the interactive prompts:

```
App name: my-ml-app
AWS profile: dev
AWS region: eu-west-1
Python version: 3.13
Requirements file: requirements.txt
```

This creates:

```
my-ml-app/
├── my-ml-app.json              # Configuration
├── requirements.txt            # Dependencies
└── my-ml-app/
    └── easy_sm_base/
        ├── Dockerfile
        ├── training/
        │   ├── train           # Entry point
        │   └── training.py     # Your training code goes here
        ├── prediction/
        │   └── serve           # Your serving code goes here
        ├── processing/         # Processing scripts
        └── local_test/
            └── test_dir/       # Test data
```

## Step 2: Add Your Code

### Training Code

Edit `my-ml-app/easy_sm_base/training/training.py`:

```python
import pandas as pd
import joblib
import os

def train(input_data_path, model_save_path, hyperparams_path=None):
    """Train your model."""
    # Load training data
    data = pd.read_csv(os.path.join(input_data_path, 'data.csv'))

    # Train model (replace with your logic)
    from sklearn.linear_model import LinearRegression
    X = data[['feature1', 'feature2']]
    y = data['target']
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    joblib.dump(model, os.path.join(model_save_path, 'model.mdl'))
    print("Model training complete!")
```

### Serving Code

Edit `my-ml-app/easy_sm_base/prediction/serve`:

```python
import joblib
import os
import numpy as np

def model_fn(model_dir):
    """Load the model."""
    return joblib.load(os.path.join(model_dir, 'model.mdl'))

def predict_fn(input_data, model):
    """Make predictions."""
    return model.predict(input_data)
```

### Test Data

Add sample training data to:
`my-ml-app/easy_sm_base/local_test/test_dir/input/data/training/data.csv`

```csv
feature1,feature2,target
1.0,2.0,3.0
2.0,3.0,5.0
3.0,4.0,7.0
```

## Step 3: Build and Test Locally

### Build Docker Image

```bash
cd my-ml-app/
easy_sm build
```

This builds a Docker image with your code and dependencies.

### Train Locally

Test training in a Docker container:

```bash
easy_sm local train
```

This runs your training code using the test data. Check output for any errors.

### Deploy Locally

Start a local inference endpoint:

```bash
easy_sm local deploy
```

This starts a Flask server on `http://localhost:8080`.

### Test Local Endpoint

In another terminal:

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0'
```

### Stop Local Endpoint

```bash
easy_sm local stop
```

## Step 4: Deploy to SageMaker

Once local testing works, deploy to AWS SageMaker.

### Set IAM Role

Set your SageMaker execution role:

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
```

!!! tip "Persist Environment Variable"
    Add this to `~/.bashrc` or `~/.zshrc` to persist across sessions:
    ```bash
    echo 'export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole' >> ~/.bashrc
    source ~/.bashrc
    ```

### Push Docker Image to ECR

```bash
easy_sm push
```

This pushes your Docker image to AWS Elastic Container Registry.

### Train on SageMaker

```bash
easy_sm train -n my-training-job -e ml.m5.large \
  -i s3://my-bucket/input -o s3://my-bucket/output
```

This starts a SageMaker training job. Monitor progress in the AWS Console.

### Deploy to Endpoint

Once training completes, deploy the model:

```bash
# Get model artifact from training job
MODEL=$(easy_sm get-model-artifacts -j my-training-job)

# Deploy to endpoint
easy_sm deploy -n my-endpoint -e ml.m5.large -m $MODEL
```

Or use a one-liner:

```bash
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j my-training-job)
```

### Test Cloud Endpoint

Test your deployed endpoint using the AWS SDK:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='text/csv',
    Body='1.0,2.0'
)

print(json.loads(response['Body'].read()))
```

## Next Steps

Now that you have a working project:

- Learn about [local development workflows](../user-guide/local-development.md)
- Explore [cloud deployment options](../user-guide/cloud-deployment.md)
- Master [piped workflows](../user-guide/piped-workflows.md) for efficiency
- Read the [command reference](../commands/index.md) for all options

## Common Issues

### Docker Build Fails

**Error**: `Cannot connect to Docker daemon`

**Solution**: Ensure Docker is running:
```bash
docker ps
```

### Training Fails Locally

**Error**: `ModuleNotFoundError`

**Solution**: Add missing dependencies to `requirements.txt`, then rebuild:
```bash
echo "scikit-learn" >> requirements.txt
easy_sm build
```

### SageMaker Training Fails

**Error**: `Could not assume role`

**Solution**: Verify your IAM role exists and has correct permissions:
```bash
aws iam get-role --role-name SageMakerRole
```

See [AWS Setup](../user-guide/aws-setup.md) for IAM configuration.

### Push to ECR Fails

**Error**: `no basic auth credentials`

**Solution**: Ensure AWS credentials are configured:
```bash
aws configure --profile dev
```

For more troubleshooting, see individual [command documentation](../commands/index.md).
