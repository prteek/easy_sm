# Training Example

Complete walkthrough of training a machine learning model with easy_sm, from initialization to cloud deployment.

## Overview

This example demonstrates:

1. Setting up a training project
2. Writing training code
3. Preparing test data
4. Testing locally with Docker
5. Training on AWS SageMaker

## Step 1: Initialize Project

Create a new easy_sm project:

```bash
easy_sm init
```

Enter these values at the prompts:

```
App name: my-ml-app
AWS profile: dev
AWS region: eu-west-1
Python version: 3.13
Requirements file: requirements.txt
```

This creates the project structure:

```
my-ml-app/
├── my-ml-app.json              # Configuration
├── requirements.txt            # Dependencies
└── my-ml-app/
    └── easy_sm_base/
        ├── Dockerfile
        ├── training/
        │   ├── train           # Entry point
        │   └── training.py     # Your code goes here
        ├── prediction/
        │   └── serve
        ├── processing/
        └── local_test/
            └── test_dir/       # Test data
```

## Step 2: Write Training Code

Edit `my-ml-app/easy_sm_base/training/training.py`:

```python
import pandas as pd
import joblib
import os

def train(input_data_path, model_save_path, hyperparams_path=None):
    """Train a simple linear regression model."""
    # Load training data
    data_file = os.path.join(input_data_path, 'data.csv')
    print(f"Loading data from: {data_file}")
    data = pd.read_csv(data_file)

    print(f"Training data shape: {data.shape}")

    # Simple model training (replace with your logic)
    from sklearn.linear_model import LinearRegression

    X = data[['feature1', 'feature2']]
    y = data['target']

    print("Training model...")
    model = LinearRegression()
    model.fit(X, y)

    # Calculate training accuracy
    score = model.score(X, y)
    print(f"Training R² score: {score:.4f}")

    # Save model
    model_file = os.path.join(model_save_path, 'model.mdl')
    print(f"Saving model to: {model_file}")
    joblib.dump(model, model_file)

    print("Training complete!")

if __name__ == '__main__':
    import sys

    # SageMaker passes paths as arguments
    train_data_path = sys.argv[1] if len(sys.argv) > 1 else '/opt/ml/input/data/training'
    model_save_path = sys.argv[2] if len(sys.argv) > 2 else '/opt/ml/model'

    train(train_data_path, model_save_path)
```

### Key Points

- **`train()` function**: Main training logic
- **Paths**: SageMaker provides standardized paths (`/opt/ml/input/data/training`, `/opt/ml/model`)
- **Logging**: Use `print()` for logs (captured by SageMaker CloudWatch)
- **Model format**: Save as `.mdl`, `.pkl`, or `.joblib`

## Step 3: Add Dependencies

Edit `requirements.txt`:

```txt
scikit-learn>=1.3.0
pandas>=2.0.0
joblib>=1.3.0
numpy>=1.24.0
```

## Step 4: Prepare Test Data

Create test data for local development:

```bash
mkdir -p my-ml-app/easy_sm_base/local_test/test_dir/input/data/training
```

Create `my-ml-app/easy_sm_base/local_test/test_dir/input/data/training/data.csv`:

```csv
feature1,feature2,target
1.0,2.0,3.0
2.0,3.0,5.0
3.0,4.0,7.0
4.0,5.0,9.0
5.0,6.0,11.0
```

This test data will be used for local training runs.

## Step 5: Build Docker Image

Navigate to project directory and build:

```bash
cd my-ml-app/
easy_sm build
```

Output:

```
Building Docker image: my-ml-app:latest
Step 1/8 : FROM python:3.13
Step 2/8 : COPY requirements.txt /opt/program/requirements.txt
...
Successfully built a1b2c3d4e5f6
Successfully tagged my-ml-app:latest
```

The Docker image includes:

- Python 3.13
- Your training code
- All dependencies from requirements.txt

## Step 6: Test Training Locally

Run training in Docker container:

```bash
easy_sm local train
```

Output:

```
Loading data from: /opt/ml/input/data/training/data.csv
Training data shape: (5, 3)
Training model...
Training R² score: 1.0000
Saving model to: /opt/ml/model/model.mdl
Training complete!
```

### Verify Model File

The model is saved to `local_test/test_dir/model/`:

```bash
ls -lh my-ml-app/easy_sm_base/local_test/test_dir/model/
```

Output:

```
model.mdl
```

## Step 7: Set Up AWS

Set your SageMaker IAM role:

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
```

!!! tip "Persist Environment Variable"
    Add to `~/.bashrc` for persistence:
    ```bash
    echo 'export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole' >> ~/.bashrc
    source ~/.bashrc
    ```

## Step 8: Upload Training Data to S3

Upload your training data to S3:

```bash
# Create S3 bucket (if not exists)
aws s3 mb s3://my-sagemaker-bucket

# Upload training data
aws s3 cp data/ s3://my-sagemaker-bucket/training-data/ --recursive
```

Or use the upload-data command:

```bash
easy_sm upload-data -i ./data -t s3://my-sagemaker-bucket/training-data
```

## Step 9: Push Docker Image to ECR

Push your Docker image to AWS Elastic Container Registry:

```bash
easy_sm push
```

Output:

```
Authenticating with ECR...
Creating ECR repository: my-ml-app
Pushing image to: 123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-ml-app:latest
The push refers to repository [123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-ml-app]
...
latest: digest: sha256:abc123... size: 2525
```

easy_sm automatically:

- Authenticates with ECR
- Creates repository if needed
- Tags and pushes the image

## Step 10: Train on SageMaker

Start a SageMaker training job:

```bash
easy_sm train -n my-training-job-001 -e ml.m5.large \
  -i s3://my-sagemaker-bucket/training-data \
  -o s3://my-sagemaker-bucket/models
```

Output:

```
s3://my-sagemaker-bucket/models/my-training-job-001/output/model.tar.gz
```

### Training Job Details

| Parameter | Value |
|-----------|-------|
| Job name | `my-training-job-001` |
| Instance type | `ml.m5.large` |
| Input data | `s3://my-sagemaker-bucket/training-data` |
| Output path | `s3://my-sagemaker-bucket/models` |

## Step 11: Monitor Training Job

Monitor the training job in the AWS Console:

1. Navigate to **SageMaker → Training jobs**
2. Find your job: `my-training-job-001`
3. Check **Status**: `InProgress` → `Completed`
4. View **CloudWatch logs** for training output

Or use the CLI:

```bash
# List recent training jobs
easy_sm list-training-jobs -m 5

# Check specific job
aws sagemaker describe-training-job --training-job-name my-training-job-001
```

## Step 12: Retrieve Model Artifacts

Get the S3 path of the trained model:

```bash
easy_sm get-model-artifacts -j my-training-job-001
```

Output:

```
s3://my-sagemaker-bucket/models/my-training-job-001/output/model.tar.gz
```

This path can be used for deployment.

## Advanced: Distributed Training

Use multiple instances for distributed training:

```bash
easy_sm train -n distributed-job -e ml.m5.xlarge \
  -c 3 \
  -i s3://my-sagemaker-bucket/training-data \
  -o s3://my-sagemaker-bucket/models
```

Your training code can access distributed training environment variables:

```python
import os
import json

def train(input_data_path, model_save_path):
    # Get distributed training info
    hosts = json.loads(os.environ.get('SM_HOSTS', '[]'))
    current_host = os.environ.get('SM_CURRENT_HOST', '')
    num_gpus = int(os.environ.get('SM_NUM_GPUS', 0))

    print(f"Running on {current_host}, total hosts: {len(hosts)}")

    if len(hosts) > 1:
        # Implement distributed training logic
        # Use frameworks like Horovod, PyTorch DDP, etc.
        pass
```

## Advanced: GPU Training

Use GPU instances for deep learning:

```bash
easy_sm train -n gpu-training-job -e ml.p3.2xlarge \
  -i s3://my-sagemaker-bucket/training-data \
  -o s3://my-sagemaker-bucket/models
```

Common GPU instance types:

- `ml.p3.2xlarge` - 1 NVIDIA V100 GPU
- `ml.p3.8xlarge` - 4 NVIDIA V100 GPUs
- `ml.g4dn.xlarge` - 1 NVIDIA T4 GPU (cost-effective)

## Troubleshooting

### Training Fails Locally

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**: Add missing dependency to requirements.txt, then rebuild:

```bash
echo "scikit-learn" >> requirements.txt
easy_sm build
```

### Training Fails on SageMaker

**Issue**: `AlgorithmError: ExecuteUserScriptError`

**Solution**: Check CloudWatch logs for the actual error. Common issues:

- Missing dependencies
- Incorrect file paths
- Data format mismatches

View logs:

```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow \
  --log-stream-name my-training-job-001/algo-1-1234567890
```

### Model Not Found

**Issue**: Model file not created

**Solution**: Ensure your code saves the model to the correct path:

```python
model_save_path = '/opt/ml/model'  # SageMaker standard path
joblib.dump(model, os.path.join(model_save_path, 'model.mdl'))
```

## Next Steps

- Learn how to [deploy the trained model](deployment-example.md)
- Explore [advanced workflows](advanced-workflows.md)
- Read about [piped workflows](../user-guide/piped-workflows.md)

## See Also

- [Training Command Reference](../commands/train.md)
- [Local Development Guide](../user-guide/local-development.md)
- [Cloud Deployment Guide](../user-guide/cloud-deployment.md)
