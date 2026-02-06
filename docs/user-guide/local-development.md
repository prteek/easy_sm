# Local Development Guide

This guide covers local development workflows using easy_sm, which allows you to train, process, and deploy models locally in Docker containers that mimic AWS SageMaker environments.

## Overview

Local development with easy_sm provides a fast iteration cycle before deploying to the cloud. All local operations run in Docker containers that replicate the SageMaker runtime environment, ensuring consistency between local testing and cloud deployment.

## Prerequisites

- Docker installed and running
- easy_sm project initialized (`easy_sm init`)
- Docker image built (`easy_sm build`)

## Local Commands

All local operations are under the `local` sub-command:

```bash
easy_sm local train      # Train models locally
easy_sm local deploy     # Start local inference server
easy_sm local process    # Run processing jobs
easy_sm local stop       # Stop local deployment server
```

## Local Training

### Overview

The `local train` command runs your training code in a Docker container using test data from your project's `local_test` directory.

### Basic Usage

```bash
# Auto-detects app name from *.json config file
easy_sm local train

# Or specify app name explicitly
easy_sm local train -a my-app
```

### Training Code Structure

Your training code should be in `app-name/easy_sm_base/training/training.py`:

```python
import pandas as pd
import joblib
import os

def train(input_data_path, model_save_path, hyperparams_path=None):
    """
    Training function called by easy_sm.

    Args:
        input_data_path: Path to input data directory
        model_save_path: Path where trained model should be saved
        hyperparams_path: Optional path to hyperparameters file
    """
    # Load training data
    data = pd.read_csv(os.path.join(input_data_path, 'data.csv'))

    # Train your model
    model = train_your_model(data)

    # Save trained model
    joblib.dump(model, os.path.join(model_save_path, 'model.mdl'))
```

### Test Data Setup

Place sample training data in:
```
app-name/easy_sm_base/local_test/test_dir/input/data/training/
```

For example:
```
app-name/easy_sm_base/local_test/test_dir/input/data/training/data.csv
```

### Container Behavior

When you run `easy_sm local train`, the following happens:

1. Docker container is created from your built image
2. Training code is mounted into the container
3. Test data from `local_test/test_dir/input/data/training/` is mounted to `/opt/ml/input/data/training/`
4. Training script executes with SageMaker-compatible paths
5. Model artifacts are saved to `local_test/test_dir/model/`
6. Container is automatically removed after training completes

### Docker Volume Mounts

```
Host Path                                           → Container Path
────────────────────────────────────────────────────────────────────
app-name/easy_sm_base/training/                     → /opt/ml/code/
app-name/easy_sm_base/local_test/test_dir/input/    → /opt/ml/input/
app-name/easy_sm_base/local_test/test_dir/model/    → /opt/ml/model/
app-name/easy_sm_base/local_test/test_dir/output/   → /opt/ml/output/
```

### Expected Output

After successful training, you'll find:
- Model artifacts in `app-name/easy_sm_base/local_test/test_dir/model/`
- Training logs in the console
- Any output files in `app-name/easy_sm_base/local_test/test_dir/output/`

## Local Deployment

### Overview

The `local deploy` command starts a local inference server on port 8080 using your trained model.

### Basic Usage

```bash
# Start local server
easy_sm local deploy

# Server starts on http://localhost:8080
```

The server runs in the background and serves predictions using your model from the `local_test/test_dir/model/` directory.

### Serving Code Structure

Your serving code should be in `app-name/easy_sm_base/prediction/serve`:

```python
import joblib
import os
import numpy as np

def model_fn(model_dir):
    """
    Load the model from the model directory.

    Args:
        model_dir: Path to the directory containing model artifacts

    Returns:
        Loaded model object
    """
    return joblib.load(os.path.join(model_dir, 'model.mdl'))

def predict_fn(input_data, model):
    """
    Make predictions using the loaded model.

    Args:
        input_data: Input data for prediction
        model: Loaded model from model_fn

    Returns:
        Model predictions
    """
    return model.predict(input_data)

def input_fn(request_body, content_type):
    """
    Optional: Parse input data from request.

    Args:
        request_body: Raw request body
        content_type: Content type of the request

    Returns:
        Parsed input data
    """
    if content_type == 'text/csv':
        return np.fromstring(request_body, sep=',').reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def output_fn(prediction, accept_type):
    """
    Optional: Format prediction output.

    Args:
        prediction: Model prediction
        accept_type: Requested output format

    Returns:
        Formatted prediction
    """
    return str(prediction)
```

### Testing the Endpoint

Once the server is running, test it with curl:

```bash
# Send CSV data
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0,3.0,4.0,5.0'

# Send JSON data (if your input_fn supports it)
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

### Container Behavior

When you run `easy_sm local deploy`:

1. Docker container starts from your built image
2. Container runs in detached mode (background)
3. Port 8080 is exposed to localhost
4. Model artifacts from `local_test/test_dir/model/` are mounted to `/opt/ml/model/`
5. Prediction code is mounted into the container
6. Server starts and listens for requests

### Stopping the Server

```bash
# Stop the local deployment server
easy_sm local stop
```

This stops and removes the Docker container running the inference server.

## Local Processing

### Overview

The `local process` command runs data processing jobs locally in Docker containers.

### Basic Usage

```bash
# Run processing script
easy_sm local process -f script.py

# Or specify app name explicitly
easy_sm local process -f script.py -a my-app
```

### Processing Script Structure

Your processing script (`script.py`) should be in `app-name/easy_sm_base/processing/`:

```python
import pandas as pd
import os

def process():
    """
    Processing function that transforms data.
    """
    # Read input data
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # Load and process data
    df = pd.read_csv(os.path.join(input_path, 'data.csv'))

    # Apply transformations
    df_processed = transform_data(df)

    # Save processed data
    df_processed.to_csv(os.path.join(output_path, 'processed.csv'), index=False)

if __name__ == '__main__':
    process()
```

### Test Data Setup

Place sample processing data in:
```
app-name/easy_sm_base/local_test/test_dir/input/data/processing/
```

### Container Behavior

When you run `easy_sm local process`:

1. Docker container is created from your built image
2. Processing script is mounted into the container
3. Input data from `local_test/test_dir/input/data/processing/` is mounted to `/opt/ml/processing/input/`
4. Processing script executes
5. Output is saved to `local_test/test_dir/output/processing/`
6. Container is automatically removed after processing completes

## Workflow Examples

### Complete Local Development Workflow

```bash
# 1. Initialize project
easy_sm init

# 2. Add training code
# Edit app-name/easy_sm_base/training/training.py

# 3. Add test data
# Copy data to app-name/easy_sm_base/local_test/test_dir/input/data/training/

# 4. Build Docker image
easy_sm build

# 5. Train locally
easy_sm local train

# 6. Add serving code
# Edit app-name/easy_sm_base/prediction/serve

# 7. Deploy locally
easy_sm local deploy

# 8. Test endpoint
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0,3.0'

# 9. Stop local server
easy_sm local stop
```

### Iterative Development

```bash
# Make code changes
vim app-name/easy_sm_base/training/training.py

# Rebuild image
easy_sm build

# Test changes
easy_sm local train

# Verify output
ls -la app-name/easy_sm_base/local_test/test_dir/model/
```

### Processing Workflow

```bash
# Add processing script
# Edit app-name/easy_sm_base/processing/preprocess.py

# Add test data
# Copy data to app-name/easy_sm_base/local_test/test_dir/input/data/processing/

# Build image
easy_sm build

# Run processing
easy_sm local process -f preprocess.py

# Check output
ls -la app-name/easy_sm_base/local_test/test_dir/output/processing/
```

## Docker Container Details

### Image Structure

The Docker image built by `easy_sm build` includes:
- Base Python runtime (version from config)
- Dependencies from requirements.txt
- SageMaker training toolkit
- Entry point scripts for training/serving/processing

### Container Lifecycle

**Training/Processing:**
- Container is created
- Mounts are attached
- Command executes
- Container is automatically removed (`--rm` flag)

**Deployment:**
- Container is created
- Runs in detached mode (`-d` flag)
- Port 8080 exposed
- Container persists until `easy_sm local stop`

### Environment Variables

The following SageMaker-compatible environment variables are available in containers:

```bash
SM_MODEL_DIR=/opt/ml/model
SM_INPUT_DIR=/opt/ml/input
SM_OUTPUT_DIR=/opt/ml/output
SM_CHANNEL_TRAINING=/opt/ml/input/data/training
```

## Debugging Tips

### View Training Logs

All training logs are printed to stdout. To save them:

```bash
easy_sm local train 2>&1 | tee training.log
```

### Inspect Docker Container

While the deployment server is running:

```bash
# List running containers
docker ps

# View logs
docker logs <container-id>

# Execute commands inside container
docker exec -it <container-id> /bin/bash
```

### Check Model Artifacts

After training:

```bash
# List model files
ls -la app-name/easy_sm_base/local_test/test_dir/model/

# Inspect model
python -c "import joblib; m = joblib.load('app-name/easy_sm_base/local_test/test_dir/model/model.mdl'); print(m)"
```

### Test Server Manually

Debug endpoint issues:

```bash
# Check if server is responding
curl http://localhost:8080/ping

# Test with verbose output
curl -v -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0,3.0'
```

### Common Issues

**Issue: Container fails to start**
```bash
# Check if Docker is running
docker ps

# View Docker logs
docker logs <container-id>

# Rebuild image
easy_sm build
```

**Issue: Port 8080 already in use**
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process or stop existing deployment
easy_sm local stop
```

**Issue: Model not found**
```bash
# Ensure training completed successfully
ls -la app-name/easy_sm_base/local_test/test_dir/model/

# Re-run training if needed
easy_sm local train
```

**Issue: Code changes not reflected**
```bash
# Rebuild Docker image after code changes
easy_sm build

# Then re-run command
easy_sm local train
# or
easy_sm local deploy
```

### Debugging Inside Container

For complex issues, run container interactively:

```bash
# Start container with bash
docker run -it --rm \
  -v "$(pwd)/app-name/easy_sm_base:/opt/ml" \
  app-name:latest /bin/bash

# Inside container, manually run commands
cd /opt/ml/code
python training.py
```

## Performance Considerations

### Resource Allocation

Docker resource limits can be configured if needed:

```bash
# Example: Run with memory limits
docker run --memory="4g" --cpus="2" ...
```

For local development, easy_sm uses Docker's default resource allocation.

### Data Volume

For large datasets:
- Use a subset of data for local testing
- Keep full datasets in S3 for cloud training
- Consider using symbolic links to avoid data duplication

## Next Steps

After successful local development:

1. **Push to ECR**: `easy_sm push`
2. **Train on SageMaker**: See [Cloud Deployment Guide](cloud-deployment.md)
3. **Deploy to SageMaker**: See [Cloud Deployment Guide](cloud-deployment.md)

For AWS setup instructions, see [AWS Setup Guide](aws-setup.md).
