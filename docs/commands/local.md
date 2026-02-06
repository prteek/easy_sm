# local

Local operations for testing ML workflows in Docker containers.

## Synopsis

```bash
easy_sm [--docker-tag TAG] local SUBCOMMAND [OPTIONS]
```

## Description

The `local` command group provides subcommands for testing your ML workflows locally using Docker containers that mimic the SageMaker environment. This enables rapid iteration without the cost and latency of cloud deployments.

All local commands use the Docker image built with [`easy_sm build`](build.md).

## Subcommands

- [`train`](#train) - Train ML models locally
- [`deploy`](#deploy) - Deploy model locally as HTTP endpoint
- [`process`](#process) - Run processing jobs locally
- [`stop`](#stop) - Stop local deployment

---

## train

Train ML models locally in a Docker container.

### Synopsis

```bash
easy_sm [--docker-tag TAG] local train [--app-name APP_NAME]
```

### Description

Runs your training code locally in a Docker container using test data from `local_test/test_dir/`. This simulates the SageMaker training environment without requiring cloud resources.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

### Examples

#### Basic local training

```bash
easy_sm local train
```

#### Train with specific app and tag

```bash
easy_sm -t v1.0.0 local train -a my-ml-app
```

### Output

```
[Training container output...]
Local training completed
```

Trained models are saved to:
```
{app_name}/easy_sm_base/local_test/test_dir/model/
```

### Training Data Location

Place your test data in:
```
{app_name}/easy_sm_base/local_test/test_dir/input/data/training/
```

Example structure:
```
local_test/test_dir/
├── input/
│   └── data/
│       └── training/
│           ├── train.csv
│           └── data.parquet
├── model/                    # Output: trained models saved here
└── output/                   # Output: training metrics/logs
```

### Training Code Entry Point

Your training code at `training/training.py` should implement a `train()` function:

```python
import pandas as pd
import joblib
import os

def train(input_data_path, model_save_path):
    """
    Train model locally.

    Args:
        input_data_path: Path to training data
        model_save_path: Path to save trained model
    """
    # Load data
    data = pd.read_csv(os.path.join(input_data_path, 'train.csv'))

    # Train model
    model = train_your_model(data)

    # Save model
    joblib.dump(model, os.path.join(model_save_path, 'model.mdl'))
```

### Troubleshooting

**Problem**: "Not a valid easy_sm directory"

**Solution**: Ensure `local_test/test_dir/` exists in your project:
```bash
ls {app_name}/easy_sm_base/local_test/test_dir/
```

**Problem**: "No such file or directory" for training data

**Solution**: Add test data to the expected location:
```bash
mkdir -p {app_name}/easy_sm_base/local_test/test_dir/input/data/training/
cp your_data.csv {app_name}/easy_sm_base/local_test/test_dir/input/data/training/
```

---

## deploy

Deploy a trained model locally as an HTTP endpoint.

### Synopsis

```bash
easy_sm [--docker-tag TAG] local deploy [OPTIONS]
```

### Description

Starts a local HTTP server on the specified port serving your trained model. The endpoint mimics SageMaker's inference API and can be tested with HTTP requests.

The server runs in the foreground. Use [`local stop`](#stop) to terminate it.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--port` | `-p` | integer | No | `8080` | Port to run the service on |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

### Examples

#### Deploy on default port (8080)

```bash
easy_sm local deploy
```

Output:
```
Starting local deployment at localhost:8080
[Server startup logs...]
```

#### Deploy on custom port

```bash
easy_sm local deploy -p 9000
```

#### Deploy specific app with tag

```bash
easy_sm -t v1.0.0 local deploy -a my-ml-app -p 8080
```

### Testing the Endpoint

Once deployed, test with curl:

```bash
# CSV input
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0,3.0,4.0'

# JSON input
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

Or with Python:

```python
import requests
import json

# Make prediction
url = "http://localhost:8080/invocations"
data = {"features": [1.0, 2.0, 3.0, 4.0]}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### Model Location

The endpoint serves the model from:
```
{app_name}/easy_sm_base/local_test/test_dir/model/
```

This should contain the model file(s) saved during training.

### Serving Code Entry Point

Your serving code at `prediction/serve` should implement these functions:

```python
import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    """
    Load model from directory.

    Args:
        model_dir: Directory containing model files

    Returns:
        Loaded model object
    """
    return joblib.load(os.path.join(model_dir, 'model.mdl'))

def input_fn(request_body, request_content_type):
    """
    Parse input data.

    Args:
        request_body: Raw request body
        request_content_type: Content type (e.g., 'text/csv', 'application/json')

    Returns:
        Parsed input data
    """
    if request_content_type == 'text/csv':
        return np.array([float(x) for x in request_body.split(',')]).reshape(1, -1)
    elif request_content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['features']).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make prediction.

    Args:
        input_data: Parsed input from input_fn
        model: Model from model_fn

    Returns:
        Model predictions
    """
    return model.predict(input_data)

def output_fn(prediction, accept):
    """
    Format prediction output.

    Args:
        prediction: Predictions from predict_fn
        accept: Requested response type

    Returns:
        Formatted response
    """
    if accept == 'application/json':
        return json.dumps({"predictions": prediction.tolist()})
    elif accept == 'text/csv':
        return ','.join(map(str, prediction.tolist()))
    else:
        return str(prediction)
```

### Troubleshooting

**Problem**: Port already in use

**Solution**: Either stop the existing service or use a different port:
```bash
# Stop existing deployment
easy_sm local stop -p 8080

# Or use different port
easy_sm local deploy -p 9000
```

**Problem**: "Model not found" error

**Solution**: Train the model first:
```bash
easy_sm local train
# Then deploy
easy_sm local deploy
```

**Problem**: Connection refused when testing

**Solution**: Ensure the deployment is running and check the port:
```bash
# Check if container is running
docker ps | grep my-ml-app

# Check logs
docker logs <container_id>
```

---

## process

Run Python processing jobs locally.

### Synopsis

```bash
easy_sm [--docker-tag TAG] local process --file FILE [OPTIONS]
```

### Description

Executes a Python file as a processing job in a Docker container. This is useful for data preprocessing, feature engineering, or post-processing tasks.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--file` | `-f` | string | **Yes** | - | Python file name to run (relative to `processing/` directory) |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

### Examples

#### Run a processing script

```bash
easy_sm local process -f preprocess.py
```

#### Process with specific app and tag

```bash
easy_sm -t v1.0.0 local process -f feature_engineering.py -a my-ml-app
```

### Output

```
[Processing container output...]
Local processing completed
```

### Processing Script Location

Place your processing scripts in:
```
{app_name}/easy_sm_base/processing/
```

Example:
```
processing/
├── preprocess.py
├── feature_engineering.py
└── postprocess.py
```

### Processing Script Example

```python
# processing/preprocess.py
import pandas as pd
import os

def process():
    """Process data locally."""
    # Input data
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # Read data
    df = pd.read_csv(os.path.join(input_path, 'raw_data.csv'))

    # Process
    df_processed = preprocess_data(df)

    # Save
    df_processed.to_csv(
        os.path.join(output_path, 'processed_data.csv'),
        index=False
    )
    print("Processing completed")

if __name__ == '__main__':
    process()
```

### Data Paths

Processing jobs have access to:

- **Input data**: `/opt/ml/processing/input`
- **Output data**: `/opt/ml/processing/output`

Locally, these map to:
```
local_test/test_dir/
├── input/          # Maps to /opt/ml/processing/input
└── output/         # Maps to /opt/ml/processing/output
```

### Troubleshooting

**Problem**: "Processing file not found"

**Solution**: Ensure the file exists in the processing directory:
```bash
ls {app_name}/easy_sm_base/processing/
# File must be there
```

**Problem**: "Not a valid easy_sm directory"

**Solution**: Check that `local_test/test_dir/` exists:
```bash
ls {app_name}/easy_sm_base/local_test/test_dir/
```

---

## stop

Stop a local deployment.

### Synopsis

```bash
easy_sm [--docker-tag TAG] local stop [OPTIONS]
```

### Description

Stops a running local deployment by terminating the Docker container serving the model endpoint.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--port` | `-p` | integer | No | `8080` | Port the service is running on |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

### Examples

#### Stop deployment on default port

```bash
easy_sm local stop
```

Output:
```
Local deployment stopped
```

#### Stop deployment on custom port

```bash
easy_sm local stop -p 9000
```

#### Stop specific app deployment

```bash
easy_sm local stop -a my-ml-app -p 8080
```

### Troubleshooting

**Problem**: "Container not found"

**Solution**: The deployment may have already stopped. Verify with:
```bash
docker ps | grep my-ml-app
```

**Problem**: Port mismatch

**Solution**: Specify the correct port used when deploying:
```bash
# If deployed with: easy_sm local deploy -p 9000
# Stop with: easy_sm local stop -p 9000
```

---

## Common Prerequisites

All local commands require:

- Docker installed and running
- Docker image built with `easy_sm build`
- Valid easy_sm project structure
- Configuration file in current directory

## Complete Local Workflow

```bash
# 1. Initialize project
easy_sm init

# 2. Add training code and data
# Edit: {app_name}/easy_sm_base/training/training.py
# Add data to: {app_name}/easy_sm_base/local_test/test_dir/input/data/training/

# 3. Build Docker image
easy_sm build

# 4. Train locally
easy_sm local train

# 5. Deploy locally
easy_sm local deploy

# 6. Test endpoint (in another terminal)
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0,3.0'

# 7. Stop deployment
easy_sm local stop

# 8. Optional: Run processing job
easy_sm local process -f preprocess.py
```

## SageMaker Environment Simulation

Local commands simulate the SageMaker container environment:

| SageMaker Path | Local Equivalent |
|----------------|------------------|
| `/opt/ml/input/data/training/` | `local_test/test_dir/input/data/training/` |
| `/opt/ml/model/` | `local_test/test_dir/model/` |
| `/opt/ml/output/` | `local_test/test_dir/output/` |
| `/opt/ml/processing/input/` | `local_test/test_dir/input/` |
| `/opt/ml/processing/output/` | `local_test/test_dir/output/` |

## Related Commands

- [`build`](build.md) - Build Docker image for local testing
- [`train`](train.md) - Train on SageMaker after local validation
- [`deploy`](deploy.md) - Deploy to SageMaker after local testing
- [`process`](process.md) - Run processing jobs on SageMaker

## See Also

- [Local Development Guide](../user-guide/local-development.md)
- [Training Guide](../user-guide/cloud-deployment.md)
- [Inference Guide](../user-guide/cloud-deployment.md)
- [Processing Guide](../user-guide/cloud-deployment.md)
