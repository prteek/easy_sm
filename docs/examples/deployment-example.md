# Deployment Example

Complete walkthrough of deploying a trained model with easy_sm, from local testing to production endpoints.

## Overview

This example demonstrates:

1. Writing serving code
2. Testing deployment locally
3. Deploying to provisioned endpoint
4. Deploying to serverless endpoint
5. Making predictions

## Prerequisites

- Completed the [training example](training-example.md) or have a trained model
- Model artifacts in S3 (e.g., `s3://bucket/models/job/output/model.tar.gz`)
- Docker running locally
- AWS credentials configured
- `SAGEMAKER_ROLE` environment variable set

## Step 1: Write Serving Code

Edit `my-ml-app/easy_sm_base/prediction/serve`:

```python
import joblib
import os
import numpy as np
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """
    Load the model from the model directory.
    Called once when the endpoint starts.
    """
    model_path = os.path.join(model_dir, 'model.mdl')
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type):
    """
    Parse input data.
    Supports CSV and JSON formats.
    """
    print(f"Received content_type: {content_type}")

    if content_type == 'text/csv':
        # Parse CSV input
        # Expected format: "1.0,2.0" or "feature1,feature2\\n1.0,2.0"
        df = pd.read_csv(StringIO(request_body), header=None)
        return df.values

    elif content_type == 'application/json':
        # Parse JSON input
        # Expected format: {"features": [[1.0, 2.0]]}
        import json
        data = json.loads(request_body)
        return np.array(data['features'])

    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Make predictions using the loaded model.
    """
    print(f"Predicting for input shape: {input_data.shape}")
    predictions = model.predict(input_data)
    return predictions

def output_fn(predictions, accept):
    """
    Format output data.
    Supports CSV and JSON formats.
    """
    print(f"Formatting output for accept: {accept}")

    if accept == 'text/csv':
        # Return CSV format
        return ','.join(map(str, predictions)), 'text/csv'

    elif accept == 'application/json':
        # Return JSON format
        import json
        return json.dumps({'predictions': predictions.tolist()}), 'application/json'

    else:
        # Default to JSON
        import json
        return json.dumps({'predictions': predictions.tolist()}), 'application/json'
```

### Key Components

| Function | Purpose |
|----------|---------|
| `model_fn(model_dir)` | Load model (called once at startup) |
| `input_fn(request_body, content_type)` | Parse input data |
| `predict_fn(input_data, model)` | Make predictions |
| `output_fn(predictions, accept)` | Format output |

## Step 2: Build Docker Image

Rebuild the image with serving code:

```bash
cd my-ml-app/
easy_sm build
```

## Step 3: Test Locally

### Start Local Server

Deploy the model locally:

```bash
easy_sm local deploy
```

Output:

```
Starting local deployment on port 8080...
Model loaded successfully
Serving at: http://localhost:8080
```

The server runs in a Docker container and listens on port 8080.

### Test with CSV Input

In another terminal, send a prediction request:

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: text/csv' \
  -d '1.0,2.0'
```

Response:

```
3.0
```

### Test with JSON Input

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"features": [[1.0, 2.0]]}'
```

Response:

```json
{"predictions": [3.0]}
```

### Test with Multiple Predictions

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}'
```

Response:

```json
{"predictions": [3.0, 7.0, 11.0]}
```

### Stop Local Server

```bash
easy_sm local stop
```

## Step 4: Deploy to Provisioned Endpoint

Deploy to AWS SageMaker provisioned endpoint.

### Set Environment

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
```

### Push Docker Image

```bash
easy_sm push
```

### Get Model Path

If you trained with easy_sm, get the model path:

```bash
MODEL=$(easy_sm get-model-artifacts -j my-training-job-001)
echo $MODEL
```

Output:

```
s3://my-sagemaker-bucket/models/my-training-job-001/output/model.tar.gz
```

### Deploy

```bash
easy_sm deploy -n my-endpoint -e ml.m5.large -m $MODEL
```

Output:

```
my-endpoint
```

### Deployment Details

| Parameter | Value |
|-----------|-------|
| Endpoint name | `my-endpoint` |
| Instance type | `ml.m5.large` |
| Instance count | 1 (default) |
| Model | S3 path from training |

### Monitor Deployment

Check deployment status:

```bash
aws sagemaker describe-endpoint --endpoint-name my-endpoint
```

Wait for status: `Creating` → `InService` (takes 5-10 minutes).

Or use:

```bash
easy_sm list-endpoints
```

## Step 5: Test Provisioned Endpoint

### Using AWS SDK (Python)

```python
import boto3
import json

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='eu-west-1')

# CSV input
response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='text/csv',
    Body='1.0,2.0'
)

result = response['Body'].read().decode()
print(f"Prediction: {result}")
# Output: Prediction: 3.0

# JSON input
response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='application/json',
    Accept='application/json',
    Body=json.dumps({'features': [[1.0, 2.0], [3.0, 4.0]]})
)

result = json.loads(response['Body'].read().decode())
print(f"Predictions: {result['predictions']}")
# Output: Predictions: [3.0, 7.0]
```

### Using AWS CLI

```bash
# Prepare input
echo '{"features": [[1.0, 2.0]]}' > input.json

# Invoke endpoint
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name my-endpoint \
  --content-type application/json \
  --body fileb://input.json \
  output.json

# View result
cat output.json
```

## Step 6: Deploy to Serverless Endpoint

For intermittent or unpredictable traffic, use serverless endpoints.

### Deploy Serverless

```bash
easy_sm deploy-serverless -n my-serverless-endpoint -s 2048 -m $MODEL
```

Output:

```
my-serverless-endpoint
```

### Serverless Configuration

| Parameter | Description | Value |
|-----------|-------------|-------|
| `-s, --memory-size` | Memory in MB (1024, 2048, 3072, 4096, 5120, 6144) | 2048 |
| `-c, --max-concurrency` | Max concurrent invocations (default: 20) | 20 |

### Test Serverless Endpoint

Testing is identical to provisioned endpoints:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='my-serverless-endpoint',
    ContentType='text/csv',
    Body='1.0,2.0'
)

result = response['Body'].read().decode()
print(f"Prediction: {result}")
```

!!! note "Cold Start Latency"
    First request to serverless endpoint may take 10-30 seconds (cold start). Subsequent requests are fast.

## Step 7: Scale Provisioned Endpoint

For production workloads, deploy with multiple instances:

```bash
easy_sm deploy -n prod-endpoint -e ml.m5.xlarge \
  --num-instances 3 \
  -m $MODEL
```

This deploys 3 instances with auto-scaling and load balancing.

## Provisioned vs Serverless Comparison

| Feature | Provisioned | Serverless |
|---------|-------------|------------|
| **Cost** | Pay for running instances (24/7) | Pay per inference |
| **Latency** | Low (<100ms) | Higher first request (cold start) |
| **Scaling** | Manual or auto-scaling | Automatic |
| **Best for** | Steady traffic | Intermittent traffic |
| **Min instances** | 1 | 0 (scales to zero) |

### When to Use Each

**Use Provisioned**:

- Steady, predictable traffic
- Latency-sensitive applications
- High request volume (>1000/day)
- Real-time applications

**Use Serverless**:

- Intermittent traffic
- Development/testing
- Low request volume
- Cost-sensitive applications

## Advanced: Blue-Green Deployment

Deploy new version without downtime:

```bash
# Train new model
NEW_MODEL=$(easy_sm train -n training-job-v2 -e ml.m5.large \
  -i s3://bucket/training-data \
  -o s3://bucket/models)

# Deploy to new endpoint
easy_sm deploy -n prod-endpoint-v2 -e ml.m5.large -m $NEW_MODEL

# Test new endpoint
# (your testing logic here)

# If successful, switch traffic
# Update DNS or load balancer to point to prod-endpoint-v2

# Delete old endpoint
easy_sm delete-endpoint -n prod-endpoint --delete-config
```

## Advanced: Multi-Model Endpoint

Deploy multiple models to one endpoint for cost savings:

```bash
# Deploy first model
easy_sm deploy -n multi-model-endpoint -e ml.m5.large \
  -m s3://bucket/models/model1.tar.gz

# Add additional models to S3
aws s3 cp s3://bucket/models/model2.tar.gz s3://bucket/multi-models/
aws s3 cp s3://bucket/models/model3.tar.gz s3://bucket/multi-models/

# Invoke specific model
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name multi-model-endpoint \
  --target-model model2.tar.gz \
  --body '{"features": [[1.0, 2.0]]}' \
  output.json
```

## Advanced: Auto-Scaling

Configure auto-scaling for provisioned endpoints:

```bash
# Deploy endpoint
easy_sm deploy -n prod-endpoint -e ml.m5.large --num-instances 2 -m $MODEL

# Configure auto-scaling (using AWS CLI)
aws application-autoscaling register-scalable-target \
  --service-namespace sagemaker \
  --resource-id endpoint/prod-endpoint/variant/AllTraffic \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --min-capacity 2 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --service-namespace sagemaker \
  --resource-id endpoint/prod-endpoint/variant/AllTraffic \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --policy-name scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
    }
  }'
```

This auto-scales between 2-10 instances based on request volume.

## Troubleshooting

### Local Deployment Fails

**Issue**: `Port 8080 already in use`

**Solution**: Stop existing container or use different port:

```bash
easy_sm local stop
# Or kill process using port 8080
lsof -ti:8080 | xargs kill -9
```

### Model Not Found Error

**Issue**: `ModelError: Could not load model`

**Solution**: Verify model file exists and path is correct:

```python
# In serve code
print(f"Model directory contents: {os.listdir(model_dir)}")
```

### Prediction Returns Error

**Issue**: `Invalid input format`

**Solution**: Check content type and input format match:

```bash
# Correct CSV format (no spaces after comma)
curl -d '1.0,2.0' -H 'Content-Type: text/csv' ...

# Not: '1.0, 2.0' (space after comma)
```

### Endpoint Creation Fails

**Issue**: `ResourceLimitExceeded`

**Solution**: Check service quotas:

```bash
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-11111111
```

Request quota increase if needed.

## Monitoring and Logging

### CloudWatch Metrics

Monitor endpoint metrics:

```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=my-endpoint \
  --start-time 2025-01-01T00:00:00Z \
  --end-time 2025-01-01T23:59:59Z \
  --period 3600 \
  --statistics Average
```

### CloudWatch Logs

View inference logs:

```bash
aws logs tail /aws/sagemaker/Endpoints/my-endpoint --follow
```

## Cost Optimization

### Provisioned Endpoints

- Use smallest instance that meets requirements
- Use auto-scaling to reduce idle capacity
- Delete unused endpoints
- Consider Savings Plans or Reserved Instances

### Serverless Endpoints

- Best for intermittent traffic
- Pay only for inference requests
- No cost when idle

### Example Costs (us-east-1, approximate)

| Instance Type | Cost/Hour | Use Case |
|---------------|-----------|----------|
| ml.t2.medium | $0.065 | Dev/test |
| ml.m5.large | $0.134 | Production |
| ml.m5.xlarge | $0.269 | High throughput |
| ml.c5.xlarge | $0.238 | CPU-intensive |

## Next Steps

- Explore [advanced workflows](advanced-workflows.md) for automation
- Learn about [piped workflows](../user-guide/piped-workflows.md)
- Read [endpoint management](../commands/endpoints.md) documentation

## See Also

- [Deploy Command Reference](../commands/deploy.md)
- [Cloud Deployment Guide](../user-guide/cloud-deployment.md)
- [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
