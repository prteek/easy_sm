# deploy

Deploy trained ML models to AWS SageMaker endpoints.

## Commands

- [`deploy`](#deploy) - Deploy to provisioned endpoint
- [`deploy-serverless`](#deploy-serverless) - Deploy to serverless endpoint

---

## deploy

Deploy ML model to a provisioned SageMaker endpoint.

### Synopsis

```bash
easy_sm [--docker-tag TAG] deploy --endpoint-name NAME --instance-type TYPE \
  --s3-model-location S3_PATH [OPTIONS]
```

### Description

The `deploy` command creates a SageMaker endpoint with provisioned instances that serve your trained model. This provides consistent performance and is suitable for production workloads with predictable traffic.

The command:

1. Creates a SageMaker model from your Docker image and model artifacts
2. Creates an endpoint configuration with specified instance type and count
3. Creates or updates the endpoint
4. Outputs the endpoint name for use in other commands

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--endpoint-name` | `-n` | string | **Yes** | - | Name for the SageMaker endpoint |
| `--instance-type` | `-e` | string | **Yes** | - | EC2 instance type (e.g., `ml.m5.large`) |
| `--s3-model-location` | `-m` | string | **Yes** | - | S3 location to model tar.gz |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | AWS IAM role ARN for SageMaker |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--instance-count` | `-c` | integer | No | `1` | Number of EC2 instances |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

### Examples

#### Deploy with model from training

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

easy_sm deploy \
  -n my-endpoint \
  -e ml.m5.large \
  -m s3://my-bucket/models/my-job/output/model.tar.gz
```

Output:
```
my-endpoint
```

#### Deploy with specific IAM role

```bash
easy_sm deploy \
  -n production-endpoint \
  -e ml.m5.large \
  -m s3://bucket/model.tar.gz \
  -r arn:aws:iam::123456789012:role/CustomRole
```

#### Deploy with multiple instances

For high-availability and load balancing:

```bash
easy_sm deploy \
  -n ha-endpoint \
  -e ml.m5.xlarge \
  -c 3 \
  -m s3://bucket/model.tar.gz
```

#### Deploy with specific Docker tag

```bash
easy_sm -t v1.2.0 deploy \
  -n versioned-endpoint \
  -e ml.m5.large \
  -m s3://bucket/model.tar.gz
```

#### Deploy latest trained model (one-liner)

```bash
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

### Output Format

The command outputs the endpoint name:

```
my-endpoint
```

This can be used in scripts or piped to other tools:

```bash
ENDPOINT=$(easy_sm deploy -n my-endpoint -e ml.m5.large -m s3://...)
echo "Deployed to: $ENDPOINT"
```

### Prerequisites

- Trained model uploaded to S3 (from `easy_sm train` or manual upload)
- Docker image pushed to ECR
- IAM role with SageMaker permissions
- Valid serving code in `prediction/serve`

### Serving Code Requirements

Your serving code at `prediction/serve` must implement:

```python
import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    """
    Load model from directory.

    Args:
        model_dir: /opt/ml/model directory with unpacked model.tar.gz

    Returns:
        Loaded model object
    """
    model_path = os.path.join(model_dir, 'model.mdl')
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """
    Parse and preprocess input data.

    Args:
        request_body: Raw request body from client
        request_content_type: Content type (e.g., 'text/csv', 'application/json')

    Returns:
        Parsed input data ready for prediction
    """
    if request_content_type == 'text/csv':
        # Parse CSV
        return np.array([float(x) for x in request_body.split(',')]).reshape(1, -1)
    elif request_content_type == 'application/json':
        # Parse JSON
        data = json.loads(request_body)
        return np.array(data['features']).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make predictions using the model.

    Args:
        input_data: Preprocessed input from input_fn
        model: Model loaded from model_fn

    Returns:
        Model predictions
    """
    return model.predict(input_data)

def output_fn(prediction, accept):
    """
    Format prediction output for response.

    Args:
        prediction: Predictions from predict_fn
        accept: Requested response content type

    Returns:
        Formatted response string
    """
    if accept == 'application/json':
        return json.dumps({"predictions": prediction.tolist()})
    elif accept == 'text/csv':
        return ','.join(map(str, prediction.tolist()))
    else:
        return str(prediction)
```

### Instance Types

Common instance types for inference:

| Instance Type | vCPUs | Memory | Use Case | Cost |
|--------------|-------|---------|----------|------|
| `ml.t2.medium` | 2 | 4 GB | Development/testing | $ |
| `ml.m5.large` | 2 | 8 GB | Low-traffic production | $$ |
| `ml.m5.xlarge` | 4 | 16 GB | Medium traffic | $$$ |
| `ml.m5.2xlarge` | 8 | 32 GB | High traffic | $$$$ |
| `ml.c5.xlarge` | 4 | 8 GB | Compute-intensive | $$$ |
| `ml.p3.2xlarge` | 8 | 61 GB + GPU | Deep learning inference | $$$$$ |

See [AWS pricing](https://aws.amazon.com/sagemaker/pricing/) for exact costs.

### Endpoint Lifecycle

1. **Creating**: Endpoint is being provisioned (~5-10 minutes)
2. **InService**: Endpoint is ready to serve predictions
3. **Updating**: Configuration changes being applied
4. **Failed**: Deployment failed (check logs)

Check status:

```bash
aws sagemaker describe-endpoint --endpoint-name my-endpoint \
  --query 'EndpointStatus'
```

### High Availability

Deploy multiple instances for redundancy:

```bash
easy_sm deploy \
  -n ha-endpoint \
  -e ml.m5.large \
  -c 3 \
  -m s3://bucket/model.tar.gz
```

SageMaker automatically:
- Load balances across instances
- Handles instance failures
- Distributes requests

### Auto-Scaling

After deployment, configure auto-scaling via AWS Console or CLI:

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace sagemaker \
  --resource-id endpoint/my-endpoint/variant/AllTraffic \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --min-capacity 1 \
  --max-capacity 10
```

### Testing the Endpoint

After deployment completes:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='application/json',
    Body=json.dumps({'features': [1.0, 2.0, 3.0, 4.0]})
)

result = json.loads(response['Body'].read())
print(result)
```

Or with AWS CLI:

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name my-endpoint \
  --content-type application/json \
  --body '{"features": [1.0, 2.0, 3.0, 4.0]}' \
  output.json

cat output.json
```

### Troubleshooting

#### Deployment takes too long

**Problem**: Endpoint stuck in "Creating" status.

**Solution**:
1. Check CloudWatch logs: `/aws/sagemaker/Endpoints/my-endpoint`
2. Verify model artifacts exist: `aws s3 ls s3://bucket/model.tar.gz`
3. Check Docker image in ECR

#### Model loading errors

**Problem**: Endpoint fails with model loading errors.

**Solution**: Ensure `model_fn` correctly loads your model format:

```python
def model_fn(model_dir):
    # List files to debug
    import os
    print(f"Files in model_dir: {os.listdir(model_dir)}")

    # Load model
    model_path = os.path.join(model_dir, 'model.mdl')
    return joblib.load(model_path)
```

#### Out of memory errors

**Problem**: Instance runs out of memory during inference.

**Solution**: Use a larger instance type:

```bash
easy_sm deploy -e ml.m5.xlarge ...  # Upgrade from ml.m5.large
```

#### "Model name already exists"

**Problem**: Model with this name already registered.

**Solution**: Delete old endpoint first:

```bash
easy_sm delete-endpoint -n my-endpoint
```

Or use a different endpoint name.

---

## deploy-serverless

Deploy ML model to a serverless SageMaker endpoint.

### Synopsis

```bash
easy_sm [--docker-tag TAG] deploy-serverless --endpoint-name NAME \
  --memory-size-in-mb SIZE --s3-model-location S3_PATH [OPTIONS]
```

### Description

The `deploy-serverless` command creates a serverless SageMaker endpoint that automatically scales based on traffic. This is cost-effective for variable or unpredictable workloads.

Serverless endpoints:
- Scale to zero when idle (no cost)
- Auto-scale based on demand
- No instance management required
- Pay only for inference time

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--endpoint-name` | `-n` | string | **Yes** | - | Name for the SageMaker endpoint |
| `--memory-size-in-mb` | `-s` | integer | **Yes** | - | Memory allocation (1024, 2048, 3072, 4096, 5120, or 6144 MB) |
| `--s3-model-location` | `-m` | string | **Yes** | - | S3 location to model tar.gz |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | AWS IAM role ARN for SageMaker |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--max-concurrency` | `-mc` | integer | No | `5` | Maximum concurrent invocations per instance |
| `--docker-tag` | `-t` | string | No | `latest` | Docker image tag (global option) |

### Examples

#### Deploy serverless endpoint

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

easy_sm deploy-serverless \
  -n my-serverless-endpoint \
  -s 2048 \
  -m s3://my-bucket/models/my-job/output/model.tar.gz
```

Output:
```
my-serverless-endpoint
```

#### Deploy with higher memory

```bash
easy_sm deploy-serverless \
  -n memory-intensive-endpoint \
  -s 6144 \
  -m s3://bucket/model.tar.gz
```

#### Deploy with custom concurrency

```bash
easy_sm deploy-serverless \
  -n high-concurrency-endpoint \
  -s 4096 \
  -mc 20 \
  -m s3://bucket/model.tar.gz
```

#### Deploy with specific tag

```bash
easy_sm -t v1.0.0 deploy-serverless \
  -n serverless-v1 \
  -s 2048 \
  -m s3://bucket/model.tar.gz
```

### Output Format

The command outputs the endpoint name:

```
my-serverless-endpoint
```

### Memory Sizes

Valid memory configurations:

- **1024 MB**: Small models, simple inference
- **2048 MB**: Most models (recommended starting point)
- **3072 MB**: Medium-sized models
- **4096 MB**: Large models
- **5120 MB**: Very large models
- **6144 MB**: Maximum memory

!!! tip "Choosing Memory Size"
    Start with 2048 MB and adjust based on:
    - Model size on disk
    - Peak memory usage during inference
    - Performance requirements

### Max Concurrency

The `--max-concurrency` setting controls how many simultaneous requests a single instance can handle.

- **Lower (1-5)**: Better for memory-intensive models
- **Higher (10-20)**: Better for fast, lightweight models

SageMaker automatically scales out instances based on concurrency and memory limits.

### Serverless vs Provisioned

| Feature | Serverless | Provisioned |
|---------|-----------|-------------|
| **Cost when idle** | $0 | Full instance cost |
| **Startup time** | Cold start ~10-30s | Always warm |
| **Best for** | Variable traffic | Consistent traffic |
| **Max concurrency** | Configurable | Instance-dependent |
| **Auto-scaling** | Automatic | Manual configuration |

### When to Use Serverless

✅ **Good for:**
- Development and testing
- Intermittent production traffic
- Unpredictable workloads
- Cost optimization

❌ **Not good for:**
- Latency-sensitive applications (cold starts)
- Consistent high traffic (provisioned is cheaper)
- Very large models (>6 GB memory)

### Cold Starts

Serverless endpoints have cold start latency when scaling from zero:

- **First request**: ~10-30 seconds (container initialization)
- **Subsequent requests**: Milliseconds (while warm)

Mitigation strategies:
1. **Periodic warm-up**: Send dummy requests every 5 minutes
2. **Accept latency**: For non-critical workloads
3. **Use provisioned**: For latency-sensitive apps

Example warm-up script:

```bash
#!/bin/bash
# Keep endpoint warm
while true; do
  aws sagemaker-runtime invoke-endpoint \
    --endpoint-name my-serverless-endpoint \
    --content-type application/json \
    --body '{"features": [0]}' \
    /dev/null
  sleep 300  # Every 5 minutes
done
```

### Cost Comparison

**Serverless pricing** (approximate):
- $0.20 per 100,000 inference requests
- $0.000125 per GB-second of memory

**Provisioned pricing** (ml.m5.large):
- $0.119 per hour (~$85/month running continuously)

**Example cost calculation:**

Scenario: 100 requests/day, 2GB memory, 1 second per request

- **Serverless**: 100 * 30 days * $0.000002 + (100 * 30 * 2 * 1) * $0.000125 = ~$0.75/month
- **Provisioned**: 24 * 30 * $0.119 = ~$85/month

Serverless is 100x cheaper for low-traffic workloads!

### Testing Serverless Endpoints

Same as provisioned endpoints:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

# First request (cold start - may take 10-30s)
response = runtime.invoke_endpoint(
    EndpointName='my-serverless-endpoint',
    ContentType='application/json',
    Body=json.dumps({'features': [1.0, 2.0, 3.0]})
)

print(json.loads(response['Body'].read()))

# Second request (warm - milliseconds)
response = runtime.invoke_endpoint(
    EndpointName='my-serverless-endpoint',
    ContentType='application/json',
    Body=json.dumps({'features': [4.0, 5.0, 6.0]})
)

print(json.loads(response['Body'].read()))
```

### Monitoring

Monitor serverless endpoints in CloudWatch:

```bash
# Invocations
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name Invocations \
  --dimensions Name=EndpointName,Value=my-serverless-endpoint \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Sum

# Model latency
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=my-serverless-endpoint \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

### Troubleshooting

#### "Memory size must be one of [1024, 2048, ...]"

**Problem**: Invalid memory size specified.

**Solution**: Use one of the valid values:

```bash
easy_sm deploy-serverless -s 2048 ...  # Valid
# Not: -s 2000  # Invalid
```

#### Model too large for serverless

**Problem**: Model exceeds 6144 MB memory limit.

**Solution**: Use provisioned endpoint instead:

```bash
easy_sm deploy -e ml.m5.xlarge ...
```

#### High cold start latency

**Problem**: First requests take too long.

**Solution**: Options:
1. Implement warm-up pings
2. Use provisioned endpoint
3. Optimize model size
4. Use faster serialization (pickle → joblib)

---

## Complete Deployment Workflow

### Development to Production

```bash
# 1. Test locally
easy_sm local train
easy_sm local deploy
# Test with curl

# 2. Train on SageMaker
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
easy_sm -t dev build
easy_sm -t dev push
MODEL=$(easy_sm -t dev train \
  -n dev-training \
  -e ml.m5.large \
  -i s3://bucket/data \
  -o s3://bucket/models)

# 3. Deploy to development (serverless)
easy_sm -t dev deploy-serverless \
  -n dev-endpoint \
  -s 2048 \
  -m $MODEL

# 4. Test development endpoint
# ... testing ...

# 5. Deploy to production (provisioned, multi-instance)
easy_sm -t v1.0.0 build
easy_sm -t v1.0.0 push
easy_sm -t v1.0.0 deploy \
  -n prod-endpoint \
  -e ml.m5.xlarge \
  -c 3 \
  -m $MODEL
```

### Blue-Green Deployment

```bash
# Current production: prod-endpoint (green)
# Deploy new version: prod-endpoint-blue

# 1. Deploy blue endpoint
easy_sm -t v2.0.0 build
easy_sm -t v2.0.0 push
MODEL=$(easy_sm -t v2.0.0 train ...)

easy_sm -t v2.0.0 deploy \
  -n prod-endpoint-blue \
  -e ml.m5.xlarge \
  -c 2 \
  -m $MODEL

# 2. Test blue endpoint
# ... testing ...

# 3. Switch traffic (in application code or API Gateway)
# Update config to use prod-endpoint-blue

# 4. Delete old green endpoint
easy_sm delete-endpoint -n prod-endpoint
```

## Related Commands

- [`train`](train.md) - Train models before deploying
- [`delete-endpoint`](endpoints.md#delete-endpoint) - Delete endpoints
- [`list-endpoints`](endpoints.md#list-endpoints) - List all endpoints
- [`local deploy`](local.md#deploy) - Test deployment locally

## See Also

- [Inference Guide](../user-guide/inference.md)
- [Endpoint Management](../user-guide/endpoint-management.md)
- [SageMaker Endpoints Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
