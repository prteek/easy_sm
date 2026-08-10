# Advanced Workflows

This guide demonstrates advanced patterns for automation, CI/CD integration, and production deployments with easy_sm.

## Piped Workflows

### One-Liner Deployment

Deploy the model from the latest training job:

```bash
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

**How it works**:

1. `list-training-jobs -n -m 1` → Get latest job name
2. `get-model-artifacts -j <job>` → Get model S3 path
3. `deploy -m <path>` → Deploy using that model

### Deploy Latest Completed Job

Filter for completed jobs only:

```bash
# Get latest completed training job (extract job name with $1, not $2 which is status)
JOB=$(easy_sm list-training-jobs -m 20 | grep Completed | head -1 | awk '{print $1}')

# Deploy that model
MODEL=$(easy_sm get-model-artifacts -j $JOB)
easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL
```

### Variable-Based Workflow

```bash
# Capture each step's output
JOB=$(easy_sm list-training-jobs -n -m 1)
echo "Latest job: $JOB"

MODEL=$(easy_sm get-model-artifacts -j $JOB)
echo "Model path: $MODEL"

ENDPOINT=$(easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL)
echo "Deployed to: $ENDPOINT"
```

## Automation Scripts

### Complete Training Pipeline

```bash
#!/bin/bash
set -e  # Exit on error

APP_NAME="my-app"
JOB_NAME="training-$(date +%Y%m%d-%H%M%S)"
ENDPOINT_NAME="prod-endpoint"

echo "Starting training pipeline..."

# 1. Upload data
echo "Uploading training data..."
easy_sm upload-data -i ./data -t s3://bucket/training-data

# 2. Train model
echo "Training model: $JOB_NAME"
MODEL=$(easy_sm train -n $JOB_NAME -e ml.m5.large \
  -i s3://bucket/training-data \
  -o s3://bucket/models)

echo "Model saved to: $MODEL"

# 3. Deploy to staging
echo "Deploying to staging..."
STAGING_ENDPOINT="${ENDPOINT_NAME}-staging"
easy_sm deploy -n $STAGING_ENDPOINT -e ml.t2.medium -m $MODEL

# 4. Test endpoint (placeholder)
echo "Testing endpoint..."
# Add your testing logic here

# 5. Deploy to production
echo "Deploying to production..."
easy_sm deploy -n $ENDPOINT_NAME -e ml.m5.large -m $MODEL --num-instances 2

# 6. Cleanup staging
echo "Cleaning up staging..."
easy_sm delete-endpoint -n $STAGING_ENDPOINT --delete-config

echo "Pipeline complete!"
```

Save as `deploy_pipeline.sh` and run:

```bash
chmod +x deploy_pipeline.sh
./deploy_pipeline.sh
```

### Batch Processing Pipeline

```bash
#!/bin/bash
set -e

# 1. Run processing job
echo "Processing data..."
easy_sm process -f preprocess.py -e ml.m5.large -n process-job \
  -i s3://bucket/raw-data \
  -o s3://bucket/processed-data

# 2. Train on processed data
echo "Training model..."
MODEL=$(easy_sm train -n training-job -e ml.m5.xlarge \
  -i s3://bucket/processed-data \
  -o s3://bucket/models)

# 3. Run batch transform
echo "Running batch predictions..."
easy_sm batch-transform -e ml.m5.large --num-instances 1 \
  -m $MODEL \
  -i s3://bucket/batch-input \
  -o s3://bucket/predictions

echo "Batch processing complete!"
```

### Scheduled Retraining

```bash
#!/bin/bash
# Add to crontab: 0 2 * * 0 /path/to/retrain.sh

set -e

DATE=$(date +%Y%m%d)
JOB_NAME="weekly-retrain-$DATE"

echo "[$DATE] Starting weekly retraining..."

# Train new model
MODEL=$(easy_sm train -n $JOB_NAME -e ml.m5.large \
  -i s3://bucket/latest-data \
  -o s3://bucket/models)

# Get current endpoint model
CURRENT_MODEL=$(aws sagemaker describe-endpoint-config \
  --endpoint-config-name prod-endpoint-config \
  --query 'ProductionVariants[0].ModelName' --output text)

echo "Current model: $CURRENT_MODEL"
echo "New model: $MODEL"

# Deploy new model to test endpoint
TEST_ENDPOINT="prod-test-$DATE"
easy_sm deploy -n $TEST_ENDPOINT -e ml.m5.large -m $MODEL

# Test new endpoint
# (your testing logic)

# If tests pass, update production
if [ $? -eq 0 ]; then
    easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL --num-instances 3
    easy_sm delete-endpoint -n $TEST_ENDPOINT
    echo "Production updated with new model"
else
    echo "Tests failed, keeping old model" >&2
    exit 1
fi
```

Add to crontab for weekly retraining:

```bash
crontab -e
# Add line:
0 2 * * 0 /path/to/retrain.sh
```

## Production Deployment Patterns

### Blue-Green Deployment

Deploy new version without downtime:

```bash
#!/bin/bash
set -e

# Train new model
NEW_MODEL=$(easy_sm train -n training-job-v2 -e ml.m5.large \
  -i s3://bucket/training-data \
  -o s3://bucket/models)

# Deploy to new endpoint (green)
GREEN_ENDPOINT="prod-endpoint-v2"
easy_sm deploy -n $GREEN_ENDPOINT -e ml.m5.large -m $NEW_MODEL --num-instances 2

# Test new endpoint
echo "Testing new endpoint..."
# Run your test suite against $GREEN_ENDPOINT

# If tests pass, switch traffic
if [ $? -eq 0 ]; then
    echo "Tests passed, switching traffic..."

    # Update DNS/load balancer to point to new endpoint
    # (implementation depends on your setup)

    # Wait for traffic to drain from old endpoint
    sleep 300

    # Delete old endpoint (blue)
    easy_sm delete-endpoint -n prod-endpoint-v1 --delete-config

    echo "Blue-green deployment complete"
else
    echo "Tests failed, keeping old endpoint" >&2
    easy_sm delete-endpoint -n $GREEN_ENDPOINT
    exit 1
fi
```

### Canary Deployment

Gradual rollout with monitoring:

```bash
#!/bin/bash
set -e

MODEL=$(easy_sm get-model-artifacts -j latest-job)

# Deploy canary with 1 instance
echo "Deploying canary..."
easy_sm deploy -n canary-endpoint -e ml.m5.large -m $MODEL --num-instances 1

# Monitor for 10 minutes
echo "Monitoring canary for 10 minutes..."
sleep 600

# Check error rate (placeholder - implement your monitoring logic)
ERROR_RATE=$(check_error_rate canary-endpoint)

if [ $ERROR_RATE -lt 5 ]; then
    echo "Canary successful, scaling to production..."
    easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL --num-instances 3
    easy_sm delete-endpoint -n canary-endpoint
else
    echo "Canary failed, rolling back..." >&2
    easy_sm delete-endpoint -n canary-endpoint
    exit 1
fi
```

### A/B Testing Setup

Deploy multiple model versions:

```bash
#!/bin/bash

# Deploy model A (current)
MODEL_A=$(easy_sm get-model-artifacts -j current-model)
easy_sm deploy -n endpoint-a -e ml.m5.large -m $MODEL_A --num-instances 2

# Deploy model B (experiment)
MODEL_B=$(easy_sm get-model-artifacts -j experimental-model)
easy_sm deploy -n endpoint-b -e ml.m5.large -m $MODEL_B --num-instances 2

echo "A/B test endpoints deployed:"
echo "Model A: endpoint-a"
echo "Model B: endpoint-b"
echo "Configure load balancer to split traffic 50/50"
```

## CI/CD Integration

### GitHub Actions

Complete CI/CD pipeline:

```yaml
name: Train and Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  AWS_REGION: eu-west-1
  SAGEMAKER_ROLE: ${{ secrets.SAGEMAKER_ROLE }}

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install easy_sm
        run: pip install easy-sm

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Configure Docker tag
        run: |
          # Update config with current commit SHA
          python3 - <<EOF
          import json
          with open('my-app.json', 'r') as f:
              config = json.load(f)
          config['docker_tag'] = '${{ github.sha }}'
          with open('my-app.json', 'w') as f:
              json.dump(config, f, indent=2)
          EOF

      - name: Build Docker image
        run: easy_sm build

      - name: Push to ECR
        run: easy_sm push

  train:
    needs: build
    runs-on: ubuntu-latest
    outputs:
      model_path: ${{ steps.train.outputs.model }}
    steps:
      - uses: actions/checkout@v3

      - name: Install easy_sm
        run: pip install easy-sm

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Configure Docker tag
        run: |
          python3 - <<EOF
          import json
          with open('my-app.json', 'r') as f:
              config = json.load(f)
          config['docker_tag'] = '${{ github.sha }}'
          with open('my-app.json', 'w') as f:
              json.dump(config, f, indent=2)
          EOF

      - name: Train model
        id: train
        run: |
          MODEL=$(easy_sm train \
            -n training-job-${{ github.run_number }} \
            -e ml.m5.large \
            -i s3://my-bucket/training-data \
            -o s3://my-bucket/models)
          echo "model=$MODEL" >> $GITHUB_OUTPUT

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install easy_sm
        run: pip install easy-sm

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Deploy to staging
        run: |
          easy_sm deploy -n staging-endpoint \
            -e ml.t2.medium \
            -m ${{ needs.train.outputs.model_path }}

      - name: Test staging endpoint
        run: |
          # Add your testing logic
          python test_endpoint.py staging-endpoint

      - name: Deploy to production
        if: success()
        run: |
          easy_sm deploy -n prod-endpoint \
            -e ml.m5.large \
            --num-instances 2 \
            -m ${{ needs.train.outputs.model_path }}
```

### GitLab CI

```yaml
stages:
  - build
  - train
  - deploy

variables:
  AWS_REGION: eu-west-1
  SAGEMAKER_ROLE: $SAGEMAKER_ROLE_ARN

build:
  stage: build
  image: python:3.13
  before_script:
    - pip install easy-sm
  script:
    - |
      python3 - <<EOF
      import json
      with open('my-app.json', 'r') as f:
          config = json.load(f)
      config['docker_tag'] = '$CI_COMMIT_SHA'
      with open('my-app.json', 'w') as f:
          json.dump(config, f, indent=2)
      EOF
    - easy_sm build
    - easy_sm push

train:
  stage: train
  image: python:3.13
  before_script:
    - pip install easy-sm
  script:
    - |
      python3 - <<EOF
      import json
      with open('my-app.json', 'r') as f:
          config = json.load(f)
      config['docker_tag'] = '$CI_COMMIT_SHA'
      with open('my-app.json', 'w') as f:
          json.dump(config, f, indent=2)
      EOF
    - MODEL=$(easy_sm train
        -n training-job-$CI_PIPELINE_ID
        -e ml.m5.large
        -i s3://bucket/training-data
        -o s3://bucket/models)
    - echo "MODEL_PATH=$MODEL" >> train.env
  artifacts:
    reports:
      dotenv: train.env

deploy-staging:
  stage: deploy
  image: python:3.13
  before_script:
    - pip install easy-sm
  script:
    - easy_sm deploy -n staging-endpoint -e ml.t2.medium -m $MODEL_PATH
  dependencies:
    - train

deploy-production:
  stage: deploy
  image: python:3.13
  before_script:
    - pip install easy-sm
  script:
    - easy_sm deploy -n prod-endpoint -e ml.m5.large --num-instances 3 -m $MODEL_PATH
  dependencies:
    - train
  when: manual
  only:
    - main
```

## Model Versioning

### Version Tracking Script

```bash
#!/bin/bash
set -e

VERSION=$(date +%Y%m%d-%H%M%S)
JOB_NAME="model-$VERSION"

# Train model
MODEL=$(easy_sm train -n $JOB_NAME -e ml.m5.large \
  -i s3://bucket/training-data \
  -o s3://bucket/models/$VERSION)

# Save version metadata
cat > model-$VERSION.json <<EOF
{
  "version": "$VERSION",
  "training_job": "$JOB_NAME",
  "model_path": "$MODEL",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$(git rev-parse HEAD)",
  "metrics": {
    "accuracy": $(get_accuracy $JOB_NAME),
    "f1_score": $(get_f1_score $JOB_NAME)
  }
}
EOF

# Upload metadata to S3
aws s3 cp model-$VERSION.json s3://bucket/model-registry/

echo "Model version $VERSION registered"
```

### Model Rollback

```bash
#!/bin/bash

# List recent model versions
echo "Recent models:"
aws s3 ls s3://bucket/model-registry/ | tail -5

# Select version to rollback to
read -p "Enter version to rollback to: " VERSION

# Get model path from metadata
MODEL=$(aws s3 cp s3://bucket/model-registry/model-$VERSION.json - \
  | jq -r '.model_path')

echo "Rolling back to: $MODEL"

# Deploy previous version
easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL --num-instances 3

echo "Rollback complete"
```

## Monitoring and Alerts

### Health Check Script

```bash
#!/bin/bash

ENDPOINT="prod-endpoint"

while true; do
    # Check endpoint status
    STATUS=$(aws sagemaker describe-endpoint --endpoint-name $ENDPOINT \
      --query 'EndpointStatus' --output text)

    if [ "$STATUS" != "InService" ]; then
        echo "WARNING: Endpoint $ENDPOINT is $STATUS"

        # Send alert (example using AWS SNS)
        aws sns publish \
          --topic-arn arn:aws:sns:region:account:alerts \
          --message "Endpoint $ENDPOINT is $STATUS"

        # Attempt recovery
        MODEL=$(easy_sm get-model-artifacts -j last-known-good-job)
        easy_sm delete-endpoint -n $ENDPOINT
        easy_sm deploy -n $ENDPOINT -e ml.m5.large -m $MODEL
    fi

    sleep 300  # Check every 5 minutes
done
```

### CloudWatch Metrics Script

```bash
#!/bin/bash

ENDPOINT="prod-endpoint"
START_TIME=$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%S)

# Get model latency
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=$ENDPOINT \
  --start-time $START_TIME \
  --end-time $END_TIME \
  --period 300 \
  --statistics Average,Maximum \
  --output table

# Get invocation count
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name Invocations \
  --dimensions Name=EndpointName,Value=$ENDPOINT \
  --start-time $START_TIME \
  --end-time $END_TIME \
  --period 300 \
  --statistics Sum \
  --output table
```

## Cost Optimization

### Automatic Instance Scaling

```bash
#!/bin/bash

# Deploy with auto-scaling enabled
easy_sm deploy -n prod-endpoint -e ml.m5.large --num-instances 2 -m $MODEL

# Configure auto-scaling (2-10 instances)
aws application-autoscaling register-scalable-target \
  --service-namespace sagemaker \
  --resource-id endpoint/prod-endpoint/variant/AllTraffic \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --min-capacity 2 \
  --max-capacity 10

# Add scaling policy (scale based on invocations)
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
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

### Cost Monitoring

```bash
#!/bin/bash

# Get endpoint costs for last 7 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter file://filter.json

# filter.json:
# {
#   "Tags": {
#     "Key": "Project",
#     "Values": ["easy-sm-prod"]
#   }
# }
```

## Next Steps

- Read [Piped Workflows Guide](../user-guide/piped-workflows.md) for more examples
- See [Cloud Deployment Guide](../user-guide/cloud-deployment.md) for deployment options
- Explore [Command Reference](../commands/index.md) for all available commands

## See Also

- [Training Example](training-example.md)
- [Deployment Example](deployment-example.md)
- [User Guide](../user-guide/overview.md)
