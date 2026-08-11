# Piped Workflows Guide

This guide demonstrates Unix-style command composition with easy_sm, enabling powerful one-liner deployments and automated workflows.

## Overview

Easy_sm follows Unix philosophy by outputting clean, pipable data instead of verbose messages. This design enables command composition, automation, and integration with shell scripts.

## Design Principles

### Clean Output

Commands output only the essential data:

```bash
# train outputs S3 model path
easy_sm train -n job -e ml.m5.large -i s3://in -o s3://out
# Output: s3://bucket/out/job/output/model.tar.gz

# deploy outputs endpoint name
easy_sm deploy -n endpoint -e ml.m5.large -m s3://model.tar.gz
# Output: endpoint

# get-model-artifacts outputs S3 path
easy_sm get-model-artifacts -j training-job
# Output: s3://bucket/path/model.tar.gz

# list-training-jobs with -n outputs names only
easy_sm list-training-jobs -n -m 3
# Output:
# training-job-3
# training-job-2
# training-job-1
```

### Error Handling

- Success → Output to stdout
- Errors → Messages to stderr
- Exit codes: 0 (success), non-zero (failure)

This enables proper error handling in scripts:

```bash
if MODEL=$(easy_sm get-model-artifacts -j my-job 2>/dev/null); then
    echo "Model: $MODEL"
else
    echo "Failed to get model" >&2
    exit 1
fi
```

## Basic Command Composition

### Variable Assignment

Capture command output in variables:

```bash
# Get latest training job name
JOB=$(easy_sm list-training-jobs -n -m 1)

# Get model path from job
MODEL=$(easy_sm get-model-artifacts -j $JOB)

# Deploy using model
ENDPOINT=$(easy_sm deploy -n my-endpoint -e ml.m5.large -m $MODEL)

echo "Deployed to: $ENDPOINT"
```

### Command Substitution

Use command substitution for inline composition:

```bash
# Deploy using model from latest job
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

## Training Workflows

### Train and Save Model Path

```bash
# Train and capture model path
MODEL=$(easy_sm train -n my-training-job -e ml.m5.large \
  -i s3://bucket/training-data \
  -o s3://bucket/models)

echo "Model saved to: $MODEL"

# Save to file for later use
echo $MODEL > model_path.txt
```

### Train with Logging

```bash
# Save training output and model path
easy_sm train -n my-job -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output \
  | tee model_path.txt

# Later, deploy using saved path
MODEL=$(cat model_path.txt)
easy_sm deploy -n endpoint -e ml.m5.large -m $MODEL
```

### Conditional Training

```bash
# Only train if previous job succeeded
if easy_sm train -n job-v2 -e ml.m5.large -i s3://in -o s3://out; then
    echo "Training succeeded"
    # Continue with deployment
else
    echo "Training failed" >&2
    exit 1
fi
```

## Deployment Workflows

### One-Liner Deployment

Deploy model from latest training job:

```bash
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

**Breakdown:**
1. `list-training-jobs -n -m 1` → Get latest job name
2. `get-model-artifacts -j <job>` → Get model S3 path
3. `deploy -m <path>` → Deploy model to endpoint

### Deploy Latest Completed Job

Filter for completed jobs:

```bash
# Get latest completed job (extract job name with $1, not $2 which is status)
JOB=$(easy_sm list-training-jobs -m 20 | grep Completed | head -1 | awk '{print $1}')

# Get model and deploy
MODEL=$(easy_sm get-model-artifacts -j $JOB)
easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL
```

### Deploy with Validation

```bash
# Get model path
MODEL=$(easy_sm get-model-artifacts -j latest-job)

# Validate model exists in S3
if aws s3 ls $MODEL >/dev/null 2>&1; then
    echo "Model found: $MODEL"
    easy_sm deploy -n endpoint -e ml.m5.large -m $MODEL
else
    echo "Model not found in S3" >&2
    exit 1
fi
```

### Blue-Green Deployment

```bash
# Deploy new endpoint
NEW_ENDPOINT="prod-v2-$(date +%Y%m%d)"
MODEL=$(easy_sm get-model-artifacts -j latest-job)

echo "Deploying to: $NEW_ENDPOINT"
easy_sm deploy -n $NEW_ENDPOINT -e ml.m5.large -m $MODEL

# Test new endpoint
# (your testing logic here)

# If successful, delete old endpoint
if [ $? -eq 0 ]; then
    easy_sm delete-endpoint -n prod-v1 --delete-config
    echo "Switched to: $NEW_ENDPOINT"
fi
```

## Job Management Workflows

### List and Filter Training Jobs

```bash
# Get all training jobs
easy_sm list-training-jobs -m 50

# Filter completed jobs
easy_sm list-training-jobs -m 50 | grep Completed

# Filter failed jobs
easy_sm list-training-jobs -m 50 | grep Failed

# Count failed jobs
easy_sm list-training-jobs -m 100 | grep -c Failed
```

### Get Model from Specific Pattern

```bash
# Get model from job matching pattern
JOB=$(easy_sm list-training-jobs -n -m 20 | grep "prod-" | head -1)
MODEL=$(easy_sm get-model-artifacts -j $JOB)

echo "Production model: $MODEL"
```

### Process Multiple Jobs

```bash
# Get models from multiple jobs
easy_sm list-training-jobs -n -m 5 | while read job; do
    model=$(easy_sm get-model-artifacts -j $job)
    echo "Job: $job -> Model: $model"
done
```

### Find Job by Date

```bash
# Get jobs from specific date (without -n to include timestamp in output)
DATE="2025-01"
easy_sm list-training-jobs -m 100 | grep $DATE

# Get job names from specific date
DATE="2025-01"
easy_sm list-training-jobs -m 100 | grep $DATE | awk '{print $1}'
```

**Note**: Don't use `-n` flag if you need to filter by date - the `-n` flag outputs only names without timestamps!

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

# Get current endpoint model (each deploy creates a new, uniquely-named
# endpoint config, so look up the endpoint's *current* config name first)
CURRENT_CONFIG=$(aws sagemaker describe-endpoint \
  --endpoint-name prod-endpoint \
  --query 'EndpointConfigName' --output text)
CURRENT_MODEL=$(aws sagemaker describe-endpoint-config \
  --endpoint-config-name $CURRENT_CONFIG \
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

### Multi-Region Deployment

```bash
#!/bin/bash
set -e

MODEL=$(easy_sm get-model-artifacts -j production-job)
ENDPOINT_NAME="prod-endpoint"

REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1")

echo "Deploying to multiple regions..."

for region in "${REGIONS[@]}"; do
    echo "Deploying to $region..."

    # Copy model to region's bucket
    aws s3 cp $MODEL s3://my-bucket-$region/models/model.tar.gz

    # Deploy in region (requires region-specific config)
    # Note: You'd need to adjust easy_sm config for each region

    echo "Deployed to $region"
done

echo "Multi-region deployment complete!"
```

## Integration with Other Tools

### With jq for JSON Processing

```bash
# Get training job details as JSON (using AWS CLI)
JOB=$(easy_sm list-training-jobs -n -m 1)
aws sagemaker describe-training-job --training-job-name $JOB \
  | jq '.TrainingJobStatus, .ModelArtifacts.S3ModelArtifacts'
```

### With xargs for Parallel Processing

```bash
# Delete multiple endpoints in parallel
easy_sm list-endpoints | grep "test-" | awk '{print $2}' \
  | xargs -P 5 -I {} easy_sm delete-endpoint -n {}
```

### With GNU Parallel

```bash
# Train multiple models in parallel
cat job_configs.txt | parallel -j 3 \
  'easy_sm train -n {} -e ml.m5.large -i s3://in/{} -o s3://out/{}'
```

### With Makefile

```makefile
.PHONY: build push train deploy clean

MODEL_PATH := model_path.txt
ENDPOINT := prod-endpoint

build:
	easy_sm build

push: build
	easy_sm push

train: push
	easy_sm train -n training-job-$(shell date +%Y%m%d) \
	  -e ml.m5.large -i s3://bucket/in -o s3://bucket/out \
	  | tee $(MODEL_PATH)

deploy: train
	easy_sm deploy -n $(ENDPOINT) -e ml.m5.large \
	  -m $$(cat $(MODEL_PATH))

clean:
	easy_sm delete-endpoint -n $(ENDPOINT) --delete-config
	rm -f $(MODEL_PATH)
```

Usage:
```bash
make deploy  # Build, push, train, and deploy in sequence
```

## Advanced Patterns

### Retry Logic

```bash
#!/bin/bash

MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if MODEL=$(easy_sm train -n my-job -e ml.m5.large \
                -i s3://in -o s3://out 2>&1); then
        echo "Training succeeded: $MODEL"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Training failed, retry $RETRY_COUNT/$MAX_RETRIES..." >&2
        sleep 60
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Training failed after $MAX_RETRIES retries" >&2
    exit 1
fi
```

### Progressive Deployment

```bash
#!/bin/bash
set -e

MODEL=$(easy_sm get-model-artifacts -j latest-job)

# Deploy with 1 instance
echo "Deploying with 1 instance..."
easy_sm deploy -n canary-endpoint -e ml.m5.large -m $MODEL --num-instances 1

# Monitor for 10 minutes
sleep 600

# Check error rate (placeholder)
ERROR_RATE=$(check_error_rate canary-endpoint)

if [ $ERROR_RATE -lt 5 ]; then
    echo "Canary successful, scaling to 3 instances..."
    easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL --num-instances 3
else
    echo "Canary failed, rolling back..." >&2
    easy_sm delete-endpoint -n canary-endpoint
    exit 1
fi
```

### Model Versioning

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
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# Upload metadata to S3
aws s3 cp model-$VERSION.json s3://bucket/model-registry/

echo "Model version $VERSION registered"
```

### Health Check and Auto-Recovery

```bash
#!/bin/bash

ENDPOINT="prod-endpoint"
MODEL=$(easy_sm get-model-artifacts -j latest-healthy-job)

while true; do
    # Check endpoint health
    if ! aws sagemaker describe-endpoint --endpoint-name $ENDPOINT \
         --query 'EndpointStatus' --output text | grep -q InService; then

        echo "Endpoint unhealthy, redeploying..." >&2

        # Redeploy with last known good model
        easy_sm delete-endpoint -n $ENDPOINT
        easy_sm deploy -n $ENDPOINT -e ml.m5.large -m $MODEL
    fi

    sleep 300  # Check every 5 minutes
done
```

## Shell Integration

### Bash Functions

Add to `~/.bashrc`:

```bash
# Deploy latest model
deploy_latest() {
    local endpoint=$1
    local instance=${2:-ml.m5.large}

    local model=$(easy_sm get-model-artifacts \
      -j $(easy_sm list-training-jobs -n -m 1))

    easy_sm deploy -n $endpoint -e $instance -m $model
}

# Get model from job pattern
get_model() {
    local pattern=$1
    local job=$(easy_sm list-training-jobs -n -m 20 | grep "$pattern" | head -1)
    easy_sm get-model-artifacts -j $job
}
```

Usage:
```bash
deploy_latest my-endpoint
deploy_latest my-endpoint ml.m5.xlarge

get_model "prod-"
```

### Aliases

Add to `~/.bashrc`:

```bash
alias sm-train='easy_sm train'
alias sm-deploy='easy_sm deploy'
alias sm-jobs='easy_sm list-training-jobs'
alias sm-endpoints='easy_sm list-endpoints'
alias sm-latest='easy_sm list-training-jobs -n -m 1'
```

Usage:
```bash
sm-jobs -m 10
sm-deploy -n endpoint -e ml.m5.large -m $(get_model "prod-")
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Train and Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Install easy_sm
        run: pip install easy-sm

      - name: Build and Push
        run: |
          easy_sm build
          easy_sm push

      - name: Train Model
        run: |
          MODEL=$(easy_sm train -n job-${{ github.run_number }} \
            -e ml.m5.large -i s3://bucket/in -o s3://bucket/out)
          echo "MODEL_PATH=$MODEL" >> $GITHUB_ENV

      - name: Deploy
        run: |
          easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL_PATH
```

### GitLab CI

```yaml
stages:
  - build
  - train
  - deploy

build:
  stage: build
  script:
    - easy_sm build
    - easy_sm push

train:
  stage: train
  script:
    - MODEL=$(easy_sm train -n job-$CI_PIPELINE_ID -e ml.m5.large
        -i s3://bucket/in -o s3://bucket/out)
    - echo "MODEL_PATH=$MODEL" >> train.env
  artifacts:
    reports:
      dotenv: train.env

deploy:
  stage: deploy
  script:
    - easy_sm deploy -n prod-endpoint -e ml.m5.large -m $MODEL_PATH
  dependencies:
    - train
```

## Error Handling Best Practices

### Capture Errors

```bash
# Capture stderr and stdout separately
MODEL=$(easy_sm train -n job -e ml.m5.large -i s3://in -o s3://out 2>error.log)

if [ $? -ne 0 ]; then
    echo "Training failed:" >&2
    cat error.log >&2
    exit 1
fi
```

### Logging

```bash
# Log all commands and output
exec 1> >(tee -a pipeline.log)
exec 2>&1

echo "[$(date)] Starting pipeline..."
easy_sm train -n my-job -e ml.m5.large -i s3://in -o s3://out
echo "[$(date)] Pipeline complete"
```

### Cleanup on Exit

```bash
#!/bin/bash

# Cleanup function
cleanup() {
    echo "Cleaning up temp resources..."
    easy_sm delete-endpoint -n temp-endpoint 2>/dev/null || true
}

# Register cleanup on exit
trap cleanup EXIT

# Your pipeline code
MODEL=$(easy_sm train -n job -e ml.m5.large -i s3://in -o s3://out)
easy_sm deploy -n temp-endpoint -e ml.t2.medium -m $MODEL
```

## Performance Tips

### Parallel Execution

```bash
# Run independent tasks in parallel
(easy_sm local train) &
(easy_sm process -f script.py -e ml.m5.large -n job) &
wait

echo "Both tasks complete"
```

### Caching

```bash
# Cache training job list to avoid repeated API calls
JOBS_CACHE="/tmp/training-jobs-$(date +%Y%m%d).txt"

if [ ! -f $JOBS_CACHE ]; then
    easy_sm list-training-jobs -n -m 100 > $JOBS_CACHE
fi

# Use cached list
grep "prod-" $JOBS_CACHE | head -5
```

## Next Steps

- **Local Development**: See [Local Development Guide](local-development.md)
- **Cloud Deployment**: See [Cloud Deployment Guide](cloud-deployment.md)
- **AWS Setup**: See [AWS Setup Guide](aws-setup.md)

For more examples, see the [main README](../getting-started/quick-start.md).
