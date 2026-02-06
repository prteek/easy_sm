# AWS Setup Guide

This guide covers the AWS prerequisites and configuration needed to use easy_sm with AWS SageMaker.

## Overview

Easy_sm requires:
1. AWS CLI configured with credentials
2. SageMaker execution IAM role
3. S3 bucket for data and models
4. ECR repository (auto-created by easy_sm)

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed
- Basic understanding of IAM roles and policies

## AWS CLI Configuration

### 1. Install AWS CLI

**macOS:**
```bash
brew install awscli
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Windows:**
Download and run the installer from: https://aws.amazon.com/cli/

### 2. Configure AWS Credentials

Create or obtain AWS access keys from IAM console:
1. Go to AWS Console → IAM → Users → Your User → Security Credentials
2. Create access key (if you don't have one)
3. Save the Access Key ID and Secret Access Key

Configure credentials:

```bash
aws configure
```

You'll be prompted for:
```
AWS Access Key ID: YOUR_ACCESS_KEY_ID
AWS Secret Access Key: YOUR_SECRET_ACCESS_KEY
Default region name: us-east-1  # or your preferred region
Default output format: json
```

This creates two files:
- `~/.aws/credentials` - Contains your access keys
- `~/.aws/config` - Contains configuration settings

### 3. Configure Named Profiles

For multiple AWS accounts or environments, use named profiles.

**Edit `~/.aws/config`:**
```ini
[profile dev]
region = us-east-1
output = json

[profile prod]
region = eu-west-1
output = json
```

**Edit `~/.aws/credentials`:**
```ini
[dev]
aws_access_key_id = YOUR_DEV_ACCESS_KEY
aws_secret_access_key = YOUR_DEV_SECRET_KEY

[prod]
aws_access_key_id = YOUR_PROD_ACCESS_KEY
aws_secret_access_key = YOUR_PROD_SECRET_KEY
```

**Use profiles:**
```bash
# Initialize project with specific profile
easy_sm init
# (select 'dev' profile during initialization)

# Or set environment variable
export AWS_PROFILE=dev
```

### 4. Verify Configuration

```bash
# Test AWS credentials
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "AIDAI...",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/your-user"
# }

# List S3 buckets (if you have any)
aws s3 ls
```

## IAM Role Setup

### Overview

SageMaker requires an IAM role with permissions to:
- Access S3 buckets (read training data, write models)
- Access ECR (pull Docker images)
- Create/manage SageMaker resources
- Write to CloudWatch Logs

### 1. Create SageMaker Execution Role

**Via AWS Console:**

1. Go to IAM → Roles → Create Role
2. Select "AWS service" as trusted entity
3. Choose "SageMaker" as the service
4. Click "Next: Permissions"
5. Attach policies:
   - `AmazonSageMakerFullAccess` (managed policy)
   - `AmazonS3FullAccess` (or custom S3 policy - see below)
   - `AmazonEC2ContainerRegistryFullAccess` (managed policy)
6. Name the role (e.g., `SageMakerExecutionRole`)
7. Create role

**Via AWS CLI:**

```bash
# Create role with trust policy
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
```

### 2. Trust Policy

The trust policy allows SageMaker and your user to assume the role.

**Create `trust-policy.json`:**

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

**Important**: Replace:
- `123456789012` with your AWS account ID
- `your-user` with your IAM user name

**Apply trust policy:**

```bash
aws iam update-assume-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-document file://trust-policy.json
```

### 3. Custom S3 Policy (Optional)

For better security, restrict S3 access to specific buckets:

**Create `s3-policy.json`:**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-sagemaker-bucket",
                "arn:aws:s3:::your-sagemaker-bucket/*"
            ]
        }
    ]
}
```

**Attach custom policy:**

```bash
aws iam put-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-name CustomS3Access \
  --policy-document file://s3-policy.json
```

### 4. Get Role ARN

```bash
# Get role ARN
aws iam get-role --role-name SageMakerExecutionRole --query 'Role.Arn' --output text

# Output:
# arn:aws:iam::123456789012:role/SageMakerExecutionRole
```

Save this ARN - you'll use it with easy_sm.

### 5. Set Environment Variable

Add the role ARN to your environment:

```bash
# Set for current session
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerExecutionRole

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerExecutionRole' >> ~/.bashrc
source ~/.bashrc
```

**Note**: You can also pass the role via `-r/--iam-role-arn` flag on individual commands.

## S3 Bucket Setup

### 1. Create S3 Bucket

**Via AWS Console:**
1. Go to S3 → Create Bucket
2. Choose unique bucket name (e.g., `my-sagemaker-bucket-123`)
3. Select region (same as SageMaker)
4. Keep default settings or customize as needed
5. Create bucket

**Via AWS CLI:**

```bash
# Create bucket
aws s3 mb s3://my-sagemaker-bucket-123 --region us-east-1

# Verify bucket creation
aws s3 ls | grep my-sagemaker-bucket
```

### 2. Bucket Structure

Organize your bucket with a consistent structure:

```
my-sagemaker-bucket/
├── training-data/           # Training datasets
│   ├── dataset-v1/
│   └── dataset-v2/
├── models/                  # Trained model artifacts
│   ├── model-v1/
│   └── model-v2/
├── batch-data/              # Batch transform input
├── batch-predictions/       # Batch transform output
└── processing/              # Processing job data
    ├── input/
    └── output/
```

### 3. Upload Data

```bash
# Upload local directory to S3
aws s3 cp ./local-data s3://my-sagemaker-bucket/training-data/ --recursive

# Or use easy_sm
easy_sm upload-data -i ./local-data -t s3://my-sagemaker-bucket/training-data
```

### 4. Verify Upload

```bash
# List bucket contents
aws s3 ls s3://my-sagemaker-bucket/training-data/

# Download file to verify
aws s3 cp s3://my-sagemaker-bucket/training-data/data.csv ./test.csv
```

## ECR Repository Setup

### Overview

ECR (Elastic Container Registry) stores your Docker images. Easy_sm automatically creates ECR repositories when you run `easy_sm push`, so manual setup is optional.

### Automatic Setup (Recommended)

```bash
# Build Docker image
easy_sm build

# Push to ECR (creates repository if it doesn't exist)
easy_sm push
```

Easy_sm will:
1. Create ECR repository named after your image
2. Authenticate with ECR
3. Tag and push Docker image

### Manual Setup (Optional)

If you prefer to create the repository manually:

**Via AWS Console:**
1. Go to ECR → Repositories → Create Repository
2. Name the repository (must match your image name from config)
3. Create repository

**Via AWS CLI:**

```bash
# Create repository
aws ecr create-repository --repository-name my-app

# Get repository URI
aws ecr describe-repositories --repository-names my-app --query 'repositories[0].repositoryUri' --output text

# Output:
# 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-app
```

### ECR Authentication

Easy_sm handles ECR authentication automatically when you run `easy_sm push`.

To authenticate manually:

```bash
# Get login password and authenticate Docker
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
```

## Region Configuration

### Supported Regions

SageMaker is available in most AWS regions. Common regions:

| Region Code | Region Name | Location |
|-------------|-------------|----------|
| us-east-1 | US East (N. Virginia) | USA |
| us-west-2 | US West (Oregon) | USA |
| eu-west-1 | Europe (Ireland) | Europe |
| eu-central-1 | Europe (Frankfurt) | Europe |
| ap-southeast-1 | Asia Pacific (Singapore) | Asia |
| ap-northeast-1 | Asia Pacific (Tokyo) | Asia |

### Select Region

Choose a region based on:
- **Data location** - Keep data and compute in same region
- **Compliance** - Some regions required for regulatory compliance
- **Cost** - Pricing varies slightly by region
- **Availability** - Some instance types not available in all regions

Set region in easy_sm config during `easy_sm init` or edit `app-name.json`:

```json
{
    "aws_region": "us-east-1",
    ...
}
```

## Permissions Checklist

Ensure your IAM user/role has these permissions:

**SageMaker:**
- [ ] Create training jobs
- [ ] Create endpoints
- [ ] Create processing jobs
- [ ] Describe jobs/endpoints
- [ ] Delete endpoints

**S3:**
- [ ] List buckets
- [ ] Get objects
- [ ] Put objects
- [ ] Delete objects

**ECR:**
- [ ] Create repositories
- [ ] Push images
- [ ] Pull images
- [ ] Get authorization token

**IAM:**
- [ ] Pass role to SageMaker
- [ ] Get role information

**CloudWatch:**
- [ ] Create log streams
- [ ] Put log events

## Security Best Practices

### 1. Use IAM Roles, Not Root Account

Never use AWS root account credentials. Always use IAM users with specific permissions.

### 2. Enable MFA

Enable multi-factor authentication on your IAM user:
1. IAM → Users → Your User → Security Credentials
2. Assign MFA device

### 3. Rotate Access Keys

Regularly rotate AWS access keys:

```bash
# Create new access key
aws iam create-access-key --user-name your-user

# Update ~/.aws/credentials with new keys

# Delete old access key
aws iam delete-access-key --access-key-id OLD_KEY_ID --user-name your-user
```

### 4. Use Least Privilege

Grant minimum permissions needed. Avoid using `*FullAccess` policies in production.

### 5. Encrypt S3 Buckets

Enable S3 bucket encryption:

```bash
aws s3api put-bucket-encryption \
  --bucket my-sagemaker-bucket \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

### 6. Enable CloudTrail

Enable CloudTrail for audit logging:
1. CloudTrail → Create Trail
2. Select S3 bucket for logs
3. Enable for all regions

## Cost Management

### 1. Set Up Billing Alerts

Create CloudWatch alarm for billing:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name billing-alert \
  --alarm-description "Alert when charges exceed $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

### 2. Use Cost Explorer

Monitor costs in AWS Console:
- Cost Explorer → Filter by Service → SageMaker
- View daily/monthly costs
- Identify cost trends

### 3. Tag Resources

Tag SageMaker resources for cost tracking:

```bash
easy_sm train -n my-job -e ml.m5.large \
  -i s3://bucket/in -o s3://bucket/out \
  --tags '{"Project": "ML-Pipeline", "Environment": "Dev"}'
```

View costs by tag in Cost Explorer.

### 4. Delete Unused Resources

Regularly delete:
- Unused endpoints (expensive!)
- Old models
- Training job outputs
- Unused S3 data

```bash
# Delete endpoint
easy_sm delete-endpoint -n old-endpoint --delete-config

# Delete S3 objects
aws s3 rm s3://bucket/old-data --recursive
```

## Verification

### Test Complete Setup

Verify everything is configured correctly:

```bash
# 1. Check AWS credentials
aws sts get-caller-identity

# 2. Verify IAM role
aws iam get-role --role-name SageMakerExecutionRole

# 3. Check S3 access
aws s3 ls s3://my-sagemaker-bucket

# 4. Verify SAGEMAKER_ROLE env var
echo $SAGEMAKER_ROLE

# 5. Test easy_sm commands
easy_sm init
easy_sm build
easy_sm push
```

If all commands succeed, your AWS setup is complete!

## Troubleshooting

### Issue: "Unable to locate credentials"

**Cause**: AWS credentials not configured

**Solution**:
```bash
aws configure
# Enter your access key ID and secret access key
```

### Issue: "Access Denied" errors

**Cause**: IAM user/role lacks permissions

**Solution**:
1. Check IAM policies attached to your user
2. Verify SageMaker execution role has required policies
3. Ensure trust policy allows your user to assume the role

### Issue: "InvalidParameter: Execution Role ARN is invalid"

**Cause**: Incorrect role ARN format or non-existent role

**Solution**:
```bash
# Get correct role ARN
aws iam get-role --role-name SageMakerExecutionRole --query 'Role.Arn' --output text

# Update environment variable
export SAGEMAKER_ROLE=<correct-arn>
```

### Issue: "BucketNotFound" errors

**Cause**: S3 bucket doesn't exist or wrong region

**Solution**:
```bash
# Verify bucket exists
aws s3 ls s3://my-sagemaker-bucket

# Create if missing
aws s3 mb s3://my-sagemaker-bucket --region us-east-1
```

### Issue: ECR push fails

**Cause**: Not authenticated with ECR or repository doesn't exist

**Solution**:
```bash
# Easy_sm handles this automatically, but you can verify:
aws ecr describe-repositories --repository-names my-app

# If authentication fails, re-run:
easy_sm push
```

### Issue: Different regions for S3 and SageMaker

**Cause**: S3 bucket and SageMaker in different regions (causes latency and transfer costs)

**Solution**:
- Create S3 bucket in same region as SageMaker
- Update config to use same region for both

## Next Steps

After completing AWS setup:

1. **Initialize project**: `easy_sm init`
2. **Local development**: See [Local Development Guide](local-development.md)
3. **Cloud deployment**: See [Cloud Deployment Guide](cloud-deployment.md)
4. **Piped workflows**: See [Piped Workflows Guide](piped-workflows.md)
