# push

Push a Docker image to AWS Elastic Container Registry (ECR).

## Synopsis

```bash
easy_sm [--docker-tag TAG] push [OPTIONS]
```

## Description

The `push` command uploads your locally built Docker image to AWS ECR, making it available for SageMaker training and deployment. The command handles ECR authentication and repository creation automatically.

Before pushing, the image must be built using the [`build`](build.md) command.

## Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--aws-region` | `-r` | string | No | From config | AWS region override |
| `--iam-role-arn` | `-i` | string | No | None | IAM role ARN for authentication (mutually exclusive with `--aws-profile`) |
| `--aws-profile` | `-p` | string | No | From config | AWS profile override (mutually exclusive with `--iam-role-arn`) |
| `--external-id` | `-e` | string | No | None | External ID for IAM role assumption |
| `--docker-tag` | `-t` | string | No | `latest` | Tag for the Docker image (global option, must come before command) |

## Examples

### Basic push with default settings

Push using configuration from `{app_name}.json`:

```bash
easy_sm push
```

### Push with specific AWS profile

Override the profile from config:

```bash
easy_sm push -p prod
```

### Push with IAM role

Use IAM role instead of profile:

```bash
easy_sm push -i arn:aws:iam::123456789012:role/MyRole
```

### Push with external ID

For cross-account access with external ID:

```bash
easy_sm push \
  -i arn:aws:iam::123456789012:role/MyRole \
  -e my-external-id-12345
```

### Push specific Docker tag

```bash
easy_sm -t v1.2.0 push
```

### Push to different region

```bash
easy_sm push -r us-west-2
```

### Push with all options

```bash
easy_sm -t v1.2.0 push \
  -a my-ml-app \
  -p prod \
  -r us-east-1
```

## Output

The push process displays:

```
Started pushing Docker image to AWS ECR. It will take some time. Please, be patient...

[ECR authentication and push progress...]

Docker image pushed to ECR successfully!
```

The image will be available at:

```
{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:{docker_tag}
```

## Prerequisites

- Docker image built with `easy_sm build`
- AWS credentials configured (either profile or IAM role)
- Permissions to push to ECR:
  - `ecr:CreateRepository`
  - `ecr:GetAuthorizationToken`
  - `ecr:BatchCheckLayerAvailability`
  - `ecr:InitiateLayerUpload`
  - `ecr:UploadLayerPart`
  - `ecr:CompleteLayerUpload`
  - `ecr:PutImage`

## Authentication Methods

### Method 1: AWS Profile (Recommended)

Use a profile from `~/.aws/credentials`:

```bash
easy_sm push -p dev
```

The profile should have ECR push permissions.

### Method 2: IAM Role

Assume an IAM role for cross-account or temporary credentials:

```bash
easy_sm push -i arn:aws:iam::123456789012:role/ECRPushRole
```

### Method 3: Config File

Use the profile specified in `{app_name}.json`:

```json
{
    "aws_profile": "dev",
    "aws_region": "us-east-1"
}
```

Then simply:

```bash
easy_sm push
```

!!! warning "Mutually Exclusive Options"
    You cannot use both `--iam-role-arn` and `--aws-profile` simultaneously. Choose one authentication method.

## ECR Repository

The push command automatically:

1. **Creates repository** if it doesn't exist (named after `image_name` from config)
2. **Authenticates** with ECR using provided credentials
3. **Tags image** with the full ECR URI
4. **Pushes layers** to the repository

## What Gets Pushed

The command pushes the Docker image built by `easy_sm build`, which includes:

- Your ML code and dependencies
- Python runtime
- Training and serving entry points
- All layers from the Dockerfile

## Workflow Example

Complete workflow from build to push:

```bash
# 1. Build the image
easy_sm -t v1.0.0 build

# 2. Test locally (optional)
easy_sm -t v1.0.0 local train

# 3. Push to ECR
easy_sm -t v1.0.0 push

# 4. Train on SageMaker
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
easy_sm -t v1.0.0 train \
  -n my-training-job \
  -e ml.m5.large \
  -i s3://my-bucket/input \
  -o s3://my-bucket/output
```

## Troubleshooting

### "AccessDeniedException" error

**Problem**: Insufficient permissions to push to ECR.

**Solution**: Ensure your IAM user/role has the required ECR permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:PutImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:CreateRepository"
            ],
            "Resource": "*"
        }
    ]
}
```

### "Image not found" error

**Problem**: Docker image doesn't exist locally.

**Solution**: Build the image first:

```bash
easy_sm build
docker images | grep my-ml-app
```

### "RepositoryNotFoundException"

**Problem**: ECR repository doesn't exist and can't be auto-created.

**Solution**: Either:
1. Add `ecr:CreateRepository` permission
2. Manually create the repository:
   ```bash
   aws ecr create-repository --repository-name my-ml-app
   ```

### "Only one of iam-role-arn and aws-profile can be used"

**Problem**: Specified both `--iam-role-arn` and `--aws-profile`.

**Solution**: Choose one authentication method:

```bash
# Use profile
easy_sm push -p dev

# OR use role
easy_sm push -i arn:aws:iam::123456789012:role/MyRole
```

### Slow push times

**Problem**: Large image taking long to push.

**Solution**:
1. **Minimize image size**: Remove unnecessary dependencies
2. **Use layer caching**: Only changed layers are pushed on subsequent runs
3. **Compress**: Docker automatically compresses layers
4. **Check network**: Ensure stable internet connection

### "Token has expired" error

**Problem**: AWS credentials expired during push.

**Solution**: Refresh credentials:

```bash
# For SSO profiles
aws sso login --profile dev

# For temporary credentials
# Re-export fresh credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

### Wrong region

**Problem**: Image pushed to wrong region.

**Solution**: Specify the correct region:

```bash
easy_sm push -r us-east-1
```

Or update your config file:

```json
{
    "aws_region": "us-east-1"
}
```

## Cross-Account Access

To push to an ECR repository in a different AWS account:

1. **Create IAM role** in the target account with ECR permissions
2. **Add trust relationship** allowing your source account:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Principal": {
                   "AWS": "arn:aws:iam::SOURCE_ACCOUNT:user/USERNAME"
               },
               "Action": "sts:AssumeRole",
               "Condition": {
                   "StringEquals": {
                       "sts:ExternalId": "your-external-id"
                   }
               }
           }
       ]
   }
   ```
3. **Push with role and external ID**:
   ```bash
   easy_sm push \
     -i arn:aws:iam::TARGET_ACCOUNT:role/ECRPushRole \
     -e your-external-id
   ```

## Verifying the Push

After pushing, verify the image in ECR:

```bash
# List images in repository
aws ecr list-images \
  --repository-name my-ml-app \
  --profile dev

# Describe specific image
aws ecr describe-images \
  --repository-name my-ml-app \
  --image-ids imageTag=latest \
  --profile dev
```

## Image URI

After pushing, your image URI will be:

```
{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:{docker_tag}
```

For example:

```
123456789012.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:v1.2.0
```

This URI is used internally by easy_sm commands that reference the image.

## Related Commands

- [`build`](build.md) - Build the Docker image before pushing
- [`train`](train.md) - Train on SageMaker using the pushed image
- [`deploy`](deploy.md) - Deploy using the pushed image
- [`update-scripts`](update-scripts.md) - Update push script with latest version

## See Also

- [AWS ECR Documentation](https://docs.aws.amazon.com/ecr/)
- [Cloud Deployment Guide](../user-guide/cloud-deployment.md)
- [Authentication Setup](../getting-started/aws-setup.md)
