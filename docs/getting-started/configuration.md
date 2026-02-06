# Configuration

easy_sm uses JSON configuration files and environment variables to manage project settings.

## Configuration File

Each project has a JSON configuration file named `{app-name}.json` in the project root.

### Example Configuration

```json
{
    "image_name": "my-ml-app",
    "aws_profile": "dev",
    "aws_region": "eu-west-1",
    "python_version": "3.13",
    "easy_sm_module_dir": "my-ml-app",
    "requirements_dir": "requirements.txt"
}
```

### Configuration Fields

| Field | Description | Example |
|-------|-------------|---------|
| `image_name` | Docker image name (used for ECR) | `my-ml-app` |
| `aws_profile` | AWS CLI profile name | `dev`, `prod` |
| `aws_region` | AWS region for SageMaker operations | `eu-west-1`, `us-east-1` |
| `python_version` | Python version for Docker image | `3.13`, `3.12` |
| `easy_sm_module_dir` | Directory containing `easy_sm_base/` | `my-ml-app` |
| `requirements_dir` | Path to requirements file | `requirements.txt` |

### Auto-Detection

Most commands auto-detect the configuration file:

```bash
# Automatically finds my-ml-app.json in current directory
easy_sm build
easy_sm train -n job-name -e ml.m5.large -i s3://... -o s3://...
```

You can override with the `-a/--app-name` flag:

```bash
easy_sm build -a my-ml-app
```

!!! warning "Multiple Config Files"
    If multiple `*.json` files exist in the current directory, easy_sm will fail. Either:

    - Remove extra JSON files
    - Use `-a` flag to specify which app to use

## Environment Variables

### SAGEMAKER_ROLE (Required)

The IAM role ARN for SageMaker operations:

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
```

!!! tip "Persist Across Sessions"
    Add to `~/.bashrc` or `~/.zshrc`:
    ```bash
    echo 'export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole' >> ~/.bashrc
    source ~/.bashrc
    ```

You can override with the `-r/--iam-role-arn` flag:

```bash
easy_sm train -r arn:aws:iam::123456789012:role/OtherRole ...
```

### AWS Credentials

easy_sm uses the standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS CLI credentials file (`~/.aws/credentials`)
3. IAM role (when running on EC2/ECS)

The `aws_profile` in the config file specifies which profile to use from `~/.aws/credentials`.

## Project Structure

After running `easy_sm init`, your project structure looks like:

```
my-project/
├── my-app.json                      # Configuration
├── requirements.txt                 # Python dependencies
└── my-app/                          # Module directory
    └── easy_sm_base/                # Template directory
        ├── Dockerfile               # Customize if needed
        ├── training/
        │   ├── train                # Entry point (shell script)
        │   └── training.py          # Your training code
        ├── prediction/
        │   └── serve                # Your serving code
        ├── processing/              # Processing scripts
        └── local_test/
            └── test_dir/            # Test data for local runs
                ├── input/           # Input data
                │   └── data/
                │       └── training/
                └── model/           # Model output
```

## Dockerfile Customization

The default Dockerfile is generated during `easy_sm init`. You can customize it for:

- Installing system dependencies
- Adding custom build steps
- Configuring environment variables

Example customization:

```dockerfile
FROM python:3.13

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# Copy code
COPY training /opt/program/training
COPY prediction /opt/program/prediction
COPY processing /opt/program/processing

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR /opt/program
```

!!! warning "Maintain Entry Points"
    Keep the original entry point scripts (`training/train`, `prediction/serve`) or SageMaker won't work correctly.

## Requirements File

The `requirements.txt` file lists Python dependencies:

```txt
scikit-learn>=1.3.0
pandas>=2.0.0
joblib>=1.3.0
numpy>=1.24.0
```

!!! tip "Pin Versions"
    Pin exact versions for reproducibility:
    ```txt
    scikit-learn==1.3.2
    pandas==2.1.4
    ```

## Docker Tags

Control Docker image versions with the `--docker-tag` flag:

```bash
# Build with custom tag
easy_sm --docker-tag v1.0 build

# Use tagged image for training
easy_sm --docker-tag v1.0 local train

# Push tagged image
easy_sm --docker-tag v1.0 push
```

Default tag is `latest`.

!!! tip "Versioning Strategy"
    Use semantic versioning for production:
    ```bash
    easy_sm --docker-tag v1.0.0 build
    easy_sm --docker-tag v1.0.0 push
    easy_sm --docker-tag v1.0.0 train -n prod-job-v1.0.0 ...
    ```

## Multiple Environments

Manage multiple environments (dev, staging, prod) with separate config files:

```
my-project/
├── my-app-dev.json      # Dev environment
├── my-app-staging.json  # Staging environment
├── my-app-prod.json     # Production environment
└── my-app/
    └── easy_sm_base/
```

Use the `-a` flag to select environment:

```bash
# Dev
easy_sm build -a my-app-dev
easy_sm train -a my-app-dev -n dev-job ...

# Production
easy_sm build -a my-app-prod
easy_sm train -a my-app-prod -n prod-job ...
```

## Configuration Validation

easy_sm validates configuration on each command:

- **App name**: Alphanumeric, hyphens, underscores only (prevents path traversal)
- **Config file**: Must exist and contain valid JSON
- **Required fields**: All fields must be present
- **IAM role**: Must be set via env var or `-r` flag

If validation fails, you'll see an error message:

```
Error: Configuration file 'my-app.json' not found
Error: Invalid app name: '../../../etc/passwd'
Error: SAGEMAKER_ROLE environment variable not set
```

## Next Steps

- Learn about [local development](../user-guide/local-development.md)
- Explore [cloud deployment](../user-guide/cloud-deployment.md)
- See [AWS setup requirements](../user-guide/aws-setup.md)
