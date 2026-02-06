# Architecture

This document describes the internal architecture of easy_sm, including command structure, core modules, and design patterns.

## Command Structure

### Entry Point

**`easy_sm/__main__.py`**: Defines the main Typer app with `--docker-tag` option and registers all commands.

### Top-Level Commands

Cloud operations are at the top level (no prefix needed):

- `init`: Initialize new easy_sm projects
- `build`: Build Docker images
- `push`: Push Docker images to ECR
- `update-scripts`: Update shell scripts with latest secure versions
- `upload-data`: Upload data to S3
- `train`: Train models on SageMaker
- `deploy`: Deploy to provisioned endpoint
- `deploy-serverless`: Deploy to serverless endpoint
- `batch-transform`: Run batch predictions
- `process`: Run processing jobs
- `list-endpoints`: List all endpoints
- `list-training-jobs`: List recent training jobs (supports `-n` for names-only)
- `get-model-artifacts`: Get S3 model path from training job
- `delete-endpoint`: Delete an endpoint

### Sub-Commands

- **`local`**: Local operations (commands: `train`, `deploy`, `process`, `stop`)

## Core Modules

### Config System

**Location**: `easy_sm/config/config.py`

**Components**:

- **`Config`**: Data class holding configuration
  - `image_name`: Docker image name
  - `aws_profile`: AWS CLI profile
  - `aws_region`: AWS region
  - `python_version`: Python version for Docker
  - `easy_sm_module_dir`: Directory containing `easy_sm_base/`
  - `requirements_dir`: Path to requirements file

- **`ConfigManager`**: Loads/saves config from JSON file
  - Creates default config if file doesn't exist
  - Pattern: Commands load config via `ConfigManager(f"{app_name}.json").get_config()`

### SageMaker Integration

**Location**: `easy_sm/sagemaker/sagemaker.py`

**Components**:

- **`SageMakerClient`**: Wrapper around AWS SageMaker SDK and boto3
  - Handles S3 uploads
  - Manages training jobs
  - Manages endpoints
  - Runs processing jobs
  - Deploys models
  - Session management via boto3 and sagemaker SDK

### Command Helpers

**Location**: `easy_sm/commands/helpers.py`

**Functions**:

- **`safe_run_subprocess`**: Executes subprocess commands with error handling
- **`auto_detect_app_name`**: Finds `*.json` config file in current directory
- **`get_app_name`**: Gets app name from parameter or auto-detects
- **`get_iam_role`**: Gets IAM role from parameter or `SAGEMAKER_ROLE` env var
- **`load_config`**: Loads and validates configuration from JSON files

### Update Scripts

**Location**: `easy_sm/commands/update.py`

**Purpose**: Copies latest shell scripts from package template to app directory with security fixes (proper variable quoting).

Updates 7 shell scripts:
- Training entry points
- Serving entry points
- Local test scripts

### Templates

**Location**: `easy_sm/template/easy_sm_base/`

**Contents**:

- Dockerfile and scripts for containerized training/processing
- Training entry point: `training/train`
- Serving entry point: `prediction/serve`
- Local test scripts in `local_test/`

## Configuration Flow

1. Commands receive optional `app_name` and `iam_role_arn` parameters
2. Auto-detect app_name from `*.json` file if not provided
3. Read IAM role from `SAGEMAKER_ROLE` env var if not provided
4. Validate app_name (alphanumeric, hyphens, underscores only)
5. Load config from `{app_name}.json` in current directory
6. Config specifies Docker image name, AWS credentials, Python version, and module locations
7. Commands use config to build images, run jobs, or deploy endpoints

## Auto-Detection Behavior

### App Name

- Searches for `*.json` files in current directory
- Fails if none or multiple found
- Can override with `-a/--app-name` flag

### IAM Role

- Reads from `SAGEMAKER_ROLE` environment variable
- Fails if not set and not provided via `-r/--iam-role-arn` flag

### AWS Profile/Region

- From config file
- Uses boto3 default credential chain

## Output Design

Commands output clean, pipable data suitable for Unix-style composition.

### Command Outputs

| Command | Output Format |
|---------|---------------|
| `train` | S3 model path (`s3://bucket/path/model.tar.gz`) |
| `deploy` | Endpoint name |
| `upload-data` | S3 data path |
| `get-model-artifacts` | S3 model path |
| `list-training-jobs` | Job details or names-only with `-n` flag |
| `list-endpoints` | Endpoint details (name, status, timestamp) |
| `delete-endpoint` | Endpoint name |
| **Errors** | Go to stderr (via typer) |

### Unix-Style Composition

This design enables piping and command substitution:

```bash
# One-liner deployment
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

**Breakdown**:
1. `list-training-jobs -n -m 1` outputs latest job name
2. `get-model-artifacts -j <job>` outputs S3 model path
3. `deploy -m <path>` uses the model path

## Docker Context

### Docker Tag

- Passed via CLI flag `--docker-tag` (default: "latest")
- Accessible as `helpers.docker_tag` in commands
- Full image name: `{config.image_name}:{docker_tag}`

### Container Behavior

- Source code is mounted/copied into Docker containers
- Training: `/opt/ml/input/data/training`
- Model output: `/opt/ml/model`
- Hyperparameters: `/opt/ml/input/config/hyperparameters.json`

## Security Features

### App Name Validation

Prevents path traversal attacks:

```python
# Invalid names rejected:
# - ../../../etc/passwd
# - /absolute/path
# - ~user/path

# Valid names:
# - my-app
# - app_name
# - app-2024
```

### Shell Script Quoting

All variables properly quoted to prevent injection:

```bash
# Secure
docker run -v "${INPUT_DIR}:/opt/ml/input" ...

# Insecure (not used)
docker run -v $INPUT_DIR:/opt/ml/input ...
```

### File Permissions

Scripts set to `0o755` (not world-writable):

```python
os.chmod(script_path, 0o755)  # rwxr-xr-x
```

## Project Structure

```
easy_sm/
├── easy_sm/
│   ├── __main__.py           # CLI entry point, registers commands
│   ├── commands/             # Command implementations
│   │   ├── build.py          # Build Docker image
│   │   ├── cloud.py          # Cloud operations
│   │   ├── local.py          # Local operations
│   │   ├── initialize.py     # Initialize projects
│   │   ├── push.py           # Push images to ECR
│   │   ├── update.py         # Update shell scripts
│   │   └── helpers.py        # Utilities and shared state
│   ├── config/
│   │   └── config.py         # Config and ConfigManager
│   ├── sagemaker/
│   │   └── sagemaker.py      # SageMakerClient wrapper
│   └── template/
│       └── easy_sm_base/     # Docker template
├── tests/                    # Test suite (120 tests)
├── setup.py                  # Package metadata
└── base-requirements.txt     # Development dependencies
```

## Key Implementation Details

- All commands execute in the current working directory
- Projects identified by presence of `{app_name}.json`
- Docker images built locally and pushed to registries
- SageMaker operations require valid AWS credentials via configured profile
- Local training/processing uses Docker to simulate SageMaker environment
- Configuration persisted as JSON to maintain state across command invocations
- App names validated to prevent security issues

## Dependencies

### Runtime Dependencies

- **typer** (>=0.9.0): CLI framework
- **docker** (>=7.1.0): Docker SDK
- **sagemaker** (>=2.243.0): AWS SageMaker SDK
- **boto3** (>=1.26.0): AWS SDK

### Development Dependencies

- **pytest**: Test framework
- **mypy**: Type checker
- **ruff**: Linter and formatter
- **requests**: HTTP library
- **statsmodels, joblib, pandas**: Sample app dependencies

## See Also

- [Code Style Guidelines](code-style.md)
- [Testing Guide](testing.md)
- [Contributing Guide](contributing.md)
