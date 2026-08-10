# build

Build a Docker image for SageMaker training and deployment.

## Synopsis

```bash
easy_sm build [--app-name APP_NAME]
```

## Description

The `build` command creates a Docker image containing your machine learning code, dependencies, and the necessary entry points for SageMaker training and serving. The image is built locally and tagged for use with both local testing and cloud deployment.

The build process:

1. Validates the easy_sm directory structure
2. Sets executable permissions on entry point scripts
3. Builds a Docker image with your code and dependencies
4. Tags the image as `{image_name}:{docker_tag}`

## Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration. If not specified, auto-detects from `*.json` file in current directory |

## Examples

### Basic build

Build with auto-detected app name and default tag:

```bash
easy_sm build
```

### Build with specific app name

Override auto-detection:

```bash
easy_sm build -a my-ml-app
```

### Build with custom Docker tag

Configure the Docker tag in your `my-ml-app.json`:

```json
{
    "image_name": "my-ml-app",
    "docker_tag": "v1.2.0"
}
```

Then build:

```bash
easy_sm build
```

### Build for different environments

Create separate config files for each environment:

**my-ml-app-dev.json**:
```json
{
    "docker_tag": "dev"
}
```

**my-ml-app-prod.json**:
```json
{
    "docker_tag": "v1.0.0"
}
```

Then build for each environment:

```bash
# Development
easy_sm build -a my-ml-app-dev

# Production
easy_sm build -a my-ml-app-prod
```

## Output

During the build, you'll see:

```
Started building SageMaker Docker image. It will take some minutes...

[Docker build output...]

Docker image built successfully!
```

The command creates a Docker image named `{image_name}:{docker_tag}` that you can verify with:

```bash
docker images | grep my-ml-app
```

Expected output:

```
my-ml-app    latest    abc123def456    2 minutes ago    2.1GB
```

## Prerequisites

- Docker installed and running
- A valid easy_sm project (created with `easy_sm init`)
- Configuration file (`{app_name}.json`) in current directory
- Required project structure with `easy_sm_base/` directory

## Build Process Details

The build command:

1. **Validates structure**: Checks for required files:
   - `build.sh`
   - `training/train`
   - `prediction/serve`
   - `executor.sh`
   - `Dockerfile`

2. **Sets permissions**: Makes scripts executable (755):
   - `training/train`
   - `prediction/serve`
   - `executor.sh`

3. **Builds image**: Runs Docker build with:
   - Your source code
   - Python version from config
   - Dependencies from requirements.txt
   - SageMaker entry points

## What Gets Included

The Docker image contains:

- **Python runtime** (version specified in config)
- **Your code** from the easy_sm module directory
- **Dependencies** from requirements.txt
- **Training entry point** (`training/train`)
- **Serving entry point** (`prediction/serve`)
- **Processing scripts** from `processing/` directory
- **SageMaker libraries** (boto3, sagemaker, etc.)

## Dockerfile Customization

You can customize the build by editing `{app_name}/easy_sm_base/Dockerfile`:

```dockerfile
# Example: Add system packages
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Example: Set environment variables
ENV MODEL_PATH=/opt/ml/model

# Example: Add custom build steps
COPY custom_config/ /opt/config/
```

After customizing the Dockerfile, run `easy_sm build` to rebuild the image.

## Build Context

The build uses the following from your configuration:

- `image_name`: Base name for the Docker image
- `python_version`: Python runtime version (3.10, 3.11, 3.12, or 3.13)
- `easy_sm_module_dir`: Source code location
- `requirements_dir`: Path to requirements.txt

## Troubleshooting

### "This is not a easy_sm directory" error

**Problem**: Missing required files in the project structure.

**Solution**: Ensure you have:
- Initialized the project with `easy_sm init`
- Required files exist in `{app_name}/easy_sm_base/`

```bash
# Re-initialize if needed
easy_sm init

# Or update scripts
easy_sm update-scripts
```

### "No such file or directory" during build

**Problem**: requirements.txt not found at specified location.

**Solution**: Check the `requirements_dir` in your config file:

```json
{
    "requirements_dir": "requirements.txt"
}
```

Ensure the file exists:

```bash
ls -la requirements.txt
```

### Docker daemon not running

**Problem**: `Cannot connect to the Docker daemon`

**Solution**: Start Docker:

```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

### Permission denied errors

**Problem**: Permission errors when building.

**Solution**: Ensure Docker has proper permissions:

```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER

# Then logout and login again
```

### Build fails with "No space left on device"

**Problem**: Docker has filled up available disk space.

**Solution**: Clean up Docker:

```bash
# Remove unused images and containers
docker system prune -a

# Remove all images for your app
docker rmi $(docker images my-ml-app -q)
```

### Requirements installation fails

**Problem**: Dependencies fail to install during build.

**Solution**:
1. Test requirements locally first:
   ```bash
   pip install -r requirements.txt
   ```
2. Check for version conflicts or platform-specific packages
3. Pin versions in requirements.txt

## Build Caching

Docker caches build layers for faster rebuilds. To force a complete rebuild:

```bash
# Remove existing image
docker rmi my-ml-app:latest

# Rebuild
easy_sm build
```

Or use Docker's `--no-cache` by modifying the build script temporarily.

## Performance Tips

1. **Order dependencies**: Place stable dependencies before frequently-changing code in Dockerfile
2. **Use specific tags**: Helps identify and deploy specific versions
3. **Minimize image size**: Remove unnecessary packages and files
4. **Multi-stage builds**: Use multi-stage Dockerfiles for smaller production images

## Related Commands

- [`init`](init.md) - Initialize a new project
- [`push`](push.md) - Push the built image to ECR
- [`local train`](local.md#train) - Test the built image locally
- [`train`](train.md) - Train on SageMaker using the image
- [`update-scripts`](update-scripts.md) - Update build scripts

## See Also

- [Local Development Guide](../user-guide/local-development.md)
- [Docker Customization](../getting-started/configuration.md)
- [Project Structure](../getting-started/configuration.md)
