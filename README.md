# easy_sm

A Python CLI tool that simplifies AWS SageMaker workflows by enabling rapid local prototyping with Docker before deploying to the cloud.

**easy_sm** makes it easy to work with SageMaker by enabling rapid prototyping with local training, processing, and deployment—and then seamlessly scaling to cloud execution on SageMaker.

> **Note**: This is an experimental package. APIs may evolve and break between releases. Please validate changes before updating.

## Features

- **Local Development**: Train and test models locally in Docker containers that mimic SageMaker environments
- **Cloud Deployment**: Deploy trained models to AWS SageMaker with minimal configuration changes
- **Docker Integration**: Automatically build and manage Docker images for your workflows
- **Training & Processing**: Support for training jobs and data processing pipelines
- **Easy Configuration**: Project-based config files (JSON) for managing credentials, AWS regions, and image names
- **Quick Prototyping**: Iterate rapidly locally before committing to cloud resources

## Requirements

- Python >=3.14
- Docker (for local development)
- AWS credentials configured (for cloud deployment)

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone <repository-url>
cd easy_sm

# Install in development mode
pip install -e .

# Install with development dependencies (for testing, linting, type checking)
pip install -e . -r requirements.txt
```

### From PyPI (When Available)

```bash
pip install easy_sm
```

## Quick Start

### 1. Initialize a New Project

```bash
easy_sm init --app-name myapp
```

This creates a `myapp.json` configuration file with default values. Edit this file to configure:
- `image_name`: Docker image name for your project
- `aws_profile`: AWS profile to use
- `aws_region`: AWS region for SageMaker
- `python_version`: Python version for your container
- `easy_sm_module_dir`: Directory containing your training code
- `requirements_dir`: Directory containing requirements files

### 2. Build Docker Image

```bash
easy_sm build --app-name myapp
```

This builds a Docker image locally using the configuration from `myapp.json`.

### 3. Train Locally

```bash
easy_sm local train --app-name myapp
```

Runs training in a Docker container that simulates the SageMaker environment.

### 4. Deploy to SageMaker

```bash
easy_sm cloud train --app-name myapp
```

Submits a training job to AWS SageMaker.

## Commands

### init
Initialize a new easy_sm project with configuration file.

```bash
easy_sm init --app-name myapp
```

### build
Build a Docker image for your project.

```bash
easy_sm build --app-name myapp [--docker-tag latest]
```

### local
Run local training, processing, or deployment in Docker.

```bash
easy_sm local train --app-name myapp
easy_sm local process --app-name myapp
easy_sm local deploy --app-name myapp
```

### cloud
Submit jobs to AWS SageMaker for training, processing, or deployment.

```bash
easy_sm cloud train --app-name myapp
easy_sm cloud process --app-name myapp
easy_sm cloud deploy --app-name myapp
```

### push
Push your Docker image to a registry.

```bash
easy_sm push --app-name myapp --registry-url <url>
```

### Global Options

- `--docker-tag` (default: `latest`): Specify the Docker image tag to use

## Project Structure

A typical easy_sm project looks like this:

```
my-project/
├── myapp.json              # Project configuration
├── my_module/              # Your training/processing code
│   ├── __init__.py
│   ├── train.py           # Training entry point
│   └── process.py         # Processing logic
├── requirements.txt        # Python dependencies
└── data/                   # Input data (optional)
```

## Architecture

### Configuration Flow

1. **Project Setup**: Create a project directory with `{app_name}.json` config
2. **Build**: Build Docker image using the config settings
3. **Local Testing**: Test training/processing locally in Docker
4. **Cloud Deployment**: Submit jobs to SageMaker with same code and configuration

### Docker Images

- Uses templates from `easy_sm/template/easy_sm_base/`
- Supports custom entry points for training (`training/train`) and serving (`prediction/serve`)
- Automatically mounts/copies your code into containers

### Configuration

Projects are configured via JSON files named `{app_name}.json`:

```json
{
  "image_name": "my-training-image",
  "aws_profile": "default",
  "aws_region": "us-west-2",
  "python_version": "3.14",
  "easy_sm_module_dir": "./my_module",
  "requirements_dir": "./requirements.txt"
}
```

## Development

See [CLAUDE.md](./CLAUDE.md) for detailed guidance on code style, testing, and development practices.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_build.py

# Run with coverage
pytest --cov=easy_sm
```

### Code Quality

```bash
# Type checking
mypy easy_sm/

# Linting
ruff check easy_sm/

# Formatting
ruff format easy_sm/
```

### Local Development Workflow

```bash
# Install in development mode
pip install -e .

# Make changes to code
# ... edit files ...

# Reinstall to pick up changes
pip install -e .

# Test your changes
easy_sm --help
```

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Claude Code guidance for developers
- **[AGENTS.md](./AGENTS.md)** - Detailed development guidelines and conventions
- **[LICENSE](./LICENSE)** - MIT License

## Contributing

This is an experimental project. If you'd like to contribute:

1. Follow the code style guidelines in [CLAUDE.md](./CLAUDE.md)
2. Add tests for new features
3. Run the test suite and linting before submitting changes
4. Update documentation as needed

## License

MIT License - See [LICENSE](./LICENSE) file for details

## Support & Issues

For bugs and feature requests, please use the GitHub Issues tracker.

## Author

Created by Prateek (prteek@icloud.com)

---

**Quick Links:**
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Click Documentation](https://click.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
