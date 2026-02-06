# Welcome to easy_sm

[![Release](https://github.com/prteek/easy_sm/actions/workflows/release.yml/badge.svg)](https://github.com/prteek/easy_sm/actions/workflows/release.yml)
[![Documentation](https://github.com/prteek/easy_sm/actions/workflows/docs.yml/badge.svg)](https://github.com/prteek/easy_sm/actions/workflows/docs.yml)
[![PyPI version](https://badge.fury.io/py/easy-sm.svg)](https://badge.fury.io/py/easy-sm)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python CLI tool that simplifies AWS SageMaker workflows by enabling rapid local prototyping with Docker before deploying to the cloud.

!!! tip "Credits"
    This project borrows heavily from [Sagify](https://github.com/Kenza-AI/sagify). Check it out especially if you want to work with LLMs on SageMaker.

!!! warning "Experimental Package"
    This is an experimental package. APIs may evolve between releases.

## Features

- **Local Development**: Train, process, and deploy models locally in Docker containers that mimic SageMaker
- **Cloud Deployment**: Deploy trained models to AWS SageMaker with minimal configuration changes
- **Docker Integration**: Automatically build and manage Docker images
- **Endpoint Management**: Deploy and manage SageMaker endpoints (provisioned and serverless)
- **Job Monitoring**: List and filter training jobs
- **Unix Philosophy**: Composable commands with clean, pipable output

## Quick Links

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Install easy_sm and run your first training job in 5 minutes

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)
    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Learn about local development, cloud deployment, and piped workflows

    [:octicons-arrow-right-24: User Guide](user-guide/overview.md)

-   :material-console:{ .lg .middle } __Command Reference__

    ---

    Detailed documentation for all CLI commands

    [:octicons-arrow-right-24: Commands](commands/index.md)

-   :material-code-tags:{ .lg .middle } __Developer Guide__

    ---

    Architecture, testing, and contribution guidelines

    [:octicons-arrow-right-24: Developer Guide](developer-guide/architecture.md)

</div>

## Requirements

- Python >=3.13
- Docker (for local development)
- AWS CLI configured with credentials

## Quick Example

```bash
# Initialize project
easy_sm init

# Build and test locally
easy_sm build
easy_sm local train

# Deploy to SageMaker
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
easy_sm push
easy_sm train -n job-name -e ml.m5.large \
  -i s3://bucket/input -o s3://bucket/output
```

## Design Philosophy

The CLI follows Unix philosophy:

- **Composable**: Commands output clean data for piping
- **Context-aware**: Auto-detects app name and IAM role from environment
- **Minimal flags**: Only essential options required
- **Pipe-friendly**: Output is data, not verbose messages

### Piped Workflows

```bash
# Get latest training job, extract model, and deploy in one line
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

## License

MIT License - see [License](about/license.md) for details.

## Author

Created by Prateek (prteek@icloud.com)
