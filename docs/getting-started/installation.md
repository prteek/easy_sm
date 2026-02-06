# Installation

This guide covers installing easy_sm from PyPI or from source.

## Requirements

Before installing easy_sm, ensure you have:

- **Python** >=3.13
- **Docker** (for local development)
- **AWS CLI** configured with credentials

## Install from PyPI

The simplest way to install easy_sm is via pip:

```bash
pip install easy-sm
```

Verify the installation:

```bash
easy_sm --help
```

## Install from Source

For development or to use the latest features:

### 1. Clone the Repository

```bash
git clone https://github.com/prteek/easy_sm.git
cd easy_sm
```

### 2. Install in Development Mode

This allows code changes without reinstalling:

```bash
pip install -e .
```

### 3. Install Development Dependencies

For testing and code quality tools:

```bash
pip install -e . -r base-requirements.txt
```

This installs additional tools:

- pytest (testing framework)
- mypy (type checking)
- ruff (linting and formatting)
- requests (HTTP library)
- Development dependencies for sample apps

## Verify Installation

Check that easy_sm is installed correctly:

```bash
easy_sm --version
easy_sm --help
```

## Docker Setup

easy_sm requires Docker for local development:

### macOS/Windows

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Linux

Install Docker Engine:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

Add your user to the docker group (optional, to avoid sudo):

```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

Verify Docker is running:

```bash
docker --version
docker ps
```

## AWS CLI Setup

easy_sm requires AWS CLI for cloud operations:

### Install AWS CLI

=== "macOS"

    ```bash
    brew install awscli
    ```

=== "Linux"

    ```bash
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    ```

=== "Windows"

    Download and run the [AWS CLI installer](https://aws.amazon.com/cli/)

### Configure AWS Credentials

Configure your AWS profile:

```bash
aws configure --profile dev
```

Enter your credentials:

```
AWS Access Key ID: YOUR_ACCESS_KEY
AWS Secret Access Key: YOUR_SECRET_KEY
Default region name: eu-west-1
Default output format: json
```

Verify configuration:

```bash
aws s3 ls --profile dev
```

## Next Steps

Once installed, proceed to the [Quick Start](quick-start.md) guide to create your first project.

## Troubleshooting

### Python Version Issues

easy_sm requires Python 3.13+. Check your version:

```bash
python --version
```

If you have an older version, install Python 3.13 using:

- [pyenv](https://github.com/pyenv/pyenv) (recommended)
- [Official Python downloads](https://www.python.org/downloads/)

### Docker Permission Denied

If you see "permission denied" errors with Docker:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker ps
```

### AWS Credentials Not Found

If AWS commands fail:

1. Check credentials file exists: `cat ~/.aws/credentials`
2. Verify profile configuration: `aws configure list --profile dev`
3. Test credentials: `aws sts get-caller-identity --profile dev`

For more help, see the [AWS Setup](../user-guide/aws-setup.md) guide.
