# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**easy_sm** is a Python CLI tool (Python >=3.13) that simplifies AWS SageMaker workflows by enabling rapid local prototyping with Docker before deploying to the cloud. It's built on Typer and provides commands for building Docker images, training/processing locally and in the cloud, and managing deployments.

## Build and Development Commands

```bash
# Install in development mode (allows code changes without reinstalling)
pip install -e .

# Full installation with development dependencies
pip install -e . -r base-requirements.txt

# Build the package
python setup.py build

# Run CLI
easy_sm --help
```

## Testing and Code Quality

Testing is configured with pytest. Comprehensive test suites exist for all commands and core modules. All test dependencies are in `base-requirements.txt`.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_build_command.py
pytest tests/test_cloud_commands.py
pytest tests/test_init_command.py
pytest tests/test_local_commands.py
pytest tests/test_push_command.py
pytest tests/test_config.py
pytest tests/test_helpers.py

# Run specific test function
pytest tests/test_filename.py::test_function_name

# Run with coverage report
pytest --cov=easy_sm

# Type checking (mypy is configured)
mypy easy_sm/

# Code linting and formatting (ruff is configured and cached)
ruff check easy_sm/
ruff format easy_sm/
```

### Test Suite Overview

**Total: 98 tests** covering all commands and core modules.

#### Command Tests
- **test_init_command.py** (7 tests): Project initialization with various configurations
- **test_build_command.py** (13 tests): Docker image building with parameter variations and error scenarios
- **test_local_commands.py** (16 tests): Local training and deployment with Docker simulation
- **test_cloud_commands.py** (20 tests): SageMaker operations (train, deploy, batch-transform, process, etc.)
- **test_push_command.py** (9 tests): ECR image push with IAM/profile authentication

#### Module Tests
- **test_config.py** (16 tests): Configuration loading, saving, serialization, and error handling
- **test_helpers.py** (17 tests): Subprocess execution, output handling, and error propagation

All tests use mocked external dependencies (subprocess, boto3, SageMaker SDK) for fast, reliable execution without requiring AWS credentials or Docker.

## Architecture

### Command Structure
- **Entry point**: `easy_sm/__main__.py` - Defines the main CLI group with `--docker-tag` option and registers all command groups
- **Command groups**:
  - `init`: Initialize new easy_sm projects
  - `build`: Build Docker images
  - `local`: Local training/deployment/processing (commands: `train`, `deploy`, `process`)
  - `cloud`: Cloud training/deployment/processing on SageMaker (commands: `train`, `deploy`, `process`)
  - `push`: Push Docker images to registry
  - `deploy`: Deploy to SageMaker endpoints

### Core Modules

**Config System** (`easy_sm/config/config.py`):
- `Config`: Data class holding configuration (image_name, aws_profile, aws_region, python_version, easy_sm_module_dir, requirements_dir)
- `ConfigManager`: Loads/saves config from JSON file (`{app_name}.json`). Creates default config if file doesn't exist.
- Pattern: Commands load config via `ConfigManager(f"{app_name}.json").get_config()`

**SageMaker Integration** (`easy_sm/sagemaker/sagemaker.py`):
- `SageMakerClient`: Wrapper around AWS SageMaker SDK and boto3
- Handles: S3 uploads, training jobs, endpoint management, processing jobs, model deployment
- Session management via boto3 and sagemaker SDK

**Command Helpers** (`easy_sm/commands/helpers.py`):
- `safe_run_subprocess`: Executes subprocess commands with error handling

**Templates** (`easy_sm/template/easy_sm_base/`):
- Dockerfile and scripts for containerized training/processing
- Training entry point: `training/train`
- Serving entry point: `prediction/serve`
- Local test scripts in `local_test/`

### Configuration Flow
1. Commands receive `app_name` parameter
2. Load config from `{app_name}.json` in current directory (fails if not in valid easy_sm directory)
3. Config specifies Docker image name, AWS credentials, Python version, and module locations
4. Commands use config to build images, run jobs, or deploy endpoints

### Docker Context
- Docker tag passed via CLI flag `--docker-tag` (default: "latest"), accessible as `ctx.obj['docker_tag']`
- Full image name: `{config.image_name}:{docker_tag}`
- Source code is mounted/copied into Docker containers for training/processing

## Code Style Guidelines

### Naming Conventions
- **Classes**: PascalCase (e.g., `ConfigManager`, `SageMakerClient`)
- **Functions/Methods**: snake_case
- **Private methods/variables**: Leading underscore prefix (e.g., `_config`, `_build_image`)
- **Constants**: UPPER_SNAKE_CASE

### Type Hints
- Use type hints for all function parameters and return values
- Example: `def process_data(input_path: str, output_path: str) -> bool:`

### String Formatting
- Prefer f-strings for new code
- Accept `.format()` in existing code for consistency

### Typer Commands
- Use `Annotated[type, typer.Option(...)]` for option parameters
- Always include help text for options
- Access global `docker_tag` via `helpers.docker_tag`
- Use `typer.Typer()` for sub-apps and register with `app.add_typer(sub_app, name="...")`

### Error Handling
- Use `ValueError` for validation errors
- Use `typer.BadParameter` for CLI-specific errors
- Print error messages to stdout before `sys.exit()`
- Use try/except for subprocess operations

### Imports
- Organize: standard library → third-party → local imports
- Sort alphabetically within each section
- Example:
  ```python
  import os
  from typing import Annotated, Optional

  import typer

  from easy_sm.commands import helpers
  from easy_sm.commands.helpers import load_config
  ```

## Common Patterns

### Loading Configuration in Commands
```python
from easy_sm.commands.helpers import load_config

config = load_config(app_name)
```

### Typer Command with Options
```python
from typing import Annotated
import typer
from easy_sm.commands import helpers

@app.command()
def subcommand(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name")],
) -> None:
    """Command description."""
    docker_tag = helpers.docker_tag
    # Implementation
```

## Available Skills

### Refresh (`/refresh`)
**Location**: `.claude/skills/refresh/SKILL.md`

Automatically audits and updates project documentation to reflect current state. Use this skill when:
- Adding new agents, skills, or integrations
- Making significant changes to project structure
- Updating development practices or conventions
- Tracking changes to test files or validation processes

The refresh skill performs:
1. Audits active agents, skills, and integrations
2. Reviews and updates CLAUDE.md with current configurations
3. Verifies and updates cross-references in documentation
4. Creates a git commit with documentation changes

Invoke with: `/refresh`

## External Integrations

### OpenCode Integration
**Location**: `.opencode/`

Provides external validation and code review capabilities:

#### Code Reviewer Agent
**Location**: `.opencode/agents/code-reviewer.md`

- **Purpose**: Reviews code for quality and best practices
- **Model**: OpenCode Qwen3-Coder
- **Capabilities**:
  - Code quality analysis
  - Best practices validation
  - Potential bugs and edge case detection
  - Performance implications assessment
  - Security considerations review
- **Mode**: Read-only (no direct code modifications)

#### Validate Command
**Location**: `.opencode/commands/validate.md`

- **Purpose**: Validates easy_sm CLI changes using the sample app
- **Sample App**: Located in `app/` directory (not committed to git)
- **Testing Flow**:
  1. Uses existing sample app with `mpg.csv` dataset
  2. Sample training/serving files already configured
  3. Install changes locally: `pip install -e .`
  4. Run easy_sm commands to validate functionality
  5. Supports testing of: local train, local deploy, cloud train, cloud deploy
- **AWS Resources** (for cloud testing):
  - Training data: `s3://easy-sm/train/data`
  - Sample model: `s3://easy-sm/train/job-artefacts/mpg-2024-07-16-21-23-11-417/output/model.tar.gz`
  - Model outputs: `s3://easy-sm/train/job-artefacts`
- **Environment Variables**: `SAGEMAKER_EXECUTION_ROLE`

## Claude Code Permissions

**Location**: `.claude/settings.local.json`

Configured to allow specific bash commands for development:
- `pytest:*` - Run test suite
- `pip install:*` - Install dependencies
- `pyenv versions:*` - Check Python versions
- `PYENV_VERSION=3.12.11 python -m pip install:*` - Install with specific Python version

## Dependencies

### Runtime Dependencies (from setup.py)
- **typer** (>=0.9.0): CLI framework (built on Click)
- **docker** (>=7.1.0): Docker SDK for building/pushing images
- **sagemaker** (>=2.243.0): AWS SageMaker SDK
- **boto3** (>=1.26.0): AWS SDK

### Development Dependencies (from base-requirements.txt)
- **pytest**: Test framework
- **requests**: HTTP library (for integration tests)
- **mypy**: Type checker
- **ruff**: Linter and code formatter
- **statsmodels, joblib, pandas**: Sample app dependencies

## Project Structure Reference

```
easy_sm/
├── .claude/                  # Claude Code configuration
│   └── skills/
│       └── refresh/          # Documentation refresh skill
│           └── SKILL.md      # Refresh skill documentation
├── easy_sm/
│   ├── __main__.py           # CLI entry point, registers commands
│   ├── commands/             # Command implementations
│   │   ├── build.py          # Build Docker image
│   │   ├── cloud.py          # Cloud training/deployment/processing
│   │   ├── local.py          # Local training/deployment/processing
│   │   ├── initialize.py     # Initialize projects
│   │   ├── push.py           # Push images to registry
│   │   ├── deploy.py         # Deploy to SageMaker
│   │   └── helpers.py        # Subprocess utilities
│   ├── config/
│   │   └── config.py         # Config and ConfigManager classes
│   ├── sagemaker/
│   │   └── sagemaker.py      # SageMakerClient wrapper
│   └── template/
│       └── easy_sm_base/     # Docker template and entry points
├── tests/                    # Test suite (uses pytest) - 98 tests total
│   ├── test_build_command.py         # Tests for build command (13 tests)
│   ├── test_init_command.py          # Tests for init command (7 tests)
│   ├── test_local_commands.py        # Tests for local training/deployment/processing (16 tests)
│   ├── test_cloud_commands.py        # Tests for cloud SageMaker operations (20 tests)
│   ├── test_push_command.py          # Tests for ECR push command (9 tests)
│   ├── test_config.py                # Tests for Config/ConfigManager (16 tests)
│   ├── test_helpers.py               # Tests for subprocess utilities (17 tests)
│   └── LOCAL_COMMANDS_TESTS_README.md # Documentation for local command tests
├── .github/
│   ├── README.md              # Usage guide and command examples
│   └── workflows/             # CI/CD workflows
├── .opencode/                 # OpenCode integration (external validators)
│   ├── agents/
│   │   └── code-reviewer.md   # Code review agent configuration
│   └── commands/
│       └── validate.md        # Validation command using sample app
├── setup.py                   # Package metadata and dependencies
├── base-requirements.txt      # Development dependencies (pytest, mypy, ruff, etc)
├── CLAUDE.md                  # Claude Code guidance (this file)
├── AGENTS.md                  # Development guidelines (detailed style guide)
└── README.md                  # Project overview
```

## Adding New Commands

1. Create `{command_name}.py` in `easy_sm/commands/`
2. Define Click command/group with appropriate decorators
3. Import and register in `easy_sm/__main__.py` via `cli.add_command()`
4. Follow existing patterns for config loading and subprocess calls

## Key Implementation Details

- All commands execute in the current working directory; projects identified by presence of `{app_name}.json`
- Docker images are built locally and can be pushed to registries
- SageMaker operations require valid AWS credentials via configured profile
- Local training/processing uses Docker to simulate SageMaker container environment
- Configuration is persisted as JSON to maintain state across command invocations

## Git Configuration

**Commit Style**:
- Do NOT include *Co-Authored-By* trailers in commit messages
- Use clear, concise commit messages describing the change purpose
- Organize commits by type: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:` as appropriate
- Reference issue numbers when applicable

**Tracked vs Ignored Files**:
- ✅ **Tracked**: `setup.py`, `base-requirements.txt`, all source code, tests, documentation
- ❌ **Ignored**: Credentials (`.local_credentials`, `*credentials`), build artifacts (`build/`, `*.egg-info/`), cache files (`__pycache__/`), sample app (`app/`), test data (`*.csv`)

## BEFORE WRITING CODE

Explain what you're about to do and why
Break it down into steps I can follow
Wait for my OK before proceeding

## AFTER WRITING CODE

Explain what each part does
Ask me 3 questions to verify I understood
If I answer wrong, explain again until I get it
Do NOT let me commit until I pass your questions

## GENERAL RULES

Never generate code I can't explain
If I ask for something complex, suggest simpler alternatives
Treat every session as a teaching opportunity
Be direct. Tell me when I'm doing something wrong