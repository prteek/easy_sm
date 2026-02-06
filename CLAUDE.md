# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**easy_sm** is a Python CLI tool (Python >=3.13) that simplifies AWS SageMaker workflows by enabling rapid local prototyping with Docker before deploying to the cloud. It's built on Typer and provides commands for building Docker images, training/processing locally and in the cloud, and managing deployments.

### Design Philosophy

The CLI follows Unix philosophy:
- **Composable**: Commands output clean data for piping
- **Context-aware**: Auto-detects app name and IAM role from environment
- **Minimal flags**: Only essential options required
- **Pipe-friendly**: Output is data, not verbose messages

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
pytest tests/test_update_command.py
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

**Total: 120 tests** covering all commands and core modules.

#### Command Tests
- **test_init_command.py** (7 tests): Project initialization with various configurations
- **test_build_command.py** (13 tests): Docker image building with parameter variations and error scenarios
- **test_local_commands.py** (23 tests): Local training, deployment, processing, and stop commands
- **test_cloud_commands.py** (28 tests): SageMaker operations (train, deploy, batch-transform, process, list-endpoints, list-training-jobs, get-model-artifacts, delete-endpoint)
- **test_push_command.py** (9 tests): ECR image push with IAM/profile authentication
- **test_update_command.py** (7 tests): Shell script update command with security fixes

#### Module Tests
- **test_config.py** (16 tests): Configuration loading, saving, serialization, and error handling
- **test_helpers.py** (17 tests): Subprocess execution, output handling, and error propagation

All tests use mocked external dependencies (subprocess, boto3, SageMaker SDK) for fast, reliable execution without requiring AWS credentials or Docker.

## Architecture

### Command Structure
- **Entry point**: `easy_sm/__main__.py` - Defines the main Typer app with `--docker-tag` option and registers all commands
- **Top-level commands** (cloud operations - no prefix needed):
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
- **Sub-commands**:
  - `local`: Local operations (commands: `train`, `deploy`, `process`, `stop`)

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
- `auto_detect_app_name`: Finds `*.json` config file in current directory
- `get_app_name`: Gets app name from parameter or auto-detects
- `get_iam_role`: Gets IAM role from parameter or `SAGEMAKER_ROLE` env var
- `load_config`: Loads and validates configuration from JSON files (supports auto-detection)

**Update Scripts** (`easy_sm/commands/update.py`):
- `update_scripts`: Copies latest shell scripts from package template to app directory
- Updates 7 shell scripts with security fixes (proper variable quoting)

**Templates** (`easy_sm/template/easy_sm_base/`):
- Dockerfile and scripts for containerized training/processing
- Training entry point: `training/train`
- Serving entry point: `prediction/serve`
- Local test scripts in `local_test/`

### Configuration Flow
1. Commands receive optional `app_name` and `iam_role_arn` parameters
2. Auto-detect app_name from `*.json` file if not provided
3. Read IAM role from `SAGEMAKER_ROLE` env var if not provided
4. Validate app_name (alphanumeric, hyphens, underscores only)
5. Load config from `{app_name}.json` in current directory
6. Config specifies Docker image name, AWS credentials, Python version, and module locations
7. Commands use config to build images, run jobs, or deploy endpoints

### Auto-Detection Behavior
- **App name**: Searches for `*.json` files in current directory. Fails if none or multiple found (can override with `-a`)
- **IAM role**: Reads from `SAGEMAKER_ROLE` environment variable. Fails if not set and not provided via `-r`
- **AWS profile/region**: From config file, uses boto3 default credential chain

### Output Design
Commands output clean, pipable data:
- **train**: S3 model path (`s3://bucket/path/model.tar.gz`)
- **deploy**: Endpoint name
- **upload-data**: S3 data path
- **get-model-artifacts**: S3 model path
- **list-training-jobs**: Job details or names-only with `-n` flag
- **list-endpoints**: Endpoint details (name, status, timestamp)
- **delete-endpoint**: Endpoint name
- **Errors**: Go to stderr (via typer)

This enables Unix-style composition:
```bash
easy_sm deploy -n my-endpoint -e ml.m5.large \
  -m $(easy_sm get-model-artifacts -j $(easy_sm list-training-jobs -n -m 1))
```

### Docker Context
- Docker tag passed via CLI flag `--docker-tag` (default: "latest"), accessible as `helpers.docker_tag`
- Full image name: `{config.image_name}:{docker_tag}`
- Source code is mounted/copied into Docker containers for training/processing

### Security Features
- **App name validation**: Prevents path traversal attacks (e.g., `../../../etc/passwd`)
- **Shell script quoting**: All variables properly quoted to prevent injection
- **File permissions**: Scripts set to 0o755 (not world-writable)

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
from easy_sm.commands.helpers import get_app_name, get_iam_role, load_config

# Get app name (from parameter or auto-detect)
app_name = get_app_name(app_name)

# Get IAM role (from parameter or SAGEMAKER_ROLE env var)
iam_role = get_iam_role(iam_role_arn)

# Load config
config = load_config(app_name)
```

### Typer Command with Optional Context
```python
from typing import Annotated, Optional
import typer
from easy_sm.commands import helpers
from easy_sm.commands.helpers import get_app_name, get_iam_role

@app.command()
def subcommand(
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
) -> None:
    """Command description."""
    app_name = get_app_name(app_name)
    iam_role = get_iam_role(iam_role_arn)
    docker_tag = helpers.docker_tag
    # Implementation
```

### Pipe-Friendly Output
```python
# Good: Output just the data
print(s3_path)
print(endpoint_name)

# Bad: Verbose messages
print(f"Model uploaded to {s3_path}")
print(f"Endpoint: {endpoint_name}")
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

### Validate Docs (`/validate-docs`)
**Location**: `.claude/skills/validate-docs/SKILL.md`

Validates all documentation against actual command implementations. Use this skill:
- After editing any documentation files
- Before committing documentation changes
- When adding new command parameters
- When updating command examples or workflows

The validation checks:
1. **Parameter Validation**: Extracts parameters from Python command files and compares against documented parameters
2. **Discrepancy Detection**: Reports flags documented but not implemented, wrong parameter names, unsupported features
3. **Example Validation**: Analyzes command examples for common mistakes:
   - Using `-n` flag with date filtering (won't work - names-only output lacks timestamps)
   - Incorrect `awk` field extraction for job names (should use `$1`, not `$2`)
   - Incomplete variable assignments missing field extraction

Invoke with: `/validate-docs`

Returns:
- Exit code `0` if all documentation is valid
- Exit code `1` if issues found (displayed with specific descriptions)

## Continuous Integration and Deployment

### GitHub Actions Workflows

**docs.yml**: Documentation Deployment
- **Trigger**: Push to main (when `docs/`, `mkdocs.yml`, or source docs files change)
- **Process**:
  1. Checkout repository with full history
  2. Setup Python 3.13
  3. Install documentation dependencies (`requirements-docs.txt`)
  4. Build MkDocs site with `mkdocs build --strict`
  5. Deploy to GitHub Pages using GitHub Actions (`configure-pages`, `upload-pages-artifact`, `deploy-pages`)
- **Result**: Documentation automatically available at https://prteek.github.io/easy_sm/
- **MkDocs Configuration**: Material theme, search, code copy buttons, dark mode toggle, git revision dates

**release.yml**: PyPI Release and Testing
- **Trigger**: Tag creation (e.g., `git tag v1.0.0`) or manual `workflow_dispatch`
- **Process**:
  1. Run full test suite (`pytest` all 120 tests)
  2. Run type checking (`mypy`)
  3. Run linting (`ruff check`)
  4. Build package distribution
  5. Publish to PyPI
- **Result**: New version available on PyPI after tests pass

### Pre-commit Hook

**Location**: `.githooks/pre-commit`
- Validates documentation against code implementation before commit
- Runs `python3 scripts/validate_docs.py`
- Blocks commit if validation fails
- Helps prevent documentation drift

## Claude Code Permissions

**Location**: `.claude/settings.local.json`

Configured to allow specific bash commands for development:
- `pytest:*` - Run test suite
- `pip install:*` - Install dependencies
- `pyenv versions:*` - Check Python versions
- `python3:*` - Run Python scripts
- `git add:*`, `git commit:*`, `git push:*` - Git operations
- `mypy:*` - Type checking
- `ruff check:*` - Linting
- `easy_sm:*` - Run CLI tool
- `mkdocs build:*` - Build documentation
- `WebFetch` and `WebSearch` - Web access for documentation and research

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

### Documentation Dependencies (from requirements-docs.txt)
- **mkdocs** (>=1.5.3): Static documentation generator
- **mkdocs-material** (>=9.5.0): Material Design theme
- **mkdocs-minify-plugin** (>=0.7.1): Asset minification
- **mkdocs-git-revision-date-localized-plugin** (>=1.2.0): Show last git commit date
- **pymdown-extensions** (>=10.7): Extended markdown features (admonitions, code tabs, emoji)

## Project Structure Reference

```
easy_sm/
├── .claude/                  # Claude Code configuration
│   ├── settings.local.json   # Claude Code permissions and settings
│   └── skills/
│       ├── refresh/          # Documentation refresh skill
│       │   └── SKILL.md      # Refresh skill documentation
│       └── validate-docs/    # Documentation validation skill
│           └── SKILL.md      # Validate-docs skill documentation
├── scripts/
│   └── validate_docs.py      # Documentation validation script (AST + regex)
├── easy_sm/
│   ├── __main__.py           # CLI entry point (Typer app), registers commands
│   ├── commands/             # Command implementations
│   │   ├── build.py          # Build Docker image
│   │   ├── cloud.py          # Cloud training/deployment/processing/endpoint management
│   │   ├── local.py          # Local training/deployment/processing
│   │   ├── initialize.py     # Initialize projects
│   │   ├── push.py           # Push images to ECR
│   │   ├── update.py         # Update shell scripts with security fixes
│   │   └── helpers.py        # Subprocess utilities, validation, and shared state
│   ├── config/
│   │   └── config.py         # Config and ConfigManager classes
│   ├── sagemaker/
│   │   └── sagemaker.py      # SageMakerClient wrapper
│   └── template/
│       └── easy_sm_base/     # Docker template and entry points
├── tests/                    # Test suite (uses pytest) - 120 tests total
│   ├── test_build_command.py         # Tests for build command (13 tests)
│   ├── test_init_command.py          # Tests for init command (7 tests)
│   ├── test_local_commands.py        # Tests for local commands (23 tests)
│   ├── test_cloud_commands.py        # Tests for cloud SageMaker operations (28 tests)
│   ├── test_push_command.py          # Tests for ECR push command (9 tests)
│   ├── test_update_command.py        # Tests for update-scripts command (7 tests)
│   ├── test_config.py                # Tests for Config/ConfigManager (16 tests)
│   ├── test_helpers.py               # Tests for subprocess utilities (17 tests)
│   └── LOCAL_COMMANDS_TESTS_README.md # Documentation for local command tests
├── .github/
│   ├── README.md              # Usage guide and command examples
│   └── workflows/             # CI/CD workflows
├── setup.py                   # Package metadata and dependencies
├── base-requirements.txt      # Development dependencies (pytest, mypy, ruff, etc)
├── CLAUDE.md                  # Claude Code guidance (this file)
├── AGENTS.md                  # Development guidelines (detailed style guide)
├── DOCS_WORKFLOW.md           # Documentation workflow and validation guide
├── README.md                  # Project overview
├── .githooks/
│   └── pre-commit             # Git pre-commit hook for doc validation
└── .github/
    ├── README.md              # Usage guide and command examples
    └── workflows/
        ├── docs.yml           # GitHub Actions for documentation deployment (MkDocs + GitHub Pages)
        └── release.yml        # GitHub Actions for PyPI release and testing
```

## Adding New Commands

1. Create `{command_name}.py` in `easy_sm/commands/`
2. Define Typer command using `@app.command()` decorator with `Annotated` type hints
3. Import and register in `easy_sm/__main__.py` via `app.command()` or `app.add_typer()`
4. Follow existing patterns for config loading and subprocess calls

## Key Implementation Details

- All commands execute in the current working directory; projects identified by presence of `{app_name}.json`
- Docker images are built locally and can be pushed to registries
- SageMaker operations require valid AWS credentials via configured profile
- Local training/processing uses Docker to simulate SageMaker container environment
- Configuration is persisted as JSON to maintain state across command invocations
- App names are validated to prevent security issues (path traversal, injection)

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

## Documentation Agent Workflow

**When working on documentation**, always follow this process:

1. **Before Editing**:
   - Understand what docs need changes
   - Plan the changes clearly
   - Consider impact on other docs

2. **While Editing**:
   - Update all related sections
   - Keep terminology consistent
   - Cross-reference related docs

3. **After Editing** (MANDATORY):
   - Run validation: `/validate-docs`
   - Fix any issues found
   - Verify all parameters match code
   - Check for deprecated feature references
   - Never commit docs without passing validation

4. **Validation Checks**:
   - Are all documented flags in the actual code?
   - Are deprecated features removed?
   - Do parameter names match implementation?
   - Are examples correct and tested?

### Documentation Validation Tool

**Script**: `scripts/validate_docs.py`
- Uses Python AST parsing to extract parameters from command implementations
- Uses regex to extract documented parameters from markdown files
- Analyzes bash code blocks in examples for common mistakes
- Returns exit code 0 if valid, 1 if issues found

Run manually:
```bash
python3 scripts/validate_docs.py
```

Or use the skill:
```bash
/validate-docs
```

Automatic (on commit):
- Pre-commit hook (`.githooks/pre-commit`) validates docs files automatically
- Blocks commit if validation fails
- Shows specific issues found with descriptions

### Common Documentation Issues

**Parameter Validation Issues**:
1. Flags documented but not implemented (e.g., `--tags`, `--spot-instances`)
2. Wrong parameter names (e.g., `--num-instances` vs `-c, --instance-count`)
3. Unsupported features mentioned (hyperparameters, spot instances, VPC settings, etc.)
4. Incorrect defaults or option values
5. Missing required options

**Example Validation Issues**:
1. Using `-n` flag with date filtering: `easy_sm list-training-jobs -n | grep $DATE` won't work because -n outputs only job names without timestamps
2. Incorrect `awk` field extraction: Using `$2` instead of `$1` to extract job names from command output
3. Incomplete variable assignments: Assigning full output without extracting specific fields needed for piping

**Example**: If docs say `--tags` but code doesn't have it → validation fails

### Documentation Files to Validate

- `docs/commands/*.md` - Command reference pages
- `docs/user-guide/*.md` - User guides
- `docs/examples/*.md` - Examples and workflows
- Against: `easy_sm/commands/*.py` - Actual implementation

### When Adding New Parameters

1. Add parameter to command code
2. Document in relevant markdown files
3. Run validation to confirm match
4. Test examples with correct output format
5. Commit only after validation passes

### Agent Instructions for Docs Work

When Claude is assigned documentation work:

**ALWAYS**:
✅ Run `/validate-docs` after making any doc changes
✅ Fix all validation issues before committing
✅ Test examples against actual code
✅ Verify command output format (fields, flags, timestamps)
✅ Update all related documentation sections
✅ Cross-check against implementation files

**NEVER**:
❌ Commit docs without validation
❌ Document features that don't exist in code
❌ Use outdated parameter names
❌ Reference removed/unsupported features
❌ Skip the validation check
❌ Use example patterns that don't work with actual command output
