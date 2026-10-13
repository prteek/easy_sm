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

**Total: 157 tests** covering all commands and core modules.

#### Command Tests
- **test_init_command.py** (7 tests): Project initialization with various configurations
- **test_build_command.py** (13 tests): Docker image building with parameter variations and error scenarios
- **test_local_commands.py** (33 tests): Local train, deploy, process, run and stop commands (incl. outdated-script guards and exit-code propagation)
- **test_cloud_commands.py** (40 tests): SageMaker operations (train, deploy, batch-transform, process, run, list-endpoints, list-training-jobs, get-model-artifacts, delete-endpoint) and `SageMakerClient.run`
- **test_push_command.py** (9 tests): ECR image push with IAM/profile authentication
- **test_update_command.py** (7 tests): Shell script update command with security fixes

#### Module Tests
- **test_config.py** (16 tests): Configuration loading, saving, serialization, and error handling
- **test_helpers.py** (23 tests): Subprocess execution, output handling, error propagation, and `parse_env_vars`

#### Shell Script Tests
- **test_local_scripts.py** (9 tests): Runs the real template scripts (`process_local.sh`, `deploy_local.sh`, `run_local.sh`, `executor.sh`) with a fake `docker` on PATH that records its arguments. Guards against env-var flag injection, checks localhost-only port binding, and verifies argument passthrough and exit codes.

Python tests mock external dependencies (subprocess, boto3, SageMaker SDK), so no AWS credentials or Docker daemon are needed. The shell script tests need only `sh` and `bash`.

## Architecture

### Command Structure
- **Entry point**: `easy_sm/__main__.py` - Defines the main Typer app and registers all commands (there is no global CLI option)
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
  - `run`: Run any CLI command (e.g. `make`, `dbt`) as a processing job (`easy_sm run ... -- make target`)
  - `list-endpoints`: List all endpoints
  - `list-training-jobs`: List recent training jobs (supports `-n` for names-only)
  - `get-model-artifacts`: Get S3 model path from training job
  - `delete-endpoint`: Delete an endpoint
- **Sub-commands**:
  - `local`: Local operations (commands: `train`, `deploy`, `process`, `run`, `stop`)

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
- Updates 8 shell scripts with security fixes (proper variable quoting)

**Templates** (`easy_sm/template/easy_sm_base/`):
- Dockerfile and scripts for containerized training/processing
- Training entry point: `training/train`
- Serving entry point: `prediction/serve`
- Local test scripts in `local_test/`

**Run Command** (`run` / `local run`):
- Generic replacement for the old `make` command: everything after `--` is passed as the command
- Option parsing stops at the first word of the command (`allow_interspersed_args=False`), so `dbt run -e prod` keeps its `-e` even without `--`; easy_sm options must precede the command
- `executor.sh` `run` branch: `shift; cd ./easy_sm_base/processing; exec "$@"` (no shell re-parsing)
- Cloud: `SageMakerClient.run()` → `_run_processing(["run", *command])`
- Local: `run_local.sh` receives `KEY=VALUE` env pairs, then `--`, then the command; rebuilds `"$@"` as `-e` flags + `image run command...`
- `helpers.require_run_support()` refuses to run if the project's `executor.sh` lacks the `run` branch (an old image would fall through to serving and run until timeout)
- `local run` exits with the command's exit code (other local commands still exit 0 on failure)

### Configuration Flow
1. Commands receive optional `app_name` and `iam_role_arn` parameters
2. Auto-detect app_name from `*.json` file if not provided
3. Read IAM role from `SAGEMAKER_ROLE` env var if not provided
4. Validate app_name (alphanumeric, hyphens, underscores only)
5. Load config from `{app_name}.json` in current directory
6. Config specifies Docker image name, AWS credentials, Python version, and module locations
7. Commands use config to build images, run jobs, or deploy endpoints

### Auto-Detection Behavior
- **App name**: Searches for `*.json` files in current directory. Fails if none or multiple found (can override with `-a`). Note: `local *` commands and `build`/`update-scripts` currently require `-a` explicitly
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
- Docker tag comes from `docker_tag` in `{app_name}.json` (default: "latest"), accessed as `config.docker_tag`
- Full image name: `{config.image_name}:{config.docker_tag}`
- Note: many user docs still describe a global `--docker-tag`/`-t` option; it does not exist (the CLI rejects it)
- Source code is mounted/copied into Docker containers for training/processing

### Security Features
- **App name validation**: Prevents path traversal attacks (e.g., `../../../etc/passwd`)
- **Shell script quoting**: All variables properly quoted to prevent injection
- **Argument passing**: User-supplied values (`--env`, `run` commands) reach `docker run` / the container as individual argv entries via `"$@"`, never via unquoted strings or `eval`
- **Local endpoint**: `local deploy` binds to `127.0.0.1` only
- **Outdated-script guards**: commands refuse to run with project scripts that predate a fix/feature and point the user to `update-scripts`
- **File permissions**: Scripts set to 0o755 (not world-writable)
- **Known open item**: `~/.aws` is mounted read-write into local containers (`:ro` would break SSO token refresh)

### Environment Variables for Processing Jobs

Both cloud and local processing jobs support passing environment variables via the `--env` flag (repeatable):

```bash
# Cloud processing with env vars
easy_sm process -f script.py -e ml.m5.large -n job \
  --env DEBUG=true --env LOG_LEVEL=info --env DATASET_VERSION=2024-q1

# Local processing with env vars
easy_sm local process -f script.py -a myapp \
  --env DEBUG=true --env LOG_LEVEL=info
```

**Implementation Details:**
- **Cloud**: Environment variables passed to `Processor` constructor (sagemaker SDK)
- **Local**: Each `KEY=VALUE` pair is passed to the shell script as its own argument; the script rebuilds `"$@"` as `-e KEY=VALUE` pairs so values can never be word-split into extra docker flags. `local process` refuses `--env` if the project's `process_local.sh` is the old comma-separated version (tells user to run `update-scripts`)
- **Parsing**: Shared `helpers.parse_env_vars()`; cloud additionally enforces SageMaker's 256-char value limit
- **Secrets**: Don't pass secrets via `--env` — values are visible in the SageMaker job definition
- **Validation**: `KEY=VALUE` format enforced; values can contain `=` (e.g., `API_KEY=sk-1234=5678`)
- **Empty values**: Allowed (e.g., `--env EMPTY_VAR=`)

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
- Get the image tag from config: `config.docker_tag`
- For commands that take an arbitrary trailing command, use a `list[str]` `typer.Argument` with `context_settings={"allow_interspersed_args": False}` (so the command's flags are never parsed as ours) and tell users to put it after `--`
- Top-level commands in `cloud.py` are plain undecorated functions, registered in `__main__.py` with `app.command(name=...)(func)`. Command names and `context_settings` go on that registration
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
from easy_sm.commands.helpers import get_app_name, get_iam_role, load_config

@app.command()
def subcommand(
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
) -> None:
    """Command description."""
    app_name = get_app_name(app_name)
    iam_role = get_iam_role(iam_role_arn)
    config = load_config(app_name)
    image = f"{config.image_name}:{config.docker_tag}"
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

**test.yml**: Continuous Testing
- **Trigger**: Push to main and pull requests
- **Process**:
  1. Setup Python 3.13
  2. Install package and test dependencies
  3. Run full test suite (`pytest --cov=easy_sm`, 157 tests)
  4. Run type checking (`mypy`)
  5. Run linting (`ruff check`)
- **Result**: Validates code quality on every commit
- **Badge**: Shows test status, not affected by release jobs

**publish.yml**: Release and Publish
- **Trigger**: Git tag creation matching `v*` (e.g., `git tag v1.0.1`). No manual trigger
- **Process** (three jobs, each with least-privilege `permissions`; workflow default is `permissions: {}`):
  1. `build` (`contents: read`): build sdist and wheel, upload as artifact
  2. `publish` (`id-token: write` only): publish to PyPI via **Trusted Publishing** (OIDC, no stored token) in the `pypi` environment
  3. `release` (`contents: write`): create GitHub Release with `gh release create`
- **Security**:
  - The PyPI trusted publisher is bound to `prteek/easy_sm`, `publish.yml` and environment `pypi`. Renaming the workflow file or environment breaks publishing until PyPI is updated
  - The `pypi` GitHub environment only accepts deployments from `v*` tags
  - Build steps never get a publishing credential (separate job)
- **Result**:
  - New version available on PyPI
  - GitHub release page created with commit history
- **Important**: Only triggered on version tags, not on regular commits
- **Badges**:
  - GitHub release badge updates immediately
  - PyPI badge updates within minutes
- **How to release**:
  ```bash
  # After version bump commit is merged to main:
  git tag v1.0.1
  git push origin v1.0.1
  # Publish workflow triggers automatically and handles both PyPI + GitHub
  ```

**Action pinning**: All third-party actions are pinned to full commit SHAs with a `# vX.Y.Z` comment. `.github/dependabot.yml` opens a weekly grouped PR to bump them. When adding an action, pin it the same way (resolve the tag with `git ls-remote https://github.com/OWNER/REPO refs/tags/vX.Y.Z^{}`)

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

### Pre-commit Hook

**Location**: `.githooks/pre-commit`
- Validates documentation against code implementation before commit
- Runs `python3 scripts/validate_docs.py`
- Blocks commit if validation fails
- Helps prevent documentation drift
- Only runs when staged files include `docs/`, `CLAUDE.md`, `README.md`, `DOCS_WORKFLOW.md`, `mkdocs.yml` or `requirements-docs.txt`
- Enabled via `git config core.hooksPath .githooks`

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
- `mkdocs build --strict` - Build documentation
- `docker stop:*`, `lsof:*`, `netstat:*` - Manage local deployments
- `gh run/workflow/api/release view` - Inspect GitHub Actions and releases
- `WebFetch` and `WebSearch` - Web access for documentation and research

## Dependencies

### Runtime Dependencies (from setup.py)
- **typer** (>=0.9.0): CLI framework (built on Click)
- **docker** (>=7.1.0): Docker SDK for building/pushing images
- **sagemaker** (==2.257.3): AWS SageMaker SDK (pinned exact version for stability)
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
│   │   ├── cloud.py          # Cloud training/deployment/processing/run/endpoint management
│   │   ├── local.py          # Local train/deploy/process/run/stop
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
├── tests/                    # Test suite (uses pytest) - 157 tests total
│   ├── test_build_command.py         # Tests for build command (13 tests)
│   ├── test_init_command.py          # Tests for init command (7 tests)
│   ├── test_local_commands.py        # Tests for local commands (33 tests)
│   ├── test_cloud_commands.py        # Tests for cloud SageMaker operations (40 tests)
│   ├── test_push_command.py          # Tests for ECR push command (9 tests)
│   ├── test_update_command.py        # Tests for update-scripts command (7 tests)
│   ├── test_config.py                # Tests for Config/ConfigManager (16 tests)
│   ├── test_helpers.py               # Tests for subprocess utilities and env parsing (23 tests)
│   ├── test_local_scripts.py         # Runs template shell scripts against a fake docker (9 tests)
│   └── LOCAL_COMMANDS_TESTS_README.md # Documentation for local command tests
├── docs/                      # MkDocs site (commands/, user-guide/, examples/, ...)
├── mkdocs.yml                 # MkDocs configuration and nav
├── setup.py                   # Package metadata and dependencies
├── base-requirements.txt      # Development dependencies (pytest, mypy, ruff, etc)
├── requirements-docs.txt      # Documentation dependencies
├── CLAUDE.md                  # Claude Code guidance (this file)
├── DOCS_WORKFLOW.md           # Documentation workflow and validation guide
├── .githooks/
│   └── pre-commit             # Git pre-commit hook for doc validation
└── .github/
    ├── README.md              # Project overview, usage guide and command examples
    ├── dependabot.yml         # Weekly PRs to bump SHA-pinned GitHub Actions
    └── workflows/
        ├── test.yml           # Tests, mypy and ruff on push/PR to main
        ├── publish.yml        # PyPI publish + GitHub Release on v* tags
        └── docs.yml           # MkDocs build and GitHub Pages deployment
```

## Adding New Commands

1. Create `{command_name}.py` in `easy_sm/commands/`
2. Define the command as a plain function with `Annotated` type hints (no decorator), or as `@local_app.command()` for a `local` subcommand
3. Import and register top-level commands in `easy_sm/__main__.py` via `app.command(name=...)(func)`; sub-apps via `app.add_typer()`
4. Follow existing patterns for config loading and subprocess calls
5. If the command needs a new template script (or changes `executor.sh`), add it to `SHELL_SCRIPTS` in `easy_sm/commands/update.py` and guard against outdated project copies (see `require_run_support`)
6. Add a docs page under `docs/commands/`, add it to `mkdocs.yml` nav and `docs/commands/index.md`, and register it in `scripts/validate_docs.py`

## Key Implementation Details

- All commands execute in the current working directory; projects identified by presence of `{app_name}.json`
- Docker images are built locally and can be pushed to registries
- SageMaker operations require valid AWS credentials via configured profile
- Local training/processing uses Docker to simulate SageMaker container environment
- Configuration is persisted as JSON to maintain state across command invocations
- App names are validated to prevent security issues (path traversal, injection)

## Release Process

### Creating a Release

1. **Bump version** in:
   - `setup.py`: version="X.Y.Z"
   - `easy_sm/__init__.py`: __version__ = "X.Y.Z"
   - `mkdocs.yml`: version: X.Y.Z

2. **Commit version bump** to main:
   ```bash
   git add setup.py easy_sm/__init__.py mkdocs.yml
   git commit -m "chore: bump version to X.Y.Z"
   git push origin main
   ```

3. **Create git tag** to trigger release:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. **Watch release workflow**: GitHub Actions automatically:
   - Builds distribution packages
   - Publishes to PyPI
   - Updates PyPI badge with new version

### Why This Approach

- **Separation of concerns**: Tests run on every commit, releases only on tags
- **Badge stability**: Release badge only shows failures on actual release attempts
- **Clear history**: Git tags mark stable release points
- **No spurious failures**: Minor fixes don't trigger release job failures

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
- Commands are found by function name from a hardcoded list in `extract_command_params`, and doc/impl pairs are listed in `validate_all`. Register new commands in both places
- **Known limitation**: flag comparison only reads markdown tables whose header contains `Flag`. The command pages use `| Option | Short | ...` headers, so documented flags are currently **not** compared against code. Cross-check option tables by hand

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
