# AGENTS.md - easy_sm Development Guide

This file provides guidelines and commands for agents working on the easy_sm codebase.

## Project Overview

easy_sm is a CLI tool that makes it easier to work with AWS SageMaker by enabling rapid prototyping with local training, processing, and deployment. The project is a Python package (>=3.14) using Click for CLI commands.

## Build Commands

```bash
# Install the package in development mode
pip install -e .

# Build the package
python setup.py build

# Full installation with all dependencies
pip install -e . -r requirements.txt
```

## Testing

No test framework is currently configured. To add tests:

```bash
# Install pytest
pip install pytest

# Run all tests
pytest

# Run a single test file
pytest tests/test_filename.py

# Run a specific test
pytest tests/test_filename.py::test_function_name

# Run with coverage
pytest --cov=easy_sm
```

## Linting and Formatting

No linting or formatting tools are currently configured. Consider adding:

```bash
# Install ruff (recommended - fast and comprehensive)
pip install ruff

# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Install black
pip install black

# Format with black
black .

# Install mypy for type checking
pip install mypy

# Type check
mypy easy_sm/
```

## Code Style Guidelines

### Python Version
- Minimum: Python 3.14
- Check: `.python-version` file

### Imports
- Organize imports in three sections (standard library, third-party, local)
- Sort alphabetically within each section
- Example:
```python
import os
import sys

import click

from easy_sm.commands.helpers import safe_run_subprocess
from easy_sm.config.config import ConfigManager
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `ConfigManager`, `SageMakerClient`)
- **Functions/Variables**: snake_case (e.g., `safe_run_subprocess`, `config_file_path`)
- **Private Methods**: Leading underscore (e.g., `_config`, `_build`)
- **Private Variables**: Leading underscore (e.g., `_config_file_path`)
- **Constants**: UPPER_SNAKE_CASE

### Type Hints
- Use type hints for function parameters and return values
- Be consistent with existing code (see `easy_sm/sagemaker/sagemaker.py` for examples)
```python
def _check_endpoint_exists(self, endpoint_name: str) -> bool:
```

### Error Handling
- Use `ValueError` for validation errors
- Use `Click` exceptions for CLI errors (e.g., `BadParameter`)
- Use try/except for subprocess operations
- Print error messages to stdout before sys.exit()
```python
if not os.path.isfile(config_file_path):
    raise ValueError("This is not a easy_sm directory: {}".format(os.getcwd()))
```

### String Formatting
- Use f-strings for new code
- Keep `.format()` for consistency with existing code
```python
# Preferred for new code
print(f"Error: {message}")

# Acceptable (existing code)
print("Error occurred: {}".format(error))
```

### Click Commands
- Use `u""` prefix for Unicode strings in Click decorators
- Always include help text for options
```python
@click.command()
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def build(obj, app_name):
    """Command to build SageMaker app"""
    pass
```

### Class Structure
- Use explicit `object` as base class for Python 2 compatibility
- Use docstrings with Google-style or reST format
```python
class ConfigManager(object):
    def __init__(self, config_file_path):
        """Initialize the config manager.
        
        Args:
            config_file_path: Path to the configuration file
        """
        pass
```

### File Organization
- Main entry point: `easy_sm/__main__.py`
- CLI commands in `easy_sm/commands/`
- Core logic in `easy_sm/sagemaker/`, `easy_sm/config/`
- Templates in `easy_sm/template/`

### CLI Group Commands
- Use Click groups for related commands (e.g., `local`, `cloud`)
- Register subcommands with `cli.add_command()`
```python
@click.group()
def local():
    """Commands for local operations: train and deploy"""
    pass

local.add_command(train)
local.add_command(deploy)
```

### Docker/SageMaker Specifics
- Docker tag from CLI context: `obj['docker_tag']`
- Config loaded from `{app_name}.json`
- Image naming: `{config.image_name}:{docker_tag}`

## Common Tasks

### Adding a New Command
1. Create command file in `easy_sm/commands/`
2. Import and register in `easy_sm/commands/__init__.py`
3. Add to CLI in `easy_sm/__main__.py`

### Running the CLI
```bash
easy_sm --help
easy_sm build --app-name myapp
easy_sm local train --app-name myapp
easy_sm cloud train --app-name myapp
```

### Local Development Workflow
```bash
# Make changes
pip install -e .  # Reinstall after changes
easy_sm --help  # Verify CLI works
```


## Project Structure

```
easy_sm/
├── app/                    # Sample app for testing (not git-committed)
├── easy_sm/                # Main package
│   ├── __main__.py         # CLI entry point
│   ├── commands/           # CLI command implementations
│   ├── sagemaker/          # SageMaker integration
│   ├── config/             # Configuration management
│   └── template/           # Templates
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── AGENTS.md             # This file
```
