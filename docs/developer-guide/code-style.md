# Code Style Guidelines

This document outlines coding conventions and patterns used in easy_sm.

## Naming Conventions

### Classes

Use **PascalCase** for class names:

```python
class ConfigManager:
    pass

class SageMakerClient:
    pass
```

### Functions and Methods

Use **snake_case** for functions and methods:

```python
def train_model(data):
    pass

def get_model_artifacts(job_name):
    pass
```

### Private Methods/Variables

Use leading underscore prefix for private members:

```python
class ConfigManager:
    def __init__(self):
        self._config = {}  # Private variable

    def _load_from_file(self, path):  # Private method
        pass
```

### Constants

Use **UPPER_SNAKE_CASE** for constants:

```python
DEFAULT_PYTHON_VERSION = "3.13"
MAX_RETRIES = 3
SAGEMAKER_ROLE_ENV_VAR = "SAGEMAKER_ROLE"
```

## Type Hints

Use type hints for all function parameters and return values:

```python
def process_data(input_path: str, output_path: str) -> bool:
    """Process data from input to output."""
    # Implementation
    return True

def get_config(app_name: str) -> Config:
    """Load configuration for app."""
    # Implementation
    return config
```

### Optional Types

Use `Optional` for parameters that can be `None`:

```python
from typing import Optional

def load_model(model_path: str, cache: Optional[bool] = None) -> Model:
    pass
```

## String Formatting

### Prefer f-strings

Use f-strings for new code:

```python
# Good
print(f"Training job: {job_name}")
error_msg = f"Model not found at {model_path}"

# Avoid (but acceptable in existing code)
print("Training job: {}".format(job_name))
print("Model not found at " + model_path)
```

### Accept .format() in Existing Code

For consistency with existing code, `.format()` is acceptable:

```python
# Acceptable in existing codebase
message = "Job {} completed with status {}".format(job_name, status)
```

## Typer Commands

### Command Definitions

Use `Annotated` with `typer.Option` for parameters:

```python
from typing import Annotated, Optional
import typer

@app.command()
def train(
    job_name: Annotated[str, typer.Option("--job-name", "-n", help="Training job name")],
    instance_type: Annotated[str, typer.Option("--instance-type", "-e", help="EC2 instance type")],
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name")] = None,
) -> None:
    """Train model on SageMaker."""
    # Implementation
```

### Always Include Help Text

Every option must have descriptive help text:

```python
# Good
job_name: Annotated[str, typer.Option("-n", help="Training job name")]

# Bad (no help text)
job_name: Annotated[str, typer.Option("-n")]
```

### Access Global docker_tag

Access the global `docker_tag` via `helpers.docker_tag`:

```python
from easy_sm.commands import helpers

def build_image():
    docker_tag = helpers.docker_tag  # Get global tag
    image_name = f"{config.image_name}:{docker_tag}"
```

### Sub-Apps

Use `typer.Typer()` for sub-commands and register with `app.add_typer()`:

```python
# In easy_sm/commands/local.py
from typer import Typer

local_app = Typer()

@local_app.command()
def train():
    """Train locally."""
    pass

# In easy_sm/__main__.py
from easy_sm.commands.local import local_app

app.add_typer(local_app, name="local", help="Local operations")
```

## Error Handling

### Validation Errors

Use `ValueError` for validation errors:

```python
def validate_app_name(app_name: str) -> None:
    if not re.match(r'^[a-zA-Z0-9_-]+$', app_name):
        raise ValueError(f"Invalid app name: {app_name}")
```

### CLI-Specific Errors

Use `typer.BadParameter` for CLI-specific errors:

```python
import typer

def validate_instance_type(instance_type: str) -> str:
    if not instance_type.startswith('ml.'):
        raise typer.BadParameter("Instance type must start with 'ml.'")
    return instance_type
```

### Error Messages

Print error messages to stdout before `sys.exit()`:

```python
import sys

def handle_error(error_message: str) -> None:
    print(f"Error: {error_message}")
    sys.exit(1)
```

### Subprocess Operations

Use try/except for subprocess operations:

```python
import subprocess

try:
    result = subprocess.run(
        ["docker", "build", ".", "-t", image_name],
        check=True,
        capture_output=True,
        text=True
    )
except subprocess.CalledProcessError as e:
    print(f"Build failed: {e.stderr}")
    sys.exit(1)
```

## Imports

Organize imports in three sections:

1. Standard library
2. Third-party packages
3. Local imports

Sort alphabetically within each section:

```python
# Standard library
import os
import sys
from typing import Annotated, Optional

# Third-party
import typer
from docker import DockerClient

# Local
from easy_sm.commands import helpers
from easy_sm.commands.helpers import load_config
from easy_sm.config.config import Config, ConfigManager
```

## Common Patterns

### Loading Configuration in Commands

Standard pattern for loading configuration:

```python
from easy_sm.commands.helpers import get_app_name, get_iam_role, load_config

def my_command(
    app_name: Optional[str] = None,
    iam_role_arn: Optional[str] = None
) -> None:
    # Get app name (from parameter or auto-detect)
    app_name = get_app_name(app_name)

    # Get IAM role (from parameter or SAGEMAKER_ROLE env var)
    iam_role = get_iam_role(iam_role_arn)

    # Load config
    config = load_config(app_name)

    # Use config
    print(f"Image: {config.image_name}")
```

### Typer Command with Optional Context

Template for commands with auto-detection:

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

Output only essential data for Unix-style composition:

```python
# Good: Output just the data
print(s3_path)
print(endpoint_name)

# Bad: Verbose messages (not pipable)
print(f"Model uploaded to {s3_path}")
print(f"Successfully deployed to endpoint: {endpoint_name}")
```

### Success vs Error Output

- **Success**: Print data to stdout
- **Errors**: Print to stderr or use typer's error handling

```python
# Success case
print(model_path)  # Goes to stdout

# Error case
import sys
print(f"Error: {error_message}", file=sys.stderr)
sys.exit(1)
```

## Documentation

### Docstrings

Use docstrings for all public functions and classes:

```python
def train_model(data: pd.DataFrame, model_path: str) -> Model:
    """
    Train a machine learning model.

    Args:
        data: Training data as pandas DataFrame
        model_path: Path to save trained model

    Returns:
        Trained model object

    Raises:
        ValueError: If data is empty
    """
    if data.empty:
        raise ValueError("Training data is empty")
    # Implementation
```

### Inline Comments

Use comments for non-obvious logic:

```python
# Calculate confusion matrix for binary classification
cm = confusion_matrix(y_true, y_pred)

# Skip if already processed (optimization)
if model_id in cache:
    return cache[model_id]
```

Avoid obvious comments:

```python
# Bad (obvious)
x = x + 1  # Increment x

# Good (adds context)
x = x + 1  # Account for zero-indexing offset
```

## Testing

### Test Function Names

Prefix test functions with `test_`:

```python
def test_config_loading():
    """Test configuration file loading."""
    pass

def test_invalid_app_name():
    """Test validation rejects invalid app names."""
    pass
```

### Use Descriptive Names

Test names should describe what they test:

```python
# Good
def test_train_command_with_multiple_instances():
    pass

def test_deploy_fails_without_model_path():
    pass

# Avoid
def test_train():
    pass

def test_1():
    pass
```

## See Also

- [Architecture](architecture.md) - System design and structure
- [Testing Guide](testing.md) - Testing conventions
- [Contributing Guide](contributing.md) - Development workflow
