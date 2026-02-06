# Contributing Guide

Thank you for considering contributing to easy_sm! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/easy_sm.git
cd easy_sm
```

### 2. Install in Development Mode

Install easy_sm with all dependencies:

```bash
pip install -e .
```

This allows code changes without reinstalling.

### 3. Install Development Dependencies

Install testing and quality tools:

```bash
pip install -e . -r base-requirements.txt
```

This installs:

- **pytest**: Test framework
- **mypy**: Type checker
- **ruff**: Linter and formatter
- **requests**: HTTP library for tests
- **statsmodels, joblib, pandas**: Sample app dependencies

### 4. Verify Installation

```bash
# Run CLI
easy_sm --help

# Run tests
pytest

# Type checking
mypy easy_sm/

# Linting
ruff check easy_sm/
```

## Development Workflow

### 1. Create a Branch

Create a feature branch for your changes:

```bash
git checkout -b feature/my-new-feature
```

Branch naming conventions:

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### 2. Make Changes

Edit code following the [Code Style Guide](code-style.md):

- Use type hints
- Follow naming conventions
- Add docstrings
- Write tests

### 3. Run Tests

Test your changes:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_my_feature.py

# Run with coverage
pytest --cov=easy_sm
```

### 4. Check Code Quality

```bash
# Type checking
mypy easy_sm/

# Linting
ruff check easy_sm/

# Format code
ruff format easy_sm/
```

### 5. Commit Changes

Write clear commit messages:

```bash
git add .
git commit -m "feat: add support for spot instances"
```

Commit message format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test improvements
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

### 6. Push to Your Fork

```bash
git push origin feature/my-new-feature
```

### 7. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select `main` as the base branch
4. Describe your changes clearly
5. Link any related issues

## Adding New Commands

### 1. Create Command File

Create a new file in `easy_sm/commands/`:

```bash
touch easy_sm/commands/my_command.py
```

### 2. Define Command

Follow existing patterns:

```python
# easy_sm/commands/my_command.py
from typing import Annotated, Optional
import typer
from easy_sm.commands import helpers
from easy_sm.commands.helpers import get_app_name, load_config

@app.command()
def my_command(
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name")] = None,
    param: Annotated[str, typer.Option("--param", "-p", help="Parameter description")],
) -> None:
    """
    Description of what this command does.

    This is a longer description that appears in help text.
    """
    # Auto-detect app name
    app_name = get_app_name(app_name)

    # Load configuration
    config = load_config(app_name)

    # Command logic
    result = do_something(config, param)

    # Output result (pipe-friendly)
    print(result)
```

### 3. Register Command

Add to `easy_sm/__main__.py`:

```python
# Import command
from easy_sm.commands.my_command import my_command

# Register with app
app.command()(my_command)
```

### 4. Write Tests

Create test file:

```python
# tests/test_my_command.py
import mock
import pytest
from easy_sm.commands.my_command import my_command

def test_my_command_basic():
    """Test basic functionality."""
    # Arrange
    app_name = "test-app"
    param = "value"

    # Act
    result = my_command(app_name, param)

    # Assert
    assert result.success

@mock.patch('subprocess.run')
def test_my_command_with_subprocess(mock_run):
    """Test command that calls subprocess."""
    # Setup mock
    mock_run.return_value = mock.Mock(returncode=0)

    # Test
    result = my_command(...)

    # Verify
    mock_run.assert_called_once()
```

### 5. Add Documentation

Create command documentation in `docs/commands/my-command.md`:

```markdown
# my-command

Description of the command.

## Synopsis

\`\`\`bash
easy_sm my-command [OPTIONS]
\`\`\`

## Options

| Option | Description |
|--------|-------------|
| `-a, --app-name` | App name |
| `-p, --param` | Parameter |

## Examples

\`\`\`bash
easy_sm my-command -p value
\`\`\`
```

Update `docs/commands/index.md` to include your command.

## Testing Guidelines

### Write Tests for All New Code

Every new feature should have tests:

```python
def test_new_feature():
    """Test new feature works correctly."""
    result = new_feature()
    assert result.is_valid()

def test_new_feature_error_handling():
    """Test new feature handles errors."""
    with pytest.raises(ValueError):
        new_feature(invalid_input)
```

### Mock External Dependencies

Don't make real AWS or Docker calls:

```python
@mock.patch('boto3.client')
def test_with_aws(mock_boto):
    """Test function using AWS."""
    mock_client = mock.Mock()
    mock_boto.return_value = mock_client

    # Test your function
    result = my_aws_function()

    # Verify AWS call
    mock_client.some_method.assert_called_once()
```

### Test Both Success and Failure

```python
def test_success_case():
    """Test successful execution."""
    assert my_function(valid_input).success

def test_failure_case():
    """Test error handling."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

## Documentation

### Update Documentation

If your change affects user-facing behavior:

1. Update relevant documentation in `docs/`
2. Update `README.md` if needed
3. Add examples showing new functionality

### Docstrings

Add docstrings to all public functions:

```python
def my_function(param: str) -> bool:
    """
    Brief description of what function does.

    Longer description with more details about behavior,
    edge cases, and usage examples.

    Args:
        param: Description of parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid
    """
    # Implementation
```

## Code Review Process

### What to Expect

1. **Automated Checks**: CI/CD runs tests, linting, type checking
2. **Maintainer Review**: A maintainer reviews your code
3. **Feedback**: You may be asked to make changes
4. **Approval**: Once approved, your PR will be merged

### Review Criteria

- **Tests**: All tests pass, new code has tests
- **Quality**: Code follows style guide, passes linting
- **Documentation**: Changes are documented
- **Functionality**: Code works as intended
- **Design**: Fits with overall architecture

## Best Practices

### Keep PRs Focused

- One feature or fix per PR
- Small, reviewable changes
- Clear description of changes

### Write Good Commit Messages

```bash
# Good
git commit -m "feat: add support for spot training instances"
git commit -m "fix: handle missing config file gracefully"
git commit -m "docs: update AWS setup guide with IAM details"

# Avoid
git commit -m "updates"
git commit -m "fix bug"
git commit -m "wip"
```

### Follow Existing Patterns

- Look at existing commands for examples
- Use the same error handling patterns
- Follow the same testing style

### Ask Questions

If you're unsure about anything:

- Open an issue to discuss your idea
- Ask in the pull request comments
- Reference related issues or PRs

## Building and Testing Locally

### Build Package

```bash
python setup.py build
```

### Install Locally

```bash
pip install -e .
```

### Run CLI

```bash
easy_sm --help
easy_sm build
easy_sm local train
```

### Run Full Test Suite

```bash
# All tests with coverage
pytest --cov=easy_sm

# Specific test file
pytest tests/test_build_command.py

# Type checking
mypy easy_sm/

# Linting
ruff check easy_sm/
```

## Release Process

(For maintainers)

### 1. Update Version

Edit `setup.py`:

```python
setup(
    name="easy-sm",
    version="1.0.1",  # Update version
    ...
)
```

### 2. Update Changelog

Document changes in `CHANGELOG.md`.

### 3. Create Tag

```bash
git tag -a v1.0.1 -m "Release version 1.0.1"
git push origin v1.0.1
```

### 4. Build and Publish

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

## Getting Help

- **Issues**: [github.com/prteek/easy_sm/issues](https://github.com/prteek/easy_sm/issues)
- **Discussions**: GitHub Discussions
- **Email**: prteek@icloud.com

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what's best for the project
- Show empathy towards others

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## See Also

- [Architecture](architecture.md) - System design
- [Code Style](code-style.md) - Coding conventions
- [Testing Guide](testing.md) - Testing practices
