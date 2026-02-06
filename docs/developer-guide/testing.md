# Testing Guide

easy_sm has a comprehensive test suite covering all commands and core modules.

## Test Suite Overview

**Total: 120 tests** across 8 test files, covering all commands and core modules.

All tests use mocked external dependencies (subprocess, boto3, SageMaker SDK) for fast, reliable execution without requiring AWS credentials or Docker.

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_build_command.py
pytest tests/test_cloud_commands.py
pytest tests/test_init_command.py
pytest tests/test_local_commands.py
pytest tests/test_push_command.py
pytest tests/test_update_command.py
pytest tests/test_config.py
pytest tests/test_helpers.py
```

### Run Specific Test Function

```bash
pytest tests/test_build_command.py::test_build_with_custom_tag
pytest tests/test_cloud_commands.py::test_train_command
```

### Run Tests with Coverage

```bash
pytest --cov=easy_sm
```

Example output:

```
tests/test_build_command.py ............ [20%]
tests/test_cloud_commands.py .......... [50%]
tests/test_config.py ............ [75%]
tests/test_helpers.py ............ [100%]

====== 120 passed in 2.34s ======

---------- coverage: ----------
Name                                Stmts   Miss  Cover
-------------------------------------------------------
easy_sm/__init__.py                     1      0   100%
easy_sm/__main__.py                    45      0   100%
easy_sm/commands/build.py              67      0   100%
...
-------------------------------------------------------
TOTAL                                 1234     12    99%
```

### Run Tests with Verbose Output

```bash
pytest -v
```

### Run Tests Matching Pattern

```bash
# Run all tests with "deploy" in the name
pytest -k deploy

# Run all tests except slow ones
pytest -k "not slow"
```

## Test Files

### Command Tests (87 tests)

#### test_init_command.py (7 tests)

Tests project initialization with various configurations:

- `test_init_default_values` - Initialize with default configuration
- `test_init_custom_values` - Initialize with custom values
- `test_init_creates_directory_structure` - Verify directory creation
- `test_init_creates_config_file` - Verify config file creation
- `test_init_with_invalid_python_version` - Test validation
- `test_init_with_existing_project` - Test overwrite behavior
- `test_init_with_missing_requirements` - Test requirements handling

#### test_build_command.py (13 tests)

Tests Docker image building with parameter variations and error scenarios:

- `test_build_with_default_tag` - Build with latest tag
- `test_build_with_custom_tag` - Build with specific tag
- `test_build_with_auto_detected_app_name` - Auto-detect app name
- `test_build_with_explicit_app_name` - Override app name
- `test_build_with_missing_config` - Handle missing configuration
- `test_build_with_invalid_dockerfile` - Handle Docker errors
- `test_build_with_custom_dockerfile_path` - Custom Dockerfile location
- `test_build_caching` - Verify build caching works
- `test_build_with_build_args` - Pass build arguments
- `test_build_output_parsing` - Parse Docker build output
- `test_build_with_no_cache_flag` - Disable build cache
- `test_build_failure_cleanup` - Clean up on failure
- `test_build_with_progress_display` - Progress indication

#### test_local_commands.py (23 tests)

Tests local training, deployment, processing, and stop commands:

- `test_local_train` - Train model locally
- `test_local_train_with_custom_data_path` - Custom input data
- `test_local_train_with_hyperparameters` - Pass hyperparameters
- `test_local_train_with_missing_data` - Handle missing data
- `test_local_deploy` - Deploy model locally
- `test_local_deploy_with_custom_port` - Use custom port
- `test_local_deploy_already_running` - Handle port conflict
- `test_local_deploy_with_model_path` - Specify model path
- `test_local_process` - Run processing job locally
- `test_local_process_with_script` - Process with Python script
- `test_local_process_with_inputs_outputs` - Multiple I/O paths
- `test_local_stop` - Stop local deployment
- `test_local_stop_not_running` - Handle no container running
- `test_local_commands_with_auto_detect` - Auto-detect app name
- 9 more tests...

#### test_cloud_commands.py (28 tests)

Tests SageMaker operations:

**Training**:
- `test_train_command` - Basic training job
- `test_train_with_hyperparameters` - Pass hyperparameters
- `test_train_with_multiple_instances` - Distributed training
- `test_train_with_spot_instances` - Use spot instances
- `test_list_training_jobs` - List all jobs
- `test_list_training_jobs_names_only` - Names-only output
- `test_get_model_artifacts` - Get model S3 path

**Deployment**:
- `test_deploy_provisioned` - Deploy provisioned endpoint
- `test_deploy_with_multiple_instances` - Multi-instance deployment
- `test_deploy_serverless` - Deploy serverless endpoint
- `test_deploy_with_tags` - Add resource tags

**Processing**:
- `test_process_job` - Run processing job
- `test_batch_transform` - Batch predictions
- `test_batch_transform_with_multiple_files` - Process multiple files

**Management**:
- `test_list_endpoints` - List all endpoints
- `test_delete_endpoint` - Delete endpoint
- `test_delete_endpoint_with_config` - Delete endpoint and config
- 11 more tests...

#### test_push_command.py (9 tests)

Tests ECR image push with authentication:

- `test_push_with_profile_auth` - Push with AWS profile
- `test_push_with_iam_role_auth` - Push with IAM role
- `test_push_with_external_id` - Push with external ID
- `test_push_creates_ecr_repository` - Auto-create repository
- `test_push_with_existing_repository` - Use existing repository
- `test_push_with_custom_tag` - Push specific tag
- `test_push_authentication_failure` - Handle auth errors
- `test_push_with_cross_account` - Cross-account push
- `test_push_output_format` - Verify output format

#### test_update_command.py (7 tests)

Tests shell script update command:

- `test_update_scripts_basic` - Update all scripts
- `test_update_scripts_auto_detect_app` - Auto-detect app name
- `test_update_scripts_with_customizations` - Preserve customizations warning
- `test_update_scripts_missing_app_directory` - Handle missing directory
- `test_update_scripts_file_permissions` - Verify correct permissions (0o755)
- `test_update_scripts_security_fixes` - Verify proper quoting
- `test_update_scripts_dry_run` - Dry-run mode

### Module Tests (33 tests)

#### test_config.py (16 tests)

Tests configuration loading, saving, serialization:

- `test_config_creation` - Create Config object
- `test_config_to_dict` - Serialize to dictionary
- `test_config_from_dict` - Deserialize from dictionary
- `test_config_manager_load` - Load from JSON file
- `test_config_manager_save` - Save to JSON file
- `test_config_manager_get_or_create` - Get or create default
- `test_config_manager_missing_file` - Handle missing file
- `test_config_manager_invalid_json` - Handle malformed JSON
- `test_config_validation_empty_fields` - Validate required fields
- `test_config_validation_invalid_types` - Type checking
- `test_config_with_defaults` - Default values
- 5 more tests...

#### test_helpers.py (17 tests)

Tests subprocess execution, output handling, error propagation:

- `test_safe_run_subprocess_success` - Successful execution
- `test_safe_run_subprocess_failure` - Handle process errors
- `test_safe_run_subprocess_timeout` - Handle timeouts
- `test_safe_run_subprocess_output_capture` - Capture stdout/stderr
- `test_auto_detect_app_name_single_config` - Auto-detect single config
- `test_auto_detect_app_name_multiple_configs` - Fail on multiple configs
- `test_auto_detect_app_name_no_config` - Fail on no config
- `test_get_app_name_explicit` - Use explicit app name
- `test_get_app_name_auto_detect` - Auto-detect app name
- `test_get_iam_role_from_param` - Use explicit IAM role
- `test_get_iam_role_from_env_var` - Use SAGEMAKER_ROLE env var
- `test_get_iam_role_missing` - Error when not provided
- `test_load_config` - Load configuration
- 4 more tests...

## Testing Philosophy

### Unit Tests

Each test focuses on a single unit of functionality:

```python
def test_build_with_custom_tag():
    """Test building Docker image with custom tag."""
    # Arrange
    app_name = "my-app"
    docker_tag = "v1.0"

    # Act
    result = build_command(app_name, docker_tag)

    # Assert
    assert result.success
    assert result.image_name == f"my-app:{docker_tag}"
```

### Mocked Dependencies

External dependencies are mocked for isolation:

```python
@mock.patch('subprocess.run')
@mock.patch('boto3.client')
def test_train_command(mock_boto, mock_subprocess):
    """Test training command with mocked AWS calls."""
    # Setup mocks
    mock_sagemaker = mock.Mock()
    mock_boto.return_value = mock_sagemaker

    # Run command
    result = train_command(...)

    # Verify AWS calls
    mock_sagemaker.create_training_job.assert_called_once()
```

### Fast Execution

No actual AWS calls or Docker builds:

```
120 tests completed in 2.34 seconds
```

### Test Coverage

Aim for >95% code coverage:

```bash
pytest --cov=easy_sm --cov-report=html
open htmlcov/index.html
```

## Code Quality Tools

### Type Checking

Run mypy for type checking:

```bash
mypy easy_sm/
```

### Linting and Formatting

Use ruff for linting and formatting:

```bash
# Check for issues
ruff check easy_sm/

# Fix automatically
ruff check --fix easy_sm/

# Format code
ruff format easy_sm/
```

### Pre-Commit Checks

Before committing:

```bash
# Run all checks
pytest && mypy easy_sm/ && ruff check easy_sm/
```

## Writing Tests

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_my_feature():
    """Test description."""
    # Arrange - Set up test data
    app_name = "test-app"
    config = Config(...)

    # Act - Execute functionality
    result = my_function(app_name, config)

    # Assert - Verify results
    assert result.success
    assert result.value == expected_value
```

### Mocking External Calls

Mock subprocess and AWS calls:

```python
import mock

@mock.patch('subprocess.run')
def test_with_subprocess(mock_run):
    """Test function that calls subprocess."""
    # Configure mock
    mock_run.return_value = mock.Mock(
        returncode=0,
        stdout="Success",
        stderr=""
    )

    # Run test
    result = my_function()

    # Verify subprocess was called correctly
    mock_run.assert_called_with(
        ["docker", "build", ...],
        check=True
    )
```

### Testing Error Cases

Test both success and failure paths:

```python
def test_function_with_invalid_input():
    """Test function rejects invalid input."""
    with pytest.raises(ValueError, match="Invalid app name"):
        my_function("../../../etc/passwd")
```

## Continuous Integration

Tests run automatically on:

- Pull requests
- Commits to main branch
- Tagged releases

GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.13'
      - run: pip install -e . -r base-requirements.txt
      - run: pytest --cov=easy_sm
      - run: mypy easy_sm/
      - run: ruff check easy_sm/
```

## See Also

- [Architecture](architecture.md) - System design
- [Code Style](code-style.md) - Coding conventions
- [Contributing](contributing.md) - Development workflow
