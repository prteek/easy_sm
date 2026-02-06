# Functional Tests for Local Commands

This document describes the functional tests for the `local train` and `local deploy` commands.

## Overview

The test suite in `test_local_commands.py` provides comprehensive coverage for:

1. **Local Training** (`easy_sm local train`)
   - Command execution and configuration loading
   - Docker subprocess handling
   - Custom Docker tags
   - Error cases

2. **Local Deployment** (`easy_sm local deploy`)
   - Command execution and configuration loading
   - Docker subprocess handling
   - Custom Docker tags
   - Error cases

3. **Integration Tests**
   - Complete train → deploy workflow
   - Docker tag consistency
   - Sample app configuration validation
   - Required files and scripts validation

## Sample Application

The tests use the sample application in the `app/` directory which includes:

- **Configuration**: `app/app.json`
  - Image name: `esm`
  - Python version: 3.13
  - Module directory: `easy_sm_base`

- **Training Script**: `app/easy_sm_base/training/train`
  - Entry point: `app/easy_sm_base/training/training.py`
  - Trains an OLS model using statsmodels
  - Reads input data from `mpg.csv`
  - Outputs model to `model.mdl`

- **Serving Script**: `app/easy_sm_base/prediction/serve`
  - Flask app serving on port 8080
  - Endpoints:
    - `GET /ping` - Health check
    - `POST /invocations` - Predictions (CSV input/output)

- **Test Data**: `app/easy_sm_base/local_test/test_dir/`
  - Training data: `input/data/training/mpg.csv`
  - Model output: `model/model.mdl`
  - Hyperparameters: `input/config/hyperparameters.json`

## Test Classes

### TestLocalTrain

Tests for the `local train` command:

- `test_local_train_with_mock_subprocess`: Validates command structure with mocked subprocess
- `test_local_train_config_loading`: Verifies configuration loading from app.json
- `test_local_train_missing_config`: Tests error handling when config is missing
- `test_local_train_custom_docker_tag`: Validates custom Docker tag usage
- `test_local_train_command_structure`: Verifies correct subprocess arguments

### TestLocalDeploy

Tests for the `local deploy` command:

- `test_local_deploy_with_mock_subprocess`: Validates command structure with mocked subprocess
- `test_local_deploy_config_loading`: Verifies configuration loading
- `test_local_deploy_missing_config`: Tests error handling
- `test_local_deploy_custom_docker_tag`: Validates custom Docker tag usage
- `test_local_deploy_command_structure`: Verifies correct subprocess arguments

### TestLocalTrainAndDeployIntegration

Integration tests:

- `test_training_and_deployment_workflow`: Tests the complete train → deploy sequence
- `test_docker_tag_consistency`: Verifies Docker tag consistency across commands
- `test_app_json_format_validation`: Validates app.json structure and content
- `test_training_scripts_exist`: Checks that all required scripts exist
- `test_serve_script_is_executable`: Validates serve script permissions
- `test_training_script_is_executable`: Validates training script permissions

## Running the Tests

### Prerequisites

Install development dependencies:

```bash
pip install -e . -r requirements.txt
```

### Run All Local Command Tests

```bash
pytest tests/test_local_commands.py -v
```

### Run Specific Test Class

```bash
# Run only local train tests
pytest tests/test_local_commands.py::TestLocalTrain -v

# Run only local deploy tests
pytest tests/test_local_commands.py::TestLocalDeploy -v

# Run only integration tests
pytest tests/test_local_commands.py::TestLocalTrainAndDeployIntegration -v
```

### Run Specific Test

```bash
pytest tests/test_local_commands.py::TestLocalTrain::test_local_train_config_loading -v
```

### Run with Coverage

```bash
pytest tests/test_local_commands.py --cov=easy_sm --cov-report=html
```

### Run All Tests

```bash
pytest tests/ -v
```

## Test Strategy

The tests use the following strategies:

### Unit Tests with Mocking

Most tests use `unittest.mock.patch` to mock the `subprocess.Popen` calls. This allows testing:
- Command structure and argument passing
- Configuration loading and handling
- Error cases and edge cases
- Docker tag propagation

**Advantages**:
- Fast execution (no Docker required)
- Reliable and deterministic
- Easy to debug
- No side effects

**Example**:
```python
@patch("easy_sm.commands.helpers.subprocess.Popen")
def test_local_train_with_mock_subprocess(self, mock_popen, runner, app_dir):
    mock_process = MagicMock()
    mock_process.stdout = []
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    result = runner.invoke(cli, ["local", "train", "-a", "app"])
    assert result.exit_code == 0
```

### Configuration Validation Tests

Tests that validate:
- Configuration loading from JSON files
- Required fields and their values
- File and script existence
- File permissions

**Example**:
```python
def test_app_json_format_validation(self, app_dir):
    with open("app.json", "r") as f:
        config_data = json.load(f)

    assert "image_name" in config_data
    assert config_data["python_version"] in ["3.10", "3.11", "3.12", "3.13"]
```

## Integration Test Requirements

For full integration tests with actual Docker execution (future enhancement), you would need:

1. **Docker**: Must be installed and running
2. **Docker Image**: Must be built first with `easy_sm build --app-name app`
3. **Dependencies**: Training dependencies must be installed in the Docker image
   - pandas
   - statsmodels
   - joblib
   - flask

### Future End-to-End Tests

Once Docker infrastructure is available, additional tests could be added:

```python
@pytest.mark.docker
def test_local_train_actual_training(runner, app_dir):
    """Test actual training execution with Docker."""
    result = runner.invoke(cli, ["local", "train", "-a", "app"])
    assert result.exit_code == 0

    # Verify model file was created
    model_path = os.path.join(app_dir, "easy_sm_base/local_test/test_dir/model/model.mdl")
    assert os.path.exists(model_path)

@pytest.mark.docker
def test_local_deploy_http_requests(runner, app_dir):
    """Test deployment and HTTP requests to the service."""
    # Start deployment in background
    result = runner.invoke(cli, ["local", "deploy", "-a", "app"])
    assert result.exit_code == 0

    # Wait for service to start
    time.sleep(3)

    # Test ping endpoint
    response = requests.get("http://localhost:8080/ping")
    assert response.status_code == 200

    # Test predictions endpoint
    csv_data = "1,100\n2,150"
    response = requests.post(
        "http://localhost:8080/invocations",
        data=csv_data,
        headers={"Content-Type": "text/csv"}
    )
    assert response.status_code == 200
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError` for sagemaker or other dependencies:

```bash
pip install -e . -r requirements.txt
```

### Test Discovery Issues

If tests aren't being discovered:

```bash
# Ensure __init__.py exists in tests directory
touch tests/__init__.py

# Try running with explicit paths
pytest tests/test_local_commands.py::TestLocalTrain -v
```

### Fixture Scope Issues

Tests use function-scoped fixtures. If you need session-scoped behavior:

```python
@pytest.fixture(scope="session")
def app_dir():
    # Fixture code
    pass
```

## Contributing

When adding new tests:

1. Use descriptive test names that explain what is being tested
2. Add docstrings explaining the test purpose
3. Use fixtures for setup/teardown
4. Mock external dependencies (subprocess, Docker)
5. Keep tests isolated and independent
6. Add parametrized tests for multiple scenarios

Example:

```python
@pytest.mark.parametrize("docker_tag", ["latest", "v1.0.0", "test-tag"])
def test_local_train_with_various_tags(self, runner, app_dir, docker_tag):
    """Test local train command with various Docker tags."""
    with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli, ["--docker-tag", docker_tag, "local", "train", "-a", "app"]
        )

        assert result.exit_code == 0
        call_args = mock_popen.call_args[0][0]
        assert docker_tag in " ".join(call_args)
```

## References

- [pytest documentation](https://docs.pytest.org/)
- [Click testing](https://click.palletsprojects.com/en/latest/testing/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Sample App Configuration](../app/app.json)
- [Local Commands Implementation](../easy_sm/commands/local.py)
