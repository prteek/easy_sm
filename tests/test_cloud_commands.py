"""
Unit tests for cloud command group.

Tests the cloud commands (train, deploy, upload-data, etc.) using mocked SageMaker interactions
to avoid real AWS calls. Each test verifies command structure, parameter passing, and error handling.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from easy_sm.__main__ import app
from easy_sm.config.config import Config, ConfigManager


class TestCloudUploadData:
    """Tests for the cloud upload-data command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_upload_data_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful upload-data command."""
        app_name = "test-app"
        self._create_config(app_name)

        # Create input directory
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir)
        Path(os.path.join(input_dir, "data.csv")).touch()

        # Mock SageMakerClient
        mock_client = MagicMock()
        mock_client.upload_data.return_value = "s3://bucket/data"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "upload-data",
                "-a",
                app_name,
                "-i",
                input_dir,
                "-t",
                "s3://bucket/data",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Started uploading data to S3" in result.output
        assert "Data uploaded to s3://bucket/data successfully" in result.output
        mock_client.upload_data.assert_called_once_with(input_dir, "s3://bucket/data")

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_upload_data_missing_config(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test upload-data fails without config file."""
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir)

        result = runner.invoke(
            app,
            [
                "cloud",
                "upload-data",
                "-a",
                "nonexistent",
                "-i",
                input_dir,
                "-t",
                "s3://bucket/data",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code != 0
        assert (
            "This is not a easy_sm directory" in result.output
            or result.exception is not None
        )

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_upload_data_missing_input_dir(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test upload-data fails when input directory doesn't exist."""
        app_name = "test-app"
        self._create_config(app_name)

        result = runner.invoke(
            app,
            [
                "cloud",
                "upload-data",
                "-a",
                app_name,
                "-i",
                "/nonexistent/path",
                "-t",
                "s3://bucket/data",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code != 0


class TestCloudTrain:
    """Tests for the cloud train command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_train_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud train command."""
        app_name = "test-app"
        self._create_config(app_name)

        # Mock SageMakerClient
        mock_client = MagicMock()
        mock_client.train.return_value = "s3://bucket/model.tar.gz"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "train",
                "-a",
                app_name,
                "-i",
                "s3://bucket/training",
                "-o",
                "s3://bucket/output",
                "-e",
                "ml.m5.large",
                "-c",
                "1",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "training-job",
            ],
        )

        assert result.exit_code == 0
        assert "Started training on SageMaker" in result.output
        assert "Training on SageMaker succeeded" in result.output
        assert "s3://bucket/model.tar.gz" in result.output

        # Verify the mock was called with correct parameters
        mock_client.train.assert_called_once()
        call_kwargs = mock_client.train.call_args[1]
        assert call_kwargs["image_name"] == f"{app_name}:latest"
        assert call_kwargs["input_s3_data_location"] == "s3://bucket/training"
        assert call_kwargs["train_instance_type"] == "ml.m5.large"
        assert call_kwargs["instance_count"] == 1
        assert call_kwargs["output_path"] == "s3://bucket/output"
        assert call_kwargs["base_job_name"] == "training-job"

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_train_with_custom_docker_tag(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud train with custom Docker tag."""
        app_name = "test-app"
        docker_tag = "v1.2.3"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.train.return_value = "s3://bucket/model.tar.gz"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "--docker-tag",
                docker_tag,
                "cloud",
                "train",
                "-a",
                app_name,
                "-i",
                "s3://bucket/training",
                "-o",
                "s3://bucket/output",
                "-e",
                "ml.m5.large",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "training-job",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.train.call_args[1]
        assert call_kwargs["image_name"] == f"{app_name}:{docker_tag}"

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_train_with_multiple_instances(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud train with multiple instances."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.train.return_value = "s3://bucket/model.tar.gz"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "train",
                "-a",
                app_name,
                "-i",
                "s3://bucket/training",
                "-o",
                "s3://bucket/output",
                "-e",
                "ml.m5.large",
                "-c",
                "4",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "training-job",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.train.call_args[1]
        assert call_kwargs["instance_count"] == 4

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_train_missing_config(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud train fails without config file."""
        result = runner.invoke(
            app,
            [
                "cloud",
                "train",
                "-a",
                "nonexistent",
                "-i",
                "s3://bucket/training",
                "-o",
                "s3://bucket/output",
                "-e",
                "ml.m5.large",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "training-job",
            ],
        )

        assert result.exit_code != 0


class TestCloudDeploy:
    """Tests for the cloud deploy command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_deploy_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud deploy command."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.deploy.return_value = "test-endpoint"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "deploy",
                "-a",
                app_name,
                "-m",
                "s3://bucket/model.tar.gz",
                "-e",
                "ml.m5.large",
                "-n",
                "test-endpoint",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Started deployment on SageMaker" in result.output
        assert "Endpoint name: test-endpoint" in result.output

        mock_client.deploy.assert_called_once()
        call_kwargs = mock_client.deploy.call_args[1]
        assert call_kwargs["image_name"] == f"{app_name}:latest"
        assert call_kwargs["s3_model_location"] == "s3://bucket/model.tar.gz"
        assert call_kwargs["instance_type"] == "ml.m5.large"
        assert call_kwargs["endpoint_name"] == "test-endpoint"
        assert call_kwargs["instance_count"] == 1

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_deploy_with_multiple_instances(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud deploy with multiple instances."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.deploy.return_value = "test-endpoint"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "deploy",
                "-a",
                app_name,
                "-m",
                "s3://bucket/model.tar.gz",
                "-e",
                "ml.m5.large",
                "-c",
                "3",
                "-n",
                "test-endpoint",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.deploy.call_args[1]
        assert call_kwargs["instance_count"] == 3


class TestCloudDeployServerless:
    """Tests for the cloud deploy-serverless command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_deploy_serverless_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud deploy-serverless command."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.deploy_serverless.return_value = "test-serverless-endpoint"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "deploy-serverless",
                "-a",
                app_name,
                "-m",
                "s3://bucket/model.tar.gz",
                "-s",
                "2048",
                "-n",
                "test-serverless-endpoint",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Started deployment on SageMaker" in result.output
        assert "Endpoint name: test-serverless-endpoint" in result.output

        mock_client.deploy_serverless.assert_called_once()
        call_kwargs = mock_client.deploy_serverless.call_args[1]
        assert call_kwargs["memory_size_in_mb"] == 2048
        assert call_kwargs["max_concurrency"] == 5  # default

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_deploy_serverless_with_max_concurrency(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud deploy-serverless with custom max concurrency."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.deploy_serverless.return_value = "test-serverless-endpoint"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "deploy-serverless",
                "-a",
                app_name,
                "-m",
                "s3://bucket/model.tar.gz",
                "-s",
                "4096",
                "-n",
                "test-serverless-endpoint",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-mc",
                "10",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.deploy_serverless.call_args[1]
        assert call_kwargs["max_concurrency"] == 10


class TestCloudBatchTransform:
    """Tests for the cloud batch-transform command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_batch_transform_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud batch-transform command."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.batch_transform.return_value = "Completed"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "batch-transform",
                "-a",
                app_name,
                "-m",
                "s3://bucket/model.tar.gz",
                "-i",
                "s3://bucket/input",
                "-o",
                "s3://bucket/output",
                "--ec2-type",
                "ml.m5.large",
                "--num-instances",
                "2",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Started configuration of batch transform on SageMaker" in result.output

        mock_client.batch_transform.assert_called_once()
        call_kwargs = mock_client.batch_transform.call_args[1]
        assert call_kwargs["s3_model_location"] == "s3://bucket/model.tar.gz"
        assert call_kwargs["s3_input_location"] == "s3://bucket/input"
        assert call_kwargs["s3_output_location"] == "s3://bucket/output"
        assert call_kwargs["transform_instance_type"] == "ml.m5.large"
        assert call_kwargs["transform_instance_count"] == 2

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_batch_transform_with_wait(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud batch-transform with wait flag."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.batch_transform.return_value = "Completed"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "batch-transform",
                "-a",
                app_name,
                "-m",
                "s3://bucket/model.tar.gz",
                "-i",
                "s3://bucket/input",
                "-o",
                "s3://bucket/output",
                "--ec2-type",
                "ml.m5.large",
                "--num-instances",
                "1",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-w",
            ],
        )

        assert result.exit_code == 0
        assert "Batch transform on SageMaker finished with status: Completed" in result.output

        call_kwargs = mock_client.batch_transform.call_args[1]
        assert call_kwargs["wait"] is True

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_batch_transform_with_custom_job_name(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud batch-transform with custom job name."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.batch_transform.return_value = "Completed"
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "batch-transform",
                "-a",
                app_name,
                "-m",
                "s3://bucket/model.tar.gz",
                "-i",
                "s3://bucket/input",
                "-o",
                "s3://bucket/output",
                "--ec2-type",
                "ml.m5.large",
                "--num-instances",
                "1",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "custom-job-name",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.batch_transform.call_args[1]
        assert call_kwargs["job_name"] == "custom-job-name"


class TestCloudProcess:
    """Tests for the cloud process command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_process_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud process command."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "process",
                "-a",
                app_name,
                "-e",
                "ml.m5.large",
                "-f",
                "process.py",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "process-job",
            ],
        )

        assert result.exit_code == 0
        assert "Started processing job on SageMaker" in result.output
        assert "Processing job on SageMaker succeeded" in result.output

        mock_client.process.assert_called_once()
        call_kwargs = mock_client.process.call_args[1]
        assert call_kwargs["file"] == "process.py"
        assert call_kwargs["processing_instance_type"] == "ml.m5.large"
        assert call_kwargs["instance_count"] == 1

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_process_with_s3_locations(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud process with S3 input/output locations."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "process",
                "-a",
                app_name,
                "-e",
                "ml.m5.large",
                "-f",
                "process.py",
                "-i",
                "s3://bucket/input",
                "-o",
                "s3://bucket/output",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "process-job",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.process.call_args[1]
        assert call_kwargs["s3_input_location"] == "s3://bucket/input"
        assert call_kwargs["s3_output_location"] == "s3://bucket/output"

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_process_with_input_sharded(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud process with input sharded flag."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "process",
                "-a",
                app_name,
                "-e",
                "ml.m5.large",
                "-f",
                "process.py",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "process-job",
                "-is",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.process.call_args[1]
        assert call_kwargs["input_sharded"] is True


class TestCloudDeleteEndpoint:
    """Tests for the cloud delete-endpoint command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_delete_endpoint_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud delete-endpoint command."""
        app_name = "test-app"
        endpoint_name = "test-endpoint"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "delete-endpoint",
                "-a",
                app_name,
                "-n",
                endpoint_name,
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert f"Endpoint {endpoint_name} has been deleted" in result.output

        mock_client.shutdown_endpoint.assert_called_once_with(endpoint_name)
        mock_client.delete_endpoint_config.assert_not_called()

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_delete_endpoint_with_config(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud delete-endpoint command with --delete-config flag."""
        app_name = "test-app"
        endpoint_name = "test-endpoint"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "delete-endpoint",
                "-a",
                app_name,
                "-n",
                endpoint_name,
                "--delete-config",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert f"Endpoint {endpoint_name} has been deleted" in result.output
        assert f"Endpoint config {endpoint_name}-config has been deleted" in result.output

        mock_client.shutdown_endpoint.assert_called_once_with(endpoint_name)
        mock_client.delete_endpoint_config.assert_called_once_with(
            f"{endpoint_name}-config"
        )


class TestCloudListEndpoints:
    """Tests for the cloud list-endpoints command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_list_endpoints_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud list-endpoints command."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = [
            {
                "EndpointName": "endpoint-1",
                "EndpointStatus": "InService",
                "CreationTime": "2024-01-01T00:00:00Z",
            },
            {
                "EndpointName": "endpoint-2",
                "EndpointStatus": "Creating",
                "CreationTime": "2024-01-02T00:00:00Z",
            },
        ]
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "list-endpoints",
                "-a",
                app_name,
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Found 2 endpoint(s)" in result.output
        assert "endpoint-1" in result.output
        assert "endpoint-2" in result.output
        assert "InService" in result.output
        assert "Creating" in result.output

        mock_client.list_endpoints.assert_called_once()

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_list_endpoints_empty(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud list-endpoints command with no endpoints."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = []
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "list-endpoints",
                "-a",
                app_name,
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "No endpoints found" in result.output

        mock_client.list_endpoints.assert_called_once()


class TestCloudListTrainingJobs:
    """Tests for the cloud list-training-jobs command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_list_training_jobs_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud list-training-jobs command."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = [
            {
                "TrainingJobName": "job-1",
                "TrainingJobStatus": "Completed",
                "CreationTime": "2024-01-01T00:00:00Z",
            },
            {
                "TrainingJobName": "job-2",
                "TrainingJobStatus": "InProgress",
                "CreationTime": "2024-01-02T00:00:00Z",
            },
        ]
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "list-training-jobs",
                "-a",
                app_name,
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Found 2 training job(s)" in result.output
        assert "job-1" in result.output
        assert "job-2" in result.output
        assert "Completed" in result.output
        assert "InProgress" in result.output

        mock_client.list_training_jobs.assert_called_once_with(
            max_results=5, name_contains=None
        )

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_list_training_jobs_with_max_results(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud list-training-jobs command with custom max results."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = [
            {
                "TrainingJobName": "job-1",
                "TrainingJobStatus": "Completed",
                "CreationTime": "2024-01-01T00:00:00Z",
            }
        ]
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "list-training-jobs",
                "-a",
                app_name,
                "-m",
                "10",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Found 1 training job(s)" in result.output

        mock_client.list_training_jobs.assert_called_once_with(
            max_results=10, name_contains=None
        )

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_list_training_jobs_with_base_job_name(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud list-training-jobs command with base job name filter."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = [
            {
                "TrainingJobName": "my-job-1",
                "TrainingJobStatus": "Completed",
                "CreationTime": "2024-01-01T00:00:00Z",
            }
        ]
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "list-training-jobs",
                "-a",
                app_name,
                "-b",
                "my-job",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "Found 1 training job(s)" in result.output
        assert "my-job-1" in result.output

        mock_client.list_training_jobs.assert_called_once_with(
            max_results=5, name_contains="my-job"
        )

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_list_training_jobs_empty(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud list-training-jobs command with no jobs."""
        app_name = "test-app"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = []
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "list-training-jobs",
                "-a",
                app_name,
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
            ],
        )

        assert result.exit_code == 0
        assert "No training jobs found" in result.output

        mock_client.list_training_jobs.assert_called_once()


class TestCloudMake:
    """Tests for the cloud make command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def _create_config(self, app_name: str) -> None:
        """Helper to create a config file."""
        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_make_success(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful cloud make command."""
        app_name = "test-app"
        target = "build-feature"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "make",
                "-a",
                app_name,
                "-e",
                "ml.m5.large",
                "-t",
                target,
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "make-job",
            ],
        )

        assert result.exit_code == 0
        assert f"Building {target} on SageMaker" in result.output
        assert f"{target} built on SageMaker successfully!" in result.output

        mock_client.make.assert_called_once()
        call_kwargs = mock_client.make.call_args[1]
        assert call_kwargs["target"] == target

    @patch("easy_sm.sagemaker.sagemaker.SageMakerClient")
    def test_make_with_s3_locations(
        self, mock_sagemaker_client: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test cloud make with S3 input/output locations."""
        app_name = "test-app"
        target = "build-feature"
        self._create_config(app_name)

        mock_client = MagicMock()
        mock_sagemaker_client.return_value = mock_client

        result = runner.invoke(
            app,
            [
                "cloud",
                "make",
                "-a",
                app_name,
                "-e",
                "ml.m5.large",
                "-t",
                target,
                "-i",
                "s3://bucket/input",
                "-o",
                "s3://bucket/output",
                "-r",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-n",
                "make-job",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_client.make.call_args[1]
        assert call_kwargs["s3_input_location"] == "s3://bucket/input"
        assert call_kwargs["s3_output_location"] == "s3://bucket/output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
