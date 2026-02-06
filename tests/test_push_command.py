"""
Tests for the push command.

Tests the push command for pushing Docker images to AWS ECR using mocked subprocess
interactions to avoid real AWS calls.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from easy_sm.__main__ import cli
from easy_sm.config.config import Config, ConfigManager


class TestPushCommand:
    """Tests for the push command."""

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

    def _create_config(self, app_name: str, easy_sm_module_dir: str | None = None) -> None:
        """Helper to create a config file."""
        if easy_sm_module_dir is None:
            easy_sm_module_dir = app_name

        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.13",
            easy_sm_module_dir=easy_sm_module_dir,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    def _create_easy_sm_structure(self, base_path: str) -> None:
        """Helper to create the easy_sm_base directory structure."""
        easy_sm_base = os.path.join(base_path, "easy_sm_base")
        os.makedirs(easy_sm_base, exist_ok=True)

        # Create push.sh script
        Path(os.path.join(easy_sm_base, "push.sh")).touch()
        os.chmod(os.path.join(easy_sm_base, "push.sh"), 0o755)

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_push_success(
        self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test successful push command."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        # Mock subprocess to succeed
        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli,
            [
                "--docker-tag",
                "v1.0.0",
                "push",
                "-a",
                app_name,
                "-p",
                "test-profile",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code == 0
        assert "Started pushing Docker image to AWS ECR" in result.output
        assert "Docker image pushed to ECR successfully!" in result.output
        mock_popen.assert_called_once()

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_push_with_iam_role(
        self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test push command with IAM role."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli,
            [
                "push",
                "-a",
                app_name,
                "-i",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code == 0
        assert "Docker image pushed to ECR successfully!" in result.output

        # Verify the command was called with correct parameters
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "arn:aws:iam::123456789012:role/SageMakerRole" in call_args

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_push_with_external_id(
        self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test push command with external ID."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli,
            [
                "push",
                "-a",
                app_name,
                "-i",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-e",
                "external-id-123",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_popen.call_args[0][0]
        assert "external-id-123" in call_args

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_push_with_custom_docker_tag(
        self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test push command with custom Docker tag."""
        app_name = "test-app"
        docker_tag = "custom-tag-123"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli,
            [
                "--docker-tag",
                docker_tag,
                "push",
                "-a",
                app_name,
                "-p",
                "test-profile",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_popen.call_args[0][0]
        assert docker_tag in call_args

    @patch("subprocess.check_output")
    def test_push_missing_config(
        self, mock_subprocess: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test push command fails without config file."""
        result = runner.invoke(
            cli,
            [
                "push",
                "-a",
                "nonexistent",
                "-p",
                "test-profile",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code != 0
        assert (
            "This is not a easy_sm directory" in result.output
            or result.exception is not None
        )

    @patch("subprocess.check_output")
    def test_push_missing_push_script(
        self, mock_subprocess: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test push command fails when push.sh is missing."""
        app_name = "test-app"
        self._create_config(app_name)
        # Create easy_sm_base directory but without push.sh
        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        os.makedirs(easy_sm_base, exist_ok=True)

        result = runner.invoke(
            cli,
            [
                "push",
                "-a",
                app_name,
                "-p",
                "test-profile",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code != 0
        assert (
            "This is not a easy_sm directory" in result.output
            or result.exception is not None
        )

    @patch("subprocess.check_output")
    def test_push_both_iam_role_and_profile_error(
        self, mock_subprocess: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test push command fails when both IAM role and profile are provided."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        result = runner.invoke(
            cli,
            [
                "push",
                "-a",
                app_name,
                "-i",
                "arn:aws:iam::123456789012:role/SageMakerRole",
                "-p",
                "test-profile",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code == 2
        assert "Only one of iam-role-arn and aws-profile can be used" in result.output

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_push_uses_config_defaults(
        self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test push command uses config file defaults when options not provided."""
        app_name = "test-app"
        config_aws_profile = "config-profile"
        config_aws_region = "eu-west-1"

        config = Config(
            image_name=app_name,
            aws_profile=config_aws_profile,
            aws_region=config_aws_region,
            python_version="3.13",
            easy_sm_module_dir=app_name,
            requirements_dir="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli,
            [
                "push",
                "-a",
                app_name,
            ],
        )

        assert result.exit_code == 0
        call_args = mock_popen.call_args[0][0]
        # Verify that config defaults were used
        assert config_aws_profile in call_args
        assert config_aws_region in call_args

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_push_subprocess_output_printed(
        self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str
    ) -> None:
        """Test that push command prints subprocess output."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        output_text = "Custom output from push.sh"
        mock_process = MagicMock()
        mock_process.stdout = [output_text]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli,
            [
                "push",
                "-a",
                app_name,
                "-p",
                "test-profile",
                "-r",
                "us-west-2",
            ],
        )

        assert result.exit_code == 0
        assert output_text in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
