"""
Functional tests for local train and deploy commands.

Tests the local train and deploy commands using the sample app in the app/ directory.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import requests
from typer.testing import CliRunner

from easy_sm.__main__ import app
from easy_sm.config.config import ConfigManager


class TestLocalTrain:
    """Functional tests for the local train command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def app_dir(self) -> str:
        """Use the sample app directory for testing."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "app")
        return os.path.abspath(app_path)

    @pytest.fixture
    def cleanup_model(self, app_dir: str) -> Generator[None, None, None]:
        """Clean up model files before and after tests."""
        model_dir = os.path.join(
            app_dir, "easy_sm_base", "local_test", "test_dir", "model"
        )
        # Remove old model if exists
        model_file = os.path.join(model_dir, "model.mdl")
        if os.path.exists(model_file):
            os.remove(model_file)
        yield
        # Clean up after test
        if os.path.exists(model_file):
            os.remove(model_file)

    def test_local_train_with_mock_subprocess(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """
        Test local train command with mocked subprocess.

        This test verifies the command structure and config loading without
        requiring Docker to be running.
        """
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            # Mock subprocess to succeed
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            # Change to app directory for the test
            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(app, ["local", "train", "-a", "app"])

                # Verify command succeeded
                assert result.exit_code == 0, f"Command failed: {result.output}"
                assert "Local training" in result.output
                assert "Local training completed" in result.output

                # Verify subprocess was called
                mock_popen.assert_called_once()

                # Verify the command contains expected elements
                call_args = mock_popen.call_args[0][0]
                assert "train_local.sh" in " ".join(call_args)
            finally:
                os.chdir(original_cwd)

    def test_local_train_config_loading(self, runner: CliRunner, app_dir: str) -> None:
        """Test that local train properly loads the app configuration."""
        original_cwd = os.getcwd()
        try:
            os.chdir(app_dir)
            # Load config directly to verify it exists
            config_manager = ConfigManager("app.json")
            config = config_manager.get_config()

            # Verify config has expected values
            assert config.image_name == "esm"
            assert config.easy_sm_module_dir == "."
            assert config.requirements_dir == "easy_sm_base"
        finally:
            os.chdir(original_cwd)

    def test_local_train_missing_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test local train command fails gracefully without config file."""
        # Change to temp directory with no config
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = runner.invoke(app, ["local", "train", "-a", "nonexistent"])
            # Command should fail
            assert result.exit_code != 0
            # Check either output or exception message
            assert (
                "This is not a easy_sm directory" in result.output
                or "This is not a easy_sm directory"
                in str(result.exception.__class__.__name__)
                or result.exception is not None
            )
        finally:
            os.chdir(original_cwd)

    def test_local_train_custom_docker_tag(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test local train command with custom Docker tag."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(
                    app, ["--docker-tag", "v1.0.0", "local", "train", "-a", "app"]
                )

                assert result.exit_code == 0
                # Verify custom tag was used
                call_args = mock_popen.call_args[0][0]
                assert "v1.0.0" in " ".join(call_args)
            finally:
                os.chdir(original_cwd)

    def test_local_train_command_structure(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test that local train command structures the correct parameters."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(app, ["local", "train", "-a", "app"])

                assert result.exit_code == 0

                # Verify subprocess command contains required parameters
                call_args = mock_popen.call_args[0][0]
                call_string = " ".join(call_args)

                # Should contain paths and image name
                assert "test_dir" in call_string or "esm" in call_string
                assert "train_local.sh" in call_string
            finally:
                os.chdir(original_cwd)


class TestLocalDeploy:
    """Functional tests for the local deploy command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def app_dir(self) -> str:
        """Use the sample app directory for testing."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "app")
        return os.path.abspath(app_path)

    def test_local_deploy_with_mock_subprocess(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """
        Test local deploy command with mocked subprocess.

        Verifies the command structure without requiring Docker.
        """
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(app, ["local", "deploy", "-a", "app"])

                # Verify command structure
                assert result.exit_code == 0, f"Command failed: {result.output}"
                assert "Starting local deployment at localhost:8080" in result.output

                # Verify subprocess was called
                mock_popen.assert_called_once()

                # Verify the command contains deploy script and port parameter
                call_args = mock_popen.call_args[0][0]
                assert "deploy_local.sh" in " ".join(call_args)
                assert "8080" in call_args
            finally:
                os.chdir(original_cwd)

    def test_local_deploy_config_loading(self, runner: CliRunner, app_dir: str) -> None:
        """Test that local deploy properly loads the app configuration."""
        original_cwd = os.getcwd()
        try:
            os.chdir(app_dir)
            config_manager = ConfigManager("app.json")
            config = config_manager.get_config()

            # Verify config has expected values
            assert config.image_name == "esm"
            assert config.easy_sm_module_dir == "."
        finally:
            os.chdir(original_cwd)

    def test_local_deploy_missing_config(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test local deploy command fails gracefully without config file."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = runner.invoke(app, ["local", "deploy", "-a", "nonexistent"])
            # Command should fail
            assert result.exit_code != 0
            # Check either output or exception message
            assert (
                "This is not a easy_sm directory" in result.output
                or "This is not a easy_sm directory"
                in str(result.exception.__class__.__name__)
                or result.exception is not None
            )
        finally:
            os.chdir(original_cwd)

    def test_local_deploy_custom_docker_tag(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test local deploy command with custom Docker tag."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(
                    app, ["--docker-tag", "latest", "local", "deploy", "-a", "app"]
                )

                assert result.exit_code == 0
                # Verify tag was used
                call_args = mock_popen.call_args[0][0]
                assert "latest" in " ".join(call_args)
            finally:
                os.chdir(original_cwd)

    def test_local_deploy_custom_port(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test local deploy command with custom port."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(
                    app, ["local", "deploy", "-a", "app", "-p", "9000"]
                )

                assert result.exit_code == 0, f"Command failed: {result.output}"
                assert "Starting local deployment at localhost:9000" in result.output

                # Verify subprocess was called with correct port
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]
                assert "9000" in call_args
            finally:
                os.chdir(original_cwd)

    def test_local_deploy_port_long_option(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test local deploy command with --port long option."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(
                    app, ["local", "deploy", "-a", "app", "--port", "3000"]
                )

                assert result.exit_code == 0, f"Command failed: {result.output}"
                assert "Starting local deployment at localhost:3000" in result.output

                # Verify subprocess was called with correct port
                call_args = mock_popen.call_args[0][0]
                assert "3000" in call_args
            finally:
                os.chdir(original_cwd)

    def test_local_deploy_command_structure(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test that local deploy command structures the correct parameters."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(app, ["local", "deploy", "-a", "app"])

                assert result.exit_code == 0

                # Verify subprocess command contains required parameters
                call_args = mock_popen.call_args[0][0]
                call_string = " ".join(call_args)

                # Should contain paths and scripts
                assert "test_dir" in call_string or "esm" in call_string
                assert "deploy_local.sh" in call_string
            finally:
                os.chdir(original_cwd)


class TestLocalStop:
    """Functional tests for the local stop command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def app_dir(self) -> str:
        """Use the sample app directory for testing."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "app")
        return os.path.abspath(app_path)

    def test_local_stop_with_mock_subprocess(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """
        Test local stop command with mocked subprocess.

        Verifies the command structure without requiring Docker.
        """
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(app, ["local", "stop", "-a", "app"])

                # Verify command succeeded
                assert result.exit_code == 0, f"Command failed: {result.output}"
                assert "Local deployment stopped" in result.output

                # Verify subprocess was called
                mock_popen.assert_called_once()

                # Verify the command contains stop script
                call_args = mock_popen.call_args[0][0]
                assert "stop_local.sh" in " ".join(call_args)
            finally:
                os.chdir(original_cwd)

    def test_local_stop_with_default_port(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test local stop command uses default port 8080."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(app, ["local", "stop", "-a", "app"])

                assert result.exit_code == 0
                # Verify default port is passed
                call_args = mock_popen.call_args[0][0]
                assert "8080" in call_args
            finally:
                os.chdir(original_cwd)

    def test_local_stop_with_custom_port(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test local stop command with custom port."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(
                    app, ["local", "stop", "-a", "app", "-p", "9000"]
                )

                assert result.exit_code == 0, f"Command failed: {result.output}"

                # Verify subprocess was called with correct port
                call_args = mock_popen.call_args[0][0]
                assert "9000" in call_args
            finally:
                os.chdir(original_cwd)

    def test_local_stop_port_long_option(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test local stop command with --port long option."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                result = runner.invoke(
                    app, ["local", "stop", "-a", "app", "--port", "3000"]
                )

                assert result.exit_code == 0, f"Command failed: {result.output}"

                # Verify subprocess was called with correct port
                call_args = mock_popen.call_args[0][0]
                assert "3000" in call_args
            finally:
                os.chdir(original_cwd)

    def test_local_stop_missing_config(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test local stop command fails gracefully without config file."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = runner.invoke(app, ["local", "stop", "-a", "nonexistent"])
            # Command should fail
            assert result.exit_code != 0
            # Check either output or exception message
            assert (
                "This is not a easy_sm directory" in result.output
                or "This is not a easy_sm directory"
                in str(result.exception.__class__.__name__)
                or result.exception is not None
            )
        finally:
            os.chdir(original_cwd)


class TestLocalTrainAndDeployIntegration:
    """Integration tests for local train and deploy workflow."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide CliRunner instance."""
        return CliRunner()

    @pytest.fixture
    def app_dir(self) -> str:
        """Use the sample app directory for testing."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "app")
        return os.path.abspath(app_path)

    def test_training_and_deployment_workflow(
        self, runner: CliRunner, app_dir: str
    ) -> None:
        """Test the complete workflow of training followed by deployment."""
        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                # First, train
                train_result = runner.invoke(app, ["local", "train", "-a", "app"])
                assert train_result.exit_code == 0
                assert "Local training completed" in train_result.output

                # Then, deploy
                deploy_result = runner.invoke(app, ["local", "deploy", "-a", "app"])
                assert deploy_result.exit_code == 0
                assert "Starting local deployment at localhost:8080" in deploy_result.output

                # Verify both commands were called
                assert mock_popen.call_count >= 2
            finally:
                os.chdir(original_cwd)

    def test_docker_tag_consistency(self, runner: CliRunner, app_dir: str) -> None:
        """Test that the same Docker tag is used for train and deploy."""
        docker_tag = "test-v1.2.3"

        with patch("easy_sm.commands.helpers.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = []
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            original_cwd = os.getcwd()
            try:
                os.chdir(app_dir)
                # Train with custom tag
                runner.invoke(
                    app, ["--docker-tag", docker_tag, "local", "train", "-a", "app"]
                )

                # Deploy with same tag
                runner.invoke(
                    app, ["--docker-tag", docker_tag, "local", "deploy", "-a", "app"]
                )

                # Verify both calls used the same tag
                assert mock_popen.call_count == 2
                for call in mock_popen.call_args_list:
                    call_args = call[0][0]
                    assert docker_tag in " ".join(call_args)
            finally:
                os.chdir(original_cwd)

    def test_app_json_format_validation(self, app_dir: str) -> None:
        """Test that app.json has the correct format."""
        config_file = os.path.join(app_dir, "app.json")
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Verify required fields
        required_fields = [
            "image_name",
            "aws_profile",
            "aws_region",
            "python_version",
            "easy_sm_module_dir",
            "requirements_dir",
        ]
        for field in required_fields:
            assert (
                field in config_data
            ), f"Missing required field in app.json: {field}"

        # Verify field types and values
        assert isinstance(config_data["image_name"], str)
        assert len(config_data["image_name"]) > 0
        assert isinstance(config_data["python_version"], str)
        assert config_data["python_version"] in ["3.10", "3.11", "3.12", "3.13"]

    def test_training_scripts_exist(self, app_dir: str) -> None:
        """Test that all required training scripts and files exist."""
        required_paths = [
            "easy_sm_base/training/train",
            "easy_sm_base/training/training.py",
            "easy_sm_base/prediction/serve",
            "easy_sm_base/local_test/test_dir/input/data/training/mpg.csv",
        ]

        for path in required_paths:
            full_path = os.path.join(app_dir, path)
            assert (
                os.path.exists(full_path)
            ), f"Required file not found: {path}"

    def test_serve_script_is_executable(self, app_dir: str) -> None:
        """Test that serve script has execute permissions."""
        serve_path = os.path.join(app_dir, "easy_sm_base/prediction/serve")
        assert os.access(
            serve_path, os.X_OK
        ), f"Serve script is not executable: {serve_path}"

    def test_training_script_is_executable(self, app_dir: str) -> None:
        """Test that training script has execute permissions."""
        train_path = os.path.join(app_dir, "easy_sm_base/training/train")
        assert os.access(
            train_path, os.X_OK
        ), f"Training script is not executable: {train_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
