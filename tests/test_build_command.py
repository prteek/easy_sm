import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from easy_sm.__main__ import cli
from easy_sm.config.config import Config, ConfigManager


class TestBuildCommand:
    """Test suite for the build command"""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Fixture to provide CliRunner instance"""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Fixture to provide a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    def _create_config(self, app_name: str, easy_sm_module_dir: str | None = None) -> None:
        """Helper to create a config file"""
        if easy_sm_module_dir is None:
            easy_sm_module_dir = app_name

        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=easy_sm_module_dir,
            requirements_file_name="requirements.txt",
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)

    def _create_easy_sm_structure(self, base_path: str) -> None:
        """Helper to create the easy_sm_base directory structure"""
        easy_sm_base = os.path.join(base_path, "easy_sm_base")
        os.makedirs(os.path.join(easy_sm_base, "training"), exist_ok=True)
        os.makedirs(os.path.join(easy_sm_base, "prediction"), exist_ok=True)

        # Create required files
        Path(os.path.join(easy_sm_base, "build.sh")).touch()
        Path(os.path.join(easy_sm_base, "Dockerfile")).touch()
        Path(os.path.join(easy_sm_base, "training", "train")).touch()
        Path(os.path.join(easy_sm_base, "prediction", "serve")).touch()
        Path(os.path.join(easy_sm_base, "executor.sh")).touch()

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_successful(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test successful build command execution"""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        # Mock subprocess to succeed
        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code == 0
        assert "Started building SageMaker Docker image" in result.output
        assert "Docker image built successfully" in result.output

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_with_custom_docker_tag(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build with custom docker tag via CLI option"""
        app_name = "my-app"
        docker_tag = "v1.2.3"

        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli, ["--docker-tag", docker_tag, "build", "-a", app_name]
        )

        assert result.exit_code == 0
        # Verify the subprocess was called with correct tag
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert docker_tag in call_args

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_with_default_docker_tag(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build with default docker tag (latest)"""
        app_name = "default-tag-app"

        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        # Pass obj with default docker_tag
        result = runner.invoke(
            cli, ["build", "-a", app_name], obj={"docker_tag": "latest"}
        )

        assert result.exit_code == 0

    def test_build_missing_config_file(self, runner: CliRunner, temp_dir: str) -> None:
        """Test build fails when config file is missing"""
        app_name = "nonexistent-app"

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "This is not a easy_sm directory" in result.output

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_missing_easy_sm_base_directory(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build fails when easy_sm_base directory is missing"""
        app_name = "no-easy-sm-app"
        self._create_config(app_name)
        # Don't create the easy_sm_base structure

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "This is not a easy_sm directory" in result.output

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_missing_build_script(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build fails when build.sh is missing"""
        app_name = "no-build-script-app"
        self._create_config(app_name)

        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        os.makedirs(os.path.join(easy_sm_base, "training"), exist_ok=True)
        os.makedirs(os.path.join(easy_sm_base, "prediction"), exist_ok=True)

        # Create all files except build.sh
        Path(os.path.join(easy_sm_base, "Dockerfile")).touch()
        Path(os.path.join(easy_sm_base, "training", "train")).touch()
        Path(os.path.join(easy_sm_base, "prediction", "serve")).touch()
        Path(os.path.join(easy_sm_base, "executor.sh")).touch()

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "This is not a easy_sm directory" in result.output

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_missing_train_file(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build fails when training/train file is missing"""
        app_name = "no-train-file-app"
        self._create_config(app_name)

        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        os.makedirs(os.path.join(easy_sm_base, "training"), exist_ok=True)
        os.makedirs(os.path.join(easy_sm_base, "prediction"), exist_ok=True)

        # Create all files except training/train
        Path(os.path.join(easy_sm_base, "build.sh")).touch()
        Path(os.path.join(easy_sm_base, "Dockerfile")).touch()
        Path(os.path.join(easy_sm_base, "prediction", "serve")).touch()
        Path(os.path.join(easy_sm_base, "executor.sh")).touch()

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "This is not a easy_sm directory" in result.output

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_missing_serve_file(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build fails when prediction/serve file is missing"""
        app_name = "no-serve-file-app"
        self._create_config(app_name)

        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        os.makedirs(os.path.join(easy_sm_base, "training"), exist_ok=True)
        os.makedirs(os.path.join(easy_sm_base, "prediction"), exist_ok=True)

        # Create all files except prediction/serve
        Path(os.path.join(easy_sm_base, "build.sh")).touch()
        Path(os.path.join(easy_sm_base, "Dockerfile")).touch()
        Path(os.path.join(easy_sm_base, "training", "train")).touch()
        Path(os.path.join(easy_sm_base, "executor.sh")).touch()

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "This is not a easy_sm directory" in result.output

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_sets_file_permissions(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test that build sets executable permissions on required files"""
        app_name = "permissions-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        # Remove executable permissions initially
        train_path = os.path.join(app_name, "easy_sm_base", "training", "train")
        serve_path = os.path.join(app_name, "easy_sm_base", "prediction", "serve")
        executor_path = os.path.join(app_name, "easy_sm_base", "executor.sh")

        os.chmod(train_path, 0o644)
        os.chmod(serve_path, 0o644)
        os.chmod(executor_path, 0o644)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code == 0
        # Verify permissions were set to 0o777
        assert stat.S_IMODE(os.stat(train_path).st_mode) == 0o777
        assert stat.S_IMODE(os.stat(serve_path).st_mode) == 0o777
        assert stat.S_IMODE(os.stat(executor_path).st_mode) == 0o777

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_subprocess_called_with_correct_args(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test that build subprocess is called with correct arguments"""
        app_name = "args-test-app"
        docker_tag = "test-tag"
        python_version = "3.14"

        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(
            cli, ["--docker-tag", docker_tag, "build", "-a", app_name]
        )

        assert result.exit_code == 0
        mock_popen.assert_called_once()

        # Get the command that was passed to subprocess
        call_args = mock_popen.call_args[0][0]

        # Verify command contains expected elements
        assert app_name in call_args  # image_name
        assert docker_tag in call_args
        assert python_version in call_args
        assert any("build.sh" in arg for arg in call_args)
        assert any("Dockerfile" in arg for arg in call_args)

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_subprocess_failure(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build command when subprocess fails"""
        app_name = "failing-build-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        # Mock subprocess to fail
        mock_process = MagicMock()
        mock_process.stdout = iter(["Error building image"])
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        result = runner.invoke(cli, ["build", "-a", app_name])

        # The command completes but subprocess returns error (safe_run_subprocess handles it)
        assert "Error occurred while running the command" in result.output or result.exit_code == 0

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_with_different_source_dirs(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build with different source directory paths"""
        app_name = "source-dir-app"
        source_dir = "src"

        self._create_config(app_name, source_dir)
        self._create_easy_sm_structure(source_dir)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code == 0
        call_args = mock_popen.call_args[0][0]
        assert source_dir in call_args or "src" in " ".join(call_args)

    @patch("easy_sm.commands.helpers.subprocess.Popen")
    def test_build_with_custom_requirements_path(self, mock_popen: MagicMock, runner: CliRunner, temp_dir: str) -> None:
        """Test build with custom requirements file path"""
        app_name = "custom-req-app"
        requirements_path = "config/requirements.txt"

        config = Config(
            image_name=app_name,
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=app_name,
            requirements_file_name=requirements_path,
        )
        config_manager = ConfigManager(f"{app_name}.json")
        config_manager.set_config(config)
        self._create_easy_sm_structure(app_name)

        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        result = runner.invoke(cli, ["build", "-a", app_name])

        assert result.exit_code == 0
        call_args = mock_popen.call_args[0][0]
        assert requirements_path in call_args or "requirements" in " ".join(call_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
