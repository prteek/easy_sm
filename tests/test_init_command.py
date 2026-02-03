import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from click.testing import CliRunner

from easy_sm.__main__ import cli
from easy_sm.config.config import ConfigManager


class TestInitCommand:
    """Test suite for the init command"""

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

    def test_init_new_project(self, runner: CliRunner, temp_dir: str) -> None:
        """Test init command for a new project"""
        app_name = "test-app"
        root_dir = app_name
        python_version = "3"  # Select Python 3.12
        aws_profile = "1"  # First available profile
        aws_region = "us-east-1"
        requirements_dir = "requirements.txt"

        # Simulate user input
        user_input = f"{app_name}\ny\n{python_version}\n{aws_profile}\n{aws_region}\n{requirements_dir}\n"

        result = runner.invoke(cli, ["init"], input=user_input)

        # Check command execution
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        assert "easy_sm module is created" in result.output

        # Check that config file was created
        config_file = f"{app_name}.json"
        assert os.path.isfile(config_file), f"Config file {config_file} was not created"

        # Load and validate config
        config_manager = ConfigManager(config_file)
        config = config_manager.get_config()

        assert config.image_name == app_name
        assert config.aws_region == aws_region
        assert config.python_version == "3.12"
        assert config.requirements_dir == requirements_dir
        assert config.easy_sm_module_dir == root_dir

        # Verify template directory was created
        assert os.path.isdir(os.path.join(root_dir, "easy_sm_base"))
        assert os.path.isfile(os.path.join(root_dir, "__init__.py"))

    def test_init_existing_project(self, runner: CliRunner, temp_dir: str) -> None:
        """Test init command for an existing project"""
        app_name = "existing-app"
        root_dir = "src"
        python_version = "1"  # Select Python 3.10
        aws_profile = "1"
        aws_region = "eu-west-1"
        requirements_dir = "config/requirements.txt"

        # Create the source directory
        Path(root_dir).mkdir(exist_ok=True)

        # Simulate user input (is_new_project=False)
        user_input = f"{app_name}\nn\n{root_dir}\n{python_version}\n{aws_profile}\n{aws_region}\n{requirements_dir}\n"

        result = runner.invoke(cli, ["init"], input=user_input)

        # Check command execution
        assert result.exit_code == 0, f"Command failed with: {result.output}"
        assert "easy_sm module is created" in result.output

        # Check that config file was created
        config_file = f"{app_name}.json"
        assert os.path.isfile(config_file)

        # Load and validate config
        config_manager = ConfigManager(config_file)
        config = config_manager.get_config()

        assert config.image_name == app_name
        assert config.aws_region == aws_region
        assert config.python_version == "3.10"
        assert config.requirements_dir == requirements_dir
        assert config.easy_sm_module_dir == root_dir

        # Verify template was created in the specified directory
        assert os.path.isdir(os.path.join(root_dir, "easy_sm_base"))

    def test_init_config_json_structure(self, runner: CliRunner, temp_dir: str) -> None:
        """Test that the generated config.json has correct structure"""
        app_name = "json-test-app"
        python_version = "3"  # Select Python 3.12
        aws_profile = "1"
        aws_region = "ap-south-1"
        requirements_dir = "requirements.txt"

        user_input = f"{app_name}\ny\n{python_version}\n{aws_profile}\n{aws_region}\n{requirements_dir}\n"

        result = runner.invoke(cli, ["init"], input=user_input)
        assert result.exit_code == 0

        # Load config file and verify JSON structure
        config_file = f"{app_name}.json"
        with open(config_file, "r") as f:
            config_json = json.load(f)

        # Verify all required fields exist
        required_fields = [
            "image_name",
            "aws_profile",
            "aws_region",
            "python_version",
            "easy_sm_module_dir",
            "requirements_dir",
        ]
        for field in required_fields:
            assert field in config_json, f"Missing required field: {field}"

        # Verify values
        assert config_json["image_name"] == app_name
        assert config_json["python_version"] == "3.12"
        assert config_json["requirements_dir"] == requirements_dir

    def test_init_invalid_app_name(self, runner: CliRunner, temp_dir: str) -> None:
        """Test init command with invalid app name"""
        invalid_app_name = "test@app!"  # Invalid characters
        user_input = f"{invalid_app_name}\n"

        result = runner.invoke(cli, ["init"], input=user_input)

        # Should fail with invalid app name
        assert result.exit_code != 0
        assert "invalid app name" in result.output

    def test_init_template_files_created(self, runner: CliRunner, temp_dir: str) -> None:
        """Test that all template files are created during init"""
        app_name = "template-test"
        python_version = "3"
        aws_profile = "1"
        aws_region = "us-east-1"
        requirements_dir = "requirements.txt"

        user_input = f"{app_name}\ny\n{python_version}\n{aws_profile}\n{aws_region}\n{requirements_dir}\n"

        result = runner.invoke(cli, ["init"], input=user_input)
        assert result.exit_code == 0

        # Check for expected template directories and files
        base_path = os.path.join(app_name, "easy_sm_base")
        assert os.path.isdir(os.path.join(base_path, "training"))
        assert os.path.isdir(os.path.join(base_path, "processing"))
        assert os.path.isdir(os.path.join(base_path, "prediction"))
        assert os.path.isdir(os.path.join(base_path, "local_test"))

        # Check for key files
        assert os.path.isfile(os.path.join(base_path, "Dockerfile"))
        assert os.path.isfile(os.path.join(base_path, "build.sh"))
        assert os.path.isfile(os.path.join(base_path, "push.sh"))
        assert os.path.isfile(os.path.join(base_path, "training", "train"))
        assert os.path.isfile(os.path.join(base_path, "prediction", "serve"))

    def test_init_app_name_with_dashes(self, runner: CliRunner, temp_dir: str) -> None:
        """Test init with app name containing dashes"""
        app_name = "my-test-app-123"
        python_version = "3"
        aws_profile = "1"
        aws_region = "us-west-2"
        requirements_dir = "requirements.txt"

        user_input = f"{app_name}\ny\n{python_version}\n{aws_profile}\n{aws_region}\n{requirements_dir}\n"

        result = runner.invoke(cli, ["init"], input=user_input)
        assert result.exit_code == 0

        config_file = f"{app_name}.json"
        assert os.path.isfile(config_file)

        config_manager = ConfigManager(config_file)
        config = config_manager.get_config()
        assert config.image_name == app_name

    def test_init_easy_sm_directory_conflict(self, runner: CliRunner, temp_dir: str) -> None:
        """Test init fails when easy_sm directory already exists"""
        app_name = "conflict-app"

        # Create directory structure that would conflict
        os.makedirs(os.path.join(app_name, "easy_sm_base"))

        python_version = "3"
        aws_profile = "1"
        aws_region = "us-east-1"
        requirements_dir = "requirements.txt"

        user_input = f"{app_name}\ny\n{python_version}\n{aws_profile}\n{aws_region}\n{requirements_dir}\n"

        result = runner.invoke(cli, ["init"], input=user_input)

        # Should fail due to existing easy_sm_base directory
        assert result.exit_code != 0
        assert "easy_sm directory/module already" in result.output or isinstance(
            result.exception, ValueError
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
