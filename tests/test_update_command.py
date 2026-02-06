"""Tests for the update-scripts command."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from typer.testing import CliRunner

from easy_sm.__main__ import app
from easy_sm.commands.update import SHELL_SCRIPTS
from easy_sm.config.config import Config, ConfigManager


class TestUpdateScriptsCommand:
    """Tests for the update-scripts command."""

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

    def _create_easy_sm_structure(self, app_name: str) -> None:
        """Helper to create minimal easy_sm directory structure."""
        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        os.makedirs(os.path.join(easy_sm_base, "local_test"), exist_ok=True)
        os.makedirs(os.path.join(easy_sm_base, "training"), exist_ok=True)
        os.makedirs(os.path.join(easy_sm_base, "prediction"), exist_ok=True)

    def test_update_scripts_success(self, runner: CliRunner, temp_dir: str) -> None:
        """Test successful script update."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        result = runner.invoke(app, ["update-scripts", "-a", app_name])

        assert result.exit_code == 0
        assert "Updating shell scripts" in result.output
        assert "Successfully updated" in result.output

        # Verify all scripts were copied
        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        for script in SHELL_SCRIPTS:
            script_path = os.path.join(easy_sm_base, script)
            assert os.path.isfile(script_path), f"Script not found: {script}"

    def test_update_scripts_sets_permissions(self, runner: CliRunner, temp_dir: str) -> None:
        """Test that updated scripts have correct permissions."""
        import stat

        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        result = runner.invoke(app, ["update-scripts", "-a", app_name])

        assert result.exit_code == 0

        # Verify permissions are 0o755
        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        for script in SHELL_SCRIPTS:
            script_path = os.path.join(easy_sm_base, script)
            if os.path.isfile(script_path):
                mode = stat.S_IMODE(os.stat(script_path).st_mode)
                assert mode == 0o755, f"Wrong permissions on {script}: {oct(mode)}"

    def test_update_scripts_replaces_existing(self, runner: CliRunner, temp_dir: str) -> None:
        """Test that existing scripts are replaced."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        # Create an existing script with different content
        easy_sm_base = os.path.join(app_name, "easy_sm_base")
        build_script = os.path.join(easy_sm_base, "build.sh")
        with open(build_script, "w") as f:
            f.write("#!/bin/sh\necho 'old script'\n")

        result = runner.invoke(app, ["update-scripts", "-a", app_name])

        assert result.exit_code == 0

        # Verify script was replaced (should contain docker buildx)
        with open(build_script) as f:
            content = f.read()
        assert "docker buildx build" in content
        assert "old script" not in content

    def test_update_scripts_missing_config(self, runner: CliRunner, temp_dir: str) -> None:
        """Test update-scripts fails when config file is missing."""
        app_name = "nonexistent-app"

        result = runner.invoke(app, ["update-scripts", "-a", app_name])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "not a easy_sm directory" in str(result.exception)

    def test_update_scripts_missing_easy_sm_base(self, runner: CliRunner, temp_dir: str) -> None:
        """Test update-scripts fails when easy_sm_base directory is missing."""
        app_name = "test-app"
        self._create_config(app_name)
        # Don't create easy_sm_base directory

        result = runner.invoke(app, ["update-scripts", "-a", app_name])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "not found" in str(result.exception)

    def test_update_scripts_invalid_app_name(self, runner: CliRunner, temp_dir: str) -> None:
        """Test update-scripts fails with invalid app name."""
        result = runner.invoke(app, ["update-scripts", "-a", "../../../etc/passwd"])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError) or "Invalid app name" in str(result.exception)

    def test_update_scripts_output_lists_updated_files(self, runner: CliRunner, temp_dir: str) -> None:
        """Test that output lists each updated file."""
        app_name = "test-app"
        self._create_config(app_name)
        self._create_easy_sm_structure(app_name)

        result = runner.invoke(app, ["update-scripts", "-a", app_name])

        assert result.exit_code == 0
        assert "Updated: build.sh" in result.output
        assert "Updated: push.sh" in result.output
        assert "Updated: executor.sh" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
