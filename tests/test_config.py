"""
Unit tests for the config module.

Tests Config and ConfigManager classes for configuration loading,
saving, serialization, and error handling.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from easy_sm.config.config import Config, ConfigManager


class TestConfig:
    """Tests for the Config class."""

    def test_config_initialization(self) -> None:
        """Test Config initialization with all parameters."""
        config = Config(
            image_name="test-image",
            aws_profile="test-profile",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir="./app",
            requirements_file_name="requirements.txt",
        )

        assert config.image_name == "test-image"
        assert config.aws_profile == "test-profile"
        assert config.aws_region == "us-east-1"
        assert config.python_version == "3.14"
        assert config.easy_sm_module_dir == "./app"
        assert config.requirements_file_name == "requirements.txt"

    def test_config_to_dict(self) -> None:
        """Test Config.to_dict() serialization."""
        config = Config(
            image_name="my-app",
            aws_profile="prod",
            aws_region="eu-west-1",
            python_version="3.11",
            easy_sm_module_dir="src",
            requirements_file_name="config/reqs.txt",
        )

        config_dict = config.to_dict()

        assert config_dict["image_name"] == "my-app"
        assert config_dict["aws_profile"] == "prod"
        assert config_dict["aws_region"] == "eu-west-1"
        assert config_dict["python_version"] == "3.11"
        assert config_dict["easy_sm_module_dir"] == "src"
        assert config_dict["requirements_file_name"] == "config/reqs.txt"

    def test_config_from_dict(self) -> None:
        """Test Config.from_dict() deserialization."""
        input_dict = {
            "image_name": "test-image",
            "aws_profile": "test-profile",
            "aws_region": "us-west-2",
            "python_version": "3.10",
            "easy_sm_module_dir": "/path/to/app",
            "requirements_file_name": "/path/to/requirements.txt",
        }

        config = Config.from_dict(input_dict)

        assert config.image_name == "test-image"
        assert config.aws_profile == "test-profile"
        assert config.aws_region == "us-west-2"
        assert config.python_version == "3.10"
        assert config.easy_sm_module_dir == "/path/to/app"
        assert config.requirements_file_name == "/path/to/requirements.txt"

    def test_config_roundtrip_serialization(self) -> None:
        """Test Config serialization and deserialization roundtrip."""
        original_config = Config(
            image_name="roundtrip-test",
            aws_profile="roundtrip-profile",
            aws_region="ap-southeast-1",
            python_version="3.14",
            easy_sm_module_dir="./module",
            requirements_file_name="requirements/prod.txt",
        )

        # Serialize to dict
        config_dict = original_config.to_dict()

        # Deserialize from dict
        restored_config = Config.from_dict(config_dict)

        # Verify all fields match
        assert restored_config.image_name == original_config.image_name
        assert restored_config.aws_profile == original_config.aws_profile
        assert restored_config.aws_region == original_config.aws_region
        assert restored_config.python_version == original_config.python_version
        assert restored_config.easy_sm_module_dir == original_config.easy_sm_module_dir
        assert restored_config.requirements_file_name == original_config.requirements_file_name

    def test_config_with_special_characters(self) -> None:
        """Test Config with special characters in values."""
        config = Config(
            image_name="my-app-v1.0.0",
            aws_profile="prod-env_profile",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir="./my-app/src",
            requirements_file_name="requirements/prod-requirements.txt",
        )

        config_dict = config.to_dict()
        restored_config = Config.from_dict(config_dict)

        assert restored_config.image_name == "my-app-v1.0.0"
        assert restored_config.easy_sm_module_dir == "./my-app/src"

    def test_config_from_dict_missing_field(self) -> None:
        """Test Config.from_dict() with missing required field."""
        incomplete_dict = {
            "image_name": "test",
            "aws_profile": "test",
            # Missing aws_region
        }

        with pytest.raises(KeyError):
            Config.from_dict(incomplete_dict)


class TestConfigManager:
    """Tests for the ConfigManager class."""

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)

    def test_config_manager_set_and_get(self, temp_dir: str) -> None:
        """Test ConfigManager set_config and get_config."""
        config_file = "test.json"
        config = Config(
            image_name="test-app",
            aws_profile="test",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=".",
            requirements_file_name="requirements.txt",
        )

        config_manager = ConfigManager(config_file)
        config_manager.set_config(config)

        # Load config again
        loaded_config = config_manager.get_config()

        assert loaded_config.image_name == config.image_name
        assert loaded_config.aws_profile == config.aws_profile
        assert loaded_config.aws_region == config.aws_region

    def test_config_manager_file_created(self, temp_dir: str) -> None:
        """Test that ConfigManager creates config file."""
        config_file = "new_config.json"
        config = Config(
            image_name="app",
            aws_profile="default",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=".",
            requirements_file_name="requirements.txt",
        )

        config_manager = ConfigManager(config_file)
        config_manager.set_config(config)

        # Verify file exists
        assert os.path.isfile(config_file)

        # Verify file contains valid JSON
        with open(config_file) as f:
            data = json.load(f)
            assert data["image_name"] == "app"

    def test_config_manager_file_format(self, temp_dir: str) -> None:
        """Test that ConfigManager stores config as valid JSON."""
        config_file = "format_test.json"
        config = Config(
            image_name="format-app",
            aws_profile="format-profile",
            aws_region="eu-west-1",
            python_version="3.11",
            easy_sm_module_dir="src",
            requirements_file_name="config/requirements.txt",
        )

        config_manager = ConfigManager(config_file)
        config_manager.set_config(config)

        # Read JSON file directly
        with open(config_file) as f:
            raw_json = json.load(f)

        assert raw_json["image_name"] == "format-app"
        assert raw_json["aws_profile"] == "format-profile"
        assert raw_json["aws_region"] == "eu-west-1"
        assert raw_json["python_version"] == "3.11"
        assert raw_json["easy_sm_module_dir"] == "src"
        assert raw_json["requirements_file_name"] == "config/requirements.txt"

    def test_config_manager_with_nonexistent_file(self, temp_dir: str) -> None:
        """Test ConfigManager with nonexistent file creates default config."""
        config_file = "nonexistent.json"

        # File should not exist
        assert not os.path.isfile(config_file)

        # Creating ConfigManager should create the file with default config
        config_manager = ConfigManager(config_file)

        assert os.path.isfile(config_file)

        # Load the default config
        config = config_manager.get_config()

        assert config.image_name == ""
        assert config.aws_profile == ""
        assert config.aws_region == ""

    def test_config_manager_multiple_instances(self, temp_dir: str) -> None:
        """Test that multiple ConfigManager instances can read/write same file."""
        config_file = "shared.json"

        # First manager writes config
        config1 = Config(
            image_name="app-v1",
            aws_profile="profile1",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=".",
            requirements_file_name="requirements.txt",
        )
        manager1 = ConfigManager(config_file)
        manager1.set_config(config1)

        # Second manager reads same file
        manager2 = ConfigManager(config_file)
        config2 = manager2.get_config()

        assert config2.image_name == "app-v1"
        assert config2.aws_profile == "profile1"

    def test_config_manager_update_config(self, temp_dir: str) -> None:
        """Test updating config through ConfigManager."""
        config_file = "update.json"

        # Set initial config
        initial_config = Config(
            image_name="app",
            aws_profile="dev",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=".",
            requirements_file_name="requirements.txt",
        )
        manager = ConfigManager(config_file)
        manager.set_config(initial_config)

        # Update config
        updated_config = Config(
            image_name="app-updated",
            aws_profile="prod",
            aws_region="eu-west-1",
            python_version="3.11",
            easy_sm_module_dir="src",
            requirements_file_name="config/requirements.txt",
        )
        manager.set_config(updated_config)

        # Verify update persisted
        loaded_config = manager.get_config()
        assert loaded_config.image_name == "app-updated"
        assert loaded_config.aws_profile == "prod"
        assert loaded_config.aws_region == "eu-west-1"

    def test_config_manager_json_formatting(self, temp_dir: str) -> None:
        """Test that ConfigManager stores JSON with proper formatting."""
        config_file = "formatted.json"
        config = Config(
            image_name="format-test",
            aws_profile="test",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=".",
            requirements_file_name="requirements.txt",
        )

        config_manager = ConfigManager(config_file)
        config_manager.set_config(config)

        # Read raw file content
        with open(config_file) as f:
            content = f.read()

        # Verify it's properly formatted with indentation
        assert "    " in content  # Should have indentation
        data = json.loads(content)
        assert len(data) == 6  # Should have all 6 fields

    def test_config_manager_get_nonexistent_file_after_init(self, temp_dir: str) -> None:
        """Test getting config from file initialized with defaults."""
        config_file = "default_init.json"

        manager = ConfigManager(config_file)
        config = manager.get_config()

        # Should get default empty config
        assert config.image_name == ""
        assert config.aws_profile == ""
        assert config.aws_region == ""
        assert config.python_version == ""
        assert config.easy_sm_module_dir == ""
        assert config.requirements_file_name == ""

    def test_config_manager_corrupted_json(self, temp_dir: str) -> None:
        """Test ConfigManager with corrupted JSON file."""
        config_file = "corrupted.json"

        # Write corrupted JSON
        with open(config_file, "w") as f:
            f.write("{invalid json")

        manager = ConfigManager(config_file)

        with pytest.raises(json.JSONDecodeError):
            manager.get_config()

    def test_config_manager_preserves_order(self, temp_dir: str) -> None:
        """Test that ConfigManager preserves field order."""
        config_file = "order.json"
        config = Config(
            image_name="order-app",
            aws_profile="order-profile",
            aws_region="us-east-1",
            python_version="3.14",
            easy_sm_module_dir=".",
            requirements_file_name="requirements.txt",
        )

        config_manager = ConfigManager(config_file)
        config_manager.set_config(config)

        # Read file and parse to verify order
        with open(config_file) as f:
            content = f.read()
            # Parse to OrderedDict to check order
            data = json.loads(content)

        # Convert to list to check order
        keys = list(data.keys())
        assert keys[0] == "image_name"
        assert keys[1] == "aws_profile"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
