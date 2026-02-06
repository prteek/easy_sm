import os
import re
import subprocess

from easy_sm.config.config import Config, ConfigManager

# Global state for docker_tag (set by main CLI callback)
docker_tag: str = "latest"

# Pattern for valid app names: alphanumeric, hyphens, underscores only
APP_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def safe_run_subprocess(command: list[str], success_message: str | None = None) -> int:
    """Run subprocess and stream output. Returns exit code."""
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        if success_message:
            print(success_message)
        return return_code

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}: {e.cmd}")
        return e.returncode


def load_config(app_name: str) -> Config:
    """Load configuration from app_name.json in current directory."""
    if not app_name or not APP_NAME_PATTERN.match(app_name):
        raise ValueError(
            f"Invalid app name: {app_name}. "
            "Must contain only alphanumeric characters, hyphens, and underscores."
        )

    config_file = f"{app_name}.json"
    if not os.path.isfile(config_file):
        raise ValueError(f"Config file not found: {config_file}")

    return ConfigManager(config_file).get_config()
