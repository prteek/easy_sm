import glob
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


def auto_detect_app_name() -> str:
    """Auto-detect app name from *.json config file in current directory.

    Returns:
        App name (filename without .json extension).

    Raises:
        ValueError: If no config file or multiple config files found.
    """
    config_files = glob.glob("*.json")

    if len(config_files) == 0:
        raise ValueError(
            "No config file found in current directory. "
            "Run 'easy_sm init' or specify app name with -a/--app-name."
        )

    if len(config_files) > 1:
        raise ValueError(
            f"Multiple config files found: {', '.join(config_files)}. "
            "Specify app name with -a/--app-name."
        )

    # Remove .json extension to get app name
    app_name = config_files[0][:-5]
    return app_name


def get_app_name(app_name: str | None = None) -> str:
    """Get app name from parameter or auto-detect.

    Args:
        app_name: Explicit app name, or None to auto-detect.

    Returns:
        App name string.
    """
    if app_name is None:
        return auto_detect_app_name()
    return app_name


def get_iam_role(iam_role_arn: str | None = None) -> str:
    """Get IAM role ARN from parameter or environment variable.

    Args:
        iam_role_arn: Explicit role ARN, or None to read from env.

    Returns:
        IAM role ARN string.

    Raises:
        ValueError: If no role specified and SAGEMAKER_ROLE env var not set.
    """
    if iam_role_arn is not None:
        return iam_role_arn

    env_role = os.environ.get("SAGEMAKER_ROLE")
    if env_role:
        return env_role

    raise ValueError(
        "IAM role not specified. Set SAGEMAKER_ROLE environment variable "
        "or use -r/--iam-role-arn option."
    )


def load_config(app_name: str | None = None) -> Config:
    """Load configuration from app_name.json in current directory.

    Args:
        app_name: Name of the app. If None, auto-detects from *.json files.

    Returns:
        Config object loaded from the JSON file.

    Raises:
        ValueError: If app name is invalid or config file not found.
    """
    app_name = get_app_name(app_name)

    if not app_name or not APP_NAME_PATTERN.match(app_name):
        raise ValueError(
            f"Invalid app name: {app_name}. "
            "Must contain only alphanumeric characters, hyphens, and underscores."
        )

    config_file = f"{app_name}.json"
    if not os.path.isfile(config_file):
        raise ValueError(f"Config file not found: {config_file}")

    return ConfigManager(config_file).get_config()


def serialize_env_vars(env_vars: dict[str, str]) -> str:
    """Serialize environment variables dict to a shell-friendly string.

    Args:
        env_vars: Dictionary of environment variables

    Returns:
        Comma-separated KEY=VALUE pairs (or empty string if no vars)

    Example:
        {"DEBUG": "true", "LOG_LEVEL": "info"} -> "DEBUG=true,LOG_LEVEL=info"
    """
    if not env_vars:
        return ""
    return ",".join(f"{k}={v}" for k, v in env_vars.items())


def deserialize_env_vars(env_string: str) -> dict[str, str]:
    """Deserialize environment variables from shell-friendly string.

    Args:
        env_string: Comma-separated KEY=VALUE pairs (or empty string)

    Returns:
        Dictionary of environment variables

    Example:
        "DEBUG=true,LOG_LEVEL=info" -> {"DEBUG": "true", "LOG_LEVEL": "info"}
    """
    if not env_string:
        return {}
    return dict(pair.split("=", 1) for pair in env_string.split(","))
