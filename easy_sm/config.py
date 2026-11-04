"""Read easy_sm.toml and work out each setting's value.

A setting's value comes from the first of these that has one:
command-line flag → environment variable → easy_sm.toml → built-in default.
"""

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_FILE = "easy_sm.toml"


@dataclass(frozen=True)
class Setting:
    type: type
    env: str | None = None  # environment variable that overrides the file
    default: Any = None


# Every setting easy_sm knows, as "section.name". Anything else in the file is an error,
# so a typo like `instnace` fails loudly instead of being silently ignored.
SETTINGS: dict[str, Setting] = {
    "app_name": Setting(str),
    "aws.region": Setting(str, env="AWS_REGION"),
    "aws.role": Setting(str, env="SAGEMAKER_ROLE"),
    "train.instance": Setting(str, default="ml.m5.large"),
    "train.count": Setting(int, default=1),
    "train.input": Setting(str),
    "train.output": Setting(str),
    "train.timeout": Setting(str, default="24h"),
    "run.instance": Setting(str, default="ml.t3.medium"),
    "run.timeout": Setting(str, default="24h"),
    "run.env": Setting(dict),
    "run.input": Setting(str),  # S3 folder copied to /opt/ml/processing/input
    "run.output": Setting(str),  # S3 folder that /opt/ml/processing/output is uploaded to
    "endpoint.name": Setting(str),
    "endpoint.instance": Setting(str, default="ml.m5.large"),
    "endpoint.serverless": Setting(bool, default=False),
    "endpoint.memory": Setting(int, default=2048),
    "endpoint.concurrency": Setting(int, default=5),
}
SECTIONS = {key.split(".")[0] for key in SETTINGS if "." in key}

# app_name becomes the Docker image name, the ECR repository and the start of SageMaker
# job and endpoint names. SageMaker allows only letters, digits and hyphens there, Docker
# only lowercase, and job names max 63 characters, leaving room for a timestamp and hash.
APP_NAME_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
APP_NAME_MAX_LENGTH = 32


class ConfigError(Exception):
    """easy_sm.toml is missing or invalid."""


@dataclass(frozen=True)
class Config:
    path: Path
    values: dict[str, Any]  # only what the file sets, keyed "section.name"

    def resolve(
        self, key: str, flag: Any = None, env: Mapping[str, str] = os.environ
    ) -> tuple[Any, str]:
        """Return (value, where it came from) for a setting."""
        setting = SETTINGS[key]
        if flag is not None:
            return flag, "flag"
        if setting.env and env.get(setting.env):
            return env[setting.env], setting.env
        if key in self.values:
            return self.values[key], self.path.name
        return setting.default, "default"

    def get(self, key: str, flag: Any = None) -> Any:
        """Return just the value of a setting."""
        return self.resolve(key, flag)[0]


def check_app_name(name: str) -> None:
    if not APP_NAME_PATTERN.fullmatch(name) or len(name) > APP_NAME_MAX_LENGTH:
        raise ConfigError(
            f"Invalid app_name '{name}': use lowercase letters, digits and single hyphens, "
            f"at most {APP_NAME_MAX_LENGTH} characters (e.g. 'churn-model')"
        )


def load(path: Path) -> Config:
    """Read and check a config file."""
    if not path.is_file():
        raise ConfigError(f"{path} not found. Run 'easy_sm init' to create a project.")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e

    values = _flatten(data)
    for key, value in values.items():
        setting = SETTINGS.get(key)
        if setting is None:
            raise ConfigError(f"{path}: unknown setting '{key}'")
        if not isinstance(value, setting.type):
            raise ConfigError(
                f"{path}: '{key}' should be {setting.type.__name__}, got {type(value).__name__}"
            )

    if "app_name" not in values:
        raise ConfigError(f"{path}: app_name is missing")
    check_app_name(values["app_name"])

    env = values.get("run.env", {})
    if not all(isinstance(v, str) for v in env.values()):
        raise ConfigError(f"{path}: run.env values must be strings, e.g. DEBUG = \"true\"")

    return Config(path, values)


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    """{"train": {"instance": "x"}} → {"train.instance": "x"}. Only known sections are opened,
    so a table value such as run.env stays a dict."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if key in SECTIONS and isinstance(value, dict):
            for name, inner in value.items():
                flat[f"{key}.{name}"] = inner
        else:
            flat[key] = value
    return flat
