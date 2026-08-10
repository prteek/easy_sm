import json
import os
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Config:
    image_name: str
    aws_profile: str
    aws_region: str
    python_version: str
    easy_sm_module_dir: str
    requirements_dir: str
    docker_tag: str = "latest"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(**data)


class ConfigManager:
    def __init__(self, config_file_path: str) -> None:
        self._config_file_path = config_file_path
        # Create default config if file doesn't exist
        if not os.path.isfile(config_file_path):
            self.set_config(Config("", "", "", "", "", ""))

    def get_config(self) -> Config:
        with open(self._config_file_path) as f:
            return Config.from_dict(json.load(f))

    def set_config(self, config: Config) -> None:
        with open(self._config_file_path, "w") as f:
            json.dump(config.to_dict(), f, indent=4)
