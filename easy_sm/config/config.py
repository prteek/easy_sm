import json
import os
from collections import OrderedDict
from typing import Dict, Any


class Config(object):
    image_name: str
    aws_profile: str
    aws_region: str
    python_version: str
    easy_sm_module_dir: str
    requirements_dir: str

    def __init__(
        self,
        image_name: str,
        aws_profile: str,
        aws_region: str,
        python_version: str,
        easy_sm_module_dir: str,
        requirements_dir: str,
    ) -> None:
        self.image_name = image_name
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        self.python_version = python_version
        self.requirements_dir = requirements_dir
        self.easy_sm_module_dir = easy_sm_module_dir

    def to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict(self.__dict__.items())

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> "Config":
        return Config(
            image_name=input_dict["image_name"],
            aws_profile=input_dict["aws_profile"],
            aws_region=input_dict["aws_region"],
            python_version=input_dict["python_version"],
            easy_sm_module_dir=input_dict["easy_sm_module_dir"],
            requirements_dir=input_dict["requirements_dir"],
        )


class ConfigManager(object):
    _config_file_path: str

    def __init__(self, config_file_path: str) -> None:
        self._config_file_path = config_file_path

        if not os.path.isfile(config_file_path):
            self.set_config(
                Config(
                    image_name="",
                    aws_profile="",
                    aws_region="",
                    python_version="",
                    easy_sm_module_dir="",
                    requirements_dir="",
                )
            )

    def get_config(self) -> Config:
        with open(self._config_file_path) as config_file:
            config_content = config_file.read()

        return Config.from_dict(json.loads(config_content))

    def set_config(self, config: Config) -> None:
        with open(self._config_file_path, "w") as config_file:
            json.dump(config.to_dict(), config_file, indent=4)
