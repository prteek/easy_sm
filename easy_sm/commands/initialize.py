import os
import re
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import boto3
import click
from click import BadParameter

from easy_sm.config.config import ConfigManager

_FILE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def _template_creation(
    app_name: str,
    aws_profile: str,
    aws_region: str,
    python_version: str,
    output_dir: str,
    requirements_dir: str,
) -> None:
    easy_sm_module_name = "easy_sm_base"

    easy_sm_exists = os.path.exists(os.path.join(output_dir, easy_sm_module_name))
    if easy_sm_exists:
        raise ValueError(
            "There is a easy_sm directory/module already. "
            "Please, rename it in order to use easy_sm."
        )

    Path(output_dir).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, "__init__.py")).touch()

    template_path = os.path.join(_FILE_DIR_PATH, "../template")
    for item in os.listdir(template_path):
        src = os.path.join(template_path, item)
        dst = os.path.join(output_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    config_manager = ConfigManager(os.path.join(f"{app_name}.json"))
    config = config_manager.get_config()

    config.image_name = app_name
    config.aws_region = aws_region
    config.aws_profile = aws_profile
    config.easy_sm_module_dir = output_dir
    config.python_version = python_version
    config.requirements_dir = requirements_dir
    config_manager.set_config(config)


def _get_local_aws_profiles() -> List[str]:
    return boto3.Session().available_profiles


def ask_for_app_name() -> str:
    app_name = click.prompt(
        text="Type in a name for your SageMaker app (Only alphanumeric characters and - are allowed))",
        type=str,
    )

    if not bool(re.fullmatch(r"[a-zA-Z0-9\-]+", app_name)):
        raise BadParameter(
            message=f"invalid app name: {app_name}. App name must only have alphanumeric characters or dashes '-' "
        )

    return app_name


def ask_if_existing_project_exists() -> bool:
    return click.confirm(text="Are you starting a new project?")


def ask_for_root_dir() -> str:
    return click.prompt(
        text="Type in the directory where your code lives. Example: src", type=str
    ).strip("/")


def ask_for_python_version() -> str:
    print("Select Python interpreter:")
    print("{}".format("\n".join(["1 - Python310", "2 - Python311", "3 - Python312", "4 - Python314"])))

    def _validate_python_option(input_value: Any) -> int:
        if int(input_value) not in {1, 2, 3, 4}:
            raise BadParameter(
                message="invalid choice: {}. (choose from 1, 2, 3, 4)".format(
                    str(input_value)
                )
            )

        return int(input_value)

    chosen_python_index = click.prompt(
        text="Choose from 1, 2, 3, 4",
        default=4,
        value_proc=lambda x: _validate_python_option(x),
    )

    _index_to_version = {1: "3.10", 2: "3.11", 3: "3.12", 4: "3.14"}

    return _index_to_version[chosen_python_index]


def ask_for_aws_details() -> Tuple[str, str]:
    available_profiles = _get_local_aws_profiles()

    if len(available_profiles) == 0:
        print("\nNo AWS profiles found in ~/.aws/credentials")
        print("You can use AWS credentials in two ways:")
        print("  1. Set environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("  2. Configure AWS CLI profiles: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html")
        print("For now, leaving aws_profile empty will use the environment variables or default credential chain.\n")
        region = click.prompt(
            text="Type in your preferred AWS region name",
            default="us-east-1",
            type=str,
        )
        return "", region

    valid_positions = list(range(1, len(available_profiles) + 1))
    print("Select AWS profile:")
    print(
        "{}".format(
            "\n".join(
                [
                    "{} - {}".format(pos, profile)
                    for pos, profile in zip(valid_positions, available_profiles)
                ]
            )
        )
    )

    def _validate_profile_position(input_pos: Any) -> int:
        if int(input_pos) not in valid_positions:
            raise BadParameter(
                message="invalid choice: {}. (choose from {})".format(
                    input_pos,
                    ", ".join([str(pos) for pos in valid_positions]),
                )
            )
        return int(input_pos)

    chosen_profile_index = click.prompt(
        text="Choose from {}".format(", ".join([str(pos) for pos in valid_positions])),
        default=1,
        value_proc=lambda x: _validate_profile_position(x),
    )
    chosen_profile_index -= 1

    chosen_profile = available_profiles[chosen_profile_index]

    chosen_region = click.prompt(
        text="Type in your preferred AWS region name",
        default="eu-west-1",
        type=str,
    )

    return chosen_profile, chosen_region


def ask_for_requirements_dir() -> str:
    return click.prompt(
        text="Type in the path to requirements.txt. Example: requirements.txt",
        type=str,
    ).strip("/")


@click.command()
def init() -> None:
    """
    Command to initialize SageMaker template
    """
    easy_sm_app_name = ask_for_app_name()

    is_new_project = ask_if_existing_project_exists()

    root_dir: Optional[str] = None
    if not is_new_project:
        root_dir = ask_for_root_dir()

    python_version = ask_for_python_version()

    aws_profile, aws_region = ask_for_aws_details()

    requirements_dir = ask_for_requirements_dir()
    _template_creation(
        app_name=easy_sm_app_name,
        aws_profile=aws_profile,
        aws_region=aws_region,
        python_version=python_version,
        output_dir=root_dir if root_dir else easy_sm_app_name,
        requirements_dir=requirements_dir,
    )

    print("\neasy_sm module is created! ヽ(´▽`)/")
