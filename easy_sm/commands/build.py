import os
from typing import Any, Dict

import click

from easy_sm.commands.helpers import (
    app_name_option,
    load_config,
    safe_run_subprocess,
)


def _build(
    source_dir: str,
    requirements_file_name: str,
    image_name: str,
    docker_tag: str,
    python_version: str,
) -> None:
    """
    Builds a Docker image that contains code under the given source root directory.

    Assumes that Docker is installed and running locally.

    :param source_dir: [str], source root directory
    :param requirements_file_name: [str], filename of requirements file (e.g., requirements.txt)
    :param image_name: [str], The name of the Docker image
    :param docker_tag: [str], the Docker tag for the image
    :param python_version: [str], Python version for the Docker image
    """
    easy_sm_module_path = os.path.relpath(os.path.join(source_dir, "easy_sm_base/"))

    build_script_path = os.path.join(easy_sm_module_path, "build.sh")
    dockerfile_path = os.path.join(easy_sm_module_path, "Dockerfile")

    train_file_path = os.path.join(easy_sm_module_path, "training", "train")
    serve_file_path = os.path.join(easy_sm_module_path, "prediction", "serve")
    executor_file_path = os.path.join(easy_sm_module_path, "executor.sh")

    if (
        not os.path.isfile(build_script_path)
        or not os.path.isfile(train_file_path)
        or not os.path.isfile(serve_file_path)
    ):
        raise ValueError("This is not a easy_sm directory: {}".format(source_dir))

    os.chmod(train_file_path, 0o777)
    os.chmod(serve_file_path, 0o777)
    os.chmod(executor_file_path, 0o777)

    target_dir_name = os.path.basename(os.path.normpath(source_dir))

    command = [
        "{}".format(build_script_path),
        "{}".format(os.path.relpath(source_dir)),
        "{}".format(os.path.relpath(target_dir_name)),
        "{}".format(dockerfile_path),
        "{}".format(requirements_file_name),
        docker_tag,
        image_name,
        python_version,
    ]
    safe_run_subprocess(command, success_message="Docker image built successfully!")


@click.command()
@app_name_option
@click.pass_obj
def build(obj: Dict[str, Any], app_name: str) -> None:
    """
    Command to build SageMaker app
    """
    print("Started building SageMaker Docker image. It will take some minutes...\n")

    config = load_config(app_name)
    _build(
        source_dir=config.easy_sm_module_dir,
        requirements_file_name=config.requirements_file_name,
        docker_tag=obj["docker_tag"],
        image_name=config.image_name,
        python_version=config.python_version,
    )
