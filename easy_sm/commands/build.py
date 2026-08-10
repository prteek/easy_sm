import os
from typing import Annotated

import typer

from easy_sm.commands.helpers import load_config, safe_run_subprocess


def _build(
    source_dir: str,
    requirements_dir: str,
    image_name: str,
    docker_tag: str,
    python_version: str,
) -> None:
    """Build a Docker image containing the source code."""
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
        raise ValueError(f"This is not a easy_sm directory: {source_dir}")

    os.chmod(train_file_path, 0o755)
    os.chmod(serve_file_path, 0o755)
    os.chmod(executor_file_path, 0o755)

    target_dir_name = os.path.basename(os.path.normpath(source_dir))

    command = [
        build_script_path,
        os.path.relpath(source_dir),
        os.path.relpath(target_dir_name),
        dockerfile_path,
        os.path.relpath(requirements_dir),
        docker_tag,
        image_name,
        python_version,
    ]
    safe_run_subprocess(command, success_message="Docker image built successfully!")


def build(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
) -> None:
    """Build SageMaker Docker image."""
    print("Started building SageMaker Docker image. It will take some minutes...\n")

    config = load_config(app_name)
    _build(
        source_dir=config.easy_sm_module_dir,
        requirements_dir=config.requirements_dir,
        docker_tag=config.docker_tag,
        image_name=config.image_name,
        python_version=config.python_version,
    )
