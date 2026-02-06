import os
from typing import Annotated

import typer

from easy_sm.commands import helpers
from easy_sm.commands.helpers import load_config, safe_run_subprocess

local_app = typer.Typer(help="Commands for local operations: train and deploy")


@local_app.command()
def train(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
) -> None:
    """Train ML model(s) locally."""
    print("Started local training...\n")
    config = load_config(app_name)

    easy_sm_module_path = os.path.join(config.easy_sm_module_dir, "easy_sm_base")
    local_train_script_path = os.path.join(easy_sm_module_path, "local_test", "train_local.sh")
    test_path = os.path.join(easy_sm_module_path, "local_test", "test_dir")

    if not os.path.isdir(test_path):
        raise ValueError(f"This is not a easy_sm directory: {config.easy_sm_module_dir}")

    command = [
        local_train_script_path,
        os.path.abspath(test_path),
        helpers.docker_tag,
        config.image_name,
    ]

    safe_run_subprocess(command, success_message="Local training completed successfully!")


@local_app.command()
def process(
    file: Annotated[str, typer.Option("--file", "-f", help="Python file name to run as processing job")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
) -> None:
    """Run python files locally as processing job."""
    print("Started local processing job...\n")
    config = load_config(app_name)

    easy_sm_module_path = os.path.join(config.easy_sm_module_dir, "easy_sm_base")
    local_process_script_path = os.path.join(easy_sm_module_path, "local_test", "process_local.sh")
    test_path = os.path.join(easy_sm_module_path, "local_test", "test_dir")
    job_file_path = os.path.join(easy_sm_module_path, "processing", file)

    if not os.path.isdir(test_path):
        raise ValueError(f"This is not a easy_sm directory: {config.easy_sm_module_dir}")

    if not os.path.isfile(job_file_path):
        raise ValueError(f"Processing file does not exist: {job_file_path}")

    command = [
        local_process_script_path,
        os.path.abspath(test_path),
        helpers.docker_tag,
        config.image_name,
        file,
        config.aws_profile,
        config.aws_region,
    ]

    safe_run_subprocess(command, success_message="Local processing completed successfully!")


@local_app.command()
def deploy(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    port: Annotated[int, typer.Option("--port", "-p", help="Port to run the service on")] = 8080,
) -> None:
    """Deploy ML model(s) locally."""
    config = load_config(app_name)

    easy_sm_module_path = os.path.join(config.easy_sm_module_dir, "easy_sm_base")
    local_deploy_script_path = os.path.join(easy_sm_module_path, "local_test", "deploy_local.sh")
    test_path = os.path.join(easy_sm_module_path, "local_test", "test_dir")

    if not os.path.isdir(test_path):
        raise ValueError(f"This is not a easy_sm directory: {config.easy_sm_module_dir}")

    print(f"Started local deployment at localhost:{port} ...\n")
    command = [
        local_deploy_script_path,
        os.path.abspath(test_path),
        helpers.docker_tag,
        config.image_name,
        str(port),
    ]

    safe_run_subprocess(command)


@local_app.command()
def stop(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    port: Annotated[int, typer.Option("--port", "-p", help="Port the service is running on")] = 8080,
) -> None:
    """Stop a local deployment."""
    config = load_config(app_name)

    easy_sm_module_path = os.path.join(config.easy_sm_module_dir, "easy_sm_base")
    local_stop_script_path = os.path.join(easy_sm_module_path, "local_test", "stop_local.sh")
    test_path = os.path.join(easy_sm_module_path, "local_test", "test_dir")

    if not os.path.isdir(test_path):
        raise ValueError(f"This is not a easy_sm directory: {config.easy_sm_module_dir}")

    command = [
        local_stop_script_path,
        config.image_name,
        str(port),
    ]

    safe_run_subprocess(command, success_message="Local deployment stopped successfully!")
