import os
from typing import Annotated, Optional

import typer

from easy_sm.commands.helpers import load_config, safe_run_subprocess, serialize_env_vars

local_app = typer.Typer(help="Commands for local operations: train and deploy")


def _get_paths(app_name: str) -> tuple[str, str]:
    """Get easy_sm_base path and test_dir path from config."""
    config = load_config(app_name)
    base_path = os.path.join(config.easy_sm_module_dir, "easy_sm_base")
    test_path = os.path.join(base_path, "local_test", "test_dir")
    if not os.path.isdir(test_path):
        raise ValueError(f"Not a valid easy_sm directory: {config.easy_sm_module_dir}")
    return base_path, test_path


@local_app.command()
def train(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
) -> None:
    """Train ML model(s) locally."""
    config = load_config(app_name)
    base_path, test_path = _get_paths(app_name)

    safe_run_subprocess(
        [
            os.path.join(base_path, "local_test", "train_local.sh"),
            os.path.abspath(test_path),
            config.docker_tag,
            config.image_name,
        ],
        success_message="Local training completed",
    )


@local_app.command()
def process(
    file: Annotated[str, typer.Option("--file", "-f", help="Python file name to run as processing job")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    env: Annotated[Optional[list[str]], typer.Option("--env", help="Environment variables in KEY=VALUE format")] = None,
) -> None:
    """Run python files locally as processing job."""
    config = load_config(app_name)
    base_path, test_path = _get_paths(app_name)

    job_file = os.path.join(base_path, "processing", file)
    if not os.path.isfile(job_file):
        raise ValueError(f"Processing file not found: {job_file}")

    # Parse and validate environment variables
    env_vars = {}
    if env:
        for env_var in env:
            if "=" not in env_var:
                raise ValueError(f"Invalid environment variable format: {env_var}. Use KEY=VALUE")
            key, value = env_var.split("=", 1)
            env_vars[key] = value

    safe_run_subprocess(
        [
            os.path.join(base_path, "local_test", "process_local.sh"),
            os.path.abspath(test_path),
            config.docker_tag,
            config.image_name,
            file,
            config.aws_profile,
            config.aws_region,
            serialize_env_vars(env_vars),
        ],
        success_message="Local processing completed",
    )


@local_app.command()
def deploy(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    port: Annotated[int, typer.Option("--port", "-p", help="Port to run the service on")] = 8080,
) -> None:
    """Deploy ML model(s) locally."""
    config = load_config(app_name)
    base_path, test_path = _get_paths(app_name)

    print(f"Starting local deployment at localhost:{port}")
    safe_run_subprocess([
        os.path.join(base_path, "local_test", "deploy_local.sh"),
        os.path.abspath(test_path),
        config.docker_tag,
        config.image_name,
        str(port),
    ])


@local_app.command()
def stop(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    port: Annotated[int, typer.Option("--port", "-p", help="Port the service is running on")] = 8080,
) -> None:
    """Stop a local deployment."""
    config = load_config(app_name)
    base_path, _ = _get_paths(app_name)

    safe_run_subprocess(
        [os.path.join(base_path, "local_test", "stop_local.sh"), config.image_name, str(port)],
        success_message="Local deployment stopped",
    )
