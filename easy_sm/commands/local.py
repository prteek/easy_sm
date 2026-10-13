import os
from typing import Annotated

import typer

from easy_sm.commands.helpers import (
    load_config,
    parse_env_vars,
    require_run_support,
    safe_run_subprocess,
)

local_app = typer.Typer(help="Commands for local operations: train, deploy, process and run")


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
    env: Annotated[list[str] | None, typer.Option("--env", help="Environment variables in KEY=VALUE format")] = None,
) -> None:
    """Run python files locally as processing job."""
    config = load_config(app_name)
    base_path, test_path = _get_paths(app_name)

    job_file = os.path.join(base_path, "processing", file)
    if not os.path.isfile(job_file):
        raise ValueError(f"Processing file not found: {job_file}")

    env_vars = parse_env_vars(env)
    process_script = os.path.join(base_path, "local_test", "process_local.sh")

    # Scripts from before easy_sm 1.3.0 joined env vars with commas and would silently drop all but the first
    if env_vars:
        with open(process_script) as f:
            if "IFS=','" in f.read():
                raise ValueError(
                    f"{process_script} is outdated and can't pass --env safely. "
                    f"Run 'easy_sm update-scripts -a {app_name}'"
                )

    safe_run_subprocess(
        [
            process_script,
            os.path.abspath(test_path),
            config.docker_tag,
            config.image_name,
            file,
            config.aws_profile,
            config.aws_region,
            *(f"{key}={value}" for key, value in env_vars.items()),
        ],
        success_message="Local processing completed",
    )


# Stop option parsing at the command, so its flags (e.g. `dbt run -e prod`) are never read as easy_sm options
@local_app.command(context_settings={"allow_interspersed_args": False})
def run(
    command: Annotated[list[str], typer.Argument(help="Command to run in easy_sm_base/processing, after -- (e.g. -- make train)")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    env: Annotated[list[str] | None, typer.Option("--env", help="Environment variables in KEY=VALUE format")] = None,
) -> None:
    """Run any CLI command (e.g. make, dbt) locally as a processing job."""
    config = load_config(app_name)
    base_path, test_path = _get_paths(app_name)
    require_run_support(config, app_name)
    env_vars = parse_env_vars(env)

    run_script = os.path.join(base_path, "local_test", "run_local.sh")
    if not os.path.isfile(run_script):
        raise ValueError(f"{run_script} not found. Run 'easy_sm update-scripts -a {app_name}'")

    return_code = safe_run_subprocess(
        [
            run_script,
            os.path.abspath(test_path),
            config.docker_tag,
            config.image_name,
            config.aws_profile,
            config.aws_region,
            *(f"{key}={value}" for key, value in env_vars.items()),
            "--",
            *command,
        ],
        success_message="Local run completed",
    )
    if return_code != 0:
        raise typer.Exit(return_code)


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
