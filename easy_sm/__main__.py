"""The easy_sm command line."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from easy_sm import config, image, local
from easy_sm.config import CONFIG_FILE, SETTINGS, Config, ConfigError
from easy_sm.image import ImageError
from easy_sm.init import create_project

app = typer.Typer(
    help="Prototype SageMaker training, jobs and endpoints locally, then run the same commands in the cloud.",
    no_args_is_help=True,
)
endpoints = typer.Typer(help="Serve, call and manage endpoints.", no_args_is_help=True)
app.add_typer(endpoints, name="endpoints")

LocalFlag = Annotated[bool, typer.Option("--local", help="Run in local Docker instead of SageMaker")]
PortOption = Annotated[int, typer.Option("--port", help="Local port of the endpoint")]


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Annotated[
        Path, typer.Option("--config", help="Config file to use instead of ./easy_sm.toml")
    ] = Path(CONFIG_FILE),
) -> None:
    ctx.obj = config_path


def fail(message: str, code: int = 1) -> NoReturn:
    """Print the message on stderr and exit."""
    typer.echo(message, err=True)
    raise typer.Exit(code)


def load_config(ctx: typer.Context) -> Config:
    """Load the config file chosen by --config, or exit with the error."""
    try:
        return config.load(ctx.obj)
    except ConfigError as e:
        fail(str(e))


def cloud_not_ready() -> NoReturn:
    fail("Only --local is available so far; SageMaker support comes in a later version of v2.", 2)


def build_image(cfg: Config) -> str:
    """Build the image if the code changed; return its tag."""
    try:
        return image.build(cfg.path.parent, cfg.get("app_name"))
    except ImageError as e:
        fail(str(e))


def docker_or_exit(command: list[str]) -> None:
    """Run a container; exit with its exit code if that isn't 0."""
    try:
        code = local.docker(command)
    except ImageError as e:
        fail(str(e))
    if code != 0:
        raise typer.Exit(code)


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Folder to create the project in")] = Path("."),
    name: Annotated[
        str | None, typer.Option("--name", help="App name (default: from the folder name)")
    ] = None,
) -> None:
    """Create a new project: easy_sm.toml, pyproject.toml, src/train.py, src/predict.py and sample data."""
    try:
        created = create_project(path, name)
    except ConfigError as e:
        fail(str(e))
    for file in created:
        typer.echo(file)


@app.command()
def env(ctx: typer.Context) -> None:
    """Show every setting, its value and where the value came from."""
    cfg = load_config(ctx)
    try:
        tag, source = image.image_tag(cfg.path.parent, cfg.get("app_name")), "computed"
    except ImageError:
        tag, source = "-", "run 'easy_sm build'"
    typer.echo(f"{'image':<22} {tag:<45} {source}")
    for key in SETTINGS:
        value, source = cfg.resolve(key)
        shown = "-" if value is None else str(value)
        typer.echo(f"{key:<22} {shown:<45} {source}")


@app.command()
def build(ctx: typer.Context) -> None:
    """Build the project's Docker image (skipped if this exact image already exists) and print its tag."""
    typer.echo(build_image(load_config(ctx)))


@app.command()
def train(
    ctx: typer.Context,
    is_local: LocalFlag = False,
    input_dir: Annotated[
        Path, typer.Option("--input", help="Local folder with the training data")
    ] = Path("data"),
) -> None:
    """Train a model: calls train() in src/train.py. Prints where the model was saved."""
    if not is_local:
        cloud_not_ready()
    cfg = load_config(ctx)
    project = cfg.path.parent
    data_dir = input_dir if input_dir.is_absolute() else project / input_dir
    if not data_dir.is_dir():
        fail(f"{data_dir} not found")
    tag = build_image(cfg)
    model_dir, output_dir = local.local_dirs(project)
    local.empty(model_dir)
    docker_or_exit(local.train_command(tag, data_dir, model_dir, output_dir))
    typer.echo(model_dir)


@app.command(context_settings={"allow_interspersed_args": False})
def run(
    ctx: typer.Context,
    command: Annotated[
        list[str], typer.Argument(help="Command to run from src/, e.g. clean.py --date 2026-11-01")
    ],
    is_local: LocalFlag = False,
    env_pairs: Annotated[
        list[str] | None, typer.Option("--env", help="Environment variable KEY=VALUE (repeatable)")
    ] = None,
) -> None:
    """Run any command from src/ in the project's image. A .py file runs with python.

    easy_sm options go before the command; everything after it is passed to the command.
    """
    if not is_local:
        cloud_not_ready()
    cfg = load_config(ctx)
    project = cfg.path.parent
    try:
        local.check_script(project / "src", command)
        env_vars = {**(cfg.get("run.env") or {}), **local.parse_env(env_pairs or [])}
    except ValueError as e:
        fail(str(e))
    tag = build_image(cfg)
    _, output_dir = local.local_dirs(project)
    docker_or_exit(local.run_command(tag, project / "data", output_dir, env_vars, command))


@endpoints.command()
def serve(ctx: typer.Context, is_local: LocalFlag = False, port: PortOption = 8080) -> None:
    """Serve the model trained with 'train --local' on 127.0.0.1. Ctrl-C stops it."""
    if not is_local:
        fail("serve only runs locally: use 'easy_sm endpoints serve --local'", 2)
    cfg = load_config(ctx)
    model_dir, _ = local.local_dirs(cfg.path.parent)
    if not any(model_dir.iterdir()):
        fail(f"{model_dir} is empty. Run 'easy_sm train --local' first.")
    tag = build_image(cfg)
    typer.echo(f"Serving {tag} on http://127.0.0.1:{port} (Ctrl-C to stop)", err=True)
    try:
        docker_or_exit(local.serve_command(tag, model_dir, port))
    except KeyboardInterrupt:
        pass  # Ctrl-C also reaches docker, which stops the container


@endpoints.command()
def invoke(
    is_local: LocalFlag = False,
    data: Annotated[
        str, typer.Option("--data", "-d", help="Request body, or @file to send a file")
    ] = "",
    content_type: Annotated[
        str, typer.Option("--content-type", help="application/json or text/csv")
    ] = "application/json",
    port: PortOption = 8080,
) -> None:
    """Send one prediction request and print the response."""
    if not is_local:
        cloud_not_ready()
    try:
        status, body = local.invoke(port, local.request_body(data), content_type)
    except (ImageError, OSError) as e:
        fail(str(e))
    if status != 200:
        fail(f"HTTP {status}: {body.decode()}")
    typer.echo(body.decode())


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
