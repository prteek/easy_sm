"""The easy_sm command line."""

from pathlib import Path
from typing import Annotated

import typer

from easy_sm import config
from easy_sm.config import CONFIG_FILE, SETTINGS, Config, ConfigError
from easy_sm.init import create_project

app = typer.Typer(
    help="Prototype SageMaker training, jobs and endpoints locally, then run the same commands in the cloud.",
    no_args_is_help=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Annotated[
        Path, typer.Option("--config", help="Config file to use instead of ./easy_sm.toml")
    ] = Path(CONFIG_FILE),
) -> None:
    ctx.obj = config_path


def load_config(ctx: typer.Context) -> Config:
    """Load the config file chosen by --config, or exit with the error on stderr."""
    try:
        return config.load(ctx.obj)
    except ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


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
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    for file in created:
        typer.echo(file)


@app.command()
def env(ctx: typer.Context) -> None:
    """Show every setting, its value and where the value came from."""
    cfg = load_config(ctx)
    for key in SETTINGS:
        value, source = cfg.resolve(key)
        shown = "-" if value is None else str(value)
        typer.echo(f"{key:<22} {shown:<45} {source}")


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
