from typing import Optional

import typer

from easy_sm.commands import helpers
from easy_sm.commands.build import build
from easy_sm.commands.cloud import cloud_app
from easy_sm.commands.initialize import init
from easy_sm.commands.local import local_app
from easy_sm.commands.push import push

app = typer.Typer(
    help="easy_sm enables training and deploying machine learning models on AWS SageMaker in a few minutes!"
)


def docker_tag_callback(tag: str) -> str:
    """Set global docker_tag when provided."""
    helpers.docker_tag = tag
    return tag


@app.callback()
def main(
    docker_tag: str = typer.Option(
        "latest",
        "--docker-tag",
        "-t",
        help="Specify tag for Docker image",
        callback=docker_tag_callback,
    ),
) -> None:
    """easy_sm CLI - Train and deploy ML models on AWS SageMaker."""
    pass


# Register commands
app.command(name="init")(init)
app.command(name="build")(build)
app.command(name="push")(push)

# Register sub-apps
app.add_typer(local_app, name="local")
app.add_typer(cloud_app, name="cloud")


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
