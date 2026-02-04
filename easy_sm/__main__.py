import os

import click

from easy_sm.commands.cloud import cloud
from easy_sm.commands.initialize import init
from easy_sm.commands.build import build
from easy_sm.commands.local import local
from easy_sm.commands.push import push

os.environ["DISABLE_JUMPSTART_LOGGING"] = "1"


@click.group()
@click.option(
    "-t",
    "--docker-tag",
    default="latest",
    help="Specify tag for Docker image",
)
@click.pass_context
def cli(ctx: click.Context, docker_tag: str) -> None:
    """
    easy_sm enables training and deploying machine learning models on AWS SageMaker in a few minutes!
    """
    ctx.obj = {"docker_tag": docker_tag}


def add_commands(cli: click.Group) -> None:
    cli.add_command(init)
    cli.add_command(build)
    cli.add_command(local)
    cli.add_command(push)
    cli.add_command(cloud)


add_commands(cli)
