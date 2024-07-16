import click
from easy_sm.commands.local import local
from easy_sm.commands.initialize import init
from easy_sm.commands.build import build
from easy_sm.commands.push import push
from easy_sm.commands.cloud import cloud

@click.group()
@click.option(u"-t", u"--docker-tag", default=u"latest", help=u"Specify tag for Docker image")
@click.pass_context
def cli(ctx, docker_tag):
    """
    easy_sm enables training and deploying machine learning models on AWS SageMaker in a few minutes!
    """
    ctx.obj = {'docker_tag': docker_tag}


def add_commands(cli):
    cli.add_command(init)
    cli.add_command(build)
    cli.add_command(local)
    cli.add_command(push)
    cli.add_command(cloud)
    # cli.add_command(configure)


add_commands(cli)
