
import typer

from easy_sm.commands.build import build
from easy_sm.commands.cloud import (
    batch_transform,
    delete_endpoint,
    deploy,
    deploy_serverless,
    get_model_artifacts,
    list_endpoints,
    list_training_jobs,
    process,
    train,
    upload_data,
)
from easy_sm.commands.initialize import init
from easy_sm.commands.local import local_app
from easy_sm.commands.push import push
from easy_sm.commands.update import update_scripts

app = typer.Typer(
    help="easy_sm enables training and deploying machine learning models on AWS SageMaker in a few minutes!"
)


# Register commands
app.command(name="init")(init)
app.command(name="build")(build)
app.command(name="push")(push)
app.command(name="update-scripts")(update_scripts)

# Register cloud commands at top level
app.command(name="upload-data")(upload_data)
app.command(name="train")(train)
app.command(name="deploy")(deploy)
app.command(name="deploy-serverless")(deploy_serverless)
app.command(name="batch-transform")(batch_transform)
app.command(name="delete-endpoint")(delete_endpoint)
app.command(name="list-endpoints")(list_endpoints)
app.command(name="list-training-jobs")(list_training_jobs)
app.command(name="get-model-artifacts")(get_model_artifacts)
app.command(name="process")(process)

# Register local sub-app
app.add_typer(local_app, name="local")


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
