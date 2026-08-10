import os
import sys
from typing import Annotated, Optional

import typer

from easy_sm.commands.helpers import load_config, safe_run_subprocess


def _push(
    dir: str,
    docker_tag: str,
    aws_region: str,
    iam_role_arn: str,
    aws_profile: str,
    external_id: str,
    image_name: str,
) -> None:
    """Push Docker image to AWS ECR."""
    easy_sm_module_path = os.path.relpath(os.path.join(dir, "easy_sm_base/"))
    push_script_path = os.path.join(easy_sm_module_path, "push.sh")

    if not os.path.isfile(push_script_path):
        raise ValueError(f"This is not a easy_sm directory: {dir}")

    command = [
        push_script_path,
        docker_tag,
        aws_region,
        iam_role_arn,
        aws_profile,
        external_id,
        image_name,
    ]
    safe_run_subprocess(command)


def push(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    aws_region: Annotated[Optional[str], typer.Option("--aws-region", "-r", help="AWS region")] = None,
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-i", help="AWS IAM role ARN")] = None,
    aws_profile: Annotated[Optional[str], typer.Option("--aws-profile", "-p", help="AWS profile")] = None,
    external_id: Annotated[Optional[str], typer.Option("--external-id", "-e", help="External ID for IAM role")] = None,
) -> None:
    """Push Docker image to AWS ECR."""
    if iam_role_arn is not None and aws_profile is not None:
        print("Only one of iam-role-arn and aws-profile can be used.")
        sys.exit(2)

    if iam_role_arn is not None:
        aws_profile = ""

    config = load_config(app_name)
    image_name = config.image_name
    aws_region = config.aws_region if aws_region is None else aws_region
    aws_profile = (
        config.aws_profile
        if (aws_profile is None and iam_role_arn is None)
        else (aws_profile or "")
    )
    external_id = external_id or ""
    iam_role_arn = iam_role_arn or ""

    print("Started pushing Docker image to AWS ECR. It will take some time. Please, be patient...\n")

    _push(
        dir=config.easy_sm_module_dir,
        docker_tag=config.docker_tag,
        aws_region=aws_region,
        iam_role_arn=iam_role_arn,
        aws_profile=aws_profile,
        external_id=external_id,
        image_name=image_name,
    )

    print("Docker image pushed to ECR successfully!")
