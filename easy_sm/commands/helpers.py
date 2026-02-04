import os
import subprocess
from typing import List, Optional

import click

from easy_sm.config.config import Config, ConfigManager
from easy_sm.sagemaker import sagemaker


def safe_run_subprocess(
    command: List[str], success_message: Optional[str] = None
) -> int:
    """Safely run any subprocesses and print error messages sensibly"""
    return_code: int = 0
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        stdout = process.stdout
        if stdout:
            for line in stdout:
                print(line, end="")

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        if success_message:
            print(success_message)

    except subprocess.CalledProcessError as e:
        print("Error occurred while running the command:")
        print(f"Return code: {e.returncode}")
        print(f"Command: {e.cmd}")
        print("Error output:")
        print(e.output)

    return return_code


def load_config(app_name: str) -> Config:
    """Load configuration from app_name.json in current directory.

    Args:
        app_name: The app name whose json file will be loaded

    Returns:
        Config object with application settings

    Raises:
        ValueError: If not in a valid easy_sm directory
    """
    config_file_path = os.path.join(f"{app_name}.json")
    if not os.path.isfile(config_file_path):
        raise ValueError("This is not a easy_sm directory: {}".format(os.getcwd()))
    return ConfigManager(config_file_path).get_config()


def build_image_name(image_name: str, docker_tag: str) -> str:
    """Build full Docker image name with tag.

    Args:
        image_name: Base image name
        docker_tag: Docker tag to append

    Returns:
        Full image name in format: image_name:tag
    """
    return f"{image_name}:{docker_tag}"


def create_sagemaker_client(
    aws_profile: str, aws_region: str, iam_role_arn: str
) -> sagemaker.SageMakerClient:
    """Create and return a SageMaker client.

    Args:
        aws_profile: AWS profile name
        aws_region: AWS region
        iam_role_arn: IAM role ARN

    Returns:
        SageMakerClient instance
    """
    return sagemaker.SageMakerClient(aws_profile, aws_region, iam_role_arn)


# Reusable Click option decorators
def app_name_option(f: object) -> object:
    """Add --app-name option to Click command"""
    return click.option(
        "-a",
        "--app-name",
        required=True,
        help="The app name whose json file will be referenced for setting up command",
    )(f)


def iam_role_option(f: object) -> object:
    """Add --iam-role-arn option to Click command"""
    return click.option(
        "-r",
        "--iam-role-arn",
        required=True,
        help="The AWS role to use for this command",
    )(f)


def iam_role_optional_option(f: object) -> object:
    """Add optional --iam-role-arn option to Click command"""
    return click.option(
        "-i",
        "--iam-role-arn",
        required=False,
        help="The AWS role to use for this command",
    )(f)


def aws_region_option(f: object) -> object:
    """Add --aws-region option to Click command"""
    return click.option(
        "-r",
        "--aws-region",
        required=False,
        help="The AWS region",
    )(f)


def ec2_type_option(f: object) -> object:
    """Add --ec2-type option to Click command"""
    return click.option(
        "-e",
        "--ec2-type",
        required=True,
        help="EC2 instance type",
    )(f)


def instance_count_option(f: object) -> object:
    """Add --instance-count option to Click command"""
    return click.option(
        "-c",
        "--instance-count",
        required=False,
        default=1,
        help="EC2 instance count",
        type=int,
    )(f)


def base_job_name_option(f: object) -> object:
    """Add --base-job-name option to Click command"""
    return click.option(
        "-n",
        "--base-job-name",
        required=True,
        help="Prefix for the SageMaker job",
    )(f)
