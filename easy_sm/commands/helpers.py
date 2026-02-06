import os
import subprocess
from typing import List, Optional

from easy_sm.config.config import Config, ConfigManager
from easy_sm.sagemaker import sagemaker

# Global state for docker_tag (set by main CLI callback)
docker_tag: str = "latest"


def safe_run_subprocess(
    command: List[str], success_message: Optional[str] = None
) -> int:
    """Safely run any subprocesses and print error messages sensibly."""
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
    """Load configuration from app_name.json in current directory."""
    config_file_path = os.path.join(f"{app_name}.json")
    if not os.path.isfile(config_file_path):
        raise ValueError(f"This is not a easy_sm directory: {os.getcwd()}")
    return ConfigManager(config_file_path).get_config()


def build_image_name(image_name: str, tag: str) -> str:
    """Build full Docker image name with tag."""
    return f"{image_name}:{tag}"


def create_sagemaker_client(
    aws_profile: str, aws_region: str, iam_role_arn: str
) -> sagemaker.SageMakerClient:
    """Create and return a SageMaker client."""
    return sagemaker.SageMakerClient(aws_profile, aws_region, iam_role_arn)
