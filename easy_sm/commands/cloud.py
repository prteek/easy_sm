import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from easy_sm.commands import helpers
from easy_sm.commands.helpers import get_app_name, get_iam_role, load_config
from easy_sm.sagemaker.sagemaker import SageMakerClient

cloud_app = typer.Typer(help="Commands for AWS operations: upload data, train and deploy")


def _get_client(app_name: str | None, iam_role_arn: str | None) -> SageMakerClient:
    """Create SageMaker client from app config."""
    app_name = get_app_name(app_name)
    iam_role_arn = get_iam_role(iam_role_arn)
    config = load_config(app_name)
    return SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)


def _get_image(app_name: str | None) -> str:
    """Get full image name with tag from app config."""
    app_name = get_app_name(app_name)
    config = load_config(app_name)
    return f"{config.image_name}:{helpers.docker_tag}"


@cloud_app.command(name="upload-data")
def upload_data(
    input_dir: Annotated[Path, typer.Option("--input-dir", "-i", help="Path to data input directory", exists=True, file_okay=False, dir_okay=True)],
    target_dir: Annotated[str, typer.Option("--target-dir", "-t", help="S3 location to upload data")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
) -> None:
    """Upload data to S3."""
    client = _get_client(app_name, iam_role_arn)
    target_path = client.upload_data(str(input_dir), target_dir)
    print(target_path)


@cloud_app.command(name="train")
def train(
    input_s3_dir: Annotated[str, typer.Option("--input-s3-dir", "-i", help="S3 location for input data")],
    output_s3_dir: Annotated[str, typer.Option("--output-s3-dir", "-o", help="S3 location to save output")],
    ec2_type: Annotated[str, typer.Option("--ec2-type", "-e", help="EC2 instance type")],
    base_job_name: Annotated[str, typer.Option("--base-job-name", "-n", help="Prefix for the SageMaker job")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="EC2 instance count")] = 1,
) -> Optional[str]:
    """Train ML model(s) on SageMaker."""
    client = _get_client(app_name, iam_role_arn)
    image = _get_image(app_name)

    s3_model_location = client.train(
        image_name=image,
        input_s3_data_location=input_s3_dir,
        train_instance_type=ec2_type,
        instance_count=instance_count,
        output_path=output_s3_dir,
        base_job_name=base_job_name,
    )

    print(s3_model_location)
    return s3_model_location


@cloud_app.command(name="deploy")
def deploy(
    s3_model_location: Annotated[str, typer.Option("--s3-model-location", "-m", help="S3 location to model tar.gz")],
    instance_type: Annotated[str, typer.Option("--instance-type", "-e", help="EC2 instance type for endpoint")],
    endpoint_name: Annotated[str, typer.Option("--endpoint-name", "-n", help="Name for the SageMaker endpoint")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="EC2 instance count")] = 1,
) -> None:
    """Deploy ML model(s) on SageMaker as a regular endpoint."""
    client = _get_client(app_name, iam_role_arn)
    image = _get_image(app_name)

    endpoint = client.deploy(
        image_name=image,
        s3_model_location=s3_model_location,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        instance_count=instance_count,
    )
    print(endpoint)


@cloud_app.command(name="deploy-serverless")
def deploy_serverless(
    s3_model_location: Annotated[str, typer.Option("--s3-model-location", "-m", help="S3 location to model tar.gz")],
    memory_size_in_mb: Annotated[int, typer.Option("--memory-size-in-mb", "-s", help="Memory size in MB for serverless endpoint")],
    endpoint_name: Annotated[str, typer.Option("--endpoint-name", "-n", help="Name for the SageMaker endpoint")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    max_concurrency: Annotated[int, typer.Option("--max-concurrency", "-mc", help="Max concurrency for the endpoint")] = 5,
) -> None:
    """Deploy ML model(s) on SageMaker as serverless endpoint."""
    client = _get_client(app_name, iam_role_arn)
    image = _get_image(app_name)

    endpoint = client.deploy_serverless(
        image_name=image,
        s3_model_location=s3_model_location,
        memory_size_in_mb=memory_size_in_mb,
        endpoint_name=endpoint_name,
        max_concurrency=max_concurrency,
    )
    print(endpoint)


@cloud_app.command(name="batch-transform")
def batch_transform(
    s3_model_location: Annotated[str, typer.Option("--s3-model-location", "-m", help="S3 location to model tar.gz")],
    s3_input_location: Annotated[str, typer.Option("--s3-input-location", "-i", help="S3 input data location")],
    s3_output_location: Annotated[str, typer.Option("--s3-output-location", "-o", help="S3 location to save predictions")],
    num_instances: Annotated[int, typer.Option("--num-instances", help="Number of EC2 instances")],
    ec2_type: Annotated[str, typer.Option("--ec2-type", "-e", help="EC2 instance type")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    wait: Annotated[bool, typer.Option("--wait", "-w", help="Wait until Batch Transform is finished")] = False,
    job_name: Annotated[Optional[str], typer.Option("--job-name", "-n", help="Name for the SageMaker batch transform job")] = None,
) -> None:
    """Execute a batch transform job on SageMaker."""
    client = _get_client(app_name, iam_role_arn)
    image = _get_image(app_name)

    status = client.batch_transform(
        image_name=image,
        s3_model_location=s3_model_location,
        s3_input_location=s3_input_location,
        s3_output_location=s3_output_location,
        transform_instance_count=num_instances,
        transform_instance_type=ec2_type,
        wait=wait,
        job_name=job_name,
    )

    if wait and status:
        print(status)
        if status == "Failed":
            sys.exit(1)


@cloud_app.command(name="delete-endpoint")
def delete_endpoint(
    endpoint_name: Annotated[str, typer.Option("--endpoint-name", "-n", help="Name of the SageMaker endpoint")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    delete_config: Annotated[bool, typer.Option("--delete-config", help="Also delete the associated endpoint config")] = False,
) -> None:
    """Delete a SageMaker endpoint."""
    client = _get_client(app_name, iam_role_arn)
    client.shutdown_endpoint(endpoint_name)
    print(endpoint_name)

    if delete_config:
        config_name = f"{endpoint_name}-config"
        client.delete_endpoint_config(config_name)


@cloud_app.command(name="list-endpoints")
def list_endpoints(
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
) -> None:
    """List all SageMaker endpoints."""
    client = _get_client(app_name, iam_role_arn)
    endpoints = client.list_endpoints()

    for ep in endpoints:
        print(f"{ep.get('EndpointName')}  {ep.get('EndpointStatus')}  {ep.get('CreationTime')}")


@cloud_app.command(name="list-training-jobs")
def list_training_jobs(
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    max_results: Annotated[int, typer.Option("--max-results", "-m", help="Maximum number of jobs to return")] = 5,
    names_only: Annotated[bool, typer.Option("--names-only", "-n", help="Output only job names (one per line)")] = False,
) -> None:
    """List recent SageMaker training jobs."""
    client = _get_client(app_name, iam_role_arn)
    jobs = client.list_training_jobs(max_results=max_results)

    for job in jobs:
        if names_only:
            print(job.get("TrainingJobName", ""))
        else:
            print(f"{job.get('TrainingJobName')}  {job.get('TrainingJobStatus')}  {job.get('CreationTime')}")


@cloud_app.command(name="get-model-artifacts")
def get_model_artifacts(
    training_job_name: Annotated[str, typer.Option("--training-job-name", "-j", help="Training job name")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
) -> None:
    """Get S3 model artifacts location from training job name."""
    client = _get_client(app_name, iam_role_arn)
    s3_location = client.get_model_artifacts(training_job_name)
    print(s3_location)


@cloud_app.command(name="process")
def process(
    file: Annotated[str, typer.Option("--file", "-f", help="Python file name to run as processing job")],
    ec2_type: Annotated[str, typer.Option("--ec2-type", "-e", help="EC2 instance type")],
    base_job_name: Annotated[str, typer.Option("--base-job-name", "-n", help="Prefix for the SageMaker job")],
    iam_role_arn: Annotated[Optional[str], typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN (or set SAGEMAKER_ROLE env var)")] = None,
    app_name: Annotated[Optional[str], typer.Option("--app-name", "-a", help="App name (auto-detected if not specified)")] = None,
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="EC2 instance count")] = 1,
    s3_input_location: Annotated[Optional[str], typer.Option("--s3-input-location", "-i", help="S3 input data location")] = None,
    s3_output_location: Annotated[Optional[str], typer.Option("--s3-output-location", "-o", help="S3 location to save output")] = None,
    input_sharded: Annotated[bool, typer.Option("--input-sharded", "-is", help="Shard input data across machines")] = False,
) -> None:
    """Run python file as processing job on SageMaker."""
    client = _get_client(app_name, iam_role_arn)
    image = _get_image(app_name)

    client.process(
        image_name=image,
        processing_instance_type=ec2_type,
        instance_count=instance_count,
        file=file,
        s3_input_location=s3_input_location,
        input_sharded=input_sharded,
        s3_output_location=s3_output_location,
        base_job_name=base_job_name,
    )
    print(base_job_name)
