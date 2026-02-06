import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from easy_sm.commands import helpers
from easy_sm.commands.helpers import (
    build_image_name,
    create_sagemaker_client,
    load_config,
)

cloud_app = typer.Typer(help="Commands for AWS operations: upload data, train and deploy")


@cloud_app.command(name="upload-data")
def upload_data(
    input_dir: Annotated[Path, typer.Option("--input-dir", "-i", help="Path to data input directory", exists=True, file_okay=False, dir_okay=True)],
    target_dir: Annotated[str, typer.Option("--target-dir", "-t", help="S3 location to upload data")],
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
) -> None:
    """Upload data to S3."""
    print("Started uploading data to S3...\n")
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)
    target_path = sage_maker_client.upload_data(str(input_dir), target_dir)
    print(f"Data uploaded to {target_path} successfully")


@cloud_app.command(name="train")
def train(
    input_s3_dir: Annotated[str, typer.Option("--input-s3-dir", "-i", help="S3 location for input data")],
    output_s3_dir: Annotated[str, typer.Option("--output-s3-dir", "-o", help="S3 location to save output")],
    ec2_type: Annotated[str, typer.Option("--ec2-type", "-e", help="EC2 instance type")],
    base_job_name: Annotated[str, typer.Option("--base-job-name", "-n", help="Prefix for the SageMaker job")],
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="EC2 instance count")] = 1,
) -> Optional[str]:
    """Train ML model(s) on SageMaker."""
    print("Started training on SageMaker...\n")
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)

    image_name = build_image_name(config.image_name, helpers.docker_tag)

    s3_model_location = sage_maker_client.train(
        image_name=image_name,
        input_s3_data_location=input_s3_dir,
        train_instance_type=ec2_type,
        instance_count=instance_count,
        output_path=output_s3_dir,
        base_job_name=base_job_name,
    )

    print("Training on SageMaker succeeded")
    print(f"Model S3 location: {s3_model_location}")
    return s3_model_location


@cloud_app.command(name="deploy")
def deploy(
    s3_model_location: Annotated[str, typer.Option("--s3-model-location", "-m", help="S3 location to model tar.gz")],
    instance_type: Annotated[str, typer.Option("--instance-type", "-e", help="EC2 instance type for endpoint")],
    endpoint_name: Annotated[str, typer.Option("--endpoint-name", "-n", help="Name for the SageMaker endpoint")],
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="EC2 instance count")] = 1,
) -> None:
    """Deploy ML model(s) on SageMaker as a regular endpoint."""
    print("Started deployment on SageMaker ...\n")
    config = load_config(app_name)
    image_name = build_image_name(config.image_name, helpers.docker_tag)

    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)
    endpoint_name = sage_maker_client.deploy(
        image_name=image_name,
        s3_model_location=s3_model_location,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        instance_count=instance_count,
    )

    print(f"Endpoint name: {endpoint_name}")


@cloud_app.command(name="deploy-serverless")
def deploy_serverless(
    s3_model_location: Annotated[str, typer.Option("--s3-model-location", "-m", help="S3 location to model tar.gz")],
    memory_size_in_mb: Annotated[int, typer.Option("--memory-size-in-mb", "-s", help="Memory size in MB for serverless endpoint")],
    endpoint_name: Annotated[str, typer.Option("--endpoint-name", "-n", help="Name for the SageMaker endpoint")],
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    max_concurrency: Annotated[int, typer.Option("--max-concurrency", "-mc", help="Max concurrency for the endpoint")] = 5,
) -> None:
    """Deploy ML model(s) on SageMaker as serverless endpoint."""
    print("Started deployment on SageMaker ...\n")
    config = load_config(app_name)
    image_name = build_image_name(config.image_name, helpers.docker_tag)

    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)
    endpoint_name = sage_maker_client.deploy_serverless(
        image_name=image_name,
        s3_model_location=s3_model_location,
        memory_size_in_mb=memory_size_in_mb,
        endpoint_name=endpoint_name,
        max_concurrency=max_concurrency,
    )

    print(f"Endpoint name: {endpoint_name}")


@cloud_app.command(name="batch-transform")
def batch_transform(
    s3_model_location: Annotated[str, typer.Option("--s3-model-location", "-m", help="S3 location to model tar.gz")],
    s3_input_location: Annotated[str, typer.Option("--s3-input-location", "-i", help="S3 input data location")],
    s3_output_location: Annotated[str, typer.Option("--s3-output-location", "-o", help="S3 location to save predictions")],
    num_instances: Annotated[int, typer.Option("--num-instances", help="Number of EC2 instances")],
    ec2_type: Annotated[str, typer.Option("--ec2-type", "-e", help="EC2 instance type")],
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    wait: Annotated[bool, typer.Option("--wait", "-w", help="Wait until Batch Transform is finished")] = False,
    job_name: Annotated[Optional[str], typer.Option("--job-name", "-n", help="Name for the SageMaker batch transform job")] = None,
) -> None:
    """Execute a batch transform job on SageMaker."""
    print("Started configuration of batch transform on SageMaker ...\n")

    config = load_config(app_name)
    image_name = build_image_name(config.image_name, helpers.docker_tag)

    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)
    status = sage_maker_client.batch_transform(
        image_name=image_name,
        s3_model_location=s3_model_location,
        s3_input_location=s3_input_location,
        s3_output_location=s3_output_location,
        transform_instance_count=num_instances,
        transform_instance_type=ec2_type,
        wait=wait,
        job_name=job_name,
    )

    if wait:
        print(f"Batch transform on SageMaker finished with status: {status}")
        if status == "Failed":
            sys.exit(1)
    else:
        print("Started batch transform on SageMaker successfully")


@cloud_app.command(name="delete-endpoint")
def delete_endpoint(
    endpoint_name: Annotated[str, typer.Option("--endpoint-name", "-n", help="Name of the SageMaker endpoint")],
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    delete_config: Annotated[bool, typer.Option("--delete-config", help="Also delete the associated endpoint config")] = False,
) -> None:
    """Delete a SageMaker endpoint."""
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)
    sage_maker_client.shutdown_endpoint(endpoint_name)
    print(f"Endpoint {endpoint_name} has been deleted")

    if delete_config:
        endpoint_config_name = f"{endpoint_name}-config"
        sage_maker_client.delete_endpoint_config(endpoint_config_name)
        print(f"Endpoint config {endpoint_config_name} has been deleted")


@cloud_app.command(name="list-endpoints")
def list_endpoints(
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
) -> None:
    """List all SageMaker endpoints."""
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)
    endpoints = sage_maker_client.list_endpoints()

    if not endpoints:
        print("No endpoints found")
        return

    print(f"\nFound {len(endpoints)} endpoint(s):\n")
    for endpoint in endpoints:
        endpoint_name = endpoint.get("EndpointName", "N/A")
        endpoint_status = endpoint.get("EndpointStatus", "N/A")
        creation_time = endpoint.get("CreationTime", "N/A")
        print(f"  • Name: {endpoint_name}")
        print(f"    Status: {endpoint_status}")
        print(f"    Created: {creation_time}\n")


@cloud_app.command(name="list-training-jobs")
def list_training_jobs(
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    max_results: Annotated[int, typer.Option("--max-results", "-m", help="Maximum number of jobs to return")] = 5,
    names_only: Annotated[bool, typer.Option("--names-only", "-n", help="Output only job names (one per line)")] = False,
) -> None:
    """List recent SageMaker training jobs."""
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)
    training_jobs = sage_maker_client.list_training_jobs(max_results=max_results)

    if not training_jobs:
        if not names_only:
            print("No training jobs found")
        return

    if names_only:
        for job in training_jobs:
            print(job.get("TrainingJobName", ""))
    else:
        print(f"\nFound {len(training_jobs)} training job(s):\n")
        for job in training_jobs:
            job_name = job.get("TrainingJobName", "N/A")
            job_status = job.get("TrainingJobStatus", "N/A")
            creation_time = job.get("CreationTime", "N/A")
            print(f"  • Name: {job_name}")
            print(f"    Status: {job_status}")
            print(f"    Created: {creation_time}\n")


@cloud_app.command(name="process")
def process(
    file: Annotated[str, typer.Option("--file", "-f", help="Python file name to run as processing job")],
    ec2_type: Annotated[str, typer.Option("--ec2-type", "-e", help="EC2 instance type")],
    base_job_name: Annotated[str, typer.Option("--base-job-name", "-n", help="Prefix for the SageMaker job")],
    iam_role_arn: Annotated[str, typer.Option("--iam-role-arn", "-r", help="AWS IAM role ARN")],
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
    instance_count: Annotated[int, typer.Option("--instance-count", "-c", help="EC2 instance count")] = 1,
    s3_input_location: Annotated[Optional[str], typer.Option("--s3-input-location", "-i", help="S3 input data location")] = None,
    s3_output_location: Annotated[Optional[str], typer.Option("--s3-output-location", "-o", help="S3 location to save output")] = None,
    input_sharded: Annotated[bool, typer.Option("--input-sharded", "-is", help="Shard input data across machines")] = False,
) -> None:
    """Run python file as processing job on SageMaker."""
    print("Started processing job on SageMaker...\n")
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(config.aws_profile, config.aws_region, iam_role_arn)

    image_name = build_image_name(config.image_name, helpers.docker_tag)

    sage_maker_client.process(
        image_name=image_name,
        processing_instance_type=ec2_type,
        instance_count=instance_count,
        file=file,
        s3_input_location=s3_input_location,
        input_sharded=input_sharded,
        s3_output_location=s3_output_location,
        base_job_name=base_job_name,
    )

    print("Processing job on SageMaker succeeded")


