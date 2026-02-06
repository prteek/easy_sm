import sys
from typing import Any, Dict, Optional

import click

from easy_sm.commands.helpers import (
    app_name_option,
    base_job_name_option,
    build_image_name,
    create_sagemaker_client,
    ec2_type_option,
    iam_role_option,
    instance_count_option,
    load_config,
)




@click.group()
def cloud() -> None:
    """
    Commands for AWS operations: upload data, train and deploy
    """
    pass


@click.command(name="upload-data")
@click.option(
    "-i",
    "--input-dir",
    required=True,
    help="Path to data input directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "-t",
    "--target-dir",
    required=True,
    help="s3 location to upload data",
    type=str,
)
@iam_role_option
@app_name_option
def upload_data(
    input_dir: str, target_dir: str, iam_role_arn: str, app_name: str
) -> None:
    """
    Command to upload data to S3
    """
    print("Started uploading data to S3...\n")
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )
    target_path = sage_maker_client.upload_data(input_dir, target_dir)
    print("Data uploaded to {} successfully".format(target_path))


@click.command(name="train")
@click.option(
    "-i",
    "--input-s3-dir",
    required=True,
    help="s3 location to input data",
    type=str,
)
@click.option(
    "-o",
    "--output-s3-dir",
    required=True,
    help="s3 location to save output (models, etc)",
    type=str,
)
@ec2_type_option
@instance_count_option
@iam_role_option
@base_job_name_option
@app_name_option
@click.pass_obj
def train(
    obj: Dict[str, Any],
    input_s3_dir: str,
    output_s3_dir: str,
    ec2_type: str,
    instance_count: int,
    iam_role_arn: str,
    base_job_name: str,
    app_name: str,
) -> Optional[str]:
    """
    Command to train ML model(s) on SageMaker
    """
    print("Started training on SageMaker...\n")
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )

    image_name = build_image_name(config.image_name, obj["docker_tag"])

    s3_model_location = sage_maker_client.train(
        image_name=image_name,
        input_s3_data_location=input_s3_dir,
        train_instance_type=ec2_type,
        instance_count=instance_count,
        output_path=output_s3_dir,
        base_job_name=base_job_name,
    )

    print("Training on SageMaker succeeded")
    print("Model S3 location: {}".format(s3_model_location))
    return s3_model_location


@click.command(name="deploy")
@click.option(
    "-m",
    "--s3-model-location",
    required=True,
    help="s3 location to model tar.gz",
    type=str,
)
@click.option(
    "-e", "--instance-type", required=True, help="ec2 instance type for the endpoint"
)
@instance_count_option
@click.option(
    "-n",
    "--endpoint-name",
    required=True,
    default=None,
    help="Name for the SageMaker endpoint",
)
@iam_role_option
@app_name_option
@click.pass_obj
def deploy(
    obj: Dict[str, Any],
    s3_model_location: str,
    instance_type: str,
    instance_count: int,
    iam_role_arn: str,
    endpoint_name: str,
    app_name: str,
) -> None:
    """
    Command to deploy ML model(s) on SageMaker as a regular endpoint
    """
    print("Started deployment on SageMaker ...\n")
    config = load_config(app_name)
    image_name = build_image_name(config.image_name, obj["docker_tag"])

    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )
    endpoint_name = sage_maker_client.deploy(
        image_name=image_name,
        s3_model_location=s3_model_location,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        instance_count=instance_count,
    )

    print("Endpoint name: {}".format(endpoint_name))


@click.command(name="deploy-serverless")
@click.option(
    "-m",
    "--s3-model-location",
    required=True,
    help="s3 location to model tar.gz",
    type=str,
)
@click.option(
    "-s",
    "--memory-size-in-mb",
    required=True,
    type=click.INT,
    help="memory size in MB for serverless endpoint",
)
@iam_role_option
@click.option(
    "-n",
    "--endpoint-name",
    required=True,
    default=None,
    help="Name for the SageMaker endpoint",
)
@app_name_option
@click.option(
    "-mc",
    "--max-concurrency",
    help="Max concurrency for the endpoint (default=5)",
    default=5,
    type=int,
)
@click.pass_obj
def deploy_serverless(
    obj: Dict[str, Any],
    s3_model_location: str,
    memory_size_in_mb: int,
    iam_role_arn: str,
    endpoint_name: str,
    app_name: str,
    max_concurrency: int,
) -> None:
    """
    Command to deploy ML model(s) on SageMaker
    """
    print("Started deployment on SageMaker ...\n")
    config = load_config(app_name)
    image_name = build_image_name(config.image_name, obj["docker_tag"])

    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )
    endpoint_name = sage_maker_client.deploy_serverless(
        image_name=image_name,
        s3_model_location=s3_model_location,
        memory_size_in_mb=memory_size_in_mb,
        endpoint_name=endpoint_name,
        max_concurrency=max_concurrency,
    )

    print("Endpoint name: {}".format(endpoint_name))


@click.command(name="batch-transform")
@click.option(
    "-m",
    "--s3-model-location",
    required=True,
    help="s3 location to model tar.gz",
    type=str,
)
@click.option(
    "-i",
    "--s3-input-location",
    required=True,
    help="s3 input data location",
    type=str,
)
@click.option(
    "-o",
    "--s3-output-location",
    required=True,
    help="s3 location to save predictions",
    type=str,
)
@click.option(
    "--num-instances", required=True, type=int, help="Number of ec2 instances"
)
@ec2_type_option
@iam_role_option
@click.option(
    "-w",
    "--wait",
    default=False,
    is_flag=True,
    help="Wait until Batch Transform is finished. Default: don't wait",
)
@click.option(
    "-n",
    "--job-name",
    required=False,
    default=None,
    help="Name for the SageMaker batch transform job.",
)
@app_name_option
@click.pass_obj
def batch_transform(
    obj: Dict[str, Any],
    s3_model_location: str,
    s3_input_location: str,
    s3_output_location: str,
    num_instances: int,
    ec2_type: str,
    iam_role_arn: str,
    wait: bool,
    job_name: Optional[str],
    app_name: str,
) -> None:
    """
    Command to execute a batch transform job given a trained ML model on SageMaker
    """
    print("Started configuration of batch transform on SageMaker ...\n")

    config = load_config(app_name)
    image_name = build_image_name(config.image_name, obj["docker_tag"])

    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )
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
        print("Batch transform on SageMaker finished with status: {}".format(status))
        if status == "Failed":
            sys.exit(1)
    else:
        print("Started batch transform on SageMaker successfully")


@click.command(name="delete-endpoint")
@click.option(
    "-n",
    "--endpoint-name",
    required=True,
    default=None,
    help="Name of the SageMaker endpoint",
)
@click.option(
    "--delete-config",
    is_flag=True,
    default=False,
    help="Also delete the associated endpoint config",
)
@iam_role_option
@app_name_option
def delete_endpoint(
    endpoint_name: str, delete_config: bool, iam_role_arn: str, app_name: str
) -> None:
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )
    sage_maker_client.shutdown_endpoint(endpoint_name)
    print(f"Endpoint {endpoint_name} has been deleted")

    if delete_config:
        endpoint_config_name = f"{endpoint_name}-config"
        sage_maker_client.delete_endpoint_config(endpoint_config_name)
        print(f"Endpoint config {endpoint_config_name} has been deleted")


@click.command(name="list-endpoints")
@iam_role_option
@app_name_option
def list_endpoints(iam_role_arn: str, app_name: str) -> None:
    """
    Command to list all SageMaker endpoints
    """
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )
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


@click.command(name="list-training-jobs")
@click.option(
    "-m",
    "--max-results",
    type=int,
    default=5,
    help="Number of training jobs to fetch (default: 5)",
)
@click.option(
    "-b",
    "--base-job-name",
    required=False,
    default=None,
    help="Filter by base job name (optional)",
)
@iam_role_option
@app_name_option
def list_training_jobs(
    max_results: int, base_job_name: Optional[str], iam_role_arn: str, app_name: str
) -> None:
    """
    Command to list recent SageMaker training jobs
    """
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )
    training_jobs = sage_maker_client.list_training_jobs(
        max_results=max_results, name_contains=base_job_name
    )

    if not training_jobs:
        print("No training jobs found")
        return

    print(f"\nFound {len(training_jobs)} training job(s):\n")
    for job in training_jobs:
        job_name = job.get("TrainingJobName", "N/A")
        job_status = job.get("TrainingJobStatus", "N/A")
        creation_time = job.get("CreationTime", "N/A")
        print(f"  • Name: {job_name}")
        print(f"    Status: {job_status}")
        print(f"    Created: {creation_time}\n")


@click.command(name="process")
@ec2_type_option
@instance_count_option
@iam_role_option
@base_job_name_option
@click.option(
    "-f",
    "--file",
    required=True,
    help="The name (not path) of python file to run as processing job",
)
@click.option(
    "-i",
    "--s3-input-location",
    required=False,
    default=None,
    help="s3 input data location",
    type=str,
)
@click.option(
    "-o",
    "--s3-output-location",
    required=False,
    default=None,
    help="s3 location to save output",
    type=str,
)
@click.option(
    "-is",
    "--input-sharded",
    is_flag=True,
    default=False,
    help="Flag to indicate if input data should be sharded (distributed on machines)",
)
@app_name_option
@click.pass_obj
def process(
    obj: Dict[str, Any],
    ec2_type: str,
    instance_count: int,
    iam_role_arn: str,
    base_job_name: str,
    file: str,
    s3_input_location: Optional[str],
    s3_output_location: Optional[str],
    input_sharded: bool,
    app_name: str,
) -> None:
    """
    Command to run python file as processing job on SageMaker
    """
    print("Started processing job on SageMaker...\n")
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )

    image_name = build_image_name(config.image_name, obj["docker_tag"])

    _ = sage_maker_client.process(
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


@click.command(name="make")
@ec2_type_option
@instance_count_option
@iam_role_option
@base_job_name_option
@click.option(
    "-t", "--target", required=True, help="The name of the target to be built"
)
@click.option(
    "-i",
    "--s3-input-location",
    required=False,
    default=None,
    help="s3 input data location",
    type=str,
)
@click.option(
    "-o",
    "--s3-output-location",
    required=False,
    default=None,
    help="s3 location to save output",
    type=str,
)
@click.option(
    "-is",
    "--input-sharded",
    is_flag=True,
    default=False,
    help="Flag to indicate if input data should be sharded (distributed on machines)",
)
@app_name_option
@click.pass_obj
def make(
    obj: Dict[str, Any],
    ec2_type: str,
    instance_count: int,
    iam_role_arn: str,
    base_job_name: str,
    target: str,
    s3_input_location: Optional[str],
    input_sharded: bool,
    s3_output_location: Optional[str],
    app_name: str,
) -> None:
    """
    Command to build make targets defined in a Makefile in easy_sm_base/processing on SageMaker
    """
    print(f"Building {target} on SageMaker...\n")
    config = load_config(app_name)
    sage_maker_client = create_sagemaker_client(
        config.aws_profile, config.aws_region, iam_role_arn
    )

    image_name = build_image_name(config.image_name, obj["docker_tag"])

    _ = sage_maker_client.make(
        image_name=image_name,
        processing_instance_type=ec2_type,
        instance_count=instance_count,
        target=target,
        s3_input_location=s3_input_location,
        input_sharded=input_sharded,
        s3_output_location=s3_output_location,
        base_job_name=base_job_name,
    )

    print(f"{target} built on SageMaker successfully!")


cloud.add_command(upload_data)
cloud.add_command(train)
cloud.add_command(deploy)
cloud.add_command(deploy_serverless)
cloud.add_command(batch_transform)
cloud.add_command(delete_endpoint)
cloud.add_command(list_endpoints)
cloud.add_command(list_training_jobs)
cloud.add_command(process)
cloud.add_command(make)
