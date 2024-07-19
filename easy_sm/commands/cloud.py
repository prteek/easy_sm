import os
import sys
import click
from easy_sm.config.config import ConfigManager
from easy_sm.sagemaker import sagemaker


def _config():
    if not os.path.isfile('.easy_sm.json'):
        raise ValueError("This is not a easy_sm directory: {}".format(os.getcwd()))
    else:
        return ConfigManager('.easy_sm.json').get_config()


@click.group()
def cloud():
    """
    Commands for AWS operations: upload data, train and deploy
    """
    pass


@click.command(name='upload-data')
@click.option(u"-i", u"--input-dir", required=True, help="Path to data input directory")
@click.option(
    u"-t", u"--target-dir",
    required=True,
    help="s3 location to upload data",
    type=click.Path()
)
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use for the upload command"
)
def upload_data(input_dir, target_dir, iam_role_arn):
    """
    Command to upload data to S3
    """
    print("Started uploading data to S3...\n")
    config = _config()
    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)
    target_path = sage_maker_client.upload_data(input_dir, target_dir)
    print("Data uploaded to {} successfully".format(target_path))


@click.command(name='train')
@click.option(
    u"-i", u"--input-s3-dir",
    required=True,
    help="s3 location to input data",
    type=click.Path()
)
@click.option(
    u"-o", u"--output-s3-dir",
    required=True,
    help="s3 location to save output (models, etc)",
    type=click.Path()
)
@click.option(u"-e", u"--ec2-type", required=True, help="ec2 instance type")
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use for the train command"
)
@click.option(
    u"-n",
    u"--base-job-name",
    required=False,
    help="Optional prefix for the SageMaker training job."
    "If not specified, the estimator generates a default job name, based on the training image name and current timestamp."
)
@click.pass_obj
def train(
        obj,
        input_s3_dir,
        output_s3_dir,
        ec2_type,
        iam_role_arn,
        base_job_name
):
    """
    Command to train ML model(s) on SageMaker
    """

    print("Started training on SageMaker...\n")
    config = _config()
    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)

    image_name = config.image_name+':'+obj['docker_tag']

    s3_model_location = sage_maker_client.train(
        image_name=image_name,
        input_s3_data_location=input_s3_dir,
        train_instance_type=ec2_type,
        output_path=output_s3_dir,
        base_job_name=base_job_name
    )

    print("Training on SageMaker succeeded")
    print("Model S3 location: {}".format(s3_model_location))
    return s3_model_location  # To pipe into other commands


@click.command(name='deploy-serverless')
@click.option(
    u"-m", u"--s3-model-location",
    required=True,
    help="s3 location to model tar.gz",
    type=click.Path()
)
@click.option(u"-s",
              u"--memory-size-in-mb",
              required=True,
              type=click.INT,
              help="memory size in MB for serverless endpoint")
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use for the deploy command"
)
@click.option(
u"-n",
    u"--endpoint-name",
    required=True,
    default=None,
    help="Name for the SageMaker endpoint"
)
@click.pass_obj
def deploy_serverless(
        obj,
        s3_model_location,
        memory_size_in_mb,
        iam_role_arn,
        endpoint_name
):
    """
    Command to deploy ML model(s) on SageMaker
    """

    print("Started deployment on SageMaker ...\n")
    config = _config()
    image_name = config.image_name+':'+obj['docker_tag']

    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)
    endpoint_name = sage_maker_client.deploy_serverless(
        image_name=image_name,
        s3_model_location=s3_model_location,
        memory_size_in_mb=memory_size_in_mb,
        endpoint_name=endpoint_name
    )

    print("Endpoint name: {}".format(endpoint_name))


@click.command(name='batch-transform')
@click.option(
    u"-m", u"--s3-model-location",
    required=True,
    help="s3 location to model tar.gz",
    type=click.Path()
)
@click.option(
    u"-i", u"--s3-input-location",
    required=True,
    help="s3 input data location",
    type=click.Path()
)
@click.option(
    u"-o", u"--s3-output-location",
    required=True,
    help="s3 location to save predictions",
    type=click.Path()
)
@click.option(u"--num-instances", required=True, type=int, help="Number of ec2 instances")
@click.option(u"--ec2-type", required=True, help="ec2 instance type")
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use for batch transform command"
)
@click.option(
    u"-w",
    u"--wait",
    default=False,
    is_flag=True,
    help="Wait until Batch Transform is finished. "
         "Default: don't wait"
)
@click.option(
    u"-n",
    u"--job-name",
    required=True,
    default=None,
    help="Name for the SageMaker batch transform job."
)
@click.pass_obj
def batch_transform(
        obj,
        s3_model_location,
        s3_input_location,
        s3_output_location,
        num_instances,
        ec2_type,
        iam_role_arn,
        wait,
        job_name
):
    """
    Command to execute a batch transform job given a trained ML model on SageMaker
    """
    print("Started configuration of batch transform on SageMaker ...\n")

    config = _config()
    image_name = config.image_name+':'+obj['docker_tag']

    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)
    status = sage_maker_client.batch_transform(
        image_name=image_name,
        s3_model_location=s3_model_location,
        s3_input_location=s3_input_location,
        s3_output_location=s3_output_location,
        transform_instance_count=num_instances,
        transform_instance_type=ec2_type,
        wait=wait,
        job_name=job_name
    )

    if wait:
        print("Batch transform on SageMaker finished with status: {}".format(status))
        if status == "Failed":
            sys.exit(1)
    else:
        print("Started batch transform on SageMaker successfully")


@click.command(name='delete-endpoint')
@click.option(
u"-n",
    u"--endpoint-name",
    required=True,
    default=None,
    help="Name of the SageMaker endpoint"
)
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use for delete command"
)
def delete_endpoint(endpoint_name, iam_role_arn):
    config = _config()
    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)
    sage_maker_client.shutdown_endpoint(endpoint_name)
    print(f"Endpoint {endpoint_name} has been deleted")


cloud.add_command(upload_data)
cloud.add_command(train)
cloud.add_command(deploy_serverless)
cloud.add_command(batch_transform)
cloud.add_command(delete_endpoint)
