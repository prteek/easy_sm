import os
import sys
import click
from easy_sm.config.config import ConfigManager
from easy_sm.sagemaker import sagemaker


def _config(app_name):
    config_file_path = os.path.join(f'{app_name}.json')
    if not os.path.isfile(config_file_path):
        raise ValueError("This is not a easy_sm directory: {}".format(os.getcwd()))
    else:
        return ConfigManager(config_file_path).get_config()


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
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
def upload_data(input_dir, target_dir, iam_role_arn, app_name):
    """
    Command to upload data to S3
    """
    print("Started uploading data to S3...\n")
    config = _config(app_name)
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
@click.option(u"-c", u"--instance-count", required=False, default=1, help="ec2 instance count")
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use for the train command"
)
@click.option(
    u"-n",
    u"--base-job-name",
    required=True,
    help="Prefix for the SageMaker training job."
)
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def train(
        obj,
        input_s3_dir,
        output_s3_dir,
        ec2_type,
        instance_count,
        iam_role_arn,
        base_job_name,
        app_name
):
    """
    Command to train ML model(s) on SageMaker
    """

    print("Started training on SageMaker...\n")
    config = _config(app_name)
    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)

    image_name = config.image_name+':'+obj['docker_tag']

    s3_model_location = sage_maker_client.train(
        image_name=image_name,
        input_s3_data_location=input_s3_dir,
        train_instance_type=ec2_type,
        instance_count=instance_count,
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
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.option(
    u"-mc",
    u"--max-concurrency",
    help="Max concurrency for the endpoint (default=5)",
    default=5,
    type=int
)
@click.pass_obj
def deploy_serverless(
        obj,
        s3_model_location,
        memory_size_in_mb,
        iam_role_arn,
        endpoint_name,
        app_name,
        max_concurrency
):
    """
    Command to deploy ML model(s) on SageMaker
    """

    print("Started deployment on SageMaker ...\n")
    config = _config(app_name)
    image_name = config.image_name+':'+obj['docker_tag']

    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)
    endpoint_name = sage_maker_client.deploy_serverless(
        image_name=image_name,
        s3_model_location=s3_model_location,
        memory_size_in_mb=memory_size_in_mb,
        endpoint_name=endpoint_name,
        max_concurrency=max_concurrency
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
    required=False,
    default=None,
    help="Name for the SageMaker batch transform job."
)
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
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
        job_name,
        app_name
):
    """
    Command to execute a batch transform job given a trained ML model on SageMaker
    """
    print("Started configuration of batch transform on SageMaker ...\n")

    config = _config(app_name)
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
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
def delete_endpoint(endpoint_name, iam_role_arn, app_name):
    config = _config(app_name)
    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)
    sage_maker_client.shutdown_endpoint(endpoint_name)
    print(f"Endpoint {endpoint_name} has been deleted")


@click.command(name='process')
@click.option(u"-e", u"--ec2-type", required=True, help="ec2 instance type")
@click.option(u"-c", u"--instance-count", required=False, default=1, help="ec2 instance count")
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use"
)
@click.option(
    u"-n",
    u"--base-job-name",
    required=True,
    help="Prefix for the SageMaker processing job."
    "If not specified, default job name is generated, based on the docker image name and current timestamp."
)
@click.option(
    u"-f",
    u"--file",
    required=True,
    help="The name (not path) of python file to run as processing job"
)
@click.option(
    u"-i", u"--s3-input-location",
    required=False,
    default=None,
    help="s3 input data location",
    type=click.Path()
)
@click.option(
    u"-o", u"--s3-output-location",
    required=False,
    default=None,
    help="s3 location to save output",
    type=click.Path()
)
@click.option(
    u"-is", u"--input-sharded",
    is_flag=True,
    default=False,
    help="Flag to indicate if input data should be sharded (distributed on machines)",
)
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def process(
        obj,
        ec2_type,
        instance_count,
        iam_role_arn,
        base_job_name,
        file,
        s3_input_location,
        s3_output_location,
        input_sharded,
        app_name
):
    """
    Command to run python file as processing job on Sagemaker
    """

    print("Started processing job on SageMaker...\n")
    config = _config(app_name)
    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)

    image_name = config.image_name+':'+obj['docker_tag']

    _ = sage_maker_client.process(
        image_name=image_name,
        processing_instance_type=ec2_type,
        instance_count=instance_count,
        file=file,
        s3_input_location=s3_input_location,
        input_sharded=input_sharded,
        s3_output_location=s3_output_location,
        base_job_name=base_job_name
    )

    print("Processing job on SageMaker succeeded")



@click.command(name='make')
@click.option(u"-e", u"--ec2-type", required=True, help="ec2 instance type")
@click.option(u"-c", u"--instance-count", required=False, default=1, help="ec2 instance count")
@click.option(
    u"-r",
    u"--iam-role-arn",
    required=True,
    help="The AWS role to use"
)
@click.option(
    u"-n",
    u"--base-job-name",
    required=True,
    help="Prefix for the SageMaker processing job."
    "If not specified, default job name is generated, based on the docker image name and current timestamp."
)
@click.option(
    u"-t",
    u"--target",
    required=True,
    help="The name of the target to be built"
)
@click.option(
    u"-i", u"--s3-input-location",
    required=False,
    default=None,
    help="s3 input data location",
    type=click.Path()
)
@click.option(
    u"-o", u"--s3-output-location",
    required=False,
    default=None,
    help="s3 location to save output",
    type=click.Path()
)
@click.option(
    u"-is", u"--input-sharded",
    is_flag=True,
    default=False,
    help="Flag to indicate if input data should be sharded (distributed on machines)",
)
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def make(
        obj,
        ec2_type,
        instance_count,
        iam_role_arn,
        base_job_name,
        target,
        s3_input_location,
        input_sharded,
        s3_output_location,
        app_name
):
    """
    Command to build make targets defined in a Makefile in easy_sm_base/processing on Sagemaker
    """

    print(f"Building {target} on SageMaker...\n")
    config = _config(app_name)
    sage_maker_client = sagemaker.SageMakerClient(config.aws_profile, config.aws_region, iam_role_arn)

    image_name = config.image_name+':'+obj['docker_tag']

    _ = sage_maker_client.make(
        image_name=image_name,
        processing_instance_type=ec2_type,
        instance_count=instance_count,
        target=target,
        s3_input_location=s3_input_location,
        input_sharded=input_sharded,
        s3_output_location=s3_output_location,
        base_job_name=base_job_name
    )

    print(f"{target} built on Sagemaker successfully!")


cloud.add_command(upload_data)
cloud.add_command(train)
cloud.add_command(deploy_serverless)
cloud.add_command(batch_transform)
cloud.add_command(delete_endpoint)
cloud.add_command(process)
cloud.add_command(make)