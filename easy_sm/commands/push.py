import sys
import click
import subprocess
import os
from easy_sm.config.config import ConfigManager

def _config(app_name):
    config_file_path = os.path.join(f'{app_name}.json')
    if not os.path.isfile(config_file_path):
        raise ValueError("This is not a easy_sm directory: {}".format(os.getcwd()))
    else:
        return ConfigManager(config_file_path).get_config()


def _push(dir, docker_tag, aws_region, iam_role_arn, aws_profile, external_id, image_name):
    """
    Push Docker image to AWS ECR

    :param dir: [str], source root directory
    :param docker_tag: [str], the Docker tag for the image
    :param aws_region: [str], the AWS region to push the image to
    :param iam_role_arn: [str], the AWS role used to push the image to ECR
    :param aws_profile: [str], the AWS profile used to push the image to ECR
    :param external_id: [str], Optional external id used when using an IAM role
    :param image_name: [str], The name of the Docker image
    """

    easy_sm_module_path = os.path.relpath(os.path.join(dir, 'easy_sm_base/'))
    push_script_path = os.path.join(easy_sm_module_path, 'push.sh')

    if not os.path.isfile(push_script_path):
        raise ValueError("This is not a easy_sm directory: {}".format(dir))

    output = subprocess.check_output([
                                     "{}".format(push_script_path),
                                     docker_tag,
                                     aws_region,
                                     iam_role_arn,
                                     aws_profile,
                                     external_id,
                                     image_name])
    print(output)

@click.command()
@click.option(u"-r", u"--aws-region", required=False, help="The AWS region to push the image to")
@click.option(u"-i", u"--iam-role-arn", required=False, help="The AWS role to use for the push command")
@click.option(u"-p", u"--aws-profile", required=False, help="The AWS profile to use for the push command")
@click.option(u"-e", u"--external-id", required=False, help="Optional external id used when using an IAM role")
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def push(obj, aws_region, iam_role_arn, aws_profile, external_id, app_name):
    """
    Command to push Docker image to AWS ECR
    """

    if iam_role_arn is not None and aws_profile is not None:
        print('Only one of iam-role-arn and aws-profile can be used.')
        sys.exit(2)

    if iam_role_arn is not None:
        aws_profile = ''

    config = _config(app_name)
    image_name = config.image_name
    aws_region = config.aws_region if aws_region is None else aws_region
    aws_profile = config.aws_profile if (aws_profile is None and iam_role_arn is None) else aws_profile
    external_id = "" if external_id is None else external_id
    iam_role_arn = "" if iam_role_arn is None else iam_role_arn

    print("Started pushing Docker image to AWS ECR. It will take some time. Please, be patient...\n")

    _push(
        dir=config.easy_sm_module_dir,
        docker_tag=obj['docker_tag'],
        aws_region=aws_region,
        iam_role_arn=iam_role_arn,
        aws_profile=aws_profile,
        external_id=external_id,
        image_name=image_name)

    print("Docker image pushed to ECR successfully!")
