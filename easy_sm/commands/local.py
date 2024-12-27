import os
import sys
import click
from easy_sm.commands.helpers import safe_run_subprocess
from easy_sm.config.config import ConfigManager

def _config(app_name):
    config_file_path = os.path.join(f'{app_name}.json')
    if not os.path.isfile(config_file_path):
        raise ValueError("This is not a easy_sm directory: {}".format(os.getcwd()))
    else:
        return ConfigManager(config_file_path).get_config()


@click.group()
def local():
    """
    Commands for local operations: train and deploy
    """
    pass


@click.command()
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def train(obj, app_name):
    """
    Command to train ML model(s) locally
    """
    print("Started local training...\n")
    config = _config(app_name)
    dir = config.easy_sm_module_dir
    docker_tag = obj['docker_tag']
    image_name = config.image_name
    easy_sm_module_path = os.path.join(dir, 'easy_sm_base')
    local_train_script_path = os.path.join(easy_sm_module_path, 'local_test', 'train_local.sh')
    test_path = os.path.join(easy_sm_module_path, 'local_test', 'test_dir')

    if not os.path.isdir(test_path):
        raise ValueError("This is not a easy_sm directory: {}".format(dir))

    command = [
                "{}".format(local_train_script_path),
                "{}".format(os.path.abspath(test_path)),
                docker_tag,
                image_name
            ]

    safe_run_subprocess(command, success_message="Local training completed successfully!")


@click.command()
@click.option(
    u"-f",
    u"--file",
    required=True,
    help="The name (not path) of python file to run as processing job"
)
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def process(obj, file, app_name):
    """
    Command to run python files locally as processing job
    """
    print("Started local processing job...\n")
    config = _config(app_name)
    dir = config.easy_sm_module_dir
    docker_tag = obj['docker_tag']
    image_name = config.image_name
    aws_profile = config.aws_profile
    aws_region = config.aws_region
    easy_sm_module_path = os.path.join(dir, 'easy_sm_base')
    local_process_script_path = os.path.join(easy_sm_module_path, 'local_test', 'process_local.sh')
    test_path = os.path.join(easy_sm_module_path, 'local_test', 'test_dir')
    job_file_path = os.path.join(easy_sm_module_path, 'processing', file)

    if not os.path.isdir(test_path):
        raise ValueError("This is not a easy_sm directory: {}".format(dir))

    if not os.path.isfile(job_file_path):
        raise ValueError("Processing file does not exist: {}".format(job_file_path))

    command = [
            "{}".format(local_process_script_path),
            "{}".format(os.path.abspath(test_path)),
            docker_tag,
            image_name,
            file,
            aws_profile,
            aws_region
        ]

    safe_run_subprocess(command, success_message="Local processing completed successfully!")


@click.command()
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def deploy(obj, app_name):
    """
    Command to deploy ML model(s) locally
    """

    config = _config(app_name)
    dir = config.easy_sm_module_dir
    docker_tag = obj['docker_tag']
    image_name = config.image_name

    easy_sm_module_path = os.path.join(dir, 'easy_sm_base')
    local_deploy_script_path = os.path.join(easy_sm_module_path, 'local_test', 'deploy_local.sh')
    test_path = os.path.join(easy_sm_module_path, 'local_test', 'test_dir')

    if not os.path.isdir(test_path):
        raise ValueError("This is not a easy_sm directory: {}".format(dir))

    print("Started local deployment at localhost:8080 ...\n")
    command = [
                "{}".format(local_deploy_script_path),
                "{}".format(os.path.abspath(test_path)),
                docker_tag,
                image_name
            ]

    safe_run_subprocess(command, success_message=None)


@click.command()
@click.option(
    u"-t",
    u"--target",
    required=True,
    help="The name of target that needs to be built"
)
@click.option(
    u"-a",
    u"--app-name",
    required=True,
    help="The app name whose json file will be referenced for setting up command"
)
@click.pass_obj
def make(obj, target, app_name):
    """
    Command to build make targets defined in a Makefile in easy_sm_base/processing
    """
    config = _config(app_name)
    dir = config.easy_sm_module_dir
    docker_tag = obj['docker_tag']
    image_name = config.image_name
    aws_profile = config.aws_profile
    aws_region = config.aws_region
    easy_sm_module_path = os.path.join(dir, 'easy_sm_base')
    local_make_script_path = os.path.join(easy_sm_module_path, 'local_test', 'make_local.sh')
    test_path = os.path.join(easy_sm_module_path, 'local_test', 'test_dir')
    makefile_path = os.path.join(easy_sm_module_path, 'processing', 'Makefile')

    if not os.path.isdir(test_path):
        raise ValueError("This is not a easy_sm directory: {}".format(dir))

    if not os.path.isfile(makefile_path):
        raise ValueError("Makefile does not exist: {}".format(makefile_path))

    command = [
                "{}".format(local_make_script_path),
                "{}".format(os.path.abspath(test_path)),
                docker_tag,
                image_name,
                target,
                aws_profile,
                aws_region
            ]

    safe_run_subprocess(command, success_message=f"{target} built successfully!")

local.add_command(train)
local.add_command(deploy)
local.add_command(process)
local.add_command(make)
