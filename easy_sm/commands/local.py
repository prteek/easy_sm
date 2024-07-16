import os
import sys
import click
import subprocess

from easy_sm.config.config import ConfigManager


@click.group()
def local():
    """
    Commands for local operations: train and deploy
    """
    pass


@click.command()
@click.pass_obj
def train(obj):
    """
    Command to train ML model(s) locally
    """
    print("Started local training...\n")
    config = ConfigManager(os.path.join('.easy_sm.json')).get_config()
    dir = config.easy_sm_module_dir
    docker_tag = obj['docker_tag']
    image_name = config.image_name
    easy_sm_module_path = os.path.join(dir, 'easy_sm_base')
    local_train_script_path = os.path.join(easy_sm_module_path, 'local_test', 'train_local.sh')
    test_path = os.path.join(easy_sm_module_path, 'local_test', 'test_dir')

    if not os.path.isdir(test_path):
        raise ValueError("This is not a easy_sm directory: {}".format(dir))

    output = subprocess.check_output(
        [
            "{}".format(local_train_script_path),
            "{}".format(os.path.abspath(test_path)),
            docker_tag,
            image_name
        ]
    )
    print(output)

    print("Local training completed successfully!")


@click.command()
@click.pass_obj
def deploy(obj):
    """
    Command to deploy ML model(s) locally
    """

    config = ConfigManager(os.path.join('.easy_sm.json')).get_config()
    dir = config.easy_sm_module_dir
    docker_tag = obj['docker_tag']
    image_name = config.image_name

    easy_sm_module_path = os.path.join(dir, 'easy_sm_base')
    local_deploy_script_path = os.path.join(easy_sm_module_path, 'local_test', 'deploy_local.sh')
    test_path = os.path.join(easy_sm_module_path, 'local_test', 'test_dir')

    if not os.path.isdir(test_path):
        raise ValueError("This is not a easy_sm directory: {}".format(dir))

    print("Started local deployment at localhost:8080 ...\n")
    output = subprocess.check_output(
        [
            "{}".format(local_deploy_script_path),
            "{}".format(os.path.abspath(test_path)),
            docker_tag,
            image_name
        ]
    )
    print(output)


local.add_command(train)
local.add_command(deploy)
