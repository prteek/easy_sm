import os
import site
import json
from setuptools import setup, find_packages


def fix_sagemaker_jumpstart():
    sagemaker_jumpstart_path = os.path.join(
        site.getsitepackages()[0], "sagemaker", "jumpstart"
    )
    region_config_path = os.path.join(sagemaker_jumpstart_path, "region_config.json")

    if not os.path.exists(region_config_path):
        region_config = {
            "us-east-1": {"content_bucket": "sagemaker-us-east-1"},
            "us-east-2": {"content_bucket": "sagemaker-us-east-2"},
            "us-west-1": {"content_bucket": "sagemaker-us-west-1"},
            "us-west-2": {"content_bucket": "sagemaker-us-west-2"},
            "eu-west-1": {"content_bucket": "sagemaker-eu-west-1"},
            "eu-west-2": {"content_bucket": "sagemaker-eu-west-2"},
            "eu-central-1": {"content_bucket": "sagemaker-eu-central-1"},
            "ap-southeast-1": {"content_bucket": "sagemaker-ap-southeast-1"},
            "ap-southeast-2": {"content_bucket": "sagemaker-ap-southeast-2"},
            "ap-northeast-1": {"content_bucket": "sagemaker-ap-northeast-1"},
            "ap-northeast-2": {"content_bucket": "sagemaker-ap-northeast-2"},
            "ap-south-1": {"content_bucket": "sagemaker-ap-south-1"},
            "ca-central-1": {"content_bucket": "sagemaker-ca-central-1"},
        }

        os.makedirs(sagemaker_jumpstart_path, exist_ok=True)
        with open(region_config_path, "w") as f:
            json.dump(region_config, f)


fix_sagemaker_jumpstart()

setup(
    name="easy_sm",
    description="Easy Sagemaker Ops",
    long_description="""This package makes it easier to work with Sagemaker by enabling rapid prototyping with local training, processing and deployment.
And correspondingly training, processing and deployment on cloud.
This is very much an experimental package and API is likely to evolve and may break.
Recommended to validate before updating.
    """,
    author="Prateek",
    author_email="prteek@icloud.com",
    version="0.1.12",
    python_requires=">=3.11",
    packages=find_packages(where="."),
    package_data={
        "easy_sm": [
            "template/easy_sm_base/*.sh",
            "template/easy_sm_base/Dockerfile",
            "template/easy_sm_base/__init__.py",
            "template/easy_sm_base/training/train",
            "template/easy_sm_base/training/*.py",
            "template/easy_sm_base/processing/*.py",
            "template/easy_sm_base/prediction/*.py",
            "template/easy_sm_base/prediction/serve",
            "template/easy_sm_base/local_test/*.sh",
            "template/easy_sm_base/local_test/test_dir/output/.gitkeep",
            "template/easy_sm_base/local_test/test_dir/model/.gitkeep",
            "template/easy_sm_base/local_test/test_dir/input/data/training/.gitkeep",
        ]
    },
    install_requires=[
        "click>=8.1.7, <8.1.99",
        "docker>=7.1.0, <7.2.0",
        "sagemaker>=2.243.0, <3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "easy_sm=easy_sm.__main__:cli",
        ],
    },
)
