import os
import site
import json
from setuptools import setup, find_packages

setup(
    name="easy_sm",
    description="Easy SageMaker Ops",
    long_description="""This package makes it easier to work with SageMaker by enabling rapid prototyping with local training, processing, and deployment.
Provides seamless integration between local Docker-based development and AWS SageMaker cloud operations.
Features include: local training/processing/deployment, cloud job management, endpoint management, and job monitoring.
This is an experimental package and API is likely to evolve and may break.
Recommended to validate before updating.
    """,
    author="Prateek",
    author_email="prteek@icloud.com",
    version="0.2.0",
    python_requires=">=3.13",
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
        "typer>=0.9.0",
        "docker>=7.1.0",
        "sagemaker>=2.243.0",
        "boto3>=1.26.0",
    ],
    entry_points={
        "console_scripts": [
            "easy_sm=easy_sm.__main__:cli",
        ],
    },
)
