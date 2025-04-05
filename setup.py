from setuptools import setup, find_packages

setup(
    name="easy_sm",
    description='Easy Sagemaker Ops',
    long_description="""This package makes it easier to work with Sagemaker by enabling rapid prototyping with local training, processing and deployment.
And correspondingly training, processing and deployment on cloud.
This is very much an experimental package and API is likely to evolve and may break.
Recommended to validate before updating.
    """,
    author='Prateek',
    author_email='prteek@icloud.com',
    version="0.1.12",
    python_requires='>=3.11',
    packages=find_packages(where='.'),
    package_data={
        'easy_sm': [
            'template/easy_sm_base/*.sh',
            'template/easy_sm_base/Dockerfile',
            'template/easy_sm_base/__init__.py',
            'template/easy_sm_base/training/train',
            'template/easy_sm_base/training/*.py',
            'template/easy_sm_base/processing/*.py',
            'template/easy_sm_base/prediction/*.py',
            'template/easy_sm_base/prediction/serve',
            'template/easy_sm_base/local_test/*.sh',
            'template/easy_sm_base/local_test/test_dir/output/.gitkeep',
            'template/easy_sm_base/local_test/test_dir/model/.gitkeep',
            'template/easy_sm_base/local_test/test_dir/input/data/training/.gitkeep'
        ]
    },
    install_requires=[
        'click>=8.1.7, <8.1.99',
        'docker>=7.1.0, <7.2.0',
        'sagemaker>=2.243.0',
    ],
    entry_points={
        'console_scripts': [
            'easy_sm=easy_sm.__main__:cli',
        ],
    }
)