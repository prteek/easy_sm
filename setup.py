from setuptools import setup, find_packages

setup(
    name="easy_sm",
    description='Making working with Sagemaker Easy',
    author='Prateek',
    author_email='prteek@icloud.com',
    version="0.1.3",
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
        'sagemaker>=2.226.0, <2.228.0',
    ],
    entry_points={
        'console_scripts': [
            'easy_sm=easy_sm.__main__:cli',
        ],
    }
)