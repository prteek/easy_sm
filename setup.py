from setuptools import setup, find_packages

setup(
    name="easy_sm",
    version="0.1.0",
    python_requires='>=3.11',
    packages=find_packages(where='.'),
    package_data={
        'easy_sm': [
            'template/easy_sm_base/config.json',
            'template/easy_sm_base/*.sh',
            'template/easy_sm_base/Dockerfile',
            'template/easy_sm_base/__init__.py',
            'template/easy_sm_base/training/__init__.py',
            'template/easy_sm_base/training/train',
            'template/easy_sm_base/training/*.py',
            'template/easy_sm_base/prediction/*.py',
            # 'template/easy_sm_base/prediction/serve',
            # 'template/easy_sm_base/prediction/nginx.conf',
            # 'template/easy_sm_base/local_test/*.sh',
            # 'template/easy_sm_base/local_test/test_dir/output/.gitkeep',
            # 'template/easy_sm_base/local_test/test_dir/model/.gitkeep',
            # 'template/easy_sm_base/local_test/test_dir/input/config/*.json',
            # 'template/easy_sm_base/local_test/test_dir/input/data/training/'
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