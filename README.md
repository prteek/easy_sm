# easy_sm
Easing Sagemaker serverless deployments

Offers following commands to help work with Sagemaker
Commands:
  build  Command to build SageMaker app
  cloud  Commands for AWS operations: upload data, train and deploy
  init   Command to initialize SageMaker template
  local  Commands for local operations: train and deploy
  push   Command to push Docker image to AWS ECR


## Installation
```shell
pip install easy-sm
```

## Getting help
```shell
easy_sm --help

```

## Usage
`Note: It is assumed that AWS cli is setup and an AWS profile defined for the app to use. This profile would be required when initialising easy_sm`

There are 5 broad steps to initialise build and test any project
1. Initialise easy_sm in the repository where code lives. Follow the prompts after running the `init` command
```shell
easy_sm init
```

2. Copy the relevant code in either `easy_sm_base/processing` or `easy_sm_base/training` or `easy_sm_prediction` folder

3. Build and Push Docker image with all the code and dependency (this is where `easy_sm` shines)
```shell
easy_sm build -a app_name
easy_sm push -a app_name
```
The Dockerfile that is used here is located at `app_name/easy_sm_base/Dockerfile`.
So any additional dependencies can be introduced in this file.

4. Test locally
```shell
easy_sm local process -f file.py -a app_name
```
Similarly there are commands for training a model or running a pipeline defined in a Makefile

5. Deploy/Run on Sagemaker
```shell
easy_sm cloud process -f file.py -a app_name -r $SAGEMAKER_EXCUTION_ROLE -e ml.t3.medium
```
