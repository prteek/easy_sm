# easy_sm
Easing Sagemaker Ops

**Credits**: This Project borrows heavily from [Sagify](https://github.com/Kenza-AI/sagify). It's a great project do check it out specially if you want to work with LLMs on Sagemaker.

---
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
And similarly for any sub commands `easy_sm cloud --help`

## Usage
`Note: It is assumed that AWS cli is setup and an AWS profile defined for the app to use. This profile would be required when initialising easy_sm` [See](https://github.com/prteek/easy_sm/tree/main?tab=readme-ov-file#aws-setup)

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

## Features

### Model training
**easy_sm** enables seamless transition from local environment to training models on Sagemaker. Additionally such trained models could be deployed to a serverless endpoint!. A serverless endpoint can be very useful to have from cost and scale perspective. This avoids having to deploy a lambda function etc. for inference.

#### Getting started local training
##### Dependencies
First of all a *requirements.txt* that captures all dependencies for training code is required. This needs to be specified when using `easy_sm init` as it is subsequently used for building Docker container.
Additionally a *Dockerfile* in *app_name/easy_sm_base/Dockerfile* can be modified for flexibility in how the container is built.

##### Code
The code for training needs to be copied in **app_name/easy_sm_base/training/training.py** under the function *train* with any import statements at the top of the file
e.g.
```python
import statsmodels.api as sm
from patsy import dmatrices
import pandas as pd
import joblib
import os

def train(input_data_path, model_save_path, hyperparams_path=None):
    """
    The function to execute the training.

    :param input_data_path: [str], input directory path where all the training file(s) reside in
    :param model_save_path: [str], directory path to save your model(s)
    """
    # TODO: Write your modeling logic
    mpg = pd.read_csv(os.path.join(input_data_path, 'mpg.csv'))
    y, X = dmatrices('mpg ~ weight + horsepower', mpg, return_type="dataframe")
    ols = sm.OLS(y.values.ravel(), X.values).fit()
    print(ols.summary())

    # TODO: save the model(s) under 'model_save_path'
    joblib.dump(ols, os.path.join(model_save_path, 'model.mdl'))
```

##### Data
With the code and dependencies out of the way, a small sample of test data needs to be places at **app_name/easy_sm_base/local_test/test_dir/input/data/training**

##### Prepare container
Last step before training is preparing the container to include all dependencies, code and data
```shell
easy_sm build -a app_name
```

##### Train
With all this out of the way training can be started *easily*

```shell
easy_sm train -a app_name
```

This runs the training code inside the container so rest assured if everything worked here, it should work on Sagemaker

#### Getting started cloud training
##### AWS Setup
There are primarily 2 things required from AWS side
1. AWS Profile with credentials that can enable permissions to work with ECR, Sagemaker and S3.
This is specified in *~/.aws/config* file like following, along with accompanying set of credentials in *~/.aws/credentials* file

```bash
[profile dev]
aws_account_id = 10987654321
region = eu-west-1
output = json
```

2. Sagemaker execution role to run training and processing jobs. This is generally of the form *arn:aws:iam::10987654321:role/AVMSagemakerExecutionRole* and it will be referred to as the variable *$SAGEMAKER_EXECUTION_ROLE* in the doumentation.

Additionally, specify a trust relationship for the user (relevant for profile) to assume the Sagemaker execution role.

This is done by adding the following json blob to Trust entities under Trust relationship tab of the role in IAM console. Any number of users can be added within the Statement field.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::109876543210:user/dev",
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

##### Push to ECR
If the container was built properly during local training it can be pushed to ECR *easily*
```shell
easy_sm push -a app_name
```

##### Data in S3
The dataset to train on needs to be present in s3. There is a command for copying local files to s3 *easily*
```shell
easy_sm cloud upload-data -i training_data.csv -s s3://bucket/folder/input -r $SAGEMAKER_EXECUTION_ROLE
```

##### Train
Once the data and ECR image are in place invoking training is *easy*
```shell
easy_sm cloud train -n training-job -r $SAGEMAKER_EXECUTION_ROLE -e ml.m5.large -i s3://bucket/folder/input -o s3://bucket/folder/train/artefacts -a app_name
```
**Note**: that using *folder* as a parent leads us to nicely organise training data for the project. The folder can be anything, brownie points if it is name of the app.

##### Outputs
The training job writes text output in the console that can be useful for further steps in the pipeline
```shell
Training on SageMaker succeeded
Model S3 location: s3://bucket/folder/train/artefacts/training-job-2024-08-07-10-41-23-345/output/model.tar.gz
```

This points to the location where model is saved and this text string can be used to extract model location and deploy the model.
