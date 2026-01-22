# easy_sm
Easing SageMaker Ops

**Credits**: This Project borrows heavily from [Sagify](https://github.com/Kenza-AI/sagify). It's a great project - check it out especially if you want to work with LLMs on SageMaker.

---
Offers following commands to help work with SageMaker

```text
Commands:
  build  Command to build SageMaker app
  cloud  Commands for AWS operations: upload data, train and deploy
  init   Command to initialize SageMaker template
  local  Commands for local operations: train and deploy
  push   Command to push Docker image to AWS ECR
```

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
> **Note**: It is assumed that AWS CLI is set up and an AWS profile is defined for the app to use. This profile would be required when initializing easy_sm. [See AWS Setup](#aws-setup)

There are 5 broad steps to initialize build and test any project
1. Initialize easy_sm in the repository where code lives. Follow the prompts after running the `init` command
```shell
easy_sm init
```

2. Copy the relevant code in either `easy_sm_base/processing` or `easy_sm_base/training` or `easy_sm_prediction` folder

3. Build and Push Docker image with all the code and dependency (this is where `easy_sm` shines)
```shell
easy_sm build -a app-name
easy_sm push -a app-name
```
The Dockerfile that is used here is located at `app-name/easy_sm_base/Dockerfile`.
So any additional dependencies can be introduced in this file.

4. Test locally
```shell
easy_sm local process -f file.py -a app-name
```
Similarly there are commands for training a model or running a pipeline defined in a Makefile

5. Deploy/Run on SageMaker
```shell
easy_sm cloud process -f file.py -a app-name -r $SAGEMAKER_EXCUTION_ROLE -e ml.t3.medium
```

## Features

### Model training
**easy_sm** enables seamless transition from local environment to training models on SageMaker. Additionally, such trained models could be deployed to a serverless endpoint or a regular (provisioned) endpoint. A serverless endpoint can be very useful from a cost and scale perspective. A regular endpoint provides consistent performance with dedicated instances.

#### Getting started local training
##### Dependencies
First of all a *requirements.txt* that captures all dependencies for training code is required. This needs to be specified when using `easy_sm init` as it is subsequently used for building Docker container.
Additionally a *Dockerfile* in *app-name/easy_sm_base/Dockerfile* can be modified for flexibility in how the container is built.

##### Code
The code for training needs to be copied in **app-name/easy_sm_base/training/training.py** under the function *train* with any import statements at the top of the file
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
    mpg = pd.read_csv(os.path.join(input_data_path, 'auto-mpg.csv'))
    y, X = dmatrices('mpg ~ weight + horsepower', mpg, return_type="dataframe")
    ols = sm.OLS(y.values.ravel(), X.values).fit()
    print(ols.summary())

    # TODO: save the model(s) under 'model_save_path'
    joblib.dump(ols, os.path.join(model_save_path, 'model.mdl'))
```

##### Data
With the code and dependencies out of the way, a small sample of test data needs to be placed at **app-name/easy_sm_base/local_test/test_dir/input/data/training**

For this example the dataset used is at https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv

##### Prepare container
Last step before training is preparing the container to include all dependencies, code and data
```shell
easy_sm build -a app-name
```

##### Train
With all this out of the way training can be started easily

```shell
easy_sm local train -a app-name
```

This runs the training code inside the container so rest assured if everything worked here, it should work on SageMaker

#### Getting started cloud training
#### AWS Setup
There are primarily 2 things required from AWS side:
1. AWS Profile with credentials that can enable permissions to work with ECR, SageMaker and S3.
This is specified in *~/.aws/config* file like following, along with accompanying set of credentials in *~/.aws/credentials* file

```text
[profile dev]
aws_account_id = 10987654321
region = eu-west-1
output = json
```

2. SageMaker execution role to run training and processing jobs. This is generally of the form *arn:aws:iam::10987654321:role/AVMSagemakerExecutionRole* and it will be referred to as the variable *$SAGEMAKER_EXECUTION_ROLE* in the documentation.

Additionally, specify a trust relationship for the user (relevant for profile) to assume the SageMaker execution role.

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
If the container was built properly during local training it can be pushed to ECR easily
```shell
easy_sm push -a app-name
```

##### Data in S3
The dataset to train on needs to be present in S3. There is a command for copying local files to S3 easily
```shell
easy_sm cloud upload-data -i training_data.csv -s s3://bucket/folder/input -r $SAGEMAKER_EXECUTION_ROLE -a app-name
```

##### Train
Once the data and ECR image are in place invoking training is easy
```shell
easy_sm cloud train -n training-job -r $SAGEMAKER_EXECUTION_ROLE -e ml.m5.large -i s3://bucket/folder/input -o s3://bucket/folder/train/artifacts -a app-name
```

**Note**: Using *folder* as a parent leads to nicely organized training data for the project. The folder can be anything - brownie points if it is the name of the app.

##### Outputs
The training job writes text output in the console that can be useful for further steps in the pipeline
```text
Training on SageMaker succeeded
Model S3 location: s3://bucket/folder/train/artifacts/training-job-2024-08-07-10-41-23-345/output/model.tar.gz
```

This points to the location where model is saved and this text string can be used to extract model location and deploy the model.

It is often useful to also save this output in a text file.
```shell
easy_sm cloud train -n training-job -r $SAGEMAKER_EXECUTION_ROLE -e ml.m5.large -i s3://bucket/folder/input -o s3://bucket/folder/train/artifacts -a app-name | tee train_output.txt
```

### Model deployment

#### Getting started with local deployment

##### Code
To run inference using trained model
1. Model must be loaded in the container
2. Any input data must be handled and pre processed
3. Predictions made and output data processed if necessary

The code to accomplish all this needs to be defined in **app-name/easy_sm_base/prediction/serve**. By default *text/csv* inputs are supported and results returned as *text/csv* but other formats can be introduced in the *serve* file. If the default settings are usable then the only changes to the code need to be in *model_fn* and *input_fn* along with any dependencies at the top. A sample code looks like following

```python
# Your imports here
import joblib
import statsmodels.api as sm

def model_fn(model_dir):
    """Required model loading for Sagemaker framework"""
    # TODO Load a specific model
    model = joblib.load(os.path.join(model_dir, 'model.mdl'))
    return model


def predict_fn(input_data, model):
    """Predict on the input data"""
    # TODO Add any preprocessing or prediction related logic here
    input_data = sm.add_constant(input_data, has_constant='add')
    predictions = model.predict(input_data)
    return predictions
```

##### Deploy
Having setup the code, it is required to rebuild the container with updated serving code and run a local training job
```shell
easy_sm build -a app-name
easy_sm local train -a app-name
```

This will create a model and place it in an appropriate directory where serving code can locate it.
Local serving is *easy*
```shell
easy_sm local deploy -a app-name
```

And it can be tested by passing the payload
```shell
curl -X POST \
http://localhost:8080/invocations \
-H 'Cache-Control: no-cache' \
-H 'Content-Type: text/csv' \
-d '4732.0,193.0
3302.0,88.0'


curl -X POST \
http://localhost:8080/invocations \
-H 'Cache-Control: no-cache' \
-H 'Content-Type: text/csv' \
-T payload.csv
```


#### Getting started with cloud deployment
After the container is updated with serving code it needs to be pushed to ECR and a cloud training step needs to be run to generate a model object

```shell
easy_sm push -a app-name
easy_sm cloud train -n training-job -r $SAGEMAKER_EXECUTION_ROLE -e ml.m5.large -i s3://bucket/folder/input -o s3://bucket/folder/train/artifacts -a app-name >| train_output.txt

```


#### Cloud deployment options
easy_sm supports two types of endpoints for deploying models to SageMaker:

**1. Regular (Provisioned) Endpoint**
- Uses dedicated EC2 instances that are always running
- Pay per hour for instance usage
- Consistent latency and performance
- Best for production workloads with predictable traffic

**2. Serverless Endpoint**
- Automatically scales to zero when not in use
- Pay per invocation and compute time
- Cost-effective for sporadic or unpredictable traffic
- No capacity planning required

To begin with the model location is required for deployment. It can either be manually provided or if you saved the entire output of training job to the *train_output.txt* file, the model location can be extracted and passed to deployment commands.

**Deploy to a regular (provisioned) endpoint:**
```shell
easy_sm cloud deploy -e ml.m5.large -c 1 -n endpoint-name -r $SAGEMAKER_EXECUTION_ROLE -m s3://bucket/folder/train/artifacts/training-job-2024-08-07-10-41-23-345/output/model.tar.gz -a app-name
```

**Deploy to a serverless endpoint:**
```shell
easy_sm cloud deploy-serverless -s 2048 -n endpoint-name -r $SAGEMAKER_EXECUTION_ROLE -m s3://bucket/folder/train/artifacts/training-job-2024-08-07-10-41-23-345/output/model.tar.gz -a app-name
```

Choose `deploy` for provisioned instances (pay per hour, always running) or `deploy-serverless` for serverless (pay per invocation, scales to zero). Extracting model location from training output file can be done by `$(grep -o -E "s3://[^ ]+" train_output.txt)`.
This is particularly useful when running these commands on a remote runner like GitHub Actions. Training and deployment steps can be successively run without manual intervention.