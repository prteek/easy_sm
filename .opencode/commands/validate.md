---
description: Validates easy_sm CLI changes using the sample app
subtask: true
model: opencode/qwen3-coder
---
The sample app exists to help test changes made to the library. The app/ folder contains a sample application that uses this library for testing and it is not committed to git.

To test changes, do the following from project root directory (strictly do not change directories):
1. The training and serving files are already setup and a sample dataset (mpg.csv) is already placed in appropriate location. There's no need to set these up and validation commands can be run straight away
2. Install changes locally: pip install -e .
3. Refer @.github/README.md to understand the setup and what commands exist and can be tested and in what order.
    a. e.g. for local deploy run local train first and for cloud deploy run cloud train first and get model location
    b. A sample training dataset can be found *s3://easy-sm/train/data*
    c. A sample trained model is at *s3://easy-sm/train/job-artefacts/mpg-2024-07-16-21-23-11-417/output/model.tar.gz* which can be used for deployment if needed
    d. Training outputs can be placed at *s3://easy-sm/train/job-artefacts*
    e. Use SAGEMAKER_EXECUTION_ROLE environment variable for role
    f. Use endpoint or model names as *mpg-current-time*
4. Run *easy_sm* commands to test functionality: e.g. easy_sm local train -a app
5. Limit testing to only commands that are affected by recent change
6. If docker daemon is not running start docker and try running again

This validator ensures that any changes to the easy_sm CLI work correctly with a real application setup.
