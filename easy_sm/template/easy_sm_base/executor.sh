#!/usr/bin/env bash
# Order of operations is important since when invoked from sagemaker for serving no argument is passed
# For training, train argument is passed by default from sagemaker
# Process is handled differently by overriding entrypoint during job definitions
if [ "$1" = "train" ]; then
    python ./easy_sm_base/training/train
elif [ "$1" = "process" ]; then
    cd ./easy_sm_base/processing
    python "$2"
elif [ "$1" = "run" ]; then
    # Run any command (e.g. make, dbt) from the processing directory.
    # exec "$@" passes each argument through as-is, with no shell re-parsing.
    shift
    cd ./easy_sm_base/processing
    exec "$@"
else # This case is reserved for serving for compatibility with Sagemaker
    python ./easy_sm_base/prediction/serve
fi
