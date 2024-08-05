#!/usr/bin/env bash

if [ $1 = "train" ]; then
    python ./easy_sm_base/training/train
elif [ $1 = "deploy" ]; then
    python ./easy_sm_base/prediction/serve
else
    python ./easy_sm_base/processing/$1
fi