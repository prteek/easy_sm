#!/usr/bin/env bash

if [ $1 = "train" ]; then
    python ./easy_sm_base/training/train
else
    python ./easy_sm_base/prediction/serve
fi