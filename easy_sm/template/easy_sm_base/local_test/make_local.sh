#!/bin/sh

test_path=$1
tag=$2
image=$3
target=$4
aws_profile=$5
aws_region=$6

docker run -v ~/.aws:/root/.aws -v ${test_path}:/opt/ml -e AWS_PROFILE=${aws_profile} -e AWS_DEFAULT_REGION=${aws_region} --rm "${image}:${tag}" make ${target}