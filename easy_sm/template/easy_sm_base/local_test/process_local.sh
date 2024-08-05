#!/bin/sh

test_path=$1
tag=$2
image=$3
file=$4
aws_profile=$5
aws_region=$6

docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=${aws_profile} -e AWS_DEFAULT_REGION=${aws_region} --rm "${image}:${tag}" process ${file}