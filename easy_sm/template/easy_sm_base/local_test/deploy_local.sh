#!/bin/sh

test_path=$1
tag=$2
image=$3
port=$4

docker run -d -v ${test_path}:/opt/ml -p ${port}:8080 --rm "${image}:${tag}" serve
