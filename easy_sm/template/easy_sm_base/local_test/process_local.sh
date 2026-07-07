#!/bin/sh

test_path="$1"
tag="$2"
image="$3"
file="$4"
aws_profile="$5"
aws_region="$6"
env_vars="$7"

# Build docker run command with base env vars
docker_env="-e AWS_PROFILE=${aws_profile} -e AWS_DEFAULT_REGION=${aws_region}"

# Add custom environment variables if provided
if [ -n "${env_vars}" ]; then
    # Convert comma-separated KEY=VALUE pairs to individual -e flags
    IFS=','
    for env_var in ${env_vars}; do
        docker_env="${docker_env} -e ${env_var}"
    done
    unset IFS
fi

docker run -v ~/.aws:/root/.aws -v "${test_path}:/opt/ml" ${docker_env} --rm "${image}:${tag}" process "${file}"
