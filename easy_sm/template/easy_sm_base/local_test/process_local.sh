#!/bin/sh

test_path="$1"
tag="$2"
image="$3"
file="$4"
aws_profile="$5"
aws_region="$6"
shift 6

# Remaining arguments are KEY=VALUE pairs. Rebuild "$@" as "-e" "KEY=VALUE" pairs so each
# value stays a single argument (no word splitting, so values can't inject docker flags).
remaining=$#
while [ "$remaining" -gt 0 ]; do
    set -- "$@" -e "$1"
    shift
    remaining=$((remaining - 1))
done

docker run -v "${HOME}/.aws:/root/.aws" -v "${test_path}:/opt/ml" \
    -e "AWS_PROFILE=${aws_profile}" -e "AWS_DEFAULT_REGION=${aws_region}" "$@" \
    --rm "${image}:${tag}" process "${file}"
