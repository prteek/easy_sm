#!/bin/sh

test_path="$1"
tag="$2"
image="$3"
aws_profile="$4"
aws_region="$5"
shift 5

# Remaining arguments are KEY=VALUE pairs, then "--", then the command to run.
# Rebuild "$@" as: -e KEY=VALUE ... image:tag run command...
# Each argument stays a single argument (no word splitting), so nothing can inject docker flags.
remaining=$#
in_command=false
while [ "$remaining" -gt 0 ]; do
    if [ "$in_command" = true ]; then
        set -- "$@" "$1"
    elif [ "$1" = "--" ]; then
        set -- "$@" "${image}:${tag}" run
        in_command=true
    else
        set -- "$@" -e "$1"
    fi
    shift
    remaining=$((remaining - 1))
done

docker run -v "${HOME}/.aws:/root/.aws" -v "${test_path}:/opt/ml" \
    -e "AWS_PROFILE=${aws_profile}" -e "AWS_DEFAULT_REGION=${aws_region}" \
    --rm "$@"
