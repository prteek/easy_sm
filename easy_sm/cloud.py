"""Push the image to ECR and run SageMaker training and processing jobs with boto3.

Request builders (*_request, job_name, ecr_uri, parse_timeout, log formatting) are pure
and tested; the functions that call AWS stay thin.
"""

import base64
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError

from easy_sm.config import Config

# SageMaker job names: letters, digits and hyphens, at most 63 characters.
JOB_NAME_MAX = 63
# SageMaker limits for processing job environment values and arguments.
ENV_VALUE_MAX = 256
LOG_GROUPS = {"training": "/aws/sagemaker/TrainingJobs", "processing": "/aws/sagemaker/ProcessingJobs"}
TERMINAL = {"Completed", "Failed", "Stopped"}


class CloudError(Exception):
    """Something easy_sm can explain went wrong talking to AWS."""


# --- session and identity ---


def session_kwargs(region: str | None) -> dict[str, str]:
    """Credentials are never chosen by easy_sm: boto3 finds them exactly like the AWS CLI
    (access keys in the environment, AWS_PROFILE, OIDC, ~/.aws, instance role).
    `easy_sm env` shows which account and source that gave."""
    return {"region_name": region} if region else {}


def session(cfg: Config) -> boto3.Session:
    s = boto3.Session(**session_kwargs(cfg.get("aws.region")))
    if not s.region_name:
        raise CloudError("No AWS region: set [aws] region in easy_sm.toml or AWS_REGION")
    return s


def account_id(s: boto3.Session) -> str:
    return s.client("sts").get_caller_identity()["Account"]


def default_bucket(account: str, region: str) -> str:
    """The bucket the SageMaker SDK and console use by default."""
    return f"sagemaker-{region}-{account}"


# --- names and requests (pure) ---


def ecr_uri(account: str, region: str, tag: str) -> str:
    """'churn:abc' → '<account>.dkr.ecr.<region>.amazonaws.com/churn:abc'."""
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{tag}"


def job_name(tag: str, now: datetime) -> str:
    """'churn:a1b2c3d4e5f6' → 'churn-20261104-153000-a1b2c3d4e5f6': sortable, and the
    image hash at the end says which code ran."""
    app, image_hash = tag.split(":")
    name = f"{app}-{now:%Y%m%d-%H%M%S}-{image_hash}"
    assert len(name) <= JOB_NAME_MAX  # app_name is at most 32 characters
    return name


def parse_timeout(text: str) -> int:
    """'90m' → 5400 seconds. Units: s, m, h, d; a plain number is seconds."""
    match = re.fullmatch(r"(\d+)([smhd]?)", text.strip())
    if not match or int(match.group(1)) == 0:
        raise ValueError(f"Invalid timeout '{text}': use e.g. 3600, 90m, 2h or 1d")
    return int(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]


def training_job_request(
    name: str, image: str, role: str, instance: str, count: int, input_s3: str, output_s3: str, timeout: int
) -> dict[str, Any]:
    return {
        "TrainingJobName": name,
        "RoleArn": role,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
        "InputDataConfig": [
            {
                "ChannelName": "training",  # appears as /opt/ml/input/data/training
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {"S3OutputPath": output_s3},
        "ResourceConfig": {"InstanceType": instance, "InstanceCount": count, "VolumeSizeInGB": 30},
        "StoppingCondition": {"MaxRuntimeInSeconds": timeout},
    }


def processing_job_request(
    name: str,
    image: str,
    role: str,
    instance: str,
    command: list[str],
    env: dict[str, str],
    input_s3: str | None,
    output_s3: str | None,
    timeout: int,
) -> dict[str, Any]:
    for key, value in env.items():
        if len(value) > ENV_VALUE_MAX:
            raise ValueError(f"--env {key}: SageMaker allows at most {ENV_VALUE_MAX} characters")
    request: dict[str, Any] = {
        "ProcessingJobName": name,
        "RoleArn": role,
        "AppSpecification": {"ImageUri": image, "ContainerArguments": ["run", *command]},
        "ProcessingResources": {
            "ClusterConfig": {"InstanceType": instance, "InstanceCount": 1, "VolumeSizeInGB": 30}
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": timeout},
    }
    if env:
        request["Environment"] = env
    if input_s3:
        request["ProcessingInputs"] = [
            {
                "InputName": "input",
                "S3Input": {
                    "S3Uri": input_s3,
                    "LocalPath": "/opt/ml/processing/input",  # same place as data/ in run --local
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ]
    if output_s3:
        request["ProcessingOutputConfig"] = {
            "Outputs": [
                {
                    "OutputName": "output",
                    "S3Output": {
                        "S3Uri": output_s3,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        }
    return request


# --- pushing the image ---


def push(s: boto3.Session, account: str, tag: str) -> str:
    """Push a local image to ECR unless that tag is already there. Return its ECR URI."""
    ecr = s.client("ecr")
    repository, image_tag = tag.split(":")
    uri = ecr_uri(account, s.region_name, tag)
    try:
        ecr.describe_repositories(repositoryNames=[repository])
    except ecr.exceptions.RepositoryNotFoundException:
        # Tags are content hashes, so one can never be reused for different code.
        ecr.create_repository(repositoryName=repository, imageTagMutability="IMMUTABLE")
    try:
        ecr.describe_images(repositoryName=repository, imageIds=[{"imageTag": image_tag}])
        return uri  # already pushed
    except ecr.exceptions.ImageNotFoundException:
        pass

    auth = ecr.get_authorization_token()["authorizationData"][0]
    password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)[1]
    registry = uri.split("/")[0]
    _docker(["docker", "login", "--username", "AWS", "--password-stdin", registry], stdin=password)
    _docker(["docker", "tag", tag, uri])
    _docker(["docker", "push", uri])
    return uri


def _docker(command: list[str], stdin: str | None = None) -> None:
    result = subprocess.run(command, input=stdin, text=True, stdout=sys.stderr, check=False)
    if result.returncode != 0:
        raise CloudError(f"'{' '.join(command[:2])}' failed with exit code {result.returncode}")


def split_s3(uri: str) -> tuple[str, str]:
    """'s3://bucket/a/b' → ('bucket', 'a/b')."""
    if not uri.startswith("s3://"):
        raise CloudError(f"'{uri}' is not an S3 path (s3://bucket/folder)")
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    return bucket, prefix


def check_s3_has_data(s: boto3.Session, uri: str) -> None:
    """Fail now, not minutes later on a paid instance, if the input folder is empty."""
    bucket, prefix = split_s3(uri)
    found = s.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("KeyCount", 0)
    if not found:
        raise CloudError(f"No data at {uri}. Upload it first, e.g. aws s3 cp --recursive data/ {uri}/")


def ensure_bucket(s: boto3.Session, bucket: str) -> None:
    s3 = s.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        location = {} if s.region_name == "us-east-1" else {
            "CreateBucketConfiguration": {"LocationConstraint": s.region_name}
        }
        s3.create_bucket(Bucket=bucket, **location)


# --- following a job ---


def host_of(stream: str) -> str:
    """'churn-…-a1b2/algo-1-1730000000' → 'algo-1'."""
    return stream.split("/", 1)[-1].rsplit("-", 1)[0]


def describe(s: boto3.Session, kind: str, name: str) -> dict[str, Any]:
    sm = s.client("sagemaker")
    if kind == "training":
        job = sm.describe_training_job(TrainingJobName=name)
        return {"status": job["TrainingJobStatus"], "detail": job.get("SecondaryStatus", ""),
                "reason": job.get("FailureReason", "")}  # fmt: skip
    job = sm.describe_processing_job(ProcessingJobName=name)
    return {"status": job["ProcessingJobStatus"], "detail": "", "reason": job.get("FailureReason", "")}


def new_log_lines(s: boto3.Session, kind: str, name: str, tokens: dict[str, str]) -> Iterator[str]:
    """Lines logged since the last call, as '[host] message'. tokens remembers the position
    in each log stream between calls."""
    logs = s.client("logs")
    try:
        streams = logs.describe_log_streams(logGroupName=LOG_GROUPS[kind], logStreamNamePrefix=f"{name}/")
    except logs.exceptions.ResourceNotFoundException:
        return  # no logs yet: the job hasn't started its container
    for stream in streams["logStreams"]:
        stream_name = stream["logStreamName"]
        while True:
            kwargs: dict[str, Any] = {"logGroupName": LOG_GROUPS[kind], "logStreamName": stream_name,
                                      "startFromHead": True}  # fmt: skip
            if stream_name in tokens:
                kwargs["nextToken"] = tokens[stream_name]
            page = logs.get_log_events(**kwargs)
            for event in page["events"]:
                yield f"[{host_of(stream_name)}] {event['message']}"
            if page["nextForwardToken"] == tokens.get(stream_name):
                break  # no newer events
            tokens[stream_name] = page["nextForwardToken"]


def follow(s: boto3.Session, kind: str, name: str, show_logs: bool, poll: float = 5.0) -> dict[str, Any]:
    """Wait for a job to finish, printing status changes and (optionally) its logs to stderr."""
    tokens: dict[str, str] = {}
    last = ""
    while True:
        job = describe(s, kind, name)
        if show_logs:
            label = f"{job['status']} {job['detail']}".strip()
            if label != last:
                print(f"[status] {label}", file=sys.stderr)
                last = label
            for line in new_log_lines(s, kind, name, tokens):
                print(line, file=sys.stderr)
        if job["status"] in TERMINAL:
            if show_logs:  # lines written just before the end
                for line in new_log_lines(s, kind, name, tokens):
                    print(line, file=sys.stderr)
            return job
        time.sleep(poll)


def start(s: boto3.Session, kind: str, request: dict[str, Any]) -> None:
    sm = s.client("sagemaker")
    if kind == "training":
        sm.create_training_job(**request)
    else:
        sm.create_processing_job(**request)


def now() -> datetime:
    return datetime.now(UTC)
