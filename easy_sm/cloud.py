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
from pathlib import Path
from typing import Any, TextIO

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


def follow(
    s: boto3.Session, kind: str, name: str, show_logs: bool, poll: float = 5.0, log_out: TextIO = sys.stderr
) -> dict[str, Any]:
    """Wait for a job to finish, printing status changes to stderr and (optionally) its logs to log_out."""
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
                print(line, file=log_out, flush=True)
        if job["status"] in TERMINAL:
            if show_logs:  # lines written just before the end
                for line in new_log_lines(s, kind, name, tokens):
                    print(line, file=log_out, flush=True)
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


# --- endpoints: names and requests (pure) ---

NAME_PATTERN = re.compile(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}")  # SageMaker endpoint, model and config names
SERVERLESS_MEMORY = (1024, 2048, 3072, 4096, 5120, 6144)
ENDPOINT_DONE = {"InService", "Failed", "OutOfService", "UpdateRollbackFailed"}


def check_endpoint_name(name: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise CloudError(f"Invalid endpoint name '{name}': letters, digits and hyphens, at most 63 characters")


def is_own(name: str, app: str) -> bool:
    """True for names easy_sm created for this app: <app>-YYYYMMDD-HHMMSS-<hash>. Other apps whose
    names merely start the same way (churn vs churn-v2) don't match."""
    return re.fullmatch(rf"{re.escape(app)}-\d{{8}}-\d{{6}}-[0-9a-f]{{12}}", name) is not None


def pick_latest(job_names: list[str], app: str) -> str | None:
    """First (newest) of the names that belong to this app."""
    return next((name for name in job_names if is_own(name, app)), None)


def resource_name(endpoint: str, image_hash: str, now: datetime) -> str:
    """Unique per deploy, used for both the model and the endpoint config, so a redeploy never
    collides with the config the endpoint is still using."""
    suffix = f"-{now:%Y%m%d-%H%M%S}-{image_hash}"
    return endpoint[: JOB_NAME_MAX - len(suffix)] + suffix


def model_request(name: str, image: str, model_s3: str, role: str) -> dict[str, Any]:
    return {
        "ModelName": name,
        "ExecutionRoleArn": role,
        # SageMaker unpacks model.tar.gz into /opt/ml/model, where Handler(model_dir) reads it
        "PrimaryContainer": {"Image": image, "ModelDataUrl": model_s3},
    }


def endpoint_config_request(
    name: str, model: str, instance: str, serverless: bool, memory: int, concurrency: int
) -> dict[str, Any]:
    variant: dict[str, Any] = {"VariantName": "AllTraffic", "ModelName": model}
    if serverless:
        if memory not in SERVERLESS_MEMORY:
            raise ValueError(f"endpoint.memory must be one of {', '.join(map(str, SERVERLESS_MEMORY))}")
        if not 1 <= concurrency <= 200:
            raise ValueError("endpoint.concurrency must be between 1 and 200")
        variant["ServerlessConfig"] = {"MemorySizeInMB": memory, "MaxConcurrency": concurrency}
    else:
        variant.update({"InstanceType": instance, "InitialInstanceCount": 1})
    return {"EndpointConfigName": name, "ProductionVariants": [variant]}


def deploy_outcome(endpoint: dict[str, Any], new_config: str) -> str | None:
    """None if the endpoint is serving the new config, else what went wrong. A failed update
    rolls back and ends InService on the OLD config, so InService alone proves nothing."""
    status = endpoint["EndpointStatus"]
    if status == "InService" and endpoint["EndpointConfigName"] == new_config:
        return None
    if status == "InService":
        return (f"update rolled back: still serving {endpoint['EndpointConfigName']}. "
                f"{endpoint.get('FailureReason', '')}").strip()
    return f"{status}: {endpoint.get('FailureReason', 'no reason given')}"


def image_hash_of(image_uri: str) -> str:
    """'…/churn:a1b2c3d4e5f6' → 'a1b2c3d4e5f6'."""
    return image_uri.rsplit(":", 1)[1]


# --- endpoints: AWS calls ---


def latest_job(s: boto3.Session, app: str) -> str:
    jobs = s.client("sagemaker").list_training_jobs(
        NameContains=f"{app}-", StatusEquals="Completed", SortBy="CreationTime", SortOrder="Descending",
        MaxResults=50,
    )["TrainingJobSummaries"]
    name = pick_latest([j["TrainingJobName"] for j in jobs], app)
    if not name:
        raise CloudError(f"No completed training job for '{app}'. Run 'easy_sm train' first.")
    return name


def job_source(s: boto3.Session, job: str) -> tuple[str, str]:
    """(model.tar.gz S3 path, image URI) of a completed training job: the model and the exact
    code it was trained with."""
    desc = s.client("sagemaker").describe_training_job(TrainingJobName=job)
    if desc["TrainingJobStatus"] != "Completed":
        raise CloudError(f"{job} is {desc['TrainingJobStatus']}, not Completed")
    return desc["ModelArtifacts"]["S3ModelArtifacts"], desc["AlgorithmSpecification"]["TrainingImage"]


def find_endpoint(s: boto3.Session, name: str) -> dict[str, Any] | None:
    try:
        return s.client("sagemaker").describe_endpoint(EndpointName=name)
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            return None
        raise


def deploy(s: boto3.Session, endpoint: str, model: dict[str, Any], config: dict[str, Any]) -> None:
    """Create the model and config, then create the endpoint or point it at the new config.
    Old models and configs are never deleted: SageMaker may need them to roll back."""
    sm = s.client("sagemaker")
    existing = find_endpoint(s, endpoint)
    if existing:
        try:
            sm.describe_endpoint_config(EndpointConfigName=existing["EndpointConfigName"])
        except ClientError as e:
            raise CloudError(
                f"{endpoint} uses config {existing['EndpointConfigName']}, which no longer exists, so "
                f"SageMaker can't update it. Delete the endpoint and deploy again."
            ) from e
    sm.create_model(**model)
    sm.create_endpoint_config(**config)
    if existing:
        sm.update_endpoint(EndpointName=endpoint, EndpointConfigName=config["EndpointConfigName"])
    else:
        sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=config["EndpointConfigName"])


def wait_for_endpoint(s: boto3.Session, endpoint: str, poll: float = 15.0) -> dict[str, Any]:
    """Wait until the endpoint stops changing, printing status changes to stderr."""
    sm = s.client("sagemaker")
    last = ""
    while True:
        desc = sm.describe_endpoint(EndpointName=endpoint)
        if desc["EndpointStatus"] != last:
            last = desc["EndpointStatus"]
            print(f"[status] {last}", file=sys.stderr)
        if last in ENDPOINT_DONE:
            return desc
        time.sleep(poll)


def invoke_endpoint(s: boto3.Session, endpoint: str, body: bytes, content_type: str) -> bytes:
    response = s.client("sagemaker-runtime").invoke_endpoint(
        EndpointName=endpoint, ContentType=content_type, Body=body
    )
    return response["Body"].read()


def list_endpoints(s: boto3.Session) -> list[dict[str, Any]]:
    return s.client("sagemaker").list_endpoints(SortBy="CreationTime", SortOrder="Descending", MaxResults=100)[
        "Endpoints"
    ]


def endpoint_details(s: boto3.Session, endpoint: str) -> dict[str, Any]:
    """The endpoint, its current config and that config's model, for `endpoints describe`."""
    sm = s.client("sagemaker")
    desc = sm.describe_endpoint(EndpointName=endpoint)
    config = sm.describe_endpoint_config(EndpointConfigName=desc["EndpointConfigName"])
    variant = config["ProductionVariants"][0]
    model = sm.describe_model(ModelName=variant["ModelName"])
    return {"endpoint": desc, "variant": variant, "model": model}


def describe_lines(details: dict[str, Any]) -> list[tuple[str, str]]:
    """Pure: the label/value pairs `endpoints describe` prints."""
    desc, variant, model = details["endpoint"], details["variant"], details["model"]
    serverless = variant.get("ServerlessConfig")
    size = (f"serverless, {serverless['MemorySizeInMB']} MB, max concurrency {serverless['MaxConcurrency']}"
            if serverless else f"{variant['InstanceType']} × {variant['InitialInstanceCount']}")
    lines = [
        ("name", desc["EndpointName"]),
        ("status", desc["EndpointStatus"]),
        ("config", desc["EndpointConfigName"]),
        ("size", size),
        ("model data", model["PrimaryContainer"].get("ModelDataUrl", "-")),
        ("image", model["PrimaryContainer"]["Image"]),
        ("updated", f"{desc['LastModifiedTime']:%Y-%m-%d %H:%M:%S}"),
    ]
    if desc.get("FailureReason"):
        lines.append(("failure", desc["FailureReason"]))
    return lines


def delete_endpoint(s: boto3.Session, endpoint: str, delete_config: bool) -> list[str]:
    """Delete the endpoint (and, if asked, its current config). Return what was deleted."""
    sm = s.client("sagemaker")
    config = sm.describe_endpoint(EndpointName=endpoint)["EndpointConfigName"]
    sm.delete_endpoint(EndpointName=endpoint)
    deleted = [f"endpoint {endpoint}"]
    if delete_config:
        sm.delete_endpoint_config(EndpointConfigName=config)
        deleted.append(f"endpoint config {config}")
    return deleted


# --- jobs: listing and inspecting (pure parts first) ---


def fmt_duration(seconds: float | None) -> str:
    """372 → '6m 12s', 3780 → '1h 03m'."""
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def job_rows(
    training: list[dict[str, Any]], processing: list[dict[str, Any]], app: str, limit: int, now_: datetime
) -> list[dict[str, Any]]:
    """Merge SageMaker's two job lists into one, newest first, keeping only this app's jobs."""
    rows = []
    for kind, summaries, prefix in (("train", training, "Training"), ("run", processing, "Processing")):
        for job in summaries:
            name = job[f"{prefix}JobName"]
            if not is_own(name, app):
                continue
            end = job.get(f"{prefix}EndTime") or now_
            rows.append({"name": name, "kind": kind, "status": job[f"{prefix}JobStatus"],
                         "created": job["CreationTime"],
                         "duration": fmt_duration((end - job["CreationTime"]).total_seconds())})  # fmt: skip
    rows.sort(key=lambda row: row["created"], reverse=True)
    return rows[:limit]


def inspect_lines(kind: str, job: dict[str, Any], now_: datetime) -> list[tuple[str, str]]:
    """The label/value pairs `jobs inspect` prints, from a describe_*_job response."""
    if kind == "training":
        resources = job["ResourceConfig"]
        end = job.get("TrainingEndTime") or now_
        lines = [
            ("name", job["TrainingJobName"]), ("kind", "train"),
            ("status", job["TrainingJobStatus"] + (f" ({job['SecondaryStatus']})"
                       if job.get("SecondaryStatus") not in (None, job["TrainingJobStatus"]) else "")),
            ("instance", f"{resources['InstanceType']} × {resources['InstanceCount']}"),
            ("image", job["AlgorithmSpecification"]["TrainingImage"]),
            ("input", job["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]),
            ("model", job.get("ModelArtifacts", {}).get("S3ModelArtifacts", "-")),
        ]  # fmt: skip
    else:
        cluster = job["ProcessingResources"]["ClusterConfig"]
        end = job.get("ProcessingEndTime") or now_
        inputs = job.get("ProcessingInputs") or [{}]
        outputs = job.get("ProcessingOutputConfig", {}).get("Outputs") or [{}]
        lines = [
            ("name", job["ProcessingJobName"]), ("kind", "run"), ("status", job["ProcessingJobStatus"]),
            ("instance", f"{cluster['InstanceType']} × {cluster['InstanceCount']}"),
            ("image", job["AppSpecification"]["ImageUri"]),
            ("command", " ".join(job["AppSpecification"].get("ContainerArguments", [])[1:])),
            ("env", ", ".join(sorted(job.get("Environment", {}))) or "-"),
            ("input", inputs[0].get("S3Input", {}).get("S3Uri", "-")),
            ("output", outputs[0].get("S3Output", {}).get("S3Uri", "-")),
        ]  # fmt: skip
    lines += [
        ("created", f"{job['CreationTime']:%Y-%m-%d %H:%M:%S}"),
        ("duration", fmt_duration((end - job["CreationTime"]).total_seconds())),
    ]
    if job.get("FailureReason"):
        lines.append(("failure", job["FailureReason"]))
    return lines


def upload_plan(local_dir: Path, s3_uri: str) -> list[tuple[Path, str, str]]:
    """(file, bucket, key) for every file under local_dir, skipping hidden files like the image does."""
    bucket, prefix = split_s3(s3_uri)
    prefix = prefix.rstrip("/")
    return [
        (path, bucket, f"{prefix}/{path.relative_to(local_dir).as_posix()}" if prefix
         else path.relative_to(local_dir).as_posix())
        for path in sorted(local_dir.rglob("*"))
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(local_dir).parts)
    ]  # fmt: skip


# --- jobs: AWS calls ---


def list_jobs(s: boto3.Session, app: str, limit: int) -> list[dict[str, Any]]:
    sm = s.client("sagemaker")
    common = {"NameContains": f"{app}-", "SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 100}
    training = sm.list_training_jobs(**common)["TrainingJobSummaries"]
    processing = sm.list_processing_jobs(**common)["ProcessingJobSummaries"]
    return job_rows(training, processing, app, limit, now())


def find_job(s: boto3.Session, name: str) -> tuple[str, dict[str, Any]]:
    """('training' | 'processing', description). Both kinds share easy_sm's naming, so ask for
    a training job first and fall back to a processing job."""
    sm = s.client("sagemaker")
    try:
        return "training", sm.describe_training_job(TrainingJobName=name)
    except ClientError:
        pass
    try:
        return "processing", sm.describe_processing_job(ProcessingJobName=name)
    except ClientError as e:
        raise CloudError(f"No training or processing job named {name}") from e


def stop_job(s: boto3.Session, kind: str, name: str) -> None:
    sm = s.client("sagemaker")
    if kind == "training":
        sm.stop_training_job(TrainingJobName=name)
    else:
        sm.stop_processing_job(ProcessingJobName=name)


def upload(s: boto3.Session, plan: list[tuple[Path, str, str]]) -> None:
    s3 = s.client("s3")
    for path, bucket, key in plan:
        print(f"upload {path} → s3://{bucket}/{key}", file=sys.stderr)
        s3.upload_file(str(path), bucket, key)
