"""SageMaker requests and names, built without calling AWS."""

from datetime import UTC, datetime

import pytest

from easy_sm.cloud import (
    CloudError,
    ecr_uri,
    host_of,
    job_name,
    parse_timeout,
    processing_job_request,
    session_kwargs,
    split_s3,
    training_job_request,
)

WHEN = datetime(2026, 11, 4, 15, 30, 0, tzinfo=UTC)


def test_credentials_are_left_to_boto3_like_the_aws_cli():
    assert session_kwargs("eu-west-1") == {"region_name": "eu-west-1"}  # no profile, no keys
    assert session_kwargs(None) == {}  # no region either: boto3 uses AWS_DEFAULT_REGION / ~/.aws


def test_ecr_uri():
    assert ecr_uri("123", "eu-west-1", "churn:abc") == "123.dkr.ecr.eu-west-1.amazonaws.com/churn:abc"


def test_job_name_is_sortable_and_ends_with_the_image_hash():
    assert job_name("churn:a1b2c3d4e5f6", WHEN) == "churn-20261104-153000-a1b2c3d4e5f6"


def test_longest_app_name_still_fits_sagemakers_63_characters():
    assert len(job_name("a" * 32 + ":a1b2c3d4e5f6", WHEN)) <= 63


@pytest.mark.parametrize(("text", "seconds"), [("3600", 3600), ("90m", 5400), ("2h", 7200), ("1d", 86400)])
def test_parse_timeout(text, seconds):
    assert parse_timeout(text) == seconds


@pytest.mark.parametrize("bad", ["0", "2 hours", "1.5h", ""])
def test_bad_timeouts_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_timeout(bad)


def test_split_s3():
    assert split_s3("s3://bucket/churn/data") == ("bucket", "churn/data")
    with pytest.raises(CloudError):
        split_s3("/local/data")


def test_training_request_puts_data_in_the_training_channel():
    request = training_job_request(
        "job", "img", "role", "ml.m5.large", 2, "s3://b/in", "s3://b/out", 3600
    )
    channel = request["InputDataConfig"][0]
    assert channel["ChannelName"] == "training"  # → /opt/ml/input/data/training
    assert channel["DataSource"]["S3DataSource"]["S3Uri"] == "s3://b/in"
    assert request["ResourceConfig"]["InstanceCount"] == 2
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 3600}


def test_processing_request_passes_command_and_env_unchanged():
    request = processing_job_request(
        "job", "img", "role", "ml.t3.medium", ["dbt", "build", "-t", "prod"],
        {"MSG": "hi --privileged"}, None, None, 600,
    )  # fmt: skip
    assert request["AppSpecification"]["ContainerArguments"] == ["run", "dbt", "build", "-t", "prod"]
    assert request["Environment"] == {"MSG": "hi --privileged"}
    assert "ProcessingInputs" not in request and "ProcessingOutputConfig" not in request


def test_processing_input_and_output_use_the_same_folders_as_run_local():
    request = processing_job_request(
        "job", "img", "role", "ml.t3.medium", ["x.py"], {}, "s3://b/in", "s3://b/out", 600
    )
    assert request["ProcessingInputs"][0]["S3Input"]["LocalPath"] == "/opt/ml/processing/input"
    output = request["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
    assert output["LocalPath"] == "/opt/ml/processing/output"


def test_env_values_over_sagemakers_limit_are_rejected():
    with pytest.raises(ValueError, match="256"):
        processing_job_request("j", "i", "r", "t", ["x.py"], {"A": "x" * 257}, None, None, 60)


def test_log_lines_are_labelled_with_the_host():
    assert host_of("churn-20261104-153000-a1b2/algo-1-1730000000") == "algo-1"
