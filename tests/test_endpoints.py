"""Endpoint names, requests and outcomes, built without calling AWS."""

from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from easy_sm.__main__ import app
from easy_sm.cloud import (
    CloudError,
    check_endpoint_name,
    deploy_outcome,
    describe_lines,
    endpoint_config_request,
    image_hash_of,
    model_request,
    pick_latest,
    resource_name,
)

WHEN = datetime(2026, 11, 4, 15, 30, 0, tzinfo=UTC)
HASH = "a1b2c3d4e5f6"


def test_latest_job_ignores_other_apps_with_a_similar_name():
    names = ["churn-v2-20261104-160000-ffffffffffff", "churn-20261104-150000-a1b2c3d4e5f6", "churn-old-manual"]
    assert pick_latest(names, "churn") == "churn-20261104-150000-a1b2c3d4e5f6"
    assert pick_latest(names[:1], "churn") is None


def test_each_deploy_gets_a_new_model_and_config_name():
    assert resource_name("churn", HASH, WHEN) == "churn-20261104-153000-a1b2c3d4e5f6"


def test_long_endpoint_names_are_shortened_to_fit_63_characters():
    name = resource_name("e" * 63, HASH, WHEN)
    assert len(name) == 63 and name.endswith(HASH)


@pytest.mark.parametrize("bad", ["-churn", "churn_model", "a" * 64])
def test_invalid_endpoint_names_are_rejected(bad):
    with pytest.raises(CloudError):
        check_endpoint_name(bad)


def test_model_points_at_the_jobs_image_and_model_tarball():
    request = model_request("m", "acct.dkr.ecr/churn:abc", "s3://b/model.tar.gz", "role")
    assert request["PrimaryContainer"] == {"Image": "acct.dkr.ecr/churn:abc", "ModelDataUrl": "s3://b/model.tar.gz"}


def test_instance_config_has_one_instance():
    variant = endpoint_config_request("c", "m", "ml.m5.large", False, 2048, 5)["ProductionVariants"][0]
    assert variant["InstanceType"] == "ml.m5.large" and variant["InitialInstanceCount"] == 1
    assert "ServerlessConfig" not in variant


def test_serverless_config_has_memory_and_concurrency_but_no_instance():
    variant = endpoint_config_request("c", "m", "ml.m5.large", True, 2048, 5)["ProductionVariants"][0]
    assert variant["ServerlessConfig"] == {"MemorySizeInMB": 2048, "MaxConcurrency": 5}
    assert "InstanceType" not in variant


def test_serverless_memory_must_be_a_size_sagemaker_offers():
    with pytest.raises(ValueError, match="1024"):
        endpoint_config_request("c", "m", "x", True, 1500, 5)


def test_in_service_on_the_new_config_is_success():
    assert deploy_outcome({"EndpointStatus": "InService", "EndpointConfigName": "new"}, "new") is None


def test_in_service_on_the_old_config_means_the_update_rolled_back():
    problem = deploy_outcome({"EndpointStatus": "InService", "EndpointConfigName": "old"}, "new")
    assert problem and "rolled back" in problem and "old" in problem


def test_failed_reports_the_reason():
    endpoint = {"EndpointStatus": "Failed", "EndpointConfigName": "new", "FailureReason": "ping failed"}
    assert deploy_outcome(endpoint, "new") == "Failed: ping failed"


def test_image_hash_of_ecr_uri():
    assert image_hash_of(f"123.dkr.ecr.eu-west-1.amazonaws.com/churn:{HASH}") == HASH


def test_describe_shows_serverless_size():
    details = {
        "endpoint": {"EndpointName": "e", "EndpointStatus": "InService", "EndpointConfigName": "c",
                     "LastModifiedTime": WHEN},
        "variant": {"ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 5}},
        "model": {"PrimaryContainer": {"Image": "img", "ModelDataUrl": "s3://m"}},
    }  # fmt: skip
    assert ("size", "serverless, 2048 MB, max concurrency 5") in describe_lines(details)


def test_deploy_needs_exactly_one_source():
    result = CliRunner().invoke(app, ["endpoints", "deploy", "--latest", "--job", "x"])
    assert result.exit_code == 2
    assert "exactly one" in result.stderr
