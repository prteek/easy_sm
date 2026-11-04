"""Job listing, inspecting and data upload plans, built without calling AWS."""

from datetime import UTC, datetime, timedelta

import pytest

from easy_sm.cloud import CloudError, fmt_duration, inspect_lines, job_rows, upload_plan

T0 = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)


def training(name, minutes_ago, status="Completed", ran_for=6):
    created = T0 - timedelta(minutes=minutes_ago)
    end = created + timedelta(minutes=ran_for) if status != "InProgress" else None
    return {"TrainingJobName": name, "TrainingJobStatus": status, "CreationTime": created, "TrainingEndTime": end}


def processing(name, minutes_ago, status="Failed"):
    created = T0 - timedelta(minutes=minutes_ago)
    return {"ProcessingJobName": name, "ProcessingJobStatus": status, "CreationTime": created,
            "ProcessingEndTime": created + timedelta(seconds=90)}  # fmt: skip


@pytest.mark.parametrize(("seconds", "text"), [(None, "-"), (45, "0m 45s"), (372, "6m 12s"), (3780, "1h 03m")])
def test_fmt_duration(seconds, text):
    assert fmt_duration(seconds) == text


def test_jobs_of_both_kinds_are_merged_newest_first():
    rows = job_rows(
        [training("churn-20261104-110000-aaaaaaaaaaaa", 60)],
        [processing("churn-20261104-113000-bbbbbbbbbbbb", 30)],
        "churn", 10, T0,
    )  # fmt: skip
    assert [(r["kind"], r["status"]) for r in rows] == [("run", "Failed"), ("train", "Completed")]
    assert rows[0]["duration"] == "1m 30s"


def test_other_apps_jobs_are_left_out():
    rows = job_rows([training("churn-v2-20261104-110000-aaaaaaaaaaaa", 5)], [], "churn", 10, T0)
    assert rows == []


def test_running_job_duration_counts_up_to_now():
    rows = job_rows([training("churn-20261104-115000-aaaaaaaaaaaa", 10, "InProgress")], [], "churn", 10, T0)
    assert rows[0]["duration"] == "10m 00s"


def test_limit_keeps_the_newest():
    jobs = [training(f"churn-20261104-1{i}0000-aaaaaaaaaaaa", 100 - i) for i in range(5)]
    rows = job_rows(jobs, [], "churn", 2, T0)
    assert [r["name"] for r in rows] == ["churn-20261104-140000-aaaaaaaaaaaa", "churn-20261104-130000-aaaaaaaaaaaa"]


def test_inspect_processing_job_shows_command_and_env_names_not_values():
    job = {
        "ProcessingJobName": "j", "ProcessingJobStatus": "Failed", "CreationTime": T0,
        "ProcessingEndTime": T0 + timedelta(seconds=72),
        "ProcessingResources": {"ClusterConfig": {"InstanceType": "ml.t3.medium", "InstanceCount": 1}},
        "AppSpecification": {"ImageUri": "img", "ContainerArguments": ["run", "clean.py", "--date", "x"]},
        "Environment": {"TOKEN": "secret", "LOG_LEVEL": "debug"},
        "FailureReason": "AlgorithmError: , exit code: 3",
    }  # fmt: skip
    lines = dict(inspect_lines("processing", job, T0))
    assert lines["command"] == "clean.py --date x"
    assert lines["env"] == "LOG_LEVEL, TOKEN"  # names only: values could be sensitive
    assert lines["duration"] == "1m 12s"
    assert lines["failure"].endswith("exit code: 3")


def test_upload_plan_skips_hidden_files_and_keeps_folders(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "train.csv").write_text("x")
    (tmp_path / "sub" / "extra.csv").write_text("y")
    (tmp_path / ".DS_Store").write_text("z")
    plan = upload_plan(tmp_path, "s3://bucket/churn/data/")
    assert [(b, k) for _, b, k in plan] == [("bucket", "churn/data/sub/extra.csv"), ("bucket", "churn/data/train.csv")]


def test_upload_plan_needs_an_s3_path(tmp_path):
    with pytest.raises(CloudError):
        upload_plan(tmp_path, "/not/s3")


def test_inspect_training_status_shows_the_detail_only_when_it_adds_something():
    base = {
        "TrainingJobName": "j", "CreationTime": T0, "TrainingEndTime": T0,
        "ResourceConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 1},
        "AlgorithmSpecification": {"TrainingImage": "img"},
        "InputDataConfig": [{"DataSource": {"S3DataSource": {"S3Uri": "s3://b/in"}}}],
    }  # fmt: skip
    failed = dict(inspect_lines("training", {**base, "TrainingJobStatus": "Failed", "SecondaryStatus": "Failed"}, T0))
    running = dict(inspect_lines("training", {**base, "TrainingJobStatus": "InProgress", "SecondaryStatus": "Training"}, T0))
    assert failed["status"] == "Failed"
    assert running["status"] == "InProgress (Training)"
