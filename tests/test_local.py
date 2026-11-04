"""The docker commands easy_sm builds for local runs. No Docker needed."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from easy_sm.__main__ import app
from easy_sm.local import (
    check_script,
    local_dirs,
    parse_env,
    request_body,
    run_command,
    serve_command,
    train_command,
)

DATA, MODEL, OUTPUT = Path("/p/data"), Path("/p/.easy_sm/model"), Path("/p/.easy_sm/output")


def mounts(command: list[str]) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]


# --- train ---


def test_train_mounts_folders_where_sagemaker_puts_them():
    command = train_command("churn:abc", DATA, MODEL, OUTPUT)
    assert mounts(command) == [
        "/p/data:/opt/ml/input/data/training:ro",  # read-only: training can't change your data
        "/p/.easy_sm/model:/opt/ml/model",
        "/p/.easy_sm/output:/opt/ml/output",
    ]
    assert command[-2:] == ["churn:abc", "train"]
    assert "linux/amd64" in command


# --- run ---


def test_run_passes_the_command_after_the_image():
    command = run_command("churn:abc", DATA, OUTPUT, {}, ["clean.py", "--date", "2026-11-01"])
    assert command[-5:] == ["churn:abc", "run", "clean.py", "--date", "2026-11-01"]


def test_env_value_stays_one_argument_even_if_it_looks_like_a_flag():
    command = run_command("churn:abc", DATA, OUTPUT, {"MSG": "hi --privileged"}, ["x.py"])
    i = command.index("MSG=hi --privileged")
    assert command[i - 1] == "-e"
    assert "--privileged" not in command  # never a separate docker flag


def test_parse_env_keeps_equals_signs_and_empty_values():
    assert parse_env(["A=1", "KEY=sk-12=34", "EMPTY="]) == {"A": "1", "KEY": "sk-12=34", "EMPTY": ""}


@pytest.mark.parametrize("bad", ["NOVALUE", "=x"])
def test_parse_env_rejects_pairs_without_a_key_or_equals(bad):
    with pytest.raises(ValueError):
        parse_env([bad])


def test_missing_script_is_caught_before_starting_a_container(tmp_path):
    with pytest.raises(ValueError, match="clen.py"):
        check_script(tmp_path, ["clen.py"])
    check_script(tmp_path, ["make", "features"])  # not a .py file: nothing to check


# --- serve and invoke ---


def test_serve_is_reachable_from_this_machine_only():
    command = serve_command("churn:abc", MODEL, 8765)
    assert command[command.index("-p") + 1] == "127.0.0.1:8765:8080"
    assert "/p/.easy_sm/model:/opt/ml/model:ro" in mounts(command)


def test_request_body_from_text_or_file(tmp_path):
    (tmp_path / "x.json").write_text('{"inputs": [[1]]}')
    assert request_body('{"a": 1}') == b'{"a": 1}'
    assert request_body(f"@{tmp_path / 'x.json'}") == b'{"inputs": [[1]]}'


# --- local folder and CLI ---


def test_local_folder_ignores_itself_in_git(tmp_path):
    model_dir, output_dir = local_dirs(tmp_path)
    assert model_dir.is_dir() and output_dir.is_dir()
    assert (tmp_path / ".easy_sm" / ".gitignore").read_text() == "*\n"


def test_cloud_only_options_are_rejected_with_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    CliRunner().invoke(app, ["init", "--name", "churn"])
    result = CliRunner().invoke(app, ["train", "--local", "--instance", "ml.m5.large"])
    assert result.exit_code == 2
    assert "only apply to SageMaker jobs" in result.stderr


# --- AWS credentials for local containers ---


def test_credentials_are_passed_by_name_never_by_value():
    command = run_command("churn:abc", DATA, OUTPUT, {}, ["x.py"], ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
    i = command.index("AWS_SECRET_ACCESS_KEY")
    assert command[i - 1] == "-e"  # `-e NAME`: docker copies the value from its own environment
    assert not any(arg.startswith("AWS_SECRET_ACCESS_KEY=") for arg in command)


def test_no_aws_folder_is_mounted():
    for command in (
        train_command("t", DATA, MODEL, OUTPUT, ["AWS_ACCESS_KEY_ID"]),
        run_command("t", DATA, OUTPUT, {}, ["x.py"], ["AWS_ACCESS_KEY_ID"]),
        serve_command("t", MODEL, 8080, ["AWS_ACCESS_KEY_ID"]),
    ):
        assert not any(".aws" in mount for mount in mounts(command))


def test_dry_run_shows_credential_names_but_not_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    CliRunner().invoke(app, ["init", "--name", "churn"])
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "src" / "x.py").write_text("print('hi')\n")
    fake = {"AWS_ACCESS_KEY_ID": "AKIAFAKEFAKEFAKE", "AWS_SECRET_ACCESS_KEY": "fake-secret-value",
            "AWS_SESSION_TOKEN": None, "AWS_PROFILE": None, "AWS_REGION": "eu-west-1"}  # fmt: skip
    result = CliRunner().invoke(app, ["run", "--local", "--dry-run", "x.py"], env=fake)
    assert "AWS_SECRET_ACCESS_KEY" in result.stdout
    assert "fake-secret-value" not in result.stdout and "AKIAFAKEFAKEFAKE" not in result.stdout
