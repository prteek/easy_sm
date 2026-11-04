"""Agent support: --dry-run, --format json, JSON errors, `easy_sm skill`, retired settings."""

import json
from importlib.resources import files

import pytest
from typer.testing import CliRunner

from easy_sm.__main__ import app
from easy_sm.config import ConfigError, load

runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "churn"])
    (tmp_path / "uv.lock").write_text("version = 1\n")  # stands in for `uv lock`
    return tmp_path


def test_train_local_dry_run_prints_the_docker_command_and_creates_nothing(project):
    result = runner.invoke(app, ["train", "--local", "--dry-run"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["docker"][-2:] == [plan["image"]["tag"], "train"]
    assert plan["image"]["tag"].startswith("churn:")
    assert not (project / ".easy_sm").exists()  # dry run: no folders made


def test_run_local_dry_run_keeps_each_env_value_as_one_argument(project):
    (project / "src" / "clean.py").write_text("print('hi')\n")
    result = runner.invoke(app, ["run", "--local", "--dry-run", "--env", "MSG=hi --privileged", "clean.py", "-x"])
    docker = json.loads(result.stdout)["docker"]
    assert "MSG=hi --privileged" in docker and "--privileged" not in docker
    assert docker[-3:] == ["run", "clean.py", "-x"]


def test_dry_run_never_runs_uv_lock(project):
    (project / "uv.lock").unlink()
    result = runner.invoke(app, ["train", "--local", "--dry-run"])
    assert result.exit_code == 1
    assert "never runs uv lock" in result.stderr
    assert not (project / "uv.lock").exists()


def test_env_as_json_shows_value_and_source(project):
    result = runner.invoke(app, ["env", "--format", "json"], env={"AWS_REGION": "eu-west-1"})
    data = json.loads(result.stdout)
    assert data["app_name"] == {"value": "churn", "source": "easy_sm.toml"}
    assert data["aws.region"] == {"value": "eu-west-1", "source": "AWS_REGION"}
    assert data["image"]["value"].startswith("churn:")


def test_errors_are_json_in_json_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no project here
    result = runner.invoke(app, ["env", "--format", "json"])
    error = json.loads(result.stderr)
    assert error["code"] == 1 and "easy_sm init" in error["error"]


def test_skill_is_written_once_and_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = runner.invoke(app, ["skill"])
    path = tmp_path / ".claude" / "skills" / "easy_sm" / "SKILL.md"
    assert first.exit_code == 0 and path.read_text().startswith("---\nname: easy_sm")
    path.write_text("my edits")
    second = runner.invoke(app, ["skill"])
    assert second.exit_code == 1 and path.read_text() == "my edits"


def test_skill_mentions_every_command():
    """Adding a command without documenting it for agents fails here."""
    text = (files("easy_sm") / "skill.md").read_text()
    names = [c.name or c.callback.__name__ for c in app.registered_commands]
    for group in app.registered_groups:
        names += [f"{group.name} {c.name or c.callback.__name__}" for c in group.typer_instance.registered_commands]
    missing = [n for n in names if f"`{n}" not in text and f"{n} " not in text]
    assert missing == []


def test_retired_setting_says_what_to_do_instead(tmp_path):
    path = tmp_path / "easy_sm.toml"
    path.write_text('app_name = "churn"\n[aws]\nprofile = "dev"\n')
    with pytest.raises(ConfigError, match="removed in 2.0.*AWS_PROFILE"):
        load(path)
