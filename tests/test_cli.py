from typer.testing import CliRunner

from easy_sm.__main__ import app

runner = CliRunner()


def test_init_then_env_shows_where_each_value_came_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "churn"]).exit_code == 0

    result = runner.invoke(app, ["env"], env={"AWS_REGION": "eu-west-1", "SAGEMAKER_ROLE": ""})

    lines = {line.split()[0]: line.split() for line in result.stdout.splitlines()}
    assert lines["app_name"][1:] == ["churn", "easy_sm.toml"]
    assert lines["aws.region"][1:] == ["eu-west-1", "AWS_REGION"]
    assert lines["run.instance"][1:] == ["ml.t3.medium", "easy_sm.toml"]
    assert lines["run.timeout"][1:] == ["24h", "default"]


def test_env_without_a_project_fails_with_a_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["env"])
    assert result.exit_code == 1
    assert "easy_sm init" in result.stderr


def test_config_option_picks_another_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prod.toml").write_text('app_name = "churn-prod"\n')
    result = runner.invoke(app, ["--config", "prod.toml", "env"])
    assert "churn-prod" in result.stdout
