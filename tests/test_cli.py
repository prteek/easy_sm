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


# No AWS at all, as on a CI runner: env must still work and say so.
NO_AWS = {"AWS_PROFILE": None, "AWS_ACCESS_KEY_ID": None, "AWS_SECRET_ACCESS_KEY": None, "AWS_SESSION_TOKEN": None,
          "AWS_CONFIG_FILE": "/dev/null", "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
          "AWS_EC2_METADATA_DISABLED": "true", "AWS_REGION": "eu-west-1"}  # fmt: skip


def test_env_works_without_any_aws_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "churn"])
    result = runner.invoke(app, ["env"], env=NO_AWS)
    assert result.exit_code == 0
    assert "no AWS credentials found" in result.stdout


# --- easy_sm dockerfile ---


def test_dockerfile_prints_the_rules_then_what_easy_sm_generates(tmp_path, monkeypatch):
    from easy_sm.image import dockerfile

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "churn"])
    result = runner.invoke(app, ["dockerfile"])
    assert result.exit_code == 0
    assert result.stdout.startswith("# Made by `easy_sm dockerfile`")
    assert result.stdout.endswith(dockerfile("3.13"))  # requires-python >=3.13 in the scaffold


def test_saved_dockerfile_is_used_by_the_build(tmp_path, monkeypatch):
    from easy_sm.image import image_files

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "churn"])
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "Dockerfile").write_text(runner.invoke(app, ["dockerfile"]).stdout)  # easy_sm dockerfile > Dockerfile
    assert image_files(tmp_path)["Dockerfile"] == (tmp_path / "Dockerfile").read_bytes()
