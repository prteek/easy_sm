from pathlib import Path

import pytest

from easy_sm.config import Config, ConfigError, check_app_name, load

NO_ENV: dict[str, str] = {}


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "easy_sm.toml"
    path.write_text(text)
    return path


# --- lookup order: flag → env var → easy_sm.toml → default ---


def test_flag_beats_everything():
    cfg = Config(Path("easy_sm.toml"), {"aws.region": "eu-west-2"})
    assert cfg.resolve("aws.region", flag="us-east-1", env={"AWS_REGION": "eu-west-1"}) == (
        "us-east-1",
        "flag",
    )


def test_env_var_beats_file():
    cfg = Config(Path("easy_sm.toml"), {"aws.region": "eu-west-2"})
    assert cfg.resolve("aws.region", env={"AWS_REGION": "eu-west-1"}) == ("eu-west-1", "AWS_REGION")


def test_empty_env_var_is_ignored():
    cfg = Config(Path("easy_sm.toml"), {"aws.region": "eu-west-2"})
    assert cfg.resolve("aws.region", env={"AWS_REGION": ""}) == ("eu-west-2", "easy_sm.toml")


def test_file_beats_default():
    cfg = Config(Path("easy_sm.toml"), {"run.instance": "ml.m5.xlarge"})
    assert cfg.resolve("run.instance", env=NO_ENV) == ("ml.m5.xlarge", "easy_sm.toml")


def test_default_when_nothing_is_set():
    cfg = Config(Path("easy_sm.toml"), {})
    assert cfg.resolve("run.instance", env=NO_ENV) == ("ml.t3.medium", "default")


def test_source_names_the_file_actually_used():
    cfg = Config(Path("prod.toml"), {"run.instance": "ml.m5.xlarge"})
    assert cfg.resolve("run.instance", env=NO_ENV)[1] == "prod.toml"


# --- loading and checking the file ---


def test_load_flattens_sections(tmp_path):
    cfg = load(write(tmp_path, 'app_name = "churn"\n[run]\nenv = { LOG_LEVEL = "info" }\n'))
    assert cfg.values == {"app_name": "churn", "run.env": {"LOG_LEVEL": "info"}}


def test_missing_file_says_run_init(tmp_path):
    with pytest.raises(ConfigError, match="easy_sm init"):
        load(tmp_path / "easy_sm.toml")


def test_typo_in_setting_name_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown setting 'train.instnace'"):
        load(write(tmp_path, 'app_name = "churn"\n[train]\ninstnace = "ml.m5.large"\n'))


def test_wrong_type_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="'train.count' should be int, got str"):
        load(write(tmp_path, 'app_name = "churn"\n[train]\ncount = "2"\n'))


def test_invalid_toml_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load(write(tmp_path, 'app_name = "churn\n'))


@pytest.mark.parametrize("name", ["churn", "churn-model", "model2"])
def test_valid_app_names(name):
    check_app_name(name)


@pytest.mark.parametrize("name", ["Churn", "churn_model", "-churn", "churn--model", "a" * 33, ""])
def test_invalid_app_names(name):
    with pytest.raises(ConfigError):
        check_app_name(name)
