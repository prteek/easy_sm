import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from easy_sm.config import ConfigError, load
from easy_sm.init import app_name_from_folder, create_project


def import_file(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_project_has_a_valid_config(tmp_path):
    create_project(tmp_path / "churn")
    assert load(tmp_path / "churn" / "easy_sm.toml").values["app_name"] == "churn"


def test_app_name_comes_from_folder_name():
    assert app_name_from_folder(Path("My Churn_Model")) == "my-churn-model"


def test_name_option_overrides_folder_name(tmp_path):
    create_project(tmp_path, name="churn")
    assert load(tmp_path / "easy_sm.toml").values["app_name"] == "churn"


def test_never_overwrites_existing_files(tmp_path):
    (tmp_path / "easy_sm.toml").write_text("mine")
    with pytest.raises(ConfigError, match="Not overwriting"):
        create_project(tmp_path, name="churn")
    assert (tmp_path / "easy_sm.toml").read_text() == "mine"
    assert not (tmp_path / "src").exists()  # nothing else was written either


def test_sample_train_and_predict_work_together(tmp_path):
    create_project(tmp_path, name="churn")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    import_file(tmp_path / "src" / "train.py").train(str(tmp_path / "data"), str(model_dir))
    handler = import_file(tmp_path / "src" / "predict.py").Handler(str(model_dir))

    assert handler({"inputs": [[1.0], [2.0]]}) == [5.05, 5.05]
