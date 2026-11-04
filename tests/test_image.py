"""What goes into the image, and when its tag changes. No Docker needed."""

from pathlib import Path

import pytest

from easy_sm.image import (
    ImageError,
    build_command,
    dockerfile,
    image_files,
    image_hash,
    python_version,
    write_context,
)
from easy_sm.init import create_project


@pytest.fixture
def project(tmp_path: Path) -> Path:
    create_project(tmp_path, name="churn")
    (tmp_path / "uv.lock").write_text("version = 1\n")  # stands in for `uv lock` output
    return tmp_path


def tag_of(project: Path) -> str:
    return image_hash(image_files(project))


# --- when the tag changes ---


def test_same_files_give_the_same_tag(project):
    assert tag_of(project) == tag_of(project)
    assert len(tag_of(project)) == 12


def test_editing_training_code_changes_the_tag(project):
    before = tag_of(project)
    (project / "src" / "train.py").write_text("def train(input_dir, model_dir): pass\n")
    assert tag_of(project) != before


def test_adding_a_dependency_changes_the_tag(project):
    before = tag_of(project)
    (project / "uv.lock").write_text("version = 1\n[[package]]\nname = 'pandas'\n")
    assert tag_of(project) != before


def test_readme_data_and_config_do_not_change_the_tag(project):
    before = tag_of(project)
    (project / "README.md").write_text("notes")
    (project / "data" / "train.csv").write_text("feature,target\n9,9\n")
    (project / "easy_sm.toml").write_text('app_name = "other"\n')
    assert tag_of(project) == before


def test_caches_and_hidden_files_are_left_out(project):
    before = tag_of(project)
    (project / "src" / "__pycache__").mkdir()
    (project / "src" / "__pycache__" / "train.cpython-313.pyc").write_bytes(b"\x00")
    (project / "src" / ".DS_Store").write_bytes(b"\x00")
    (project / "src" / ".env").write_text("SECRET=1")
    assert tag_of(project) == before


def test_renaming_a_file_changes_the_tag(project):
    before = tag_of(project)
    (project / "src" / "predict.py").rename(project / "src" / "serve.py")
    assert tag_of(project) != before


# --- what goes into the build folder ---


def test_build_folder_has_code_runtime_dependencies_and_dockerfile(project):
    names = set(image_files(project))
    assert {"pyproject.toml", "uv.lock", "Dockerfile", "src/train.py", "src/predict.py"} <= names
    assert "runtime/__main__.py" in names
    assert not any(name.startswith("data/") for name in names)


def test_build_folder_matches_the_files(project, tmp_path):
    files = image_files(project)
    dest = tmp_path / "context"
    write_context(files, dest)
    assert (dest / "src" / "train.py").read_bytes() == files["src/train.py"]


def test_own_dockerfile_replaces_the_generated_one(project):
    before = tag_of(project)
    (project / "Dockerfile").write_text("FROM python:3.13\n")
    assert image_files(project)["Dockerfile"] == b"FROM python:3.13\n"
    assert tag_of(project) != before


def test_missing_lock_file_is_an_error(project):
    (project / "uv.lock").unlink()
    with pytest.raises(ImageError, match="uv lock"):
        image_files(project)


# --- Dockerfile and build command ---


def test_python_version_comes_from_requires_python(project):
    assert python_version(project) == "3.13"


def test_python_version_file_wins(project):
    (project / ".python-version").write_text("3.12\n")
    assert python_version(project) == "3.12"


def test_dockerfile_installs_dependencies_before_copying_code():
    text = dockerfile("3.13")
    assert text.startswith("FROM python:3.13-slim")
    assert text.index("uv sync") < text.index("COPY src/")
    assert 'ENTRYPOINT ["python", "-m", "runtime"]' in text


def test_build_command_targets_sagemakers_x86_machines():
    command = build_command(Path("/tmp/ctx"), "churn:abc123")
    assert command[:3] == ["docker", "buildx", "build"]
    assert "linux/amd64" in command
    assert command[-3:] == ["-t", "churn:abc123", "/tmp/ctx"]
