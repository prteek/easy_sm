"""
Tests for the local_test shell scripts.

Runs the real template scripts with a fake `docker` on PATH that records the
arguments it receives, so we can check exactly what would reach `docker run`.
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "easy_sm" / "template" / "easy_sm_base" / "local_test"


@pytest.fixture
def fake_docker(tmp_path: Path) -> Path:
    """Put a fake docker on PATH that writes one argument per line to a file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "docker_args"
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > "{args_file}"\n')
    docker.chmod(0o755)
    return args_file


def run_script(name: str, args: list[str], fake_docker: Path) -> list[str]:
    env = {**os.environ, "PATH": f"{fake_docker.parent / 'bin'}:{os.environ['PATH']}"}
    subprocess.run(["sh", str(SCRIPTS_DIR / name), *args], env=env, check=True)
    return fake_docker.read_text().splitlines()


class TestProcessLocalScript:
    BASE_ARGS = ["/tmp/test_dir", "latest", "img", "job.py", "default", "eu-west-1"]

    def test_without_env_vars(self, fake_docker: Path) -> None:
        args = run_script("process_local.sh", self.BASE_ARGS, fake_docker)

        assert args[-3:] == ["img:latest", "process", "job.py"]
        assert "-e" in args and "AWS_PROFILE=default" in args

    def test_env_values_cannot_inject_docker_flags(self, fake_docker: Path) -> None:
        malicious = "MSG=hello --privileged -v /:/host"
        args = run_script("process_local.sh", [*self.BASE_ARGS, malicious], fake_docker)

        assert malicious in args
        assert "--privileged" not in args
        assert "/:/host" not in args

    def test_env_values_keep_commas_equals_and_empty(self, fake_docker: Path) -> None:
        env = ["URL=a,b", "KEY=sk=1=2", "EMPTY="]
        args = run_script("process_local.sh", [*self.BASE_ARGS, *env], fake_docker)

        for pair in env:
            index = args.index(pair)
            assert args[index - 1] == "-e"


class TestDeployLocalScript:
    def test_port_bound_to_localhost_only(self, fake_docker: Path) -> None:
        args = run_script("deploy_local.sh", ["/tmp/test_dir", "latest", "img", "9000"], fake_docker)

        assert args[args.index("-p") + 1] == "127.0.0.1:9000:8080"
