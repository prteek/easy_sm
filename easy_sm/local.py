"""Run the project's image with local Docker, laid out the way SageMaker lays it out.

The *_command functions only build the `docker run` argument list; docker() runs one.
"""

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from easy_sm.image import PLATFORM, ImageError

LOCAL_DIR = ".easy_sm"  # model and outputs of local runs, inside the project


def local_paths(project_dir: Path) -> tuple[Path, Path]:
    """(model dir, output dir) inside .easy_sm/, without creating anything."""
    base = (project_dir / LOCAL_DIR).resolve()
    return base / "model", base / "output"


def local_dirs(project_dir: Path) -> tuple[Path, Path]:
    """Create .easy_sm/ (ignored by git) and return (model dir, output dir)."""
    base = (project_dir / LOCAL_DIR).resolve()
    (base / "model").mkdir(parents=True, exist_ok=True)
    (base / "output").mkdir(parents=True, exist_ok=True)
    (base / ".gitignore").write_text("*\n")  # like .pytest_cache: the folder ignores itself
    return base / "model", base / "output"


def empty(folder: Path) -> None:
    """SageMaker starts every training job with an empty model folder; do the same."""
    shutil.rmtree(folder)
    folder.mkdir()


def docker_run() -> list[str]:
    return ["docker", "run", "--rm", "--platform", PLATFORM]


def train_command(tag: str, data_dir: Path, model_dir: Path, output_dir: Path) -> list[str]:
    return [
        *docker_run(),
        "-v", f"{data_dir.resolve()}:/opt/ml/input/data/training:ro",
        "-v", f"{model_dir.resolve()}:/opt/ml/model",
        "-v", f"{output_dir.resolve()}:/opt/ml/output",
        tag, "train",
    ]  # fmt: skip


def run_command(
    tag: str, data_dir: Path, output_dir: Path, env: dict[str, str], command: list[str]
) -> list[str]:
    env_args: list[str] = []
    for key, value in env.items():
        env_args += ["-e", f"{key}={value}"]  # one argument each: a value can't become a docker flag
    return [
        *docker_run(),
        "-v", f"{data_dir.resolve()}:/opt/ml/processing/input:ro",
        "-v", f"{output_dir.resolve()}:/opt/ml/processing/output",
        *env_args,
        tag, "run", *command,
    ]  # fmt: skip


def serve_command(tag: str, model_dir: Path, port: int) -> list[str]:
    return [
        *docker_run(),
        "-p", f"127.0.0.1:{port}:8080",  # reachable from this machine only
        "-v", f"{model_dir.resolve()}:/opt/ml/model:ro",
        tag, "serve",
    ]  # fmt: skip


def parse_env(pairs: list[str]) -> dict[str, str]:
    """['A=1', 'B=x=y'] → {'A': '1', 'B': 'x=y'}. Values may contain '=' or be empty."""
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--env '{pair}' must look like KEY=VALUE")
        env[key] = value
    return env


def check_script(src_dir: Path, command: list[str]) -> None:
    """Catch a mistyped script name before starting a container (or, later, a cloud machine)."""
    if command and command[0].endswith(".py") and not (src_dir / command[0]).is_file():
        raise ValueError(f"{src_dir / command[0]} not found")


def request_body(data: str) -> bytes:
    """'-d {...}' is sent as is; '-d @file.json' sends the file's contents."""
    if data.startswith("@"):
        return Path(data[1:]).read_bytes()
    return data.encode()


def invoke(port: int, body: bytes, content_type: str) -> tuple[int, bytes]:
    """POST to a local endpoint. Return (HTTP status, response body)."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/invocations",
        data=body,
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise ImageError(
            f"Nothing answering on port {port}. Start it with 'easy_sm endpoints serve --local'."
        ) from e


def docker(command: list[str]) -> int:
    """Run docker with its output on stderr; return the container's exit code."""
    try:
        return subprocess.run(command, stdout=sys.stderr, check=False).returncode
    except FileNotFoundError as e:
        raise ImageError("'docker' is not installed or not on PATH") from e
