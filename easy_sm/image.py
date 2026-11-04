"""Build the project's Docker image, tagged with a hash of everything that goes into it.

The same files feed both the hash and the build, so the tag changes exactly when the
image would: edit src/ or the dependencies and the tag changes; edit a README and it doesn't.
"""

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RUNTIME_DIR = Path(__file__).parent / "runtime"
UV_VERSION = "0.11.1"
GUNICORN_VERSION = "23.0.0"
# SageMaker runs x86 machines; on an ARM Mac Docker emulates this (slower, but it works).
PLATFORM = "linux/amd64"


class ImageError(Exception):
    """The image can't be built."""


def python_version(project_dir: Path) -> str:
    """'3.13' from .python-version, else the first version in requires-python ('>=3.13')."""
    pinned = project_dir / ".python-version"
    if pinned.is_file():
        return pinned.read_text().strip()
    match = re.search(r'requires-python\s*=\s*"[^"\d]*(\d+\.\d+)', _read(project_dir / "pyproject.toml"))
    if not match:
        raise ImageError("Set requires-python in pyproject.toml (e.g. \">=3.13\") or add .python-version")
    return match.group(1)


def dockerfile(python: str) -> str:
    return f"""\
FROM python:{python}-slim
COPY --from=ghcr.io/astral-sh/uv:{UV_VERSION} /uv /bin/uv
ENV PATH=/opt/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /opt/app
# Dependencies first: a change to src/ alone doesn't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \\
 && uv pip install --python /opt/app/.venv/bin/python gunicorn=={GUNICORN_VERSION}
COPY runtime/ runtime/
COPY src/ src/
ENTRYPOINT ["python", "-m", "runtime"]
"""


def image_files(project_dir: Path) -> dict[str, bytes]:
    """Every file the build uses: path in the build folder → contents."""
    if not (project_dir / "uv.lock").is_file():
        raise ImageError(f"{project_dir / 'uv.lock'} not found. Run 'uv lock' or 'easy_sm build'.")
    files = {
        "pyproject.toml": (project_dir / "pyproject.toml").read_bytes(),
        "uv.lock": (project_dir / "uv.lock").read_bytes(),
    }
    files.update(_folder(project_dir / "src", "src"))
    files.update(_folder(RUNTIME_DIR, "runtime"))
    custom = project_dir / "Dockerfile"
    files["Dockerfile"] = (
        custom.read_bytes() if custom.is_file() else dockerfile(python_version(project_dir)).encode()
    )
    return files


def image_hash(files: dict[str, bytes]) -> str:
    """12 hex characters that change whenever any file name or content changes."""
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + files[name] + b"\0")
    return digest.hexdigest()[:12]


def image_tag(project_dir: Path, app_name: str) -> str:
    return f"{app_name}:{image_hash(image_files(project_dir))}"


def write_context(files: dict[str, bytes], dest: Path) -> None:
    for name, content in files.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def build_command(context: Path, tag: str) -> list[str]:
    return ["docker", "buildx", "build", "--platform", PLATFORM, "--load", "-t", tag, str(context)]


def build(project_dir: Path, app_name: str) -> str:
    """Build the image unless an image with this tag already exists. Return the tag."""
    if not (project_dir / "uv.lock").is_file():
        _run(["uv", "lock"], cwd=project_dir)
    files = image_files(project_dir)
    tag = f"{app_name}:{image_hash(files)}"
    if _succeeds(["docker", "image", "inspect", tag]):
        return tag
    with tempfile.TemporaryDirectory() as tmp:
        write_context(files, Path(tmp))
        _run(build_command(Path(tmp), tag))
    return tag


def exists_locally(tag: str) -> bool | None:
    """True if Docker already has this image; None if Docker isn't available to ask."""
    try:
        return _succeeds(["docker", "image", "inspect", tag])
    except ImageError:
        return None


def _folder(folder: Path, prefix: str) -> dict[str, bytes]:
    """Files under folder, skipping caches and hidden files (.DS_Store, .env)."""
    if not folder.is_dir():
        raise ImageError(f"{folder} not found")
    return {
        f"{prefix}/{path.relative_to(folder).as_posix()}": path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not any(part.startswith(".") for part in path.relative_to(folder).parts)
    }


def _read(path: Path) -> str:
    if not path.is_file():
        raise ImageError(f"{path} not found")
    return path.read_text()


def _run(command: list[str], cwd: Path | None = None) -> None:
    """Run a command with its output on stderr, so stdout stays clean for the result."""
    try:
        subprocess.run(command, cwd=cwd, stdout=sys.stderr, check=True)
    except FileNotFoundError as e:
        raise ImageError(f"'{command[0]}' is not installed or not on PATH") from e
    except subprocess.CalledProcessError as e:
        raise ImageError(f"'{' '.join(command[:3])} ...' failed with exit code {e.returncode}") from e


def _succeeds(command: list[str]) -> bool:
    try:
        return subprocess.run(command, capture_output=True, check=False).returncode == 0
    except FileNotFoundError as e:
        raise ImageError(f"'{command[0]}' is not installed or not on PATH") from e
