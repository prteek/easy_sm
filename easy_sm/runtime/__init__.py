"""Code that runs inside the project's Docker image.

`easy_sm build` copies this folder into the image as the top-level package `runtime`,
and the image starts with `python -m runtime train | serve | run COMMAND...`.

Only the standard library may be imported here (easy_sm itself is not installed in the
image), and imports between these modules must be relative.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

# Folders as SageMaker lays them out inside the container.
APP_DIR = Path(__file__).resolve().parent.parent  # /opt/app in the image
SRC_DIR = APP_DIR / "src"
INPUT_DIR = Path("/opt/ml/input/data/training")  # the "training" input channel
MODEL_DIR = Path("/opt/ml/model")  # SageMaker uploads this as model.tar.gz after training
FAILURE_FILE = Path("/opt/ml/output/failure")  # SageMaker shows this as FailureReason


def load_from(src_dir: Path, filename: str, name: str) -> Any:
    """Import `name` (a function or class) from src_dir/filename.

    src_dir is also put on sys.path, so the user's file can import its neighbours.
    """
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    path = src_dir / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)
