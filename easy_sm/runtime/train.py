"""`python -m runtime train`: call the user's train(input_dir, model_dir)."""

import sys
import traceback
from pathlib import Path

from . import load_from


def run_training(src_dir: Path, input_dir: Path, model_dir: Path, failure_file: Path) -> int:
    """Run src_dir/train.py's train(). Return the exit code: 0 on success, 1 on error."""
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = load_from(src_dir, "train.py", "train")
        train(str(input_dir), str(model_dir))
    except Exception as e:  # noqa: BLE001 - any error in user code must reach the failure file
        # SageMaker shows only the first 1024 characters of the failure file,
        # so the one-line summary goes first and the full traceback after it.
        report = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        sys.stderr.write(report)
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        failure_file.write_text(report)
        return 1
    return 0
