"""`python -m runtime run COMMAND...`: run any command from the project's src/ folder."""

import os
import sys
from pathlib import Path


def build_command(args: list[str]) -> list[str]:
    """`clean.py --date x` → `python clean.py --date x`; anything else is left as it is."""
    if not args:
        raise ValueError("run needs a command, e.g. `run clean.py` or `run make features`")
    if args[0].endswith(".py"):
        return [sys.executable, *args]
    return args


def run_command(src_dir: Path, args: list[str]) -> None:
    """Replace this process with the command, so its exit code becomes the job's."""
    command = build_command(args)
    os.chdir(src_dir)
    os.execvp(command[0], command)
