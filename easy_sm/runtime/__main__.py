"""Entry point of the image: `python -m runtime train | serve | run COMMAND...`.

SageMaker starts training containers with the argument `train` and endpoints with `serve`;
easy_sm starts processing jobs with `run ...`.
"""

import sys

from . import FAILURE_FILE, INPUT_DIR, MODEL_DIR, SRC_DIR
from .run import run_command
from .serve import serve
from .train import run_training

USAGE = "usage: python -m runtime train | serve | run COMMAND..."


def main(argv: list[str]) -> int:
    mode, rest = (argv[0], argv[1:]) if argv else ("", [])
    if mode == "train":
        return run_training(SRC_DIR, INPUT_DIR, MODEL_DIR, FAILURE_FILE)
    if mode == "serve":
        serve()  # does not return: the process becomes gunicorn
    if mode == "run":
        try:
            run_command(SRC_DIR, rest)  # does not return: the process becomes the command
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
