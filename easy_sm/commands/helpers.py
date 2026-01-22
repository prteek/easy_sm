import subprocess
from typing import List, Optional


def safe_run_subprocess(
    command: List[str], success_message: Optional[str] = None
) -> int:
    """Safely run any subprocesses and print error messages sensibly"""
    return_code: int = 0
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        stdout = process.stdout
        if stdout:
            for line in stdout:
                print(line, end="")

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        if success_message:
            print(success_message)

    except subprocess.CalledProcessError as e:
        print("Error occurred while running the command:")
        print(f"Return code: {e.returncode}")
        print(f"Command: {e.cmd}")
        print("Error output:")
        print(e.output)

    return return_code
