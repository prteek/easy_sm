import subprocess

def safe_run_subprocess(command, success_message=None):
    """Safely run any subprocesses and print error messages sensibly"""
    try:
        process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,  # Ensure output is in text mode
        )

        # Print output as it's produced
        for line in process.stdout:
            print(line, end='')  # Print each line immediately

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        print(success_message)

    except subprocess.CalledProcessError as e:
        # Surface the error
        print("Error occurred while running the command:")
        print(f"Return code: {e.returncode}")
        print(f"Command: {e.cmd}")
        print("Error output:")
        print(e.output)  # This contains both stdout and stderr

    return return_code