import os
import shutil
from typing import Annotated

import typer

from easy_sm.commands.helpers import load_config

_FILE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Shell scripts to copy from template to app directory
SHELL_SCRIPTS = [
    "build.sh",
    "push.sh",
    "executor.sh",
    os.path.join("local_test", "train_local.sh"),
    os.path.join("local_test", "process_local.sh"),
    os.path.join("local_test", "deploy_local.sh"),
    os.path.join("local_test", "stop_local.sh"),
]


def update_scripts(
    app_name: Annotated[str, typer.Option("--app-name", "-a", help="App name for configuration")],
) -> None:
    """Update shell scripts in app directory with latest secure versions.

    This command copies the shell scripts from the package template to your
    app's easy_sm_base directory, replacing any existing scripts with the
    latest versions that include security fixes (proper variable quoting).
    """
    config = load_config(app_name)

    template_path = os.path.join(_FILE_DIR_PATH, "..", "template", "easy_sm_base")
    target_path = os.path.join(config.easy_sm_module_dir, "easy_sm_base")

    if not os.path.isdir(target_path):
        raise ValueError(f"easy_sm_base directory not found: {target_path}")

    print(f"Updating shell scripts in {target_path}...\n")

    updated_count = 0
    for script in SHELL_SCRIPTS:
        src = os.path.join(template_path, script)
        dst = os.path.join(target_path, script)

        if not os.path.isfile(src):
            print(f"  Warning: Source script not found: {src}")
            continue

        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)

        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        print(f"  Updated: {script}")
        updated_count += 1

    print(f"\nSuccessfully updated {updated_count} shell script(s).")
