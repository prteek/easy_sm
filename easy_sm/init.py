"""Create a new easy_sm project from the files in easy_sm/scaffold/."""

import re
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from easy_sm.config import ConfigError, check_app_name

PLACEHOLDER = "__APP_NAME__"


def app_name_from_folder(folder: Path) -> str:
    """'My Churn_Model' → 'my-churn-model'."""
    return re.sub(r"[^a-z0-9]+", "-", folder.resolve().name.lower()).strip("-")


def scaffold_files() -> dict[str, str]:
    """Relative path → file contents (still containing the placeholder)."""
    result: dict[str, str] = {}

    def walk(node: Traversable, prefix: str) -> None:
        for child in node.iterdir():
            if child.name == "__pycache__":
                continue
            if child.is_dir():
                walk(child, f"{prefix}{child.name}/")
            else:
                result[f"{prefix}{child.name}"] = child.read_text()

    walk(files("easy_sm") / "scaffold", "")
    return result


def create_project(folder: Path, name: str | None = None) -> list[Path]:
    """Write the scaffold into folder and return the files created.

    Never overwrites: if any file already exists, nothing is written.
    """
    name = name or app_name_from_folder(folder)
    check_app_name(name)

    targets = {folder / rel: text.replace(PLACEHOLDER, name) for rel, text in scaffold_files().items()}
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise ConfigError("Not overwriting existing files: " + ", ".join(sorted(existing)))

    for path, text in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sorted(targets)
