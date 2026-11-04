"""Serving code.

easy_sm creates Handler once with the folder train() saved the model into,
then calls it for every prediction request.
"""

import json
from pathlib import Path


class Handler:
    def __init__(self, model_dir: str) -> None:
        self.model = json.loads((Path(model_dir) / "model.json").read_text())

    def __call__(self, data):
        # JSON request: the parsed body, e.g. {"inputs": [[1.0], [2.0]]}
        # CSV request: a list of rows
        rows = data["inputs"] if isinstance(data, dict) else data
        return [self.model["mean"] for _ in rows]
