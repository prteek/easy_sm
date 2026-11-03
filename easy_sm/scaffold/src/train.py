"""Training code.

easy_sm calls train() with the folder holding your input data and a folder to save
the model into. Everything saved in model_dir is what Handler in predict.py loads.
This sample "model" just remembers the mean of the target column.
"""

import csv
import json
from pathlib import Path


def train(input_dir: str, model_dir: str) -> None:
    with open(Path(input_dir) / "train.csv", newline="") as f:
        targets = [float(row["target"]) for row in csv.DictReader(f)]

    model = {"mean": sum(targets) / len(targets)}
    (Path(model_dir) / "model.json").write_text(json.dumps(model))
