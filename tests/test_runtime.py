"""The runtime runs inside the image; here it is called directly with temporary folders."""

import ast
import io
import json
import sys
from pathlib import Path
from typing import Any
from wsgiref.util import setup_testing_defaults

import pytest

from easy_sm.init import create_project
from easy_sm.runtime import load_from
from easy_sm.runtime.run import build_command
from easy_sm.runtime.serve import make_app
from easy_sm.runtime.train import run_training

RUNTIME_DIR = Path(__file__).parent.parent / "easy_sm" / "runtime"


def call(app: Any, method: str, path: str, body: bytes = b"", content_type: str = "application/json"):
    """Send one request to a WSGI app; return (status line, parsed JSON body)."""
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    setup_testing_defaults(environ)
    statuses: list[str] = []
    response = app(environ, lambda status, headers: statuses.append(status))
    return statuses[0], json.loads(b"".join(response))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    create_project(tmp_path, name="churn")
    return tmp_path


# --- train ---


def test_training_saves_the_model(project):
    code = run_training(project / "src", project / "data", project / "model", project / "failure")
    assert code == 0
    assert json.loads((project / "model" / "model.json").read_text()) == {"mean": 5.05}


def test_training_error_goes_to_the_failure_file_summary_first(project):
    (project / "data" / "train.csv").write_text("feature,target\n")  # no rows → mean of nothing
    code = run_training(project / "src", project / "data", project / "model", project / "failure")
    assert code == 1
    assert (project / "failure").read_text().startswith("ZeroDivisionError: division by zero")


# --- serve ---


def test_ping_answers_ok():
    assert call(make_app(lambda data: None), "GET", "/ping") == ("200 OK", {"status": "ok"})


def test_json_request_reaches_the_handler_parsed():
    app = make_app(lambda data: data["inputs"][0])
    assert call(app, "POST", "/invocations", b'{"inputs": [[1, 2]]}') == ("200 OK", [1, 2])


def test_csv_request_arrives_as_rows_of_strings():
    app = make_app(lambda data: data)
    assert call(app, "POST", "/invocations", b"1,2\n3,4\n", "text/csv") == (
        "200 OK",
        [["1", "2"], ["3", "4"]],
    )


def test_bad_json_is_a_400():
    status, _ = call(make_app(lambda data: data), "POST", "/invocations", b"{not json")
    assert status == "400 Bad Request"


def test_unknown_content_type_is_a_400():
    status, body = call(make_app(lambda data: data), "POST", "/invocations", b"x", "image/png")
    assert status == "400 Bad Request" and "image/png" in body["error"]


def test_unknown_route_is_a_404():
    assert call(make_app(lambda data: data), "GET", "/predict")[0] == "404 Not Found"


def test_handler_error_is_a_500_with_traceback_in_the_logs(capsys):
    def broken(data):
        raise KeyError("inputs")

    status, _ = call(make_app(broken), "POST", "/invocations", b"{}")
    assert status == "500 Internal Server Error"
    assert "KeyError: 'inputs'" in capsys.readouterr().err


def test_result_that_is_not_json_is_a_500(capsys):
    status, _ = call(make_app(lambda data: {1, 2}), "POST", "/invocations", b"{}")  # a set
    assert status == "500 Internal Server Error"
    assert "not JSON serializable" in capsys.readouterr().err


def test_sample_project_trains_then_serves(project):
    run_training(project / "src", project / "data", project / "model", project / "failure")
    handler = load_from(project / "src", "predict.py", "Handler")(str(project / "model"))
    status, body = call(make_app(handler), "POST", "/invocations", b'{"inputs": [[1], [2]]}')
    assert (status, body) == ("200 OK", [5.05, 5.05])


# --- run ---


def test_py_file_runs_with_python():
    assert build_command(["clean.py", "--date", "2026-11-01"]) == [
        sys.executable,
        "clean.py",
        "--date",
        "2026-11-01",
    ]


def test_other_commands_are_unchanged():
    assert build_command(["dbt", "build", "-t", "prod"]) == ["dbt", "build", "-t", "prod"]


def test_empty_command_is_an_error():
    with pytest.raises(ValueError):
        build_command([])


# --- the runtime must work without easy_sm installed ---


def test_runtime_imports_only_the_standard_library():
    for file in RUNTIME_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:  # level > 0: relative
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in sys.stdlib_module_names, f"{file.name} imports {name}"
