"""`python -m runtime serve`: answer SageMaker's /ping and /invocations with the user's Handler.

The app is plain WSGI (the standard interface between Python web servers and apps),
so it needs no web framework. In the image it runs under gunicorn.
"""

import csv
import io
import json
import os
import sys
import traceback
from collections.abc import Callable, Iterable
from typing import Any

from . import MODEL_DIR, SRC_DIR, load_from

WSGIApp = Callable[[dict[str, Any], Callable[..., Any]], Iterable[bytes]]

STATUS = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}


def make_app(handler: Callable[[Any], Any]) -> WSGIApp:
    """Build the WSGI app around one handler (the user's Handler instance)."""

    def app(environ: dict[str, Any], start_response: Callable[..., Any]) -> Iterable[bytes]:
        method, path = environ["REQUEST_METHOD"], environ["PATH_INFO"]

        if method == "GET" and path == "/ping":  # SageMaker's health check
            return respond(start_response, 200, {"status": "ok"})

        if method == "POST" and path == "/invocations":  # one prediction request
            length = int(environ.get("CONTENT_LENGTH") or 0)
            body = environ["wsgi.input"].read(length)
            content_type = (environ.get("CONTENT_TYPE") or "application/json").split(";")[0]
            try:
                data = decode(body, content_type)
            except ValueError as e:  # bad JSON or an unsupported content type
                return respond(start_response, 400, {"error": str(e)})
            try:
                result = json.dumps(handler(data))
            except Exception:  # noqa: BLE001 - any error in user code must become a 500
                traceback.print_exc(file=sys.stderr)  # the details go to the logs
                return respond(start_response, 500, {"error": "prediction failed, see the endpoint logs"})
            return respond(start_response, 200, result, encoded=True)

        return respond(start_response, 404, {"error": f"no route {method} {path}"})

    return app


def decode(body: bytes, content_type: str) -> Any:
    """Request body → Python data: JSON is parsed, CSV becomes a list of rows of strings."""
    if content_type == "application/json":
        return json.loads(body)  # json.JSONDecodeError is a ValueError
    if content_type == "text/csv":
        return list(csv.reader(io.StringIO(body.decode())))
    raise ValueError(f"unsupported content type '{content_type}', use application/json or text/csv")


def respond(
    start_response: Callable[..., Any], status: int, payload: Any, encoded: bool = False
) -> list[bytes]:
    body = (payload if encoded else json.dumps(payload)).encode()
    start_response(
        f"{status} {STATUS[status]}",
        [("Content-Type", "application/json"), ("Content-Length", str(len(body)))],
    )
    return [body]


def create() -> WSGIApp:
    """Called by gunicorn in each worker: load the model once, then serve requests."""
    handler_class = load_from(SRC_DIR, "predict.py", "Handler")
    return make_app(handler_class(str(MODEL_DIR)))


def gunicorn_command(workers: int) -> list[str]:
    # SageMaker sends requests to port 8080 and gives each one at most 60 seconds.
    return [
        sys.executable, "-m", "gunicorn",
        "--bind", "0.0.0.0:8080",
        "--workers", str(workers),
        "--timeout", "60",
        "runtime.serve:create()",
    ]  # fmt: skip


def serve() -> None:
    """Replace this process with gunicorn, one worker per CPU."""
    command = gunicorn_command(os.cpu_count() or 1)
    os.execvp(command[0], command)
